"""Optional pymss catalog, model management and isolated separation runtime."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from omnizart.desktop.optional_models import python_path, ready


def model_root():
    root = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local' / 'share')))
    return root / 'Omnizart' / 'models' / 'pymss'


def _environment():
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYMSS_USER_MODELS'] = str(model_root() / 'user_models.json')
    return env


def _command(*arguments, runtime='pymss'):
    if not ready(runtime):
        raise RuntimeError(f'{runtime} runtime is not installed. Install it in the desktop app.')
    return [str(python_path(runtime)), '-m', 'pymss.cli', *map(str, arguments)]


def _capture(*arguments):
    try:
        result = subprocess.run(_command(*arguments), env=_environment(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('pymss model management timed out.') from exc
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-4000:])
    return result.stdout


def catalog():
    entries = json.loads(_capture('list', '--json'))
    if not isinstance(entries, list):
        raise RuntimeError('pymss returned an invalid model catalog.')
    return [item for item in entries if isinstance(item, dict) and item.get('name') and item.get('supported', True)]


def download(name, source='huggingface'):
    model_root().mkdir(parents=True, exist_ok=True)
    return _capture('download', name, '--model-dir', model_root(), '--source', source)


def register(name, architecture, weights, config=None):
    model_root().mkdir(parents=True, exist_ok=True)
    args = ['register', name, '--type', architecture, '--model', weights]
    if config:
        args.extend(['--config', config])
    return _capture(*args)


def separate(source, output_base, model, selected_stems=None, source_name='huggingface', device='cpu', gpu_index=0):
    """Run pymss and publish selected WAVs without overwriting files."""
    if not model:
        raise ValueError('Choose a pymss model.')
    if device not in ('cpu', 'cuda') or not isinstance(gpu_index, int) or not 0 <= gpu_index <= 99:
        raise ValueError('Choose CPU or CUDA and a GPU index between 0 and 99.')
    runtime = 'pymss CUDA' if device == 'cuda' else 'pymss'
    if device == 'cuda':
        _command(runtime=runtime)  # Check installation before probing.
        probe = subprocess.run([str(python_path(runtime)), '-c',
            "import sys, torch; i=int(sys.argv[1]); assert torch.cuda.is_available(), 'CUDA is unavailable; check the NVIDIA driver'; assert i < torch.cuda.device_count(), 'GPU index is unavailable'; torch.cuda.init(); print(torch.cuda.get_device_name(i))", str(gpu_index)],
            env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace', timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if probe.returncode:
            raise RuntimeError('NVIDIA CUDA could not start: ' + probe.stdout[-2000:])
    source, output_base = Path(source), Path(output_base)
    model_root().mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='omnizart-pymss-', dir=str(output_base.parent)) as temp:
        command = _command('infer', model, '--model-dir', model_root(), '--source', source_name,
                           '-i', source, '-o', temp, '--device', device, '--format', 'wav',
                           *(['--device-id', gpu_index] if device == 'cuda' else []), runtime=runtime)
        try:
            result = subprocess.run(command, env=_environment(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=7200)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('pymss separation timed out.') from exc
        if result.returncode:
            raise RuntimeError(f'pymss separation failed (exit {result.returncode}); see the log.')
        prefix = source.stem + '_'
        found = {}
        for path in Path(temp).glob('*.wav'):
            if not path.stem.startswith(prefix):
                continue
            name = re.sub(r'[^a-z0-9_-]+', '_', path.stem[len(prefix):].lower()).strip('_')
            if not name or name in found:
                raise ValueError('pymss produced ambiguous stem names.')
            found[name] = path
        wanted = list(selected_stems) if selected_stems else list(found)
        if len(set(wanted)) != len(wanted) or any(not re.fullmatch(r'[a-z0-9_-]+', name) for name in wanted):
            raise ValueError('Invalid or repeated stem name.')
        if not wanted or any(name not in found for name in wanted):
            raise ValueError('The selected model did not produce the requested stem. Available: ' + ', '.join(sorted(found)))
        targets = [output_base.with_name(output_base.name + '_' + stem + '.wav') for stem in wanted]
        if any(path.exists() for path in targets):
            raise FileExistsError('A selected stem output already exists.')
        published = []
        try:
            for stem, target in zip(wanted, targets):
                with target.open('xb') as out, found[stem].open('rb') as inp:
                    published.append(target)
                    shutil.copyfileobj(inp, out)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return targets

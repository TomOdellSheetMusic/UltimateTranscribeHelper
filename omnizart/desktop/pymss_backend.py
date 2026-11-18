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


def _command(*arguments):
    if not ready('pymss'):
        raise RuntimeError('pymss is not installed. Use Browse pymss models to install it.')
    return [str(python_path('pymss')), '-m', 'pymss.cli', *map(str, arguments)]


def _capture(*arguments):
    result = subprocess.run(_command(*arguments), env=_environment(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=3600)
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


def separate(source, output_base, model, selected_stems=None, source_name='huggingface'):
    """Run pymss on CPU, then publish selected WAVs without overwriting files."""
    if not model:
        raise ValueError('Choose a pymss model.')
    source, output_base = Path(source), Path(output_base)
    model_root().mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='omnizart-pymss-', dir=str(output_base.parent)) as temp:
        command = _command('infer', model, '--model-dir', model_root(), '--source', source_name,
                           '-i', source, '-o', temp, '--device', 'cpu', '--format', 'wav')
        result = subprocess.run(command, env=_environment(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=7200)
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

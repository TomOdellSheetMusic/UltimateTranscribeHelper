"""Optional ROSVOT singing transcription, using the bundled CPU PyTorch runtime."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import urllib.request
import zipfile

_BASE = 'https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess/resolve/main/rosvot/'
_FILES = {
    'rosvot/config.yaml': '6bccb2becf6c69dc7b60ecf4b4253854a70fda9b30d63838cd300b18da74abb4',
    'rwbd/config.yaml': '3bb41f1d9eaa85aa1b3e5b6d94fff4ab4affb39719028ab69b4505974b9a1bc7',
    'rosvot/model.pt': '7501fb5f913d971c2f51bcb3063b930027b03206581820a4d2bfdc394c9c3fcb',
    'rwbd/model.pt': '0bc2d42a6d4b7a05436deb937e2deda1c12de49e5687cfda0bdf6a430120dcd2',
    'rmvpe/model.pt': '19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2',
}


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _root():
    return Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share')) / 'Omnizart' / 'optional' / 'rosvot'


def _setup(root):
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.ready-v2'
    if not marker.is_file():
        print('Preparing ROSVOT (first use downloads approximately 630 MB)...', flush=True)
        archive = Path(__file__).parent / 'assets' / 'rosvot-3c8332b.zip'
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                target = (root / info.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Invalid path in bundled ROSVOT archive')
            zipped.extractall(root)
        source = root / 'inference' / 'rosvot.py'
        text = source.read_text(encoding='utf-8')
        device = 'torch.device(f"cuda:{int(rank)}")'
        transfer = 'batch = move_to_cuda(batch, int(rank))'
        if device not in text or transfer not in text:
            raise RuntimeError('Unsupported ROSVOT source revision.')
        text = text.replace(device, 'torch.device("cuda:0" if os.environ.get("OMNIZART_ROSVOT_DEVICE") == "cuda" else "cpu")')
        text = text.replace(transfer, 'batch = move_to_cuda(batch, int(rank)) if device.type == "cuda" else batch')
        source.write_text(text, encoding='utf-8')
        python = Path(sys.executable)
        if python.name.lower() == 'pythonw.exe':
            python = python.with_name('python.exe')
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
                        '--target', str(root / 'dependencies'), 'pyworld==0.3.5', 'einops==0.8.2',
                        'pyloudnorm==0.2.0'], check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        marker.write_text('2', encoding='utf-8')
    for name, expected in _FILES.items():
        path = root / 'checkpoints' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and _hash(path) == expected:
            continue
        print('Downloading ROSVOT ' + name + '...', flush=True)
        partial = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(_BASE + name, timeout=120) as response, partial.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if _hash(partial) != expected:
                raise ValueError('ROSVOT download checksum mismatch: ' + name)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    config = root / 'checkpoints' / 'rosvot' / 'config.yaml'
    effective_config = config.with_name('omnizart-inference.yaml')
    effective_config.write_text(config.read_text(encoding='utf-8').replace(
        'pretrained_models/rosvot/rmvpe/model.pt', 'checkpoints/rmvpe/model.pt'), encoding='utf-8')
    return effective_config


def transcribe(vocal_audio, device='cpu', gpu_index=0):
    import pretty_midi
    root = _root().resolve()
    config = _setup(root)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'dependencies')))
    if device not in ('cpu', 'cuda'):
        raise ValueError('ROSVOT device must be cpu or cuda.')
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_index) if device == 'cuda' else ''
    env['OMNIZART_ROSVOT_DEVICE'] = device
    env['PYTHONUTF8'] = '1'
    python = Path(sys.executable)
    if python.name.lower() == 'pythonw.exe':
        python = python.with_name('python.exe')
    launcher = []
    if device == 'cuda':
        from omnizart.desktop.optional_models import install, python_path, ready
        if not ready('pymss CUDA'):
            print('Installing the shared NVIDIA CUDA runtime for ROSVOT...', flush=True)
            install('pymss CUDA')
        python = python_path('pymss CUDA')
        # Append the bundled scientific libraries after the CUDA venv's packages,
        # so the venv's CUDA PyTorch takes precedence over the bundled CPU one.
        launcher = ['-c', 'import sys,runpy; sys.path.append(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name="__main__")',
                    sysconfig.get_path('purelib')]
    with TemporaryDirectory(prefix='omnizart-rosvot-') as temporary:
        print('Transcribing separated singing with ROSVOT on ' + ('NVIDIA GPU' if device == 'cuda' else 'CPU') + '...', flush=True)
        command = [str(python), *launcher, str(root / 'inference' / 'rosvot.py'), '-o', temporary,
                   '-p', str(Path(vocal_audio).resolve()), '--config', str(config), '--sync_saving']
        result = subprocess.run(command, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError('ROSVOT transcription failed: ' + result.stdout[-4000:])
        outputs = list(Path(temporary).rglob('*.mid'))
        if len(outputs) != 1:
            raise RuntimeError('ROSVOT did not produce a single MIDI file: ' + result.stdout[-2000:])
        return pretty_midi.PrettyMIDI(str(outputs[0]))

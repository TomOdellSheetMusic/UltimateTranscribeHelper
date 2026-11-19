"""Run the optional Beat This! model in its isolated environment."""
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from omnizart.desktop.optional_models import environment, python_path, ready


def detect(audio):
    if not ready('Beat This'):
        raise RuntimeError('Beat This is not installed. Enable audio bar alignment in the desktop app to set it up.')
    with TemporaryDirectory(prefix='omnizart-beats-') as folder:
        output = Path(folder) / 'beats.json'
        worker = Path(__file__).with_name('beat_worker.py')
        model = environment('Beat This') / 'model.ckpt'
        result = subprocess.run([str(python_path('Beat This')), str(worker), str(audio), str(model), str(output)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', timeout=3600,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or not output.is_file():
            raise RuntimeError('Beat This failed: ' + result.stdout[-2500:])
        values = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(values, dict) or not isinstance(values.get('beats'), list) or not isinstance(values.get('downbeats'), list):
            raise RuntimeError('Beat This returned invalid beat data.')
        if (_irregular(values['beats']) or not values['downbeats']) and values.get('beat_probabilities') and values.get('downbeat_probabilities'):
            decoded = Path(folder) / 'decoded.json'
            executable = Path(sys.executable)
            if executable.name.lower() == 'pythonw.exe':
                executable = executable.with_name('python.exe')
            worker = Path(__file__).with_name('beat_dbn_worker.py')
            try:
                result = subprocess.run([str(executable), str(worker), str(output), str(decoded)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                        errors='replace', timeout=300, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode == 0 and decoded.is_file():
                    candidate = json.loads(decoded.read_text(encoding='utf-8'))
                    if (isinstance(candidate, dict) and isinstance(candidate.get('beats'), list) and isinstance(candidate.get('downbeats'), list)
                            and not _irregular(candidate['beats']) and candidate['downbeats']):
                        print('Refined irregular beat detections using the rhythm decoder.', flush=True)
                        values = candidate
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass  # The score layer will explicitly warn and use an estimated grid.
        return values['beats'], values['downbeats']


def _irregular(beats):
    try:
        beats = [float(value) for value in beats]
        if any(not math.isfinite(value) or value < 0 for value in beats):
            return True
        steps = [b - a for a, b in zip(beats, beats[1:])]
        if len(steps) < 3 or any(not math.isfinite(step) or step <= 0 or step < .2 or step > 2 for step in steps):
            return True
        typical = statistics.median(steps)
        return sum(not .55 * typical <= step <= 1.8 * typical for step in steps) > len(steps) * .15
    except (TypeError, ValueError):
        return True

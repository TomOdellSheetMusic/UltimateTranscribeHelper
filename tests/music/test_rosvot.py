from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pretty_midi
import numpy as np
import soundfile as sf
from omnizart.desktop import rosvot


class RosvotTests(TestCase):
    def test_cuda_uses_cuda_runtime_and_keeps_base_packages_after_it(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sf.write(str(root / 'voice.wav'), np.zeros(800, dtype=np.float32), 8000)
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                midi = pretty_midi.PrettyMIDI()
                track = pretty_midi.Instrument(0)
                track.notes.append(pretty_midi.Note(80, 60, 0, 1))
                midi.instruments.append(track)
                result_dir = Path(command[command.index('-o') + 1])
                result_dir.mkdir()
                midi.write(str(result_dir / 'output.mid'))
                return type('Result', (), {'returncode': 0, 'stdout': ''})()
            with patch.object(rosvot, '_root', return_value=root), \
                 patch.object(rosvot, '_setup', return_value=root / 'config.yaml'), \
                 patch('omnizart.desktop.optional_models.ready', return_value=True), \
                 patch('omnizart.desktop.optional_models.python_path', return_value=root / 'cuda-python.exe'), \
                 patch.object(rosvot.subprocess, 'run', side_effect=run):
                midi = rosvot.transcribe(root / 'voice.wav', device='cuda', gpu_index=2)
            self.assertEqual(len(midi.instruments[0].notes), 1)
            command, kwargs = calls[0]
            self.assertEqual(command[0], str(root / 'cuda-python.exe'))
            self.assertIn('sys.path.append', command[2])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(kwargs['env']['OMNIZART_ROSVOT_DEVICE'], 'cuda')

    def test_missing_midi_reports_model_failure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sf.write(str(root / 'voice.wav'), np.zeros(800, dtype=np.float32), 8000)
            result = type('Result', (), {'returncode': 0, 'stdout': 'no pitched notes'})()
            with patch.object(rosvot, '_root', return_value=root), \
                 patch.object(rosvot, '_setup', return_value=root / 'config.yaml'), \
                 patch.object(rosvot.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(ValueError, 'found no pitched vocal notes'):
                    rosvot.transcribe(root / 'voice.wav')

    def test_long_audio_uses_overlapping_windows_without_duplicate_notes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sf.write(str(root / 'voice.wav'), np.zeros(125 * 8000, dtype=np.float32), 8000)
            windows = []
            def run(command, **kwargs):
                index = len(windows)
                segment = Path(command[command.index('-p') + 1])
                start = max(0, index * 60 - 6)
                seconds = sf.info(str(segment)).duration
                windows.append((start, seconds))
                midi = pretty_midi.PrettyMIDI()
                track = pretty_midi.Instrument(0)
                for pitch, begin, end in ((60, 10, 11), (62, 58, 62), (64, 122, 123)):
                    if start <= begin and end <= start + seconds:
                        track.notes.append(pretty_midi.Note(80, pitch, begin - start, end - start))
                midi.instruments.append(track)
                result_dir = Path(command[command.index('-o') + 1])
                result_dir.mkdir()
                midi.write(str(result_dir / 'output.mid'))
                return type('Result', (), {'returncode': 0, 'stdout': ''})()
            with patch.object(rosvot, '_root', return_value=root), \
                 patch.object(rosvot, '_setup', return_value=root / 'config.yaml'), \
                 patch.object(rosvot.subprocess, 'run', side_effect=run):
                result = rosvot.transcribe(root / 'voice.wav')
            self.assertEqual(windows, [(0, 66), (54, 71), (114, 11)])
            self.assertEqual([(n.pitch, round(n.start), round(n.end)) for n in result.instruments[0].notes],
                             [(60, 10, 11), (62, 58, 62), (64, 122, 123)])

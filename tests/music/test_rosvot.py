from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pretty_midi
from omnizart.desktop import rosvot


class RosvotTests(TestCase):
    def test_cuda_uses_cuda_runtime_and_keeps_base_packages_after_it(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                midi = pretty_midi.PrettyMIDI()
                track = pretty_midi.Instrument(0)
                track.notes.append(pretty_midi.Note(80, 60, 0, 1))
                midi.instruments.append(track)
                midi.write(str(Path(command[command.index('-o') + 1]) / 'output.mid'))
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
            result = type('Result', (), {'returncode': 0, 'stdout': 'no pitched notes'})()
            with patch.object(rosvot, '_root', return_value=root), \
                 patch.object(rosvot, '_setup', return_value=root / 'config.yaml'), \
                 patch.object(rosvot.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'did not produce'):
                    rosvot.transcribe(root / 'voice.wav')

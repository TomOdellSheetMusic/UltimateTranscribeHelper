from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import subprocess
from unittest.mock import patch

from click.testing import CliRunner
from omnizart.cli.stems import stems
from omnizart.desktop import pymss_backend


class PymssTests(unittest.TestCase):
    def test_cuda_probe_and_device_are_forwarded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'song.wav'
            source.touch()
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if '-o' in command:
                    (Path(command[command.index('-o') + 1]) / 'song_piano.wav').write_bytes(b'piano')
                return SimpleNamespace(returncode=0, stdout='GPU')
            with patch.object(pymss_backend, 'ready', return_value=True), \
                 patch.object(pymss_backend, 'python_path', return_value=Path('python')), \
                 patch.object(pymss_backend, 'model_root', return_value=root / 'models'), \
                 patch.object(pymss_backend.subprocess, 'run', side_effect=run):
                pymss_backend.separate(source, root / 'out', 'model', ['piano'], device='cuda', gpu_index=1)
            self.assertEqual(commands[-1][commands[-1].index('--device') + 1], 'cuda')
            self.assertEqual(commands[-1][commands[-1].index('--device-id') + 1], '1')
            self.assertIn('torch.cuda.is_available()', commands[0][2])

    def test_cuda_failure_is_not_silently_run_on_cpu(self):
        with TemporaryDirectory() as directory, \
             patch.object(pymss_backend, 'ready', return_value=True), \
             patch.object(pymss_backend.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='no GPU')) as run:
            with self.assertRaisesRegex(RuntimeError, 'no GPU'):
                pymss_backend.separate(Path(directory) / 'song.wav', Path(directory) / 'out', 'model', device='cuda')
            self.assertEqual(run.call_count, 1)

    def test_timeout_is_reported_as_separation_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'song.wav'
            source.touch()
            with patch.object(pymss_backend, 'ready', return_value=True), \
                 patch.object(pymss_backend, 'python_path', return_value=Path('python')), \
                 patch.object(pymss_backend, 'model_root', return_value=root / 'models'), \
                 patch.object(pymss_backend.subprocess, 'run', side_effect=subprocess.TimeoutExpired('pymss', 7200)):
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    pymss_backend.separate(source, root / 'out', 'model')

    def test_selected_stem_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'song.wav'
            source.touch()
            def run(command, **kwargs):
                out = Path(command[command.index('-o') + 1])
                (out / 'song_piano.wav').write_bytes(b'piano')
                (out / 'song_drums.wav').write_bytes(b'drums')
                return SimpleNamespace(returncode=0)
            with patch.object(pymss_backend, 'ready', return_value=True), \
                 patch.object(pymss_backend, 'python_path', return_value=Path('python')), \
                 patch.object(pymss_backend, 'model_root', return_value=root / 'models'), \
                 patch.object(pymss_backend.subprocess, 'run', side_effect=run):
                result = pymss_backend.separate(source, root / 'out', 'model', ['piano'])
                self.assertEqual([p.name for p in result], ['out_piano.wav'])
                self.assertEqual(result[0].read_bytes(), b'piano')
                self.assertFalse((root / 'out_drums.wav').exists())
                with self.assertRaises(FileExistsError):
                    pymss_backend.separate(source, root / 'out', 'model', ['piano'])
                with self.assertRaisesRegex(ValueError, 'Available'):
                    pymss_backend.separate(source, root / 'out2', 'model', ['vocals'])
                all_outputs = pymss_backend.separate(source, root / 'all', 'model')
                self.assertEqual({p.name for p in all_outputs}, {'all_piano.wav', 'all_drums.wav'})

    def test_cli_routes_pymss_model_and_stem(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'song.wav'
            source.touch()
            with patch.object(pymss_backend, 'separate') as separate:
                result = CliRunner().invoke(stems, ['transcribe', str(source), '--output', str(Path(directory) / 'out'),
                    '--backend', 'pymss', '--pymss-model', 'my-model', '--stem', 'piano'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(separate.call_args.kwargs['selected_stems'], ('piano',))


if __name__ == '__main__':
    unittest.main()

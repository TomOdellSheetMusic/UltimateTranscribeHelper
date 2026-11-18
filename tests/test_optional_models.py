import tempfile
import unittest
import io
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omnizart.desktop import muscriptor, optional_models, yourmt3


class OptionalModelsTest(unittest.TestCase):
    def test_muscriptor_requires_companion_config(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(muscriptor, "model_folder", return_value=Path(directory)):
            weights = Path(directory) / "model.safetensors"
            weights.touch()
            self.assertIsNone(muscriptor.model_checkpoint())
            (Path(directory) / "config.json").touch()
            self.assertEqual(muscriptor.model_checkpoint(), weights)
            (Path(directory) / "other.safetensors").touch()
            self.assertIsNone(muscriptor.model_checkpoint())

    def test_failed_install_never_marks_environment_ready(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, "environment", return_value=Path(directory)), patch.object(optional_models.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stdout = "installation failed"
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                optional_models.install("YourMT3")
            self.assertFalse(optional_models.ready("YourMT3"))

    def test_yourmt3_repairs_old_environment_and_pins_transformers(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, "environment", return_value=Path(directory)), patch.object(optional_models.subprocess, "run") as run:
            python = optional_models.python_path("YourMT3")
            python.parent.mkdir(parents=True)
            python.touch()
            marker = Path(directory) / ".ready"
            marker.write_text("ready", encoding="utf-8")
            self.assertFalse(optional_models.ready("YourMT3"))
            run.return_value.returncode = 0
            optional_models.install("YourMT3")
            self.assertIn("transformers==4.43.4", run.call_args_list[1].args[0])
            self.assertTrue(optional_models.ready("YourMT3"))

    def test_neural_piano_ready_requires_verified_checkpoint(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, 'environment', return_value=Path(directory)):
            python = optional_models.python_path('Piano SVSep')
            python.parent.mkdir(parents=True)
            python.touch()
            (Path(directory) / '.ready').write_text('2', encoding='utf-8')
            (Path(directory) / '.runtime-source').write_text(optional_models._runtime_identity(), encoding='utf-8')
            checkpoint = Path(directory) / 'model.ckpt'
            checkpoint.write_bytes(b'invalid')
            self.assertFalse(optional_models.ready('Piano SVSep'))
            with patch.object(optional_models, 'SVSEP_SHA256', hashlib.sha256(b'invalid').hexdigest()):
                self.assertTrue(optional_models.ready('Piano SVSep'))
                (Path(directory) / '.runtime-source').write_text('old installation', encoding='utf-8')
                self.assertFalse(optional_models.ready('Piano SVSep'))

    def test_beat_model_ready_requires_verified_checkpoint(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, 'environment', return_value=Path(directory)):
            python = optional_models.python_path('Beat This')
            python.parent.mkdir(parents=True)
            python.touch()
            (Path(directory) / '.ready').write_text('2', encoding='utf-8')
            (Path(directory) / '.runtime-source').write_text(optional_models._runtime_identity(), encoding='utf-8')
            (Path(directory) / 'model.ckpt').write_bytes(b'bad')
            self.assertFalse(optional_models.ready('Beat This'))
            with patch.object(optional_models, 'BEAT_SHA256', hashlib.sha256(b'bad').hexdigest()):
                self.assertTrue(optional_models.ready('Beat This'))

    def test_score_backends_install_isolated_cpu_torch_and_repair_old_environments(self):
        for model, checksum in [('Piano SVSep', 'SVSEP_SHA256'), ('Beat This', 'BEAT_SHA256')]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                python = folder / ('Scripts/python.exe' if optional_models.os.name == 'nt' else 'bin/python')
                python.parent.mkdir(parents=True)
                python.touch()
                (folder / '.ready').write_text('1', encoding='utf-8')
                (folder / 'model.ckpt').write_bytes(b'checkpoint')
                with patch.object(optional_models, 'environment', return_value=folder), \
                     patch.object(optional_models, checksum, hashlib.sha256(b'checkpoint').hexdigest()), \
                     patch.object(optional_models.urllib.request, 'urlopen', return_value=io.BytesIO(b'checkpoint')), \
                     patch.object(optional_models, '_verify_svsep') as verify_model, \
                     patch.object(optional_models.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='')) as run:
                    self.assertFalse(optional_models.ready(model))
                    optional_models.install(model)
                    self.assertTrue(optional_models.ready(model))
                    venv, torch_install, dependencies, verify = [call.args[0] for call in run.call_args_list]
                    self.assertIn('venv', venv)
                    self.assertNotIn('--system-site-packages', venv)
                    self.assertIn('torch==2.5.1+cpu', torch_install)
                    self.assertIn('https://download.pytorch.org/whl/cpu', torch_install)
                    self.assertEqual('torchaudio==2.5.1+cpu' in torch_install, model == 'Beat This')
                    self.assertIn('install', dependencies)
                    self.assertEqual('pandas==2.2.3' in dependencies, model == 'Piano SVSep')
                    self.assertIn('torch.version.cuda is None', verify[-1])
                    self.assertEqual(verify_model.call_count, int(model == 'Piano SVSep'))

    def test_failed_model_smoke_test_does_not_mark_ready(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, 'environment', return_value=Path(directory)), \
             patch.object(optional_models, 'SVSEP_SHA256', hashlib.sha256(b'checkpoint').hexdigest()), \
             patch.object(optional_models.urllib.request, 'urlopen', return_value=io.BytesIO(b'checkpoint')), \
             patch.object(optional_models.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='')), \
             patch.object(optional_models, '_verify_svsep', side_effect=RuntimeError('worker failed')):
            (Path(directory) / '.ready').write_text('1', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                optional_models.install('Piano SVSep')
            self.assertFalse((Path(directory) / '.ready').exists())

    def test_failed_cpu_torch_install_does_not_mark_ready(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(optional_models, 'environment', return_value=Path(directory)), \
             patch.object(optional_models.subprocess, 'run', side_effect=[SimpleNamespace(returncode=0, stdout=''),
                                                                           SimpleNamespace(returncode=1, stdout='torch failed')]):
            (Path(directory) / '.ready').write_text('1', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'torch failed'):
                optional_models.install('Piano SVSep')
            self.assertFalse((Path(directory) / '.ready').exists())

    def test_yourmt3_uses_isolated_interpreter(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(yourmt3, "ready", return_value=True), patch.object(yourmt3, "python_path", return_value=Path("isolated-python")), patch.object(yourmt3.subprocess, "run") as run:
            target = Path(directory) / "out.mid"
            target.touch()
            run.return_value.returncode = 0
            yourmt3.transcribe("input.wav", target)
            self.assertEqual(run.call_args.args[0][:4], ["isolated-python", "-m", "mt3_infer.cli", "transcribe"])


if __name__ == "__main__":
    unittest.main()

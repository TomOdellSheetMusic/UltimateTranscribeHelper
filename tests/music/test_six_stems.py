"""Six-stem separation contracts, using tiny synthetic inputs and a fake model."""
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from click.testing import CliRunner
from omnizart.desktop import six_stems
from omnizart.cli.stems import stems


class FakeSession:
    def __init__(self, *args, **kwargs):
        pass
    def run(self, outputs, inputs):
        mix = inputs["mix"]
        return [np.stack([mix * (index + 1) for index in range(6)], axis=1)]


class SixStemTests(unittest.TestCase):
    def test_selected_stems_and_custom_model(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            source, model = folder / 'song.wav', folder / 'custom.onnx'
            sf.write(str(source), np.zeros((8, 2), dtype=np.float32), 44100)
            model.touch()
            fake_ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=FakeSession)
            with patch.object(six_stems, 'SEGMENT_SAMPLES', 8), \
                 patch.dict('sys.modules', {'onnxruntime': fake_ort, 'librosa': types.SimpleNamespace()}):
                outputs = six_stems.separate(source, folder / 'out', selected_stems=['piano'], custom_model=model)
            self.assertEqual([p.name for p in outputs], ['out_piano.wav'])
            self.assertFalse((folder / 'out_vocals.wav').exists())

    def test_verified_model_download_and_invalid_import(self):
        data = b"model"
        with TemporaryDirectory() as folder:
            target = Path(folder) / "model.onnx"
            with patch.object(six_stems, "model_path", return_value=target), \
                 patch.object(six_stems, "MODEL_SIZE", len(data)), \
                 patch.object(six_stems, "MODEL_SHA256", hashlib.sha256(data).hexdigest()), \
                 patch.object(six_stems.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                self.assertEqual(six_stems.install_model(), target)
                self.assertTrue(six_stems.verify_model(target))
                bad = Path(folder) / "bad.onnx"
                bad.write_bytes(b"wrong")
                with self.assertRaises(ValueError):
                    six_stems.install_model(bad)
                self.assertEqual(target.read_bytes(), data)

    def test_chunk_reconstruction_and_six_distinct_outputs(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "song.wav"
            audio = np.linspace(-0.1, 0.1, 19, dtype=np.float32)
            sf.write(str(source), np.column_stack((audio, audio)), 44100, subtype="FLOAT")
            fake_ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(),
                                             InferenceSession=FakeSession)
            with patch.object(six_stems, "verify_model", return_value=True), \
                 patch.object(six_stems, "SEGMENT_SAMPLES", 8), \
                 patch.dict("sys.modules", {"onnxruntime": fake_ort, "librosa": types.SimpleNamespace()}):
                paths = six_stems.separate(source, folder / "out")
            self.assertEqual([path.name for path in paths], [f"out_{name}.wav" for name in six_stems.STEMS])
            for index, path in enumerate(paths):
                result, rate = sf.read(str(path), dtype="float32", always_2d=True)
                self.assertEqual(rate, 44100)
                np.testing.assert_allclose(result[:, 0], audio * (index + 1), atol=1e-6)
            with patch.object(six_stems, "verify_model", return_value=True), \
                 patch.dict("sys.modules", {"onnxruntime": fake_ort, "librosa": types.SimpleNamespace()}):
                with self.assertRaises(FileExistsError):
                    six_stems.separate(source, folder / "out")

    def test_cli_delegates_device_and_basename(self):
        with TemporaryDirectory() as folder:
            audio = Path(folder) / "song.wav"
            audio.touch()
            with patch.object(six_stems, "separate") as separator:
                result = CliRunner().invoke(stems, ["transcribe", str(audio), "--output", str(Path(folder) / "out"),
                                                    "--device", "directml", "--gpu-index", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(separator.call_args.kwargs, {"device": "directml", "gpu_index": 1})

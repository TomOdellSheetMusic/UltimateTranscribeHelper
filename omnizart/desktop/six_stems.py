"""Six-stem HT-Demucs ONNX inference; see DEMUCS-NOTICES.md."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
from contextlib import ExitStack

STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
MODEL_NAME = "htdemucs_6s_fp16weights.onnx"
MODEL_SIZE = 136_428_532
MODEL_SHA256 = "7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869"
MODEL_URL = ("https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/"
             "49df9b6989cf2150840ea65b0bef77a2e471b678/" + MODEL_NAME)
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 343980


def model_path():
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
    return root / "Omnizart" / "models" / "six-stem" / MODEL_NAME


def verify_model(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256


def install_model(source=None):
    """Import or download only the pinned model, without replacing good files on failure."""
    target = model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="six-stem-", suffix=".partial", dir=str(target.parent))
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "wb") as output:
            input_stream = urllib.request.urlopen(MODEL_URL, timeout=60) if source is None else Path(source).open("rb")
            with input_stream:
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output.write(chunk)
                    if output.tell() > MODEL_SIZE:
                        raise ValueError("The file exceeds the expected size of the six-stem model.")
        if not verify_model(temporary):
            raise ValueError("The file does not match the official six-stem model (size or SHA-256).")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def separate(source, output_base, device="cpu", gpu_index=0, selected_stems=None, custom_model=None):
    """Write six WAV stems to distinct files beside output_base.

    The model runs in overlapping windows. The source is resampled to 44.1 kHz;
    WAVs are stereo float32 at that rate. No external FFmpeg is needed.
    """
    import librosa
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    selected = tuple(STEMS if selected_stems is None else selected_stems)
    if not selected or len(set(selected)) != len(selected) or any(stem not in STEMS for stem in selected):
        raise ValueError("Choose at least one valid six-stem output without duplicates.")
    checkpoint = Path(custom_model) if custom_model else model_path()
    if custom_model:
        if not checkpoint.is_file() or checkpoint.suffix.lower() != '.onnx':
            raise FileNotFoundError('Select an existing compatible six-stem ONNX model.')
    elif not verify_model(checkpoint):
        raise FileNotFoundError("Six-stem model missing or invalid. Open Models & import to download or import it.")
    output_base = Path(output_base)
    targets = [output_base.with_name(output_base.name + "_" + stem + ".wav") for stem in selected]
    if any(path.exists() for path in targets):
        raise FileExistsError("A stem output already exists; choose a new output name.")
    try:
        audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        decoded, rate = librosa.load(str(source), sr=None, mono=False, dtype="float32")
        audio = decoded[:, None] if decoded.ndim == 1 else decoded.T
    if not len(audio):
        raise ValueError("The input audio contains no samples.")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    else:
        audio = audio[:, :2]
    mixture = audio.T
    if rate != SAMPLE_RATE:
        mixture = librosa.resample(mixture, orig_sr=rate, target_sr=SAMPLE_RATE, axis=-1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    if device == "directml":
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML GPU support is unavailable. Choose CPU.")
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": gpu_index}), "CPUExecutionProvider"]
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError("Unknown separation device.")
    try:
        session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=providers)
    except Exception as exc:
        raise RuntimeError(f'Could not load the selected six-stem ONNX model: {exc}') from exc
    if device == "directml":
        session.disable_fallback()
        if "DmlExecutionProvider" not in session.get_providers():
            raise RuntimeError("DirectML GPU initialization failed. Check the GPU index or choose CPU.")
    total = mixture.shape[1]
    overlap = SEGMENT_SAMPLES // 4
    stride = SEGMENT_SAMPLES - overlap
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    accum = np.zeros((len(STEMS), 2, SEGMENT_SAMPLES), dtype=np.float32)
    weights = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    print(f"Separating six stems on {'DirectML GPU ' + str(gpu_index) if device == 'directml' else 'CPU'}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="omnizart-stems-", dir=str(output_base.parent)) as temporary:
        files = [Path(temporary) / (stem + ".wav") for stem in selected]
        with ExitStack() as stack:
            writers = [stack.enter_context(sf.SoundFile(str(path), mode="w", samplerate=SAMPLE_RATE,
                                                       channels=2, subtype="FLOAT")) for path in files]
            starts = range(0, total, stride)
            for number, start in enumerate(starts, 1):
                length = min(SEGMENT_SAMPLES, total - start)
                chunk = np.zeros((1, 2, SEGMENT_SAMPLES), dtype=np.float32)
                chunk[0, :, :length] = mixture[:, start:start + length]
                try:
                    prediction = session.run(["stems"], {"mix": chunk})[0]
                except Exception as exc:
                    raise RuntimeError(f'The model is not compatible with the six-stem ONNX interface: {exc}') from exc
                if prediction.shape != (1, len(STEMS), 2, SEGMENT_SAMPLES) or not np.isfinite(prediction).all():
                    raise RuntimeError("Six-stem model returned invalid audio.")
                window = np.ones(length, dtype=np.float32)
                if start:
                    n = min(overlap, length)
                    window[:n] = fade[:n]
                if start + length < total:
                    n = min(overlap, length)
                    window[-n:] = fade[-n:][::-1]
                accum[:, :, :length] += prediction[0, :, :, :length] * window
                weights[:length] += window
                emitted = min(stride, total - start)
                if np.any(weights[:emitted] <= 0):
                    raise RuntimeError("Six-stem audio reconstruction has an uncovered section.")
                block = accum[:, :, :emitted] / weights[None, None, :emitted]
                for stem, writer in zip(selected, writers):
                    writer.write(block[STEMS.index(stem)].T)
                accum[:, :, :-stride] = accum[:, :, stride:].copy()
                accum[:, :, -stride:] = 0
                weights[:-stride] = weights[stride:].copy()
                weights[-stride:] = 0
                print(f"Six-stem progress: {min(start + emitted, total) / total:.0%}", flush=True)
        published = []
        try:
            for src, dst in zip(files, targets):
                with dst.open("xb") as output:
                    published.append(dst)
                    with src.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    print(f"Selected stems saved beside {output_base}", flush=True)
    return targets

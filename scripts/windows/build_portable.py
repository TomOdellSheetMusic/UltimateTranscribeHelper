"""Assemble a Windows portable tree from a verified local runtime.

Run with the project's Windows runtime after installing its dependencies and
downloading the checkpoints. The destination must not already exist.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / (sys.argv[1] if len(sys.argv) > 1 else ".runtime")
DESTINATION = ROOT / "dist" / (sys.argv[2] if len(sys.argv) > 2 else "Omnizart-Windows")
REQUIRED_MODELS = [
    "music/music_piano", "music/music_piano-v2", "music/music_pop",
    "music/music_note_stream", "drum/drum_keras", "chord/chord_v1",
    "vocal/vocal_semi", "vocal/vocal_contour", "beat/beat_blstm",
]


def ignore_runtime_files(directory, names):
    ignored = set(shutil.ignore_patterns("__pycache__", "*.pyc")(directory, names))
    # Local wheel origins can contain the build machine's username and paths.
    if "direct_url.json" in names:
        ignored.add("direct_url.json")
    # TensorFlow's C/C++ development headers are not used for inference.
    if Path(directory).name == "tensorflow" and "include" in names:
        ignored.add("include")
    return ignored


def main():
    if sys.platform != "win32":
        raise SystemExit("Build the Windows package on Windows.")
    if DESTINATION.exists():
        raise SystemExit(f"Remove or rename the existing build directory first: {DESTINATION}")
    python = RUNTIME / "python.exe"
    if not python.is_file():
        raise SystemExit(f"Missing runtime: {python}")
    for model in REQUIRED_MODELS:
        path = ROOT / "omnizart" / "checkpoints" / model / "variables" / "variables.data-00000-of-00001"
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing model weights: {path}")
    uvr_model = ROOT / "omnizart" / "checkpoints" / "vocal" / "UVR-MDX-NET-Inst_HQ_3.onnx"
    if not uvr_model.is_file():
        raise SystemExit(f"Missing UVR separator model: {uvr_model}")
    for filename in ("vocals.fp16.onnx", "accompaniment.fp16.onnx"):
        path = ROOT / "omnizart" / "checkpoints" / "vocal" / "sherpa-onnx-spleeter-2stems-fp16" / filename
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing vocal separation model: {path}")
    source_root = ROOT / "windows-wheel-check"
    for name in ("nnls-chroma-src", "vamp-plugin-sdk"):
        if not (source_root / name).is_dir():
            raise SystemExit(f"Missing plugin source tree: {source_root / name}")
    subprocess.run([str(python), "-c", "import omnizart, tensorflow, librosa, madmom, vamp, sherpa_onnx, basic_pitch, resampy, onnxruntime, tkinterdnd2; "
                    "assert basic_pitch.ICASSP_2022_MODEL_PATH.exists(); "
                    "assert 'DmlExecutionProvider' in onnxruntime.get_available_providers(); "
                    "assert 'nnls-chroma:nnls-chroma' in vamp.list_plugins()"],
                   cwd=ROOT, check=True)
    subprocess.run([str(python), "-c", "import torch, transkun; from pathlib import Path; "
                    "assert not torch.version.cuda; "
                    "assert (Path(transkun.__file__).parent / 'pretrained' / '2.0.pt').is_file()"],
                   cwd=ROOT, check=True)
    subprocess.run([str(python), "-c", "import importlib.util; "
                    "assert importlib.util.find_spec('piano_transcription_inference'); "
                    "assert importlib.util.find_spec('torchlibrosa')"], cwd=ROOT, check=True)
    DESTINATION.mkdir(parents=True)
    shutil.copytree(RUNTIME, DESTINATION / "runtime", ignore=ignore_runtime_files)
    shutil.copytree(ROOT / "omnizart", DESTINATION / "omnizart",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "scripts" / "windows" / "Omnizart.vbs", DESTINATION / "Omnizart.vbs")
    shutil.copy2(ROOT / "LICENSE", DESTINATION / "LICENSE")
    shutil.copy2(ROOT / "scripts" / "windows" / "README.txt", DESTINATION / "README.txt")
    shutil.copy2(ROOT / "docs" / "WINDOWS_MODELS.md", DESTINATION / "MODEL-LICENSES.md")
    for name in ("nnls-chroma-src", "vamp-plugin-sdk"):
        shutil.copytree(source_root / name, DESTINATION / "source" / name,
                        ignore=shutil.ignore_patterns(".git"))
    shutil.copy2(ROOT / "scripts" / "windows" / "build_chroma.cmd", DESTINATION / "source" / "build_chroma.cmd")
    print(f"Portable folder created: {DESTINATION}")
    print("Test the copied folder before creating a release archive.")


if __name__ == "__main__":
    main()

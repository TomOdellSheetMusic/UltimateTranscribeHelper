"""Exercise two desktop runs without starting a transcription process."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import patch

import mido
from omnizart.desktop.app import Application


class RepeatedRunsTests(unittest.TestCase):
    def test_pymss_selected_for_piano_isolation_and_stem_output(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / 'song.wav'
            audio.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread, \
                 patch('omnizart.desktop.optional_models.ready', return_value=True):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.stem_model.set('pymss model')
                    app.pymss_model.set('test-piano')
                    app.piano_only.set(True)
                    app.update_separation_controls()
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertIn('--piano-only', command)
                    self.assertEqual(command[command.index('--piano-separator') + 1], 'pymss')
                    self.assertEqual(command[command.index('--pymss-model') + 1], 'test-piano')
                    app.output_lock.unlink()
                    app.mode.set('Separate stems (audio)')
                    app.mode_changed()
                    app.stem_choice.set('Piano')
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertEqual(command[command.index('--backend') + 1], 'pymss')
                    self.assertEqual(command[command.index('--stem') + 1], 'piano')
                    self.assertEqual(app.active_stems, ('piano',))
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_multiple_custom_models_can_be_selected(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "song.wav"
            audio.touch()
            musescore = folder / "MuseScore4.exe"
            musescore.touch()
            models = [folder / "one" / "piano", folder / "two" / "piano"]
            for model in models:
                (model / "variables").mkdir(parents=True)
                (model / "configurations.yaml").touch()
                (model / "saved_model.pb").touch()
                (model / "variables" / "variables.index").touch()
                (model / "variables" / "variables.data-00000-of-00001").touch()
            with patch("omnizart.desktop.app.folder_settings_path", return_value=folder / "folders.json"), \
                 patch("omnizart.desktop.app.model_settings_path", return_value=folder / "models.json"), \
                 patch("omnizart.desktop.app.filedialog.askdirectory", side_effect=[str(p) for p in models]), \
                 patch("omnizart.desktop.app.threading.Thread") as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    for _ in models:
                        app.import_model()
                    self.assertEqual(set(app.custom_models.values()), {str(p) for p in models})
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    self.assertFalse(app.lint_notes.get())
                    app.lint_notes.set(True)
                    app.musescore_path.set(str(musescore))
                    app.musescore_export.set(True)
                    app.align_bars.set(False)
                    app.pdf_export.set(True)
                    app.score_tempo.set("154")
                    reference = folder / "reference.mid"
                    reference.touch()
                    app.compare_path.set(str(reference))
                    app.compare_enabled.set(True)
                    first_label = next(label for label, path in app.custom_models.items() if path == str(models[0]))
                    app.music_model.set(first_label)
                    app.mode_changed()
                    app.start()
                    command, _ = thread.call_args.kwargs["args"]
                    self.assertEqual(command[command.index("--model-path") + 1], str(models[0]))
                    self.assertEqual(command[command.index("--compare-midi") + 1], str(reference))
                    self.assertIn("--lint-notes", command)
                    self.assertEqual(command[command.index("--musescore") + 1], str(musescore))
                    self.assertIn("--score", command)
                    self.assertEqual(command[command.index("--score-tempo") + 1], "154")
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_optional_uvr_models_disable_gpu_and_have_manager(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch("omnizart.desktop.app.folder_settings_path", return_value=folder / "folders.json"):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.remove_vocals.set(True)
                    app.separator_model.set("UVR Vocal FT (CPU)")
                    app.use_gpu.set(True)
                    app.update_separation_controls()
                    self.assertEqual(str(app.gpu_option.cget("state")), "disabled")
                    app.show_separation_models()
                    root.update_idletasks()
                finally:
                    root.destroy()

    def test_augmented_transkun_is_selectable_without_other_thresholds(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio, archive = folder / "song.wav", folder / "model.zip"
            audio.touch()
            archive.touch()
            with patch("omnizart.desktop.app.folder_settings_path", return_value=folder / "folders.json"), \
                    patch("omnizart.desktop.app.threading.Thread") as thread, \
                    patch("omnizart.desktop.transkun_aug.model_path", return_value=archive), \
                    patch("omnizart.desktop.transkun_aug.ARCHIVE_SIZE", 0):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.music_model.set("Transkun V2 Aug")
                    app.mode_changed()
                    app.show_models()
                    root.update_idletasks()
                    app.start()
                    command, _ = thread.call_args.kwargs["args"]
                    self.assertEqual(command[command.index("--model-path") + 1], "Transkun V2 Aug")
                    self.assertNotIn("--onset-th", command)
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_stem_mode_uses_optional_gpu_and_reports_wav_results(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio, model = folder / "song.wav", folder / "model.onnx"
            audio.touch()
            model.touch()
            with patch("omnizart.desktop.app.folder_settings_path", return_value=folder / "folders.json"), \
                    patch("omnizart.desktop.app.threading.Thread") as thread, \
                    patch("omnizart.desktop.six_stems.model_path", return_value=model), \
                    patch("omnizart.desktop.six_stems.MODEL_SIZE", 0):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.mode.set("Separate stems (audio)")
                    app.mode_changed()
                    self.assertEqual(str(app.gpu_option.cget("state")), "normal")
                    app.use_gpu.set(True)
                    app.update_separation_controls()
                    app.start()
                    command, result_folder = thread.call_args.kwargs["args"]
                    self.assertIn("stems", command)
                    self.assertEqual(command[command.index("--device") + 1], "directml")
                    base = Path(command[command.index("--output") + 1])
                    for stem in ("drums", "bass", "other", "vocals", "guitar", "piano"):
                        Path(str(base) + "_" + stem + ".wav").touch()
                    app.messages.put(("done", (0, result_folder)))
                    app.poll()
                    self.assertIn("results are ready", app.status.get())
                finally:
                    root.destroy()

    def test_same_folder_distinct_results_and_current_note_count(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "song.wav"
            audio.touch()
            with patch("omnizart.desktop.app.folder_settings_path", return_value=folder / "folders.json"), \
                    patch("omnizart.desktop.app.threading.Thread") as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.start()
                    first_command, first_folder = thread.call_args.kwargs["args"]
                    first_base = Path(first_command[first_command.index("--output") + 1])
                    midi = mido.MidiFile()
                    midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=60, velocity=80)]))
                    midi.save(str(first_base) + ".mid")
                    app.messages.put(("done", (0, first_folder)))
                    app.poll()
                    self.assertIn("1 notes", app.status.get())
                    app.start()
                    second_command, second_folder = thread.call_args.kwargs["args"]
                    second_base = Path(second_command[second_command.index("--output") + 1])
                    self.assertEqual(first_folder, second_folder)
                    self.assertEqual(second_base.name, "song_2")
                    empty = mido.MidiFile()
                    empty.tracks.append(mido.MidiTrack())
                    empty.save(str(second_base) + ".mid")
                    app.messages.put(("done", (0, second_folder)))
                    app.poll()
                    self.assertIn("No notes", app.status.get())
                    self.assertTrue(Path(str(first_base) + ".mid").is_file())
                    self.assertFalse(list(first_folder.glob("*.omnizart.lock")))
                finally:
                    root.destroy()

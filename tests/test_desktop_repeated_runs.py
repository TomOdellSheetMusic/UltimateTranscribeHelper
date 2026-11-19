"""Exercise two desktop runs without starting a transcription process."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mido
from omnizart.desktop.app import Application


class RepeatedRunsTests(unittest.TestCase):
    def setUp(self):
        preferences = patch('omnizart.desktop.app.read_preferences', return_value={})
        save = patch.object(Application, 'save_preferences')
        preferences.start()
        save.start()
        self.addCleanup(preferences.stop)
        self.addCleanup(save.stop)

    def test_piano_and_vocal_share_stem_model_and_piano_only_clears_separation(self):
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
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.stem_model.set('pymss model')
                    app.pymss_model.set('shared-model')
                    app.piano_only_option.invoke()
                    app.vocal_option.invoke()
                    self.assertTrue(app.piano_only.get())
                    self.assertTrue(app.include_vocals.get())
                    self.assertFalse(app.remove_vocals.get())
                    self.assertEqual(app.vocal_track_backend.get(), 'Same as selected stem')
                    app.start()
                    command, _ = thread.call_args.kwargs['args']
                    self.assertIn('--piano-only', command)
                    self.assertIn('--include-vocals', command)
                    self.assertEqual(command[command.index('--vocal-track-separator') + 1], 'stem')
                    self.assertNotIn('--remove-vocals', command)
                    app.output_lock.unlink()
                    app.input_piano_option.invoke()
                    self.assertTrue(app.input_is_piano.get())
                    self.assertFalse(app.piano_only.get())
                    self.assertFalse(app.include_vocals.get())
                    self.assertFalse(app.remove_vocals.get())
                finally:
                    root.destroy()

    def test_vocal_reduction_and_vocal_track_can_use_different_backends(self):
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
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.separation_option.invoke()
                    app.vocal_option.invoke()
                    app.vocal_track_backend.set('pymss')
                    app.vocal_track_pymss_model.set('voice-model')
                    app.vocal_transcription_model.set('Basic Pitch')
                    app.use_gpu.set(True)
                    app.gpu_index.set('1')
                    app.update_separation_controls()
                    app.start()
                    command, _ = thread.call_args.kwargs['args']
                    self.assertEqual(command[command.index('--separator') + 1], 'uvr')
                    self.assertEqual(command[command.index('--vocal-track-separator') + 1], 'pymss')
                    self.assertEqual(command[command.index('--vocal-track-pymss-model') + 1], 'voice-model')
                    self.assertEqual(command[command.index('--vocal-transcription-model') + 1], 'basic-pitch')
                    self.assertEqual(command[command.index('--uvr-device') + 1], 'directml')
                    self.assertEqual(command[command.index('--pymss-device') + 1], 'cuda')
                    self.assertEqual(command.count('--gpu-index'), 1)
                    app.output_lock.unlink()
                    app.piano_only_option.invoke()
                    self.assertTrue(app.piano_only.get())
                    self.assertFalse(app.remove_vocals.get())
                    self.assertTrue(app.include_vocals.get())
                finally:
                    root.destroy()

    def test_rosvot_gpu_is_independent_of_separation_gpu(self):
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
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.vocal_option.invoke()
                    app.vocal_transcription_model.set('ROSVOT')
                    app.rosvot_gpu.set(True)
                    app.update_separation_controls()
                    self.assertEqual(app.rosvot_gpu_option.winfo_manager(), 'grid')
                    app.start()
                    command, _ = thread.call_args.kwargs['args']
                    self.assertEqual(command[command.index('--vocal-transcription-model') + 1], 'rosvot')
                    self.assertEqual(command[command.index('--vocal-transcription-device') + 1], 'cuda')
                    self.assertNotIn('--uvr-device', command)
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_vocal_backend_switches_model_controls_and_routes_pymss_cuda(self):
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
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.separation_option.invoke()
                    app.vocal_option.invoke()
                    app.vocal_backend.set('pymss')
                    app.vocal_pymss_model.set('vocal-model')
                    app.use_gpu.set(True)
                    app.gpu_index.set('1')
                    app.update_separation_controls()
                    self.assertEqual(app.separator_picker.winfo_manager(), '')
                    self.assertEqual(app.vocal_pymss_button.winfo_manager(), 'grid')
                    self.assertIn('CUDA', app.gpu_option.cget('text'))
                    app.start()
                    command, _ = thread.call_args.kwargs['args']
                    self.assertEqual(command[command.index('--separator') + 1], 'pymss')
                    self.assertEqual(command[command.index('--reduction-pymss-model') + 1], 'vocal-model')
                    self.assertEqual(command[command.index('--pymss-device') + 1], 'cuda')
                    self.assertIn('--include-vocals', command)
                    self.assertNotIn('--uvr-device', command)
                    app.output_lock.unlink()
                    app.vocal_backend.set('Built-in')
                    app.update_separation_controls()
                    self.assertEqual(app.vocal_pymss_button.winfo_manager(), '')
                    self.assertEqual(app.separator_picker.winfo_manager(), 'grid')
                finally:
                    root.destroy()

    def test_vocal_pymss_browser_buttons_open_and_select_for_each_role(self):
        catalog = [{'name': 'voice-model', 'target_stem': 'vocals/other'}]
        with TemporaryDirectory() as directory, \
             patch('omnizart.desktop.app.folder_settings_path', return_value=Path(directory) / 'folders.json'), \
             patch('omnizart.desktop.optional_models.ready', return_value=True), \
             patch('omnizart.desktop.pymss_backend.catalog', return_value=catalog):
            root = tk.Tk()
            try:
                app = Application(root)
                app.sections.select(0)
                app.separation_option.invoke()
                app.vocal_backend.set('pymss')
                app.vocal_backend_picker.event_generate('<<ComboboxSelected>>')
                root.update()
                self.assertTrue(app.vocal_pymss_button.winfo_ismapped())
                self.assertLessEqual(app.vocal_pymss_button.winfo_rooty() + app.vocal_pymss_button.winfo_height(),
                                     app.sections.winfo_rooty() + app.sections.winfo_height())
                app.vocal_pymss_button.invoke()
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                listing = next(w for w in dialog.winfo_children() if isinstance(w, tk.Listbox))
                listing.selection_set(0)
                next(w for frame in dialog.winfo_children() for w in frame.winfo_children()
                     if isinstance(w, tk.ttk.Button) and w.cget('text') == 'Use selected model').invoke()
                self.assertEqual(app.vocal_pymss_model.get(), 'voice-model')
                app.vocal_option.invoke()
                app.vocal_track_backend.set('pymss')
                app.vocal_track_backend_picker.event_generate('<<ComboboxSelected>>')
                root.update()
                self.assertTrue(app.vocal_track_pymss_button.winfo_ismapped())
                self.assertLessEqual(app.vocal_track_pymss_button.winfo_rooty() + app.vocal_track_pymss_button.winfo_height(),
                                     app.sections.winfo_rooty() + app.sections.winfo_height())
                app.vocal_track_pymss_button.invoke()
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                listing = next(w for w in dialog.winfo_children() if isinstance(w, tk.Listbox))
                listing.selection_set(0)
                next(w for frame in dialog.winfo_children() for w in frame.winfo_children()
                     if isinstance(w, tk.ttk.Button) and w.cget('text') == 'Use selected model').invoke()
                self.assertEqual(app.vocal_track_pymss_model.get(), 'voice-model')
            finally:
                root.destroy()

    def test_first_pymss_use_shows_setup_and_opens_requested_browser(self):
        with TemporaryDirectory() as directory, \
             patch('omnizart.desktop.app.folder_settings_path', return_value=Path(directory) / 'folders.json'), \
             patch('omnizart.desktop.optional_models.ready', side_effect=[False, True]), \
             patch('omnizart.desktop.optional_models.install') as install, \
             patch('omnizart.desktop.app.threading.Thread') as thread, \
             patch('omnizart.desktop.pymss_backend.catalog', return_value=[{'name': 'voice', 'target_stem': 'vocals'}]):
            root = tk.Tk()
            root.withdraw()
            try:
                app = Application(root)
                app.vocal_option.invoke()
                app.vocal_track_backend.set('pymss')
                app.update_separation_controls()
                app.vocal_track_pymss_button.invoke()
                self.assertTrue(app.pymss_setup_dialog.winfo_exists())
                thread.call_args.kwargs['target']()
                app.poll()
                install.assert_called_once_with('pymss')
                self.assertIsNone(app.pymss_setup_dialog)
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                listing = next(w for w in dialog.winfo_children() if isinstance(w, tk.Listbox))
                listing.selection_set(0)
                next(w for frame in dialog.winfo_children() for w in frame.winfo_children()
                     if isinstance(w, tk.ttk.Button) and w.cget('text') == 'Use selected model').invoke()
                self.assertEqual(app.vocal_track_pymss_model.get(), 'voice')
            finally:
                root.destroy()

    def test_vocal_model_browser_filters_and_keeps_stem_selection_separate(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            catalog = [{'name': 'piano-model', 'target_stem': 'piano'},
                       {'name': 'vocal-model', 'target_stem': 'vocals/other'}]
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.optional_models.ready', return_value=True), \
                 patch('omnizart.desktop.pymss_backend.catalog', return_value=catalog):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.stem_model.set('Custom six-stem ONNX')
                    app.pymss_model.set('existing-stem')
                    app.vocal_backend.set('pymss')
                    app.show_pymss_models('vocal')
                    dialog = next(widget for widget in root.winfo_children() if isinstance(widget, tk.Toplevel))
                    listing = next(widget for widget in dialog.winfo_children() if isinstance(widget, tk.Listbox))
                    self.assertEqual(listing.size(), 1)
                    self.assertIn('vocal-model', listing.get(0))
                    listing.selection_set(0)
                    buttons = [child for frame in dialog.winfo_children() for child in frame.winfo_children()
                               if isinstance(child, tk.ttk.Button) and child.cget('text') == 'Use selected model']
                    buttons[0].invoke()
                    self.assertEqual(app.vocal_pymss_model.get(), 'vocal-model')
                    self.assertEqual(app.pymss_model.get(), 'existing-stem')
                    self.assertEqual(app.stem_model.get(), 'Custom six-stem ONNX')
                finally:
                    root.destroy()

    def test_estimated_alignment_is_visible_in_queue(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / 'song.wav'
            audio.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.enqueue_current()
                    command, result_folder = thread.call_args.kwargs['args']
                    base = Path(command[command.index('--output') + 1])
                    midi = mido.MidiFile()
                    midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80)]))
                    midi.save(str(base) + '.mid')
                    app.messages.put(('log', 'UserWarning: Audio bar alignment unavailable; using a constant-tempo grid.\n'))
                    app.messages.put(('done', (0, result_folder)))
                    app.poll()
                    self.assertEqual(app.jobs[0]['state'], 'Review timing')
                    self.assertIn('Timing was estimated', app.status.get())
                finally:
                    root.destroy()

    def test_neural_and_alignment_produce_final_midi_without_musicxml(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / 'song.wav'
            audio.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.optional_models.ready', return_value=True), \
                 patch('omnizart.desktop.app.threading.Thread') as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set('Piano')
                    app.mode_changed()
                    app.source.set(str(audio))
                    app.destination.set(str(folder))
                    app.neural_staff.set(True)
                    app.align_bars.set(True)
                    app.musescore_export.set(False)
                    app.split_hands.set(False)
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertIn('--neural-staff', command)
                    self.assertIn('--align-bars', command)
                    self.assertIn('--split-hands', command)
                    self.assertNotIn('--score', command)
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_result_rows_have_aligned_headings_and_independent_folder_buttons(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            first, second = folder / 'one.wav', folder / 'two.wav'
            first.touch()
            second.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.os.startfile') as open_folder:
                root = tk.Tk()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.destination.set(str(folder))
                    app.running = True  # Keep both jobs waiting while their rows are arranged.
                    app.enqueue_files([first, second])
                    for index, job in enumerate(app.jobs):
                        job['output'] = folder / str(index)
                        job['output'].mkdir()
                        app._update_job(job, 'Finished')
                    app._navigate('queue')
                    root.update()
                    app._position_queue_buttons()
                    for column in ('file', 'mode', 'state', 'folder'):
                        self.assertEqual(app.queue_view.heading(column)['anchor'], 'w')
                        self.assertEqual(app.queue_view.column(column)['anchor'], 'w')
                    for job in app.jobs:
                        self.assertTrue(job['folder_button'].place_info())
                    app.jobs[1]['folder_button'].invoke()
                    open_folder.assert_called_once_with(str(folder / '1'))
                    app.running = False
                finally:
                    root.destroy()

    def test_hands_option_and_existing_midi_mode(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio, midi = folder / 'song.wav', folder / 'song.mid'
            audio.touch()
            midi.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.destination.set(str(folder))
                    app.source.set(str(audio))
                    app.split_hands.set(True)
                    app.align_bars.set(False)
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertIn('--split-hands', command)
                    app.output_lock.unlink()
                    app.source.set(str(midi))
                    app.mode.set('Split piano MIDI into hands')
                    app.mode_changed()
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertIn('hands', command)
                    self.assertNotIn('--split-hands', command)
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_drop_and_pymss_gpu_commands(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / 'song with spaces ü.wav'
            audio.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread, \
                 patch('omnizart.desktop.optional_models.ready', return_value=True):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    self.assertTrue(root.tk.call('package', 'provide', 'tkdnd'))
                    self.assertEqual(app.drop_input(SimpleNamespace(data='{' + str(audio) + '}')), 'copy')
                    self.assertEqual(app.source.get(), str(audio))
                    self.assertEqual(app.drop_input(SimpleNamespace(data='{' + str(audio) + '} {' + str(folder / 'missing.wav') + '}')), 'refuse_drop')
                    self.assertEqual(app.drop_output(SimpleNamespace(data='{' + str(folder) + '}')), 'copy')
                    app.stem_model.set('pymss model')
                    app.pymss_model.set('piano-model')
                    app.piano_only.set(True)
                    app.use_gpu.set(True)
                    app.update_separation_controls()
                    self.assertEqual(str(app.gpu_option['state']), 'normal')
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertEqual(command[command.index('--pymss-device') + 1], 'cuda')
                    self.assertNotIn('--uvr-device', command)
                    app.output_lock.unlink()
                    app.mode.set('Separate stems (audio)')
                    app.mode_changed()
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertEqual(command[command.index('--device') + 1], 'cuda')
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_queue_runs_in_order_with_captured_settings_and_separate_details(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            first, second = folder / 'one.wav', folder / 'two.wav'
            first.touch()
            second.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread:
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.destination.set(str(folder))
                    app.source.set(str(first))
                    app.music_model.set('Piano')
                    app.enqueue_current()
                    first_command, first_folder = thread.call_args.kwargs['args']
                    self.assertEqual(app.jobs[0]['state'], 'Running')
                    self.assertEqual(str(app.start_button['state']), 'normal')
                    app.source.set(str(second))
                    app.music_model.set('PianoV2')
                    app.enqueue_current()
                    app.music_model.set('Pop')
                    self.assertEqual(app.jobs[1]['state'], 'Waiting')
                    self.assertEqual(thread.call_count, 1)
                    first_base = Path(first_command[first_command.index('--output') + 1])
                    midi = mido.MidiFile()
                    midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80)]))
                    midi.save(str(first_base) + '.mid')
                    app.messages.put(('log', 'first job output\n'))
                    app.messages.put(('done', (0, first_folder)))
                    app.poll()
                    root.update()
                    self.assertEqual(app.jobs[0]['state'], 'Finished')
                    second_command, second_folder = thread.call_args.kwargs['args']
                    self.assertEqual(second_command[second_command.index('--model-path') + 1], 'PianoV2')
                    self.assertEqual(app.music_model.get(), 'Pop')
                    self.assertEqual(app.jobs[1]['state'], 'Running')
                    app.messages.put(('log', 'second job output\n'))
                    app.queue_view.selection_set(app.jobs[0]['id'])
                    app._show_selected_job()
                    app.poll()
                    self.assertIn('first job output', app.log.get('1.0', 'end'))
                    self.assertNotIn('second job output', app.log.get('1.0', 'end'))
                    app.messages.put(('done', (1, second_folder)))
                    app.poll()
                    self.assertEqual(app.jobs[1]['state'], 'Needs attention')
                    self.assertTrue(app.log_open)
                    app.clear_finished_jobs()
                    self.assertEqual(app.jobs, [])
                finally:
                    root.destroy()

    def test_dropping_multiple_files_queues_them(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            first, second = folder / 'one.wav', folder / 'two.wav'
            first.touch()
            second.touch()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread'):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.destination.set(str(folder))
                    event = SimpleNamespace(data='{' + str(first) + '} {' + str(second) + '}')
                    self.assertEqual(app.drop_input(event), 'copy')
                    self.assertEqual([Path(job['source']).name for job in app.jobs], ['one.wav', 'two.wav'])
                    self.assertEqual([job['state'] for job in app.jobs], ['Running', 'Waiting'])
                    app.queue_view.selection_set(app.jobs[1]['id'])
                    app.remove_queued_job()
                    self.assertEqual(len(app.jobs), 1)
                    app.output_lock.unlink()
                finally:
                    root.destroy()

    def test_queue_preserves_settings_across_optional_install(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            first, second = folder / 'one.wav', folder / 'two.wav'
            first.touch()
            second.touch()
            installed = set()
            with patch('omnizart.desktop.app.folder_settings_path', return_value=folder / 'folders.json'), \
                 patch('omnizart.desktop.app.threading.Thread') as thread, \
                 patch('omnizart.desktop.optional_models.ready', side_effect=lambda model: model in installed):
                root = tk.Tk()
                root.withdraw()
                try:
                    app = Application(root)
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
                    app.destination.set(str(folder))
                    app.source.set(str(first))
                    app.music_model.set('YourMT3')
                    app.enqueue_current()
                    self.assertEqual(app.jobs[0]['state'], 'Installing model')
                    app.source.set(str(second))
                    app.music_model.set('Piano')
                    app.enqueue_current()
                    installed.add('YourMT3')
                    app.messages.put(('optional_install_done', ('YourMT3', 0, '')))
                    app.poll()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertEqual(command[command.index('--model-path') + 1], 'YourMT3')
                    self.assertEqual(Path(command[command.index('transcribe') + 1]), first)
                    self.assertEqual(app.source.get(), str(second))
                    self.assertEqual(app.jobs[1]['state'], 'Waiting')
                    app.output_lock.unlink()
                finally:
                    root.destroy()

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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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
                    app.stem_choice.set('Vocals')
                    app.save_separated_stem.set(True)
                    app.start()
                    command = thread.call_args.kwargs['args'][0]
                    self.assertEqual(command[command.index('--separate-stem') + 1], 'vocals')
                    self.assertIn('--save-separated-stem', command)
                    self.assertNotIn('--piano-only', command)
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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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
                    app.music_model.set("Piano")
                    app.mode_changed()
                    app.align_bars.set(False)
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

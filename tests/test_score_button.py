"""Exercise score export command construction without a GUI display."""
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from omnizart.desktop.app import Application


def var(value):
    return SimpleNamespace(get=lambda: value)


class ScoreButtonTests(unittest.TestCase):
    def test_musicxml_needs_no_musescore(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'song.wav'
            source.touch()
            app = SimpleNamespace(
                source=var(str(source)), destination=var(str(folder)), mode=var('Piano and instruments'),
                music_model=var('Piano'), custom_models={}, custom_model_path=None,
                thresholds={key: var('') for key in ('onset', 'duration', 'instrument')},
                piano_only=var(False), input_is_piano=var(False), remove_vocals=var(False), include_vocals=var(False),
                separator_model=var('UVR MDX (recommended)'), use_gpu=var(False), gpu_index=var('0'),
                smooth_notes=var(False), remove_short_notes=var(False), compare_enabled=var(False),
                split_hands=var(False), prefer_right_octave=var(False), move_left_chords=var(False),
                shorten_pedal_tails=var(False), clarify_pedal_melody=var(False), remove_quiet_notes=var(False),
                vocal_backend=var('Built-in'),
                musescore_export=var(True), align_bars=var(False), score_downbeat=var('Auto'), score_meter=var('Auto'), score_key=var('Bb'), neural_staff=var(False), simplify_score=var(True), score_tempo=var('154'), pdf_export=var(False),
                musescore_path=var(''), lint_notes=var(False), save_folders=Mock(),
                log=Mock(), start_button=Mock(), open_button=Mock(), progress=Mock(), status=Mock(),
                run=Mock())
            with patch('omnizart.desktop.app.threading.Thread') as thread:
                Application.start(app)
            cmd = thread.call_args.kwargs['args'][0]
            self.assertIn('--score', cmd)
            self.assertIn('--simplify-score', cmd)
            self.assertEqual(cmd[cmd.index('--score-tempo') + 1], '154')
            self.assertEqual(cmd[cmd.index('--score-key') + 1], 'Bb')
            self.assertNotIn('--musescore', cmd)
            app.output_lock.unlink()
            app.neural_staff = var(True)
            with patch('omnizart.desktop.app.threading.Thread') as thread, patch('omnizart.desktop.optional_models.ready', return_value=True):
                Application.start(app)
            self.assertIn('--neural-staff', thread.call_args.kwargs['args'][0])
            app.output_lock.unlink()
            app.neural_staff = var(False)
            app.align_bars = var(True)
            app.score_downbeat = var('1.25')
            app.score_meter = var('3')
            with patch('omnizart.desktop.app.threading.Thread') as thread, patch('omnizart.desktop.optional_models.ready', return_value=True):
                Application.start(app)
            cmd = thread.call_args.kwargs['args'][0]
            self.assertIn('--align-bars', cmd)
            self.assertEqual(cmd[cmd.index('--score-meter') + 1], '3')
            self.assertEqual(cmd[cmd.index('--score-downbeat') + 1], '1.25')
            app.output_lock.unlink()
            app.align_bars = var(False)
            app.music_model = var('YourMT3')
            app.input_is_piano = var(True)
            app.neural_staff = var(True)
            with patch('omnizart.desktop.app.threading.Thread') as thread, patch('omnizart.desktop.optional_models.ready', return_value=True):
                Application.start(app)
            cmd = thread.call_args.kwargs['args'][0]
            self.assertIn('--input-is-piano', cmd)
            self.assertIn('--neural-staff', cmd)
            self.assertNotIn('--piano-only', cmd)
            app.output_lock.unlink()

    def test_simplification_omissions_are_shown_after_success(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            base = folder / 'score'
            base.with_suffix('.mid').touch()
            messages = Queue()
            messages.put(('log', 'notation.py:1: UserWarning: Score simplification omitted 2 note(s).\n'))
            messages.put(('done', (0, folder)))
            app = SimpleNamespace(messages=messages, score_warnings=[], log=Mock(), output_lock=None,
                                  progress=Mock(), start_button=Mock(), open_button=Mock(), output_folder=folder,
                                  active_mode='music', output_base=base, status=Mock(), after=Mock(), poll=Mock(),
                                  active_job=None, queue_view=Mock(selection=Mock(return_value=())),
                                  after_idle=Mock(), _start_next_job=Mock())
            with patch('omnizart.desktop.app.midi_note_count', return_value=3), patch('omnizart.desktop.app.messagebox.showwarning') as warning:
                Application.poll(app)
            self.assertIn('omitted 2 note', warning.call_args.args[1])

    def test_shortened_notes_are_shown_after_success(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            base = folder / 'score'
            base.with_suffix('.mid').touch()
            messages = Queue()
            messages.put(('log', 'notation.py:1: UserWarning: Score simplification shortened 2 held note(s).\n'))
            messages.put(('done', (0, folder)))
            app = SimpleNamespace(messages=messages, score_warnings=[], log=Mock(), output_lock=None,
                                  progress=Mock(), start_button=Mock(), open_button=Mock(), output_folder=folder,
                                  active_mode='music', output_base=base, status=Mock(), after=Mock(), poll=Mock(),
                                  active_job=None, queue_view=Mock(selection=Mock(return_value=())),
                                  after_idle=Mock(), _start_next_job=Mock())
            with patch('omnizart.desktop.app.midi_note_count', return_value=3), patch('omnizart.desktop.app.messagebox.showwarning') as warning:
                Application.poll(app)
            self.assertIn('shortened 2 held', warning.call_args.args[1])


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from click.testing import CliRunner
import mido
from unittest.mock import patch
from xml.etree import ElementTree as ET

from omnizart.cli.hands import hands
from omnizart.desktop.hand_midi import split_piano_midi
from omnizart.desktop.notation import export_score
from omnizart.desktop.beat_grid import BeatGrid
from omnizart.cli.music.transcribe import transcribe as music_transcribe


def fixture(path):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('set_tempo', tempo=420000, time=0))
    meta.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
    midi.tracks.append(meta)
    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage('track_name', name='Piano', time=0))
    piano.append(mido.Message('program_change', program=0, channel=4, time=0))
    piano.append(mido.Message('control_change', control=64, value=127, channel=4, time=10))
    piano.append(mido.Message('note_on', note=48, velocity=83, channel=4, time=7))
    piano.append(mido.Message('note_on', note=72, velocity=92, channel=4, time=0))
    piano.append(mido.Message('note_off', note=72, velocity=2, channel=4, time=250))
    piano.append(mido.Message('note_off', note=48, velocity=3, channel=4, time=160))
    piano.append(mido.Message('control_change', control=64, value=0, channel=4, time=5))
    midi.tracks.append(piano)
    midi.save(path)


def absolute(track):
    tick = 0
    result = []
    for message in track:
        tick += message.time
        result.append((tick, message))
    return result


class HandMidiTests(unittest.TestCase):
    def test_split_preserves_notes_timing_tempo_and_pedal(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'input.mid', Path(folder) / 'hands.mid'
            fixture(source)
            original = source.read_bytes()
            self.assertEqual(split_piano_midi(source, target), (1, 1))
            self.assertEqual(source.read_bytes(), original)
            midi = mido.MidiFile(target)
            right, left = midi.tracks[-2:]
            self.assertEqual((right.name, left.name), ('Right hand', 'Left hand'))
            self.assertEqual([(t, m.note, m.velocity) for t, m in absolute(right) if m.type == 'note_on'], [(17, 72, 92)])
            self.assertEqual([(t, m.note, m.velocity) for t, m in absolute(left) if m.type == 'note_on'], [(17, 48, 83)])
            self.assertEqual([(t, m.note) for t, m in absolute(right) if m.type == 'note_off'], [(267, 72)])
            self.assertEqual([(t, m.note) for t, m in absolute(left) if m.type == 'note_off'], [(427, 48)])
            self.assertEqual([m.tempo for m in midi.tracks[0] if m.type == 'set_tempo'], [420000])
            for track in (right, left):
                self.assertEqual([(t, m.value) for t, m in absolute(track) if m.type == 'control_change'], [(10, 127), (432, 0)])
            with self.assertRaises(FileExistsError):
                split_piano_midi(source, target)

    def test_rejects_multiple_instruments(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'input.mid', Path(folder) / 'hands.mid'
            fixture(source)
            midi = mido.MidiFile(source)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=90, channel=1),
                                               mido.Message('note_off', note=60, channel=1, time=20)]))
            midi.save(source)
            with self.assertRaisesRegex(ValueError, 'exactly one identifiable'):
                split_piano_midi(source, target)
            self.assertFalse(target.exists())

    def test_standalone_cli(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.mid', Path(folder) / 'result'
            fixture(source)
            result = CliRunner().invoke(hands, ['transcribe', str(source), '--output', str(output)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(output.with_suffix('.mid').is_file())

    def test_standalone_neural_preserves_original_tick_timing(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.mid', Path(folder) / 'result'
            fixture(source)
            with patch('omnizart.desktop.svsep.predict_staff', return_value=[2, 1]) as neural:
                result = CliRunner().invoke(hands, ['transcribe', str(source), '--output', str(output), '--neural-staff'])
            self.assertEqual(result.exit_code, 0, result.output)
            neural.assert_called_once()
            midi = mido.MidiFile(output.with_suffix('.mid'))
            self.assertEqual([(t, m.note) for t, m in absolute(midi.tracks[-2]) if m.type == 'note_on'], [(17, 72)])
            self.assertEqual([(t, m.note) for t, m in absolute(midi.tracks[-1]) if m.type == 'note_on'], [(17, 48)])

    def test_transcription_option_saves_an_extra_hand_midi(self):
        with TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'input.wav', Path(folder) / 'result'
            audio.touch()
            def transcribe_stub(audio_path, model, target, *args):
                fixture(Path(str(target) + '.mid'))
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_stub):
                result = CliRunner().invoke(music_transcribe, [str(audio), '--output', str(output),
                                                               '--model-path', 'Piano', '--split-hands'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(output.with_suffix('.mid').is_file())
            self.assertTrue(output.with_name('result_hands.mid').is_file())

    def test_aligned_score_and_midi_use_same_neural_hands_and_tempo_map(self):
        with TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'input.wav', Path(folder) / 'result'
            audio.touch()
            def transcribe_stub(audio_path, model, target, *args):
                fixture(Path(str(target) + '.mid'))
            beats = [1, 1.5, 2.1, 2.6, 3.2, 3.8, 4.4, 5, 5.6]
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_stub), \
                 patch('omnizart.desktop.beat_alignment.detect', return_value=(beats, [1, 3.2, 5.6])) as detect, \
                 patch('omnizart.desktop.svsep.predict_staff', return_value=[2, 1]) as neural:
                result = CliRunner().invoke(music_transcribe, [str(audio), '--output', str(output), '--model-path', 'Piano',
                    '--score', '--split-hands', '--align-bars', '--neural-staff', '--score-key', 'Bb'])
            self.assertEqual(result.exit_code, 0, result.output)
            detect.assert_called_once()
            neural.assert_called_once()
            midi = mido.MidiFile(output.with_name('result_hands.mid'))
            self.assertEqual([m.note for m in midi.tracks[-2] if m.type == 'note_on' and m.velocity], [72])
            self.assertEqual([m.note for m in midi.tracks[-1] if m.type == 'note_on' and m.velocity], [48])
            self.assertTrue(any(m.type == 'set_tempo' and abs(m.tempo - 600000) <= 1 for m in midi.tracks[0]))
            self.assertEqual([m.key for m in midi.tracks[0] if m.type == 'key_signature'], ['Bb'])
            self.assertEqual([(m.numerator, m.denominator) for m in midi.tracks[0] if m.type == 'time_signature'], [(4, 4)])
            for track in midi.tracks[-2:]:
                self.assertTrue(all(tick % 120 == 0 for tick, msg in absolute(track) if msg.type in ('note_on', 'note_off')))
            xml = ET.parse(output.with_suffix('.musicxml'))
            staff_by_pitch = {int(n.findtext('pitch/octave')) * 12 + {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[n.findtext('pitch/step')] + 12:
                              int(n.findtext('staff')) for n in xml.findall('.//note[pitch]')}
            self.assertEqual((staff_by_pitch[72], staff_by_pitch[48]), (1, 2))

    def test_alignment_failure_does_not_silently_export_unaligned_midi(self):
        with TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'input.wav', Path(folder) / 'result'
            audio.touch()
            def transcribe_stub(audio_path, model, target, *args):
                fixture(Path(str(target) + '.mid'))
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_stub), \
                 patch('omnizart.desktop.beat_alignment.detect', side_effect=RuntimeError('tracker failed')):
                result = CliRunner().invoke(music_transcribe, [str(audio), '--output', str(output), '--model-path', 'Piano',
                                                               '--split-hands', '--align-bars'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('audio bar alignment failed', result.output)
            self.assertTrue(output.with_suffix('.mid').is_file())
            self.assertFalse(output.with_name('result_hands.mid').exists())

    def test_score_render_failure_still_saves_final_midi_after_analysis(self):
        with TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'input.wav', Path(folder) / 'result'
            audio.touch()
            def transcribe_stub(audio_path, model, target, *args):
                fixture(Path(str(target) + '.mid'))
            def failing_score(source, destination, **kwargs):
                kwargs['analysis_out'].update(hands=[1, 2], positions=[(0, 1), (0, 1)],
                                              grid=None, tempo=120, meter=(3, 4), key=None, piano_key=(1, 4))
                raise ValueError('score rendering failed')
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=transcribe_stub), \
                 patch('omnizart.desktop.notation.export_score', side_effect=failing_score):
                result = CliRunner().invoke(music_transcribe, [str(audio), '--output', str(output),
                                                               '--model-path', 'Piano', '--score', '--split-hands'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('final MIDI was saved', result.output)
            self.assertTrue(output.with_name('result_hands.mid').is_file())

    def test_simplified_score_and_final_midi_omit_same_notes(self):
        with TemporaryDirectory() as folder:
            source, target, score = (Path(folder) / name for name in ('input.mid', 'hands.mid', 'score.musicxml'))
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch in range(48, 60):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
            for pitch in range(48, 60):
                track.append(mido.Message('note_off', note=pitch, time=480 if pitch == 48 else 0))
            midi.tracks.append(track)
            midi.save(source)
            analysis = {}
            export_score(source, score, tempo=120, simplify=True, analysis_out=analysis)
            analysis['grid'] = BeatGrid([0, .5, 1, 1.5], [0], tempo=120, downbeat=0)
            analysis['quantize'] = True
            split_piano_midi(source, target, analysis)
            written = mido.MidiFile(target)
            midi_pitches = sorted(msg.note for track in written.tracks for msg in track if msg.type == 'note_on' and msg.velocity)
            xml = ET.parse(score)
            xml_notes = xml.findall('.//note[pitch]')
            self.assertLess(len(midi_pitches), 12)
            self.assertEqual(len(midi_pitches), len(xml_notes))
            self.assertEqual(len([msg for msg in mido.MidiFile(source).tracks[0] if msg.type == 'note_on' and msg.velocity]), 12)

    def test_neural_hand_output_preserves_and_aligns_vocal_track(self):
        with TemporaryDirectory() as folder:
            source, target, score = (Path(folder) / name for name in ('input.mid', 'hands.mid', 'score.musicxml'))
            fixture(source)
            midi = mido.MidiFile(source)
            vocal = mido.MidiTrack([mido.MetaMessage('track_name', name='Vocals'),
                                    mido.Message('program_change', channel=0, program=54),
                                    mido.Message('note_on', channel=0, note=65, velocity=75, time=18),
                                    mido.Message('note_off', channel=0, note=65, time=240)])
            midi.tracks.append(vocal)
            midi.save(source)
            analysis = {}
            with patch('omnizart.desktop.svsep.predict_staff', return_value=[2, 1]):
                export_score(source, score, tempo=120, neural_staff=True, analysis_out=analysis)
            analysis['grid'] = BeatGrid([0, .5, 1, 1.5], [0], tempo=120, downbeat=0)
            analysis['quantize'] = True
            split_piano_midi(source, target, analysis)
            result = mido.MidiFile(target)
            vocal_out = next(track for track in result.tracks if track.name == 'Vocals')
            self.assertEqual([msg.note for msg in vocal_out if msg.type == 'note_on' and msg.velocity], [65])
            self.assertTrue(all(tick % 120 == 0 for tick, msg in absolute(vocal_out) if msg.type in ('note_on', 'note_off')))
            vocal_channel = next(msg.channel for msg in vocal_out if msg.type == 'note_on')
            hand_channels = {msg.channel for track in result.tracks[-2:] for msg in track if msg.type == 'note_on' and msg.velocity}
            self.assertEqual(len(hand_channels), 2)
            self.assertNotIn(vocal_channel, hand_channels)


if __name__ == '__main__':
    unittest.main()

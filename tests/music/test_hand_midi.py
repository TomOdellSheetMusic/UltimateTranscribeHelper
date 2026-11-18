from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from click.testing import CliRunner
import mido
from unittest.mock import patch

from omnizart.cli.hands import hands
from omnizart.desktop.hand_midi import split_piano_midi
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
            with self.assertRaisesRegex(ValueError, 'exactly one pitched'):
                split_piano_midi(source, target)
            self.assertFalse(target.exists())

    def test_standalone_cli(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.mid', Path(folder) / 'result'
            fixture(source)
            result = CliRunner().invoke(hands, ['transcribe', str(source), '--output', str(output)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(output.with_suffix('.mid').is_file())

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


if __name__ == '__main__':
    unittest.main()

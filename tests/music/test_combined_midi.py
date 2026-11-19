"""The combined output keeps instrument and vocal notes in separate tracks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pretty_midi
from omnizart.desktop.combined import add_vocals, merge_tracks


class CombinedMidiTests(unittest.TestCase):
    def test_combines_tracks_and_preserves_note_timing(self):
        with TemporaryDirectory() as temp:
            piano_path = Path(temp) / "piano.mid"
            result_path = Path(temp) / "combined.mid"
            piano = pretty_midi.PrettyMIDI()
            piano_track = pretty_midi.Instrument(0, name="Piano")
            piano_track.notes.append(pretty_midi.Note(90, 60, 0.5, 1.0))
            piano.instruments.append(piano_track)
            piano.write(str(piano_path))
            voice = pretty_midi.PrettyMIDI()
            voice_track = pretty_midi.Instrument(0)
            voice_track.notes.append(pretty_midi.Note(80, 69, 1.5, 2.0))
            voice.instruments.append(voice_track)
            merge_tracks(piano_path, voice, result_path)
            result = pretty_midi.PrettyMIDI(str(result_path))
            self.assertEqual([track.name for track in result.instruments], ["Piano", "Vocals"])
            self.assertAlmostEqual(result.instruments[1].notes[0].start, 1.5, places=2)

    def test_no_vocal_notes_is_reported(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "piano.mid"
            pretty_midi.PrettyMIDI().write(str(path))
            with self.assertRaisesRegex(ValueError, "No pitched vocal notes"):
                merge_tracks(path, pretty_midi.PrettyMIDI(), Path(temp) / "output.mid")

    def test_basic_pitch_uses_vocal_stem_and_preserves_track_order(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            piano = pretty_midi.PrettyMIDI()
            part = pretty_midi.Instrument(0, name='Piano')
            part.notes.append(pretty_midi.Note(80, 60, 0, 1))
            piano.instruments.append(part)
            piano.write(str(root / 'piano.mid'))
            def predict(source, output):
                self.assertEqual(source, str(root / 'voice.wav'))
                vocal = pretty_midi.PrettyMIDI()
                part = pretty_midi.Instrument(0)
                part.notes.append(pretty_midi.Note(80, 72, 0, 1))
                vocal.instruments.append(part)
                vocal.write(output)
            with patch('omnizart.desktop.basic_pitch.transcribe', side_effect=predict):
                add_vocals(root / 'song.wav', root / 'voice.wav', root / 'piano.mid', root / 'result.mid', model='basic-pitch')
            result = pretty_midi.PrettyMIDI(str(root / 'result.mid'))
            self.assertEqual([part.name for part in result.instruments], ['Piano', 'Vocals'])


if __name__ == "__main__":
    unittest.main()

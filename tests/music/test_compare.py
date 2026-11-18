from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import mido
from click.testing import CliRunner
from omnizart.desktop.compare import write_comparison
from omnizart.cli.music.transcribe import transcribe


def make_midi(path, notes):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    prior = 0
    for tick, pitch in notes:
        track.append(mido.Message('note_on', note=pitch, velocity=70, time=tick-prior))
        prior = tick
    midi.save(path)


class CompareTests(unittest.TestCase):
    def test_matching_uses_tempo_and_is_one_to_one(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            a, b, output = root/'new.mid', root/'old.mid', root/'compare.txt'
            make_midi(a, [(0, 60), (10, 60), (480, 62)])
            make_midi(b, [(0, 60), (480, 63)])
            self.assertEqual(write_comparison(a, b, output), 3)
            report = output.read_text(encoding='utf-8')
            self.assertIn('Matched: 1; differences: 3', report)
            self.assertIn('Precision: 0.333; recall: 0.500; F1: 0.400', report)
            self.assertIn('If the comparison MIDI is a verified reference', report)
            self.assertIn('00:00.500', report)
            with self.assertRaises(FileExistsError):
                write_comparison(a, b, output)

    def test_cli_saves_report_when_requested(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            audio, reference = root/'song.wav', root/'old.mid'
            audio.touch()
            make_midi(reference, [(0, 60)])
            with patch('omnizart.cli.music.transcribe._transcribe', side_effect=lambda *args: make_midi(root/'out.mid', [(0, 60)])):
                result = CliRunner().invoke(transcribe, [str(audio), '--output', str(root/'out'), '--compare-midi', str(reference)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn('Matched: 1', (root/'out_comparison.txt').read_text(encoding='utf-8'))

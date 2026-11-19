"""Bar alignment and variable-tempo score mapping."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET
import mido
from omnizart.desktop.beat_grid import BeatGrid
from omnizart.desktop.notation import export_score


class BeatGridTests(unittest.TestCase):
    def test_downbeat_and_pickup_align_to_bar_boundary(self):
        beats = [1 + .5 * i for i in range(13)]
        grid = BeatGrid(beats, [1, 3, 5, 7])
        self.assertEqual(grid.meter, 4)
        self.assertTrue(grid.meter_detected)
        self.assertEqual(grid.unit(1) % 16, 0)
        self.assertEqual(grid.unit(0), 8)
        self.assertEqual(grid.unit(3) - grid.unit(1), 16)

    def test_meter_is_inferred_only_when_downbeats_agree(self):
        beats = [i * .5 for i in range(13)]
        self.assertEqual(BeatGrid(beats, [0, 1.5, 3, 4.5, 6]).meter, 3)
        ambiguous = BeatGrid(beats, [0, 1.5, 3.5, 5])
        self.assertFalse(ambiguous.meter_detected)
        self.assertEqual(ambiguous.meter, 4)

    def test_variable_tempo_follows_detected_beats(self):
        beats = [1, 1.5, 2.1, 2.6, 3.2, 3.8, 4.4, 5, 5.6]
        grid = BeatGrid(beats, [1, 3.2, 5.6])
        self.assertEqual(grid.unit(2.1) - grid.unit(1), 8)
        marks = [bpm for measure in range(4) for _, bpm in grid.tempo_marks(16, measure)]
        self.assertTrue(any(abs(bpm - 100) < .01 for bpm in marks))
        self.assertTrue(any(abs(bpm - 120) < .01 for bpm in marks))

    def test_manual_tempo_and_downbeat_override(self):
        grid = BeatGrid([1 + .6 * i for i in range(12)], [1, 3.4, 5.8], tempo=120, downbeat=.35)
        self.assertEqual(grid.unit(.35) % 16, 0)
        self.assertEqual(grid.unit(.85) - grid.unit(.35), 4)
        self.assertEqual(grid.bpm, 120)
        self.assertEqual(BeatGrid([1 + .5 * i for i in range(13)], [1, 3, 5, 7], meter=3).meter, 3)

    def test_score_uses_audio_beats_and_preserves_midi(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'in.mid', Path(folder) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=90, time=960),
                                               mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            before = source.read_bytes()
            beats = [1, 1.5, 2.1, 2.6, 3.2, 3.8, 4.4, 5, 5.6]
            with patch('omnizart.desktop.beat_alignment.detect', return_value=(beats, [1, 3.2, 5.6])):
                export_score(source, target, audio='song.wav', align_bars=True)
            root = ET.parse(target).getroot()
            first_pitched = next(m for m in root.findall('.//part/measure') if m.find('.//pitch') is not None)
            self.assertEqual(first_pitched.get('number'), '2')
            self.assertAlmostEqual(float(root.findtext('.//metronome/per-minute')), 120)
            tempi = [float(sound.get('tempo')) for sound in root.findall('.//sound[@tempo]')]
            self.assertTrue(any(abs(bpm - 100) < .01 for bpm in tempi))
            self.assertTrue(any(abs(bpm - 120) < .01 for bpm in tempi))
            self.assertEqual(source.read_bytes(), before)

    def test_unsupported_meter_is_preserved_when_alignment_falls_back(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'in.mid', Path(folder) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('time_signature', numerator=6, denominator=8),
                                               mido.Message('note_on', note=60, velocity=90),
                                               mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            with patch('omnizart.desktop.beat_alignment.detect') as detector:
                with self.assertWarnsRegex(UserWarning, 'supports 3/4 and 4/4'):
                    export_score(source, target, audio='song.wav', align_bars=True)
            detector.assert_not_called()
            root = ET.parse(target).getroot()
            self.assertEqual(root.findtext('.//time/beats'), '6')
            self.assertEqual(root.findtext('.//time/beat-type'), '8')

    def test_manual_bar_line_survives_tracker_failure(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'in.mid', Path(folder) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=90, time=480),
                                               mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            with patch('omnizart.desktop.beat_alignment.detect', side_effect=RuntimeError('no model')):
                with self.assertWarnsRegex(UserWarning, 'constant-tempo grid'):
                    export_score(source, target, audio='song.wav', align_bars=True, tempo=120, downbeat=.5)
            root = ET.parse(target).getroot()
            self.assertEqual(next(m for m in root.findall('.//part/measure') if m.find('.//pitch') is not None).get('number'), '2')

    def test_unusable_audio_grid_warns_and_keeps_score(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'in.mid', Path(folder) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=90),
                                               mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            with patch('omnizart.desktop.beat_alignment.detect', side_effect=RuntimeError('no model')):
                with self.assertWarnsRegex(UserWarning, 'Audio bar alignment unavailable'):
                    export_score(source, target, audio='song.wav', align_bars=True)
            self.assertTrue(target.is_file())

    def test_irregular_beats_use_explicit_estimated_fallback(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'in.mid', Path(folder) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=90),
                                               mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            analysis = {}
            with patch('omnizart.desktop.beat_alignment.detect', return_value=([0, .3, .42, .6, .9], [.3])):
                with self.assertWarnsRegex(UserWarning, 'constant-tempo grid'):
                    export_score(source, target, audio='song.wav', align_bars=True, analysis_out=analysis)
            self.assertIsNotNone(analysis['grid'])
            self.assertAlmostEqual(analysis['grid'].anchor, .3)
            self.assertIn('Review the timing', analysis['alignment_warning'])


if __name__ == '__main__':
    unittest.main()

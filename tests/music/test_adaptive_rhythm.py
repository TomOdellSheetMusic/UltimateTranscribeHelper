from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

import mido
from omnizart.desktop.notation import export_score
from omnizart.desktop.rhythm import pieces, triplet_beats


class AdaptiveRhythmTests(unittest.TestCase):
    def test_mixed_grid_spans_are_all_representable(self):
        # All starts and ends from alternating triplet and straight beats,
        # including notes crossing beats and bar lines.
        positions = sorted({0, 4, 8, 12, 15, 18, 21, 24, 28, 32, 36, 39, 42, 45, 48, 52, 56, 60})
        for index, start in enumerate(positions):
            for end in positions[index + 1:]:
                with self.subTest(start=start, end=end):
                    result = list(pieces(start, end, 48))
                    self.assertEqual(result[0][0], start)
                    self.assertEqual(result[-1][1], end)
                    self.assertTrue(all(a[1] == b[0] for a, b in zip(result, result[1:])))
    def test_evidence_requires_multiple_distinct_third_positions(self):
        self.assertEqual(triplet_beats([0, .25, .5, .75, 1]), set())
        self.assertEqual(triplet_beats([0, .333, 1, 1.5]), set())
        self.assertEqual(triplet_beats([0, 1/3, 1/2, 2/3]), {0})
        # These measured Best of Friends onsets are expressive straight time,
        # not a quarter-note triplet despite two third-like positions.
        self.assertEqual(triplet_beats([28.016, 28.312, 28.947, 29.401, 29.43, 29.479, 29.64, 29.957]), set())
        self.assertEqual(triplet_beats([0, 2/3, 4/3, 2]), {0, 1})

    def test_quarter_triplets_are_written_with_correct_durations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch in (60, 62, 64):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
                track.append(mido.Message('note_off', note=pitch, time=320))
            midi.tracks.append(track)
            source, score = root / 'triplets.mid', root / 'score.musicxml'
            midi.save(source)
            original = source.read_bytes()
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual(xml.findtext('.//divisions'), '12')
            self.assertEqual([n.findtext('type') for n in notes], ['quarter'] * 3)
            self.assertEqual([n.findtext('duration') for n in notes], ['8'] * 3)
            self.assertTrue(all(n.findtext('time-modification/actual-notes') == '3' for n in notes))
            self.assertEqual([x.get('type') for x in xml.findall('.//tuplet')], ['start', 'stop'])
            self.assertEqual(source.read_bytes(), original)

    def test_quarter_triplet_can_start_on_an_odd_beat(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for index, pitch in enumerate((60, 62, 64)):
                track.append(mido.Message('note_on', note=pitch, velocity=80, time=480 if index == 0 else 0))
                track.append(mido.Message('note_off', note=pitch, time=320))
            midi.tracks.append(track)
            source, score = root / 'triplets.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual([n.findtext('duration') for n in notes], ['8'] * 3)
            self.assertEqual([x.get('type') for x in xml.findall('.//tuplet')], ['start', 'stop'])

    def test_straight_sixteenths_do_not_become_triplets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch in (60, 62, 64, 65):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
                track.append(mido.Message('note_off', note=pitch, time=120))
            midi.tracks.append(track)
            source, score = root / 'straight.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            self.assertEqual(xml.findtext('.//divisions'), '4')
            self.assertFalse(xml.findall('.//time-modification'))

    def test_triplets_and_straight_sixteenths_can_share_a_measure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for index, pitch in enumerate((60, 62, 64, 65, 67, 69, 71)):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
                track.append(mido.Message('note_off', note=pitch, time=160 if index < 3 else 120))
            midi.tracks.append(track)
            source, score = root / 'mixed.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, piano=False, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual([n.findtext('duration') for n in notes], ['4'] * 3 + ['3'] * 4)
            self.assertEqual([n.findtext('type') for n in notes], ['eighth'] * 3 + ['16th'] * 4)
            self.assertEqual(sum(n.find('time-modification') is not None for n in notes), 3)

    def test_simultaneous_straight_and_triplet_hands_keep_both_rhythms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            events = []
            for start, end, pitch in ((0,160,72),(160,320,74),(320,480,76),
                                      (0,240,48),(240,480,50)):
                events.extend(((start, 1, pitch), (end, 0, pitch)))
            track, previous = mido.MidiTrack(), 0
            for tick, on, pitch in sorted(events):
                track.append(mido.Message('note_on' if on else 'note_off', note=pitch,
                                          velocity=80 if on else 0, time=tick - previous))
                previous = tick
            midi.tracks.append(track)
            source, score = root / 'poly.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            upper = [n for n in notes if n.findtext('staff') == '1']
            lower = [n for n in notes if n.findtext('staff') == '2']
            self.assertEqual([n.findtext('duration') for n in upper], ['4'] * 3)
            self.assertEqual([n.findtext('duration') for n in lower], ['6'] * 2)
            self.assertTrue(all(n.find('time-modification') is not None for n in upper))
            self.assertTrue(all(n.find('time-modification') is None for n in lower))

    def test_reference_style_quarter_triplet_chords_stay_on_one_staff(self):
        # Three quarter-note triplet chords from the rhythm and pitches in
        # measure 28 of the user's Go Tell Her Now score.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for _ in range(3):
                for pitch in (56, 61, 64):
                    track.append(mido.Message('note_on', note=pitch, velocity=80))
                for index, pitch in enumerate((56, 61, 64)):
                    track.append(mido.Message('note_off', note=pitch, time=320 if index == 0 else 0))
            midi.tracks.append(track)
            source, score = root / 'reference.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual(len(notes), 9)
            self.assertEqual(len({n.findtext('staff') for n in notes}), 1)
            self.assertTrue(all(n.findtext('type') == 'quarter' and n.findtext('duration') == '8' for n in notes))
            self.assertEqual([x.get('type') for x in xml.findall('.//tuplet')], ['start', 'stop'])

    def test_reference_style_triplets_on_both_piano_staves(self):
        # The same reference passage has octave bass and triad chords together.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            pitches = (37, 49, 56, 61, 64)
            for _ in range(3):
                for pitch in pitches:
                    track.append(mido.Message('note_on', note=pitch, velocity=80))
                for index, pitch in enumerate(pitches):
                    track.append(mido.Message('note_off', note=pitch, time=320 if index == 0 else 0))
            midi.tracks.append(track)
            source, score = root / 'both-hands.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual(len(notes), 15)
            self.assertEqual(sum(n.findtext('staff') == '1' for n in notes), 9)
            self.assertEqual(sum(n.findtext('staff') == '2' for n in notes), 6)
            self.assertTrue(all(n.findtext('duration') == '8' for n in notes))
            self.assertEqual(len(xml.findall('.//tuplet[@type="start"]')), 2)
            self.assertEqual(len(xml.findall('.//tuplet[@type="stop"]')), 2)

    def test_triplets_preserve_manual_downbeat_and_partial_opening_bar(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for index, pitch in enumerate((60, 62, 64)):
                track.append(mido.Message('note_on', note=pitch, velocity=80, time=96 if index == 0 else 0))
                track.append(mido.Message('note_off', note=pitch, time=320))
            midi.tracks.append(track)
            source, score = root / 'pickup.mid', root / 'score.musicxml'
            midi.save(source)
            export_score(source, score, tempo=120, downbeat=.1, piano=False, score_key='C')
            xml = ET.parse(score).getroot()
            measures = xml.findall('.//part/measure')
            self.assertEqual(measures[0].get('implicit'), 'yes')
            self.assertEqual(sum(int(n.findtext('duration')) for n in measures[0].findall('note')), 3)
            notes = [n for n in measures[1].findall('note') if n.find('pitch') is not None]
            self.assertEqual([n.findtext('duration') for n in notes], ['8'] * 3)
            self.assertEqual(xml.findtext('.//metronome/per-minute'), '120')

    def test_triplets_follow_variable_audio_beat_grid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch, duration in ((60, 320), (62, 352), (64, 384)):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
                track.append(mido.Message('note_off', note=pitch, time=duration))
            midi.tracks.append(track)
            source, score = root / 'varying.mid', root / 'score.musicxml'
            midi.save(source)
            beats = [0, .5, 1.1, 1.6, 2.2, 2.8, 3.4, 4, 4.6]
            with patch('omnizart.desktop.beat_alignment.detect', return_value=(beats, [0, 2.2, 4.6])):
                export_score(source, score, audio='song.wav', align_bars=True, piano=False, score_key='C')
            xml = ET.parse(score).getroot()
            notes = [n for n in xml.findall('.//note') if n.find('pitch') is not None]
            self.assertEqual([n.findtext('duration') for n in notes], ['8'] * 3)
            directions = xml.findall('.//direction')
            self.assertTrue(any(d.findtext('offset') == '12' and abs(float(d.find('sound').get('tempo')) - 100) < .01
                                for d in directions if d.find('sound') is not None))


if __name__ == '__main__':
    unittest.main()

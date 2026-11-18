"""Tests for independent MIDI to MusicXML score generation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET
import mido
from omnizart.desktop.notation import export_score, _estimate_tempo, _piano_voices, _assign_piano_hands, _simplify_piano_hands, _pitch


class TestKeyAwarePitchSpelling(unittest.TestCase):
    def test_diatonic_enharmonics_and_octaves(self):
        from xml.etree import ElementTree as ET
        for midi, fifths, expected in ((65, 6, ('E', '1', '4')),
                                       (60, 7, ('B', '1', '3')),
                                       (59, -7, ('C', '-1', '4')),
                                       (64, -7, ('F', '-1', '4'))):
            with self.subTest(midi=midi, fifths=fifths):
                note = ET.Element('note')
                _pitch(note, midi, fifths)
                self.assertEqual(tuple(note.findtext('pitch/' + name) for name in ('step', 'alter', 'octave')), expected)

    def test_every_key_and_midi_pitch_preserves_sounding_pitch(self):
        from xml.etree import ElementTree as ET
        natural = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        for fifths in range(-7, 8):
            for midi in range(128):
                note = ET.Element('note')
                _pitch(note, midi, fifths)
                step = note.findtext('pitch/step')
                alter = int(note.findtext('pitch/alter', '0'))
                octave = int(note.findtext('pitch/octave'))
                self.assertEqual((octave + 1) * 12 + natural[step] + alter, midi)


class NativeScoreTests(unittest.TestCase):
    def test_downbeat_just_after_audio_start_does_not_create_empty_bar(self):
        with TemporaryDirectory() as directory:
            path, output = Path(directory) / 'input.mid', Path(directory) / 'score.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', note=60, velocity=80, time=10),
                mido.Message('note_off', note=60, time=480)]))
            midi.save(path)
            export_score(path, output, tempo=120, downbeat=.01, score_key='C')
            measures = ET.parse(output).findall('.//part/measure')
            self.assertIsNotNone(measures[0].find('.//note/pitch'))

    def test_partial_opening_measure_preserves_audio_start_time(self):
        with TemporaryDirectory() as directory:
            path, output = Path(directory) / 'input.mid', Path(directory) / 'score.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', note=60, velocity=80, time=960),
                mido.Message('note_off', note=60, time=480)]))
            midi.save(path)
            source = path.read_bytes()
            export_score(path, output, tempo=120, downbeat=1, score_key='C')
            measures = ET.parse(output).findall('.//part/measure')
            self.assertEqual(measures[0].get('implicit'), 'yes')
            self.assertEqual(sum(int(n.findtext('duration')) for n in measures[0].findall('note')
                                 if n.findtext('voice') == '1'), 8)
            self.assertTrue(all(b.findtext('duration') == '8' for b in measures[0].findall('backup')))
            self.assertIsNotNone(measures[1].find('.//note/pitch'))
            self.assertEqual(path.read_bytes(), source)

    def test_explicit_flat_key_writes_signature_and_flat_spelling(self):
        with TemporaryDirectory() as directory:
            path, output = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=70, velocity=80),
                                               mido.Message('note_off', note=70, time=480)]))
            midi.save(path)
            export_score(path, output, tempo=120, score_key='Bb')
            root = ET.parse(output).getroot()
            self.assertEqual(root.findtext('.//key/fifths'), '-2')
            self.assertEqual(root.findtext('.//key/mode'), 'major')
            self.assertEqual(root.findtext('.//note/pitch/step'), 'B')
            self.assertEqual(root.findtext('.//note/pitch/alter'), '-1')

    def test_key_estimation_does_not_claim_a_key_for_one_note(self):
        from omnizart.desktop.score_key import estimate_key
        self.assertIsNone(estimate_key({(0, 0): [(0, 1, 60, 80)]}))

    def test_simplified_piano_score_keeps_full_score_and_midi(self):
        with TemporaryDirectory() as directory:
            midi_path = Path(directory) / 'input.mid'
            full = Path(directory) / 'full.musicxml'
            simplified = Path(directory) / 'simplified.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch in range(48, 60):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
            for index, pitch in enumerate(range(48, 60)):
                track.append(mido.Message('note_off', note=pitch, time=480 if index == 0 else 0))
            midi.tracks.append(track)
            midi.save(midi_path)
            original = midi_path.read_bytes()
            with self.assertWarns(UserWarning):
                export_score(midi_path, full, tempo=120)
            full_bytes = full.read_bytes()
            with self.assertWarnsRegex(UserWarning, 'Score simplification omitted'):
                export_score(midi_path, simplified, tempo=120, simplify=True)
            full_count = len(ET.parse(full).findall('.//note/pitch'))
            simplified_count = len(ET.parse(simplified).findall('.//note/pitch'))
            self.assertEqual(full_count, 12)
            self.assertLess(simplified_count, full_count)
            self.assertEqual(full.read_bytes(), full_bytes)
            self.assertEqual(midi_path.read_bytes(), original)

    def test_piano_simplification_reports_omissions_and_preserves_source_notes(self):
        source = {1: [(0, 4, pitch, 80) for pitch in (60, 62, 64, 65, 67, 72)],
                  2: [(0, 4, 36, 80), (0, 4, 60, 80)]}
        original = {staff: list(notes) for staff, notes in source.items()}
        with self.assertWarnsRegex(UserWarning, 'omitted.*measures 1') as warning:
            result = _simplify_piano_hands(source, 16)
        self.assertIn('Omitted notes (measure, hand, MIDI pitch):', str(warning.warning))
        self.assertEqual(source, original)
        for staff in result:
            pitches = [note[2] for note in result[staff]]
            self.assertLessEqual(len(set(pitches)), 5)
            self.assertLessEqual(max(pitches) - min(pitches), 12)
        self.assertIn(72, [note[2] for note in result[1]])
        self.assertIn(36, [note[2] for note in result[2]])

    def test_overlapping_note_lengths_get_independent_voices(self):
        with TemporaryDirectory() as directory:
            midi_file = Path(directory) / 'input.mid'
            output = Path(directory) / 'score.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage('set_tempo', tempo=500000),
                mido.Message('note_on', note=64, velocity=90),
                mido.Message('note_on', note=67, velocity=90),
                mido.Message('note_off', note=67, time=480),
                mido.Message('note_on', note=69, velocity=90),
                mido.Message('note_off', note=69, time=480),
                mido.Message('note_off', note=64),
            ]))
            midi.save(midi_file)
            original = midi_file.read_bytes()
            export_score(midi_file, output, tempo=120)
            root = ET.parse(output).getroot()
            notes = [n for n in root.findall('.//note') if n.find('pitch') is not None]
            held = [n for n in notes if n.findtext('pitch/step') == 'E']
            self.assertEqual(len(held), 1)
            self.assertEqual(held[0].findtext('type'), 'half')
            self.assertIsNone(held[0].find('tie'))
            self.assertGreaterEqual(len({n.findtext('voice') for n in notes}), 2)
            self.assertEqual(midi_file.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                export_score(midi_file, output)

    def test_score_skips_percussion_and_preserves_original_midi(self):
        with TemporaryDirectory() as directory:
            path, result = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', channel=0, note=60, velocity=90),
                mido.Message('note_on', channel=9, note=36, velocity=90),
                mido.Message('note_off', channel=9, note=36, time=240),
                mido.Message('note_off', channel=0, note=60, time=240),
            ]))
            midi.save(path)
            original = path.read_bytes()
            with self.assertWarnsRegex(UserWarning, 'Percussion tracks are omitted'):
                export_score(path, result, tempo=120)
            root = ET.parse(result).getroot()
            self.assertEqual([n.findtext('pitch/step') for n in root.findall('.//note') if n.find('pitch') is not None], ['C'])
            self.assertEqual(path.read_bytes(), original)

    def test_simplify_score_avoids_voice_limit_without_changing_midi(self):
        with TemporaryDirectory() as directory:
            path, output = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            for pitch in range(60, 65):
                track.append(mido.Message('note_on', note=pitch, velocity=20 if pitch == 60 else 100))
            for pitch in range(60, 65):
                track.append(mido.Message('note_off', note=pitch, time=120))
            midi.tracks.append(track)
            midi.save(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Enable Simplify'):
                export_score(path, output, tempo=120, piano=False)
            self.assertFalse(output.exists())
            with self.assertWarnsRegex(UserWarning, 'omitted 1 note'):
                export_score(path, output, tempo=120, piano=False, simplify=True)
            root = ET.parse(output).getroot()
            self.assertEqual(len({n.findtext('voice') for n in root.findall('.//note') if n.find('pitch') is not None}), 4)
            self.assertEqual(path.read_bytes(), original)

    def test_percussion_only_reports_why_there_is_no_score(self):
        with TemporaryDirectory() as directory:
            path, result = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', channel=9, note=36, velocity=90),
                                               mido.Message('note_off', channel=9, note=36, time=240)]))
            midi.save(path)
            with self.assertRaisesRegex(ValueError, 'only percussion'):
                export_score(path, result)
            self.assertFalse(result.exists())

    def test_tempo_detection_follows_attacks_instead_of_default_midi_tempo(self):
        notes = {(0, 0): [(index * 60 / 154, index * 60 / 154 + .2, 60, 80)
                           for index in range(40)]}
        self.assertEqual(_estimate_tempo(notes, 120), 154)

    def test_overflow_preserves_staff_and_slices_dense_voices(self):
        notes = {1: [], 2: [(0, length, 40 + length, 80) for length in range(1, 6)]}
        with self.assertWarnsRegex(UserWarning, 'Dense overlapping'):
            voices = _piano_voices(notes)
        self.assertFalse(voices[1])
        self.assertEqual(len(voices[2]), 1)
        # Count note attacks, excluding continuations represented by ties.
        attacks = [pitch[0] for lane in voices[2] for _, _, chord in lane
                   for pitch in chord if not pitch[2]]
        self.assertEqual(sorted(attacks), [41, 42, 43, 44, 45])

    def test_overflow_does_not_add_a_sixth_note_to_the_other_hand(self):
        notes = {1: [(0, 8, pitch, 80) for pitch in range(60, 65)],
                 2: [(0, length, 54 + length, 80) for length in range(1, 6)]}
        with self.assertWarnsRegex(UserWarning, 'Dense overlapping'):
            voices = _piano_voices(notes)
        upper_attacks = [pitch[0] for lane in voices[1] for _, _, chord in lane
                         for pitch in chord if len(pitch) == 2 or not pitch[2]]
        self.assertEqual(sorted(upper_attacks), [60, 61, 62, 63, 64])
        lower_attacks = [pitch[0] for lane in voices[2] for _, _, chord in lane
                         for pitch in chord if len(pitch) == 2 or not pitch[2]]
        self.assertEqual(sorted(lower_attacks), [55, 56, 57, 58, 59])

    def test_voice_overflow_does_not_force_a_wide_span_on_the_other_hand(self):
        notes = {1: [(0, 8, 90, 80)],
                 2: [(0, length, 40 + length, 80) for length in range(1, 6)]}
        with self.assertWarnsRegex(UserWarning, 'Dense overlapping'):
            voices = _piano_voices(notes)
        upper = [pitch[0] for lane in voices[1] for _, _, chord in lane for pitch in chord]
        self.assertEqual(upper, [90])

    def test_neural_staff_suggestions_preserve_notes_and_respect_reach(self):
        entry = (0, 4, 60, 80)
        self.assertEqual(_assign_piano_hands([entry], preferences=[1])[1], [entry])
        self.assertEqual(_assign_piano_hands([entry], preferences=[2])[2], [entry])
        with self.assertRaisesRegex(ValueError, 'do not match'):
            _assign_piano_hands([entry], preferences=[])
        entries = [(0, 4, pitch, 80) for pitch in (48, 60, 72, 84)]
        hands = _assign_piano_hands(entries, preferences=[1, 1, 1, 1])
        self.assertEqual(sorted(hands[1] + hands[2]), sorted(entries))
        for hand in hands.values():
            self.assertLessEqual(max(n[2] for n in hand) - min(n[2] for n in hand), 12)

    def test_hand_assignment_splits_a_wide_chord_between_hands(self):
        entries = [(0, 4, pitch, 80) for pitch in (48, 60, 72, 84)]
        hands = _assign_piano_hands(entries)
        self.assertEqual([note[2] for note in hands[2]], [48, 60])
        self.assertEqual([note[2] for note in hands[1]], [72, 84])

    def test_dense_overlap_preserves_attacks_and_fixed_clefs(self):
        with TemporaryDirectory() as directory:
            path, result = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            # Nine different note ends cannot fit in eight concurrent voices.
            track = mido.MidiTrack()
            for pitch in range(48, 57):
                track.append(mido.Message('note_on', note=pitch, velocity=80))
            for pitch in range(48, 57):
                track.append(mido.Message('note_off', note=pitch, time=120))
            midi.tracks.append(track)
            midi.save(path)
            with self.assertWarnsRegex(UserWarning, 'Dense overlapping'):
                export_score(path, result, tempo=120)
            root = ET.parse(result).getroot()
            notes = [note for note in root.findall('.//note') if note.find('pitch') is not None]
            starts = [note for note in notes if not any(tie.get('type') == 'stop' for tie in note.findall('tie'))]
            self.assertEqual(len(starts), 9)
            self.assertEqual([(clef.get('number'), clef.findtext('sign')) for clef in root.findall('.//clef')], [('1', 'G'), ('2', 'F')])

    def test_ties_across_barlines_and_optional_tempo(self):
        with TemporaryDirectory() as directory:
            path, result = Path(directory) / 'in.mid', Path(directory) / 'out.musicxml'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=70, time=1440),
                                               mido.Message('note_off', note=60, time=960)]))
            midi.save(path)
            export_score(path, result, tempo=120)
            root = ET.parse(result).getroot()
            self.assertEqual([t.get('type') for t in root.findall('.//tie')], ['start', 'stop'])
            self.assertEqual(root.find('.//sound').get('tempo'), '120')
            self.assertEqual(len(root.findall('.//clef')), 2)


if __name__ == '__main__':
    unittest.main()

"""Small MIDI fixtures check timing, safety and preservation during smoothing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mido
from omnizart.desktop.smoothing import smooth_midi


def note_events(track):
    tick, result = 0, []
    for msg in track:
        tick += msg.time
        if msg.type in ("note_on", "note_off"):
            result.append((tick, msg.type, msg.note, msg.velocity))
    return result


class SmoothingTests(unittest.TestCase):
    def test_pedaled_scale_notes_end_at_the_next_attack(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'raw.mid', Path(folder) / 'clean.mid'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('control_change', control=64, value=127),
                mido.Message('note_on', note=60, velocity=80),
                mido.Message('note_on', note=62, velocity=80, time=480),
                mido.Message('note_on', note=64, velocity=80, time=480),
                mido.Message('note_off', note=60, time=480),
                mido.Message('note_off', note=62), mido.Message('note_off', note=64),
                mido.Message('control_change', control=64, value=0)]))
            midi.save(source)
            with self.assertWarnsRegex(UserWarning, 'shortened 2'):
                smooth_midi(source, target, join_notes=False, shorten_pedal=True)
            track = mido.MidiFile(target).tracks[0]
            self.assertEqual([(tick, typ, pitch) for tick, typ, pitch, _ in note_events(track)], [
                (0, 'note_on', 60), (480, 'note_off', 60), (480, 'note_on', 62),
                (960, 'note_off', 62), (960, 'note_on', 64), (1440, 'note_off', 64)])
            self.assertEqual([m.value for m in track if m.type == 'control_change' and m.control == 64], [127, 0])

    def test_pedal_cleanup_does_not_guess_without_pedal_data(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'raw.mid', Path(folder) / 'clean.mid'
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([mido.Message('note_on', note=60, velocity=80),
                                              mido.Message('note_off', note=60, time=480)]))
            midi.save(source)
            with self.assertWarnsRegex(UserWarning, 'no complete sustain-pedal'):
                smooth_midi(source, target, join_notes=False, shorten_pedal=True)
            self.assertEqual(note_events(mido.MidiFile(source).tracks[0]), note_events(mido.MidiFile(target).tracks[0]))

    def test_quiet_notes_are_removed_or_joined_to_a_matching_recent_note(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / 'raw.mid', Path(folder) / 'clean.mid'
            midi = mido.MidiFile(ticks_per_beat=500)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', note=60, velocity=80),
                mido.Message('note_off', note=60, time=100),
                mido.Message('note_on', note=60, velocity=10, time=10),
                mido.Message('note_off', note=60, time=100),
                mido.Message('note_on', note=62, velocity=15, time=10),
                mido.Message('note_off', note=62, time=30),
                mido.Message('note_on', note=64, velocity=16, time=10),
                mido.Message('note_off', note=64, time=30)]))
            midi.save(source)
            original = source.read_bytes()
            self.assertEqual(smooth_midi(source, target, join_notes=False, quiet_velocity=15), 1)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(note_events(mido.MidiFile(target).tracks[0]), [
                (0, 'note_on', 60, 80), (210, 'note_off', 60, 64),
                (260, 'note_on', 64, 16), (290, 'note_off', 64, 64)])

    def test_merges_fragments_and_preserves_tracks_and_metadata(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0),
                                               mido.MetaMessage("set_tempo", tempo=1000000, time=240)]))
            piano = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano", time=0),
                                    mido.Message("note_on", note=60, velocity=90, time=0),
                                    mido.Message("note_off", note=60, velocity=0, time=48),
                                    mido.Message("control_change", control=7, value=80, time=5),
                                    mido.Message("note_on", note=60, velocity=70, time=5),
                                    mido.Message("note_off", note=60, velocity=0, time=144),
                                    mido.Message("note_on", note=62, velocity=80, time=38),
                                    mido.Message("note_off", note=62, velocity=0, time=48),
                                    mido.Message("note_on", note=62, velocity=80, time=20),
                                    mido.Message("note_off", note=62, velocity=0, time=48)])
            vocal = mido.MidiTrack([mido.MetaMessage("track_name", name="Vocals", time=0),
                                    mido.Message("note_on", note=70, velocity=70, time=0),
                                    mido.Message("note_off", note=70, time=48),
                                    mido.Message("note_on", note=70, velocity=70, time=1),
                                    mido.Message("note_off", note=70, time=48)])
            midi.tracks.extend([piano, vocal])
            midi.save(str(source))
            original = source.read_bytes()
            self.assertEqual(smooth_midi(source, target, 30), 1)
            self.assertEqual(source.read_bytes(), original)
            result = mido.MidiFile(str(target))
            self.assertEqual(note_events(result.tracks[1]), [(0, "note_on", 60, 90), (202, "note_off", 60, 0),
                                                             (240, "note_on", 62, 80), (288, "note_off", 62, 0),
                                                             (308, "note_on", 62, 80), (356, "note_off", 62, 0)])
            self.assertEqual(note_events(result.tracks[2]), note_events(vocal))
            self.assertEqual([m.tempo for m in result.tracks[0] if m.type == "set_tempo"], [500000, 1000000])
            self.assertTrue(any(m.type == "control_change" and m.control == 7 for m in result.tracks[1]))

    def test_sustain_pitch_bends_drums_and_ambiguous_overlap_are_untouched(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            midi = mido.MidiFile()
            for control in (mido.Message("control_change", control=64, value=127), mido.Message("pitchwheel", pitch=100)):
                midi.tracks.append(mido.MidiTrack([control, mido.Message("note_on", note=60, velocity=80),
                    mido.Message("note_off", note=60, time=20), mido.Message("note_on", note=60, velocity=80, time=1),
                    mido.Message("note_off", note=60, time=20)]))
            midi.tracks.append(mido.MidiTrack([mido.Message("note_on", channel=9, note=35, velocity=80),
                mido.Message("note_off", channel=9, note=35, time=20),
                mido.Message("note_on", channel=9, note=35, velocity=80, time=1),
                mido.Message("note_off", channel=9, note=35, time=20)]))
            midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=60, velocity=80),
                mido.Message("note_on", note=60, velocity=80, time=5), mido.Message("note_off", note=60, time=5),
                mido.Message("note_off", note=60, time=5)]))
            midi.save(str(source))
            self.assertEqual(smooth_midi(source, target), 0)
            result = mido.MidiFile(str(target))
            for before, after in zip(midi.tracks, result.tracks):
                self.assertEqual(note_events(before), note_events(after))

    def test_refuses_to_overwrite_output(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            mido.MidiFile().save(str(source))
            target.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                smooth_midi(source, target)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_pedal_in_another_track_only_protects_affected_passage(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            midi = mido.MidiFile(type=1)
            midi.tracks.append(mido.MidiTrack([mido.Message("control_change", control=64, value=127, time=0),
                                               mido.Message("control_change", control=64, value=0, time=100)]))
            track = mido.MidiTrack([
                mido.Message("note_on", note=60, velocity=80, time=0), mido.Message("note_off", note=60, time=20),
                mido.Message("note_on", note=60, velocity=80, time=1), mido.Message("note_off", note=60, time=20),
                mido.Message("note_on", note=62, velocity=80, time=99), mido.Message("note_off", note=62, time=20),
                mido.Message("note_on", note=62, velocity=80, time=1), mido.Message("note_off", note=62, time=20)])
            midi.tracks.append(track)
            midi.save(str(source))
            self.assertEqual(smooth_midi(source, target), 1)
            notes = note_events(mido.MidiFile(str(target)).tracks[1])
            self.assertEqual(len(notes), 6)
            self.assertEqual([event[2] for event in notes], [60, 60, 60, 60, 62, 62])

    def test_short_note_filter_can_run_without_merging(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            midi = mido.MidiFile(ticks_per_beat=500)
            midi.tracks.append(mido.MidiTrack([
                mido.Message("note_on", note=60, velocity=80), mido.Message("note_off", note=60, time=20),
                mido.Message("note_on", note=62, velocity=80, time=1), mido.Message("note_off", note=62, time=100),
                mido.Message("note_on", note=62, velocity=80, time=0), mido.Message("note_off", note=62, time=100)]))
            midi.save(str(source))
            self.assertEqual(smooth_midi(source, target, min_note_ms=40, join_notes=False), 0)
            notes = note_events(mido.MidiFile(str(target)).tracks[0])
            self.assertEqual([event[2] for event in notes], [62, 62, 62, 62])

    def test_fragments_are_joined_before_short_note_filter(self):
        with TemporaryDirectory() as folder:
            source, target = Path(folder) / "raw.mid", Path(folder) / "clean.mid"
            midi = mido.MidiFile(ticks_per_beat=500)
            midi.tracks.append(mido.MidiTrack([
                mido.Message("note_on", note=60, velocity=80), mido.Message("note_off", note=60, time=20),
                mido.Message("note_on", note=60, velocity=80, time=10), mido.Message("note_off", note=60, time=20)]))
            midi.save(str(source))
            self.assertEqual(smooth_midi(source, target, max_gap_ms=30, min_note_ms=40), 1)
            self.assertEqual(len(note_events(mido.MidiFile(str(target)).tracks[0])), 2)

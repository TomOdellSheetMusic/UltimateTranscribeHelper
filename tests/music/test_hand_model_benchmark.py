"""Check that research comparisons preserve MIDI and interpret score ties."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import mido
from scripts.benchmark_hand_models import assemble, prepare, records_from_hands, reference_midi


class HandModelBenchmarkTests(unittest.TestCase):
    def test_assembled_predictions_keep_notes_and_pedal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mid'
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=600000)]))
            for name, channel, pitch in [('Right hand', 0, 72), ('Left hand', 1, 48)]:
                midi.tracks.append(mido.MidiTrack([
                    mido.MetaMessage('track_name', name=name),
                    mido.Message('control_change', channel=channel, control=64, value=127),
                    mido.Message('note_on', channel=channel, note=pitch, velocity=80, time=10),
                    mido.Message('note_off', channel=channel, note=pitch, time=480),
                    mido.Message('control_change', channel=channel, control=64, value=0, time=20)]))
            midi.save(source)
            folder = root / 'prepared'
            prepare(source, folder)
            records = json.loads((folder / 'notes.json').read_text())
            target = root / 'output.mid'
            assemble(source, target, records, [3 - record['hand'] for record in records])
            result, notes = records_from_hands(target)
            self.assertEqual(sorted((n['pitch'], n['hand']) for n in notes), [(48, 1), (72, 2)])
            for track in result.tracks[-2:]:
                self.assertEqual([msg.value for msg in track if msg.type == 'control_change'], [127, 0])
            self.assertEqual(result.tracks[0][0].tempo, 600000)

    def test_reference_joins_ties_and_preserves_written_staves(self):
        xml = '''<score-partwise><part><measure number="1"><attributes><divisions>4</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
        <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><tie type="start"/></note>
        <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><tie type="stop"/></note>
        <backup><duration>8</duration></backup>
        <note><pitch><step>C</step><octave>3</octave></pitch><duration>8</duration><voice>2</voice><staff>2</staff></note>
        </measure></part></score-partwise>'''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'score.musicxml', root / 'reference.mid'
            source.write_text(xml, encoding='utf-8')
            reference_midi(source, target)
            _, notes = records_from_hands(target)
            self.assertEqual(sorted((n['pitch'], n['hand'], n['start'], n['end']) for n in notes),
                             [(48, 2, 0, 960), (72, 1, 0, 960)])


if __name__ == '__main__':
    unittest.main()

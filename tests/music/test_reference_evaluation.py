import tempfile
import unittest
import zipfile
from pathlib import Path

import mido

from scripts.evaluate_reference import align_audio, compare, reference_notes


def make_score(path, extra=''):
    xml = f'''<museScore version="3.02"><Score><Staff id="1"><Measure><voice>
        <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
        <Tempo><tempo>2</tempo></Tempo>{extra}
        <Chord><durationType>quarter</durationType><Note><pitch>60</pitch></Note></Chord>
        <Chord><durationType>quarter</durationType><Note><pitch>62</pitch></Note></Chord>
        <Rest><durationType>half</durationType></Rest>
        </voice></Measure></Staff></Score></museScore>'''
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('score.mscx', xml)


class TestReferenceEvaluation(unittest.TestCase):
    def test_reference_and_prediction_use_same_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            score, midi = Path(directory) / 'score.mscz', Path(directory) / 'pred.mid'
            make_score(score)
            reference = reference_notes(score, 1)
            self.assertEqual(reference, [(0, 60), (.5, 62)])
            result = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            result.tracks.append(track)
            track.extend([mido.Message('note_on', note=60, velocity=90, time=480),
                          mido.Message('note_off', note=60, time=240),
                          mido.Message('note_on', note=62, velocity=90, time=240),
                          mido.Message('note_off', note=62, time=240)])
            result.save(midi)
            comparison = compare(reference, midi, .5, 1, 1, .05)
            self.assertEqual((comparison['matched'], comparison['predicted']), (2, 2))
            self.assertEqual(comparison['f1'], 1)

    def test_rejects_unsupported_notation(self):
        with tempfile.TemporaryDirectory() as directory:
            score = Path(directory) / 'score.mscz'
            make_score(score, '<Tuplet/>')
            with self.assertRaisesRegex(ValueError, 'Tuplets'):
                reference_notes(score, 1)

    def test_legacy_tied_continuation_is_not_a_new_attack(self):
        with tempfile.TemporaryDirectory() as directory:
            score = Path(directory) / 'score.mscz'
            xml = '''<museScore version="2.06"><Score><Staff id="1"><Measure>
                <TimeSig><sigN>3</sigN><sigD>4</sigD></TimeSig><Tempo><tempo>2</tempo></Tempo>
                <Chord><durationType>quarter</durationType><Note><Tie id="7"/><pitch>60</pitch></Note></Chord>
                <Chord><durationType>quarter</durationType><Note><endSpanner id="7"/><pitch>60</pitch></Note></Chord>
                <Rest><durationType>quarter</durationType></Rest>
                </Measure></Staff></Score></museScore>'''
            with zipfile.ZipFile(score, 'w') as archive:
                archive.writestr('score.mscx', xml)
            self.assertEqual(reference_notes(score, 1.5), [(0, 60)])

    def test_audio_alignment_uses_onsets_independently_of_predictions(self):
        import numpy as np
        import soundfile as sf
        rate = 22050
        samples = np.zeros(rate * 7, dtype=np.float32)
        for time in (1, 2, 3, 4, 5):
            start = int(time * rate)
            samples[start:start + 150] = np.hanning(150)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'clicks.wav'
            sf.write(audio, samples, rate)
            offset, scale, strength = align_audio([(float(i), 60) for i in range(5)], audio, .7, 1.3, .98, 1.02)
        self.assertAlmostEqual(offset, 1, delta=.12)
        self.assertGreater(strength, 0)
        self.assertTrue(.98 <= scale <= 1.02)


if __name__ == '__main__':
    unittest.main()

"""Compare MIDI predictions with a limited, explicitly aligned MuseScore reference.

This is a research tool, not a general MuseScore importer. Unsupported notation
is rejected rather than quietly turning an invalid reference into a benchmark.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import warnings
import zipfile
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnizart.desktop.compare import _notes


DURATIONS = {'whole': 4, 'half': 2, 'quarter': 1, 'eighth': .5,
             '16th': .25, '32nd': .125, '64th': .0625}


def reference_notes(path, seconds):
    """Read a simple constant-tempo MuseScore excerpt; reject unsupported notation."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('.mscx')]
        if len(names) != 1:
            raise ValueError('Expected exactly one score in the MuseScore archive.')
        root = ET.fromstring(archive.read(names[0]))
    score = root.find('Score')
    if score is None:
        raise ValueError('No score found.')
    staves = score.findall('Staff')
    if not staves:
        raise ValueError('No staves found.')
    tempos = [float(e.text) for e in score.findall('.//Tempo/tempo') if e.text]
    if not tempos:
        raise ValueError('An explicit tempo is required.')
    tempo = tempos[0]
    first_measure = staves[0].find('Measure')
    if first_measure is None or first_measure.find('.//Tempo/tempo') is None:
        raise ValueError('The first measure must specify the tempo.')
    if tempo <= 0:
        raise ValueError('Invalid tempo.')
    notes = []
    signatures = score.findall('.//TimeSig')
    if not signatures:
        raise ValueError('An explicit time signature is required.')
    numerator, denominator = int(signatures[0].findtext('sigN')), int(signatures[0].findtext('sigD'))
    if numerator not in (3, 4) or denominator != 4:
        raise ValueError('Only 3/4 and 4/4 are supported.')
    measure_beats = numerator
    ties = {}
    for note in score.findall('.//Chord/Note'):
        for tie in note.findall('Tie'):
            tie_id = tie.get('id')
            if tie_id is None or tie_id in ties:
                raise ValueError('Ambiguous tie in reference.')
            ties[tie_id] = int(note.findtext('pitch'))
    for staff in staves:
        beat = 0
        for measure in staff.findall('Measure'):
            if beat / tempo >= seconds:
                break
            if any(abs(float(e.text) - tempo) > 1e-6 for e in measure.findall('.//Tempo/tempo') if e.text):
                raise ValueError('Tempo changes within the excerpt are not supported.')
            if measure.get('len') is not None:
                raise ValueError('Irregular measures are not supported.')
            if any(measure.findall('.//' + tag) for tag in
                   ('Tuplet', 'startRepeat', 'endRepeat', 'Jump', 'Marker', 'tick', 'location', 'Volta', 'RepeatMeasure')):
                raise ValueError('Tuplets, repeats and explicit position changes are not supported.')
            if any('repeat' in (bar.findtext('subtype') or '').lower() for bar in measure.findall('.//BarLine')):
                raise ValueError('Repeat barlines are not supported.')
            for sig in measure.findall('.//TimeSig'):
                if (sig.findtext('sigN'), sig.findtext('sigD')) != (str(numerator), '4'):
                    raise ValueError('Time signature changes are not supported.')
            voices = measure.findall('voice')
            if not voices:
                if any(event.find('voice') is not None for event in measure.findall('Chord') + measure.findall('Rest')):
                    raise ValueError('Legacy voices are not supported.')
                voices = [measure]
            for voice in voices:
                cursor = beat
                for event in voice:
                    if event.tag not in ('Chord', 'Rest'):
                        continue
                    kind = event.findtext('durationType')
                    if kind == 'measure' and event.tag == 'Rest':
                        duration = measure_beats
                    elif kind in DURATIONS:
                        duration = DURATIONS[kind]
                    else:
                        raise ValueError('Unsupported duration: ' + str(kind))
                    duration *= sum(.5 ** i for i in range(int(event.findtext('dots', '0')) + 1))
                    if event.find('Tuplet') is not None:
                        raise ValueError('Tuplets are not supported.')
                    for note in event.findall('Note'):
                        pitch = int(note.findtext('pitch'))
                        if note.find('Spanner[@type="Tie"]') is not None:
                            raise ValueError('MuseScore 3 ties are not yet supported.')
                        endings = [end.get('id') for end in note.findall('endSpanner')]
                        if any(tie_id not in ties for tie_id in endings):
                            raise ValueError('Unrecognized note spanner in reference.')
                        tied_from_before = any(tie_id in ties for tie_id in endings)
                        if any(tie_id in ties and ties[tie_id] != pitch for tie_id in endings):
                            raise ValueError('Tie connects different pitches.')
                        onset = cursor / tempo
                        if onset < seconds and not tied_from_before:
                            notes.append((onset, pitch))
                    cursor += duration
                if cursor > beat + measure_beats + 1e-6:
                    raise ValueError('A voice exceeds its measure.')
            beat += measure_beats
    if not notes:
        raise ValueError('No reference notes in the excerpt.')
    return sorted(notes)


def compare(reference, prediction, offset, scale, duration, tolerance, start=0):
    """Maximize one-to-one onset matches per pitch under a fixed alignment."""
    if scale <= 0 or duration <= 0 or tolerance <= 0:
        raise ValueError('Scale, duration and tolerance must be positive.')
    predicted = [(t, pitch) for t, pitch, _ in _notes(prediction)
                 if offset + scale * start <= t < offset + scale * (start + duration)]
    by_pitch, expected_by_pitch = defaultdict(list), defaultdict(list)
    for time, pitch in predicted:
        by_pitch[pitch].append(time)
    for time, pitch in reference:
        expected_by_pitch[pitch].append(offset + scale * time)
    matched = 0
    for pitch, expected_times in expected_by_pitch.items():
        actual_times = sorted(by_pitch[pitch])
        i = j = 0
        expected_times.sort()
        while i < len(expected_times) and j < len(actual_times):
            difference = actual_times[j] - expected_times[i]
            if difference < -tolerance:
                j += 1
            elif difference > tolerance:
                i += 1
            else:
                matched += 1
                i += 1
                j += 1
    precision = matched / len(predicted) if predicted else 0
    recall = matched / len(reference) if reference else 0
    return {'reference': len(reference), 'predicted': len(predicted), 'matched': matched,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}


def align_audio(reference, audio, offset_min, offset_max, scale_min=.95, scale_max=1.06):
    """Fit score onsets to audio onset strength, without using model predictions.

    Constant tempo only. Repeated passages and non-piano attacks can give a
    convincing but incorrect fit; callers must review the bounds and result.
    """
    import librosa
    import numpy as np

    if offset_min < 0 or offset_max <= offset_min or scale_min <= 0 or scale_max <= scale_min:
        raise ValueError('Invalid alignment search range.')
    onsets = np.array(sorted({time for time, _ in reference}))
    if len(onsets) < 4:
        raise ValueError('At least four distinct reference onsets are required for alignment.')
    samples, rate = librosa.load(str(audio), sr=22050, mono=True)
    if not len(samples):
        raise ValueError('The alignment audio is empty.')
    envelope = librosa.onset.onset_strength(y=samples, sr=rate, hop_length=256)
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=rate, hop_length=256)
    best = None
    for scale in np.arange(scale_min, scale_max + .0001, .002):
        for offset in np.arange(offset_min, offset_max + .0001, .01):
            target = offset + scale * onsets
            if target[-1] > times[-1]:
                continue
            value = float(np.mean(np.interp(target, times, envelope)))
            if best is None or value > best[0]:
                best = (value, float(offset), float(scale))
    if best is None or best[0] <= 0:
        raise ValueError('The audio has no usable onset alignment in this range.')
    if (best[1] <= offset_min + .05 or best[1] >= offset_max - .05 or
            best[2] <= scale_min + .005 or best[2] >= scale_max - .005):
        warnings.warn('Audio alignment is at the edge of the search range; the fit may be wrong.', UserWarning)
    return best[1], best[2], best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('score', type=Path)
    parser.add_argument('predictions', type=Path, nargs='+')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--start', type=float, default=0, help='Start of the score excerpt in written seconds')
    parser.add_argument('--offset', type=float, help='Recording time of score start, in seconds')
    parser.add_argument('--scale', type=float, help='Audio seconds per written-score second')
    parser.add_argument('--align-audio', type=Path, help='Fit a constant offset and scale to audio onset strength')
    parser.add_argument('--offset-range', type=float, nargs=2, metavar=('MIN', 'MAX'))
    parser.add_argument('--scale-range', type=float, nargs=2, default=(.95, 1.06), metavar=('MIN', 'MAX'))
    parser.add_argument('--tolerance', type=float, default=.1)
    args = parser.parse_args()
    if args.start < 0:
        parser.error('--start must be non-negative')
    ref = [note for note in reference_notes(args.score, args.start + args.seconds) if note[0] >= args.start]
    if not ref:
        parser.error('No reference notes in the requested excerpt')
    if args.align_audio:
        if args.offset is not None or args.scale is not None or args.offset_range is None:
            parser.error('--align-audio requires --offset-range and cannot be combined with --offset or --scale')
        offset, scale, strength = align_audio(ref, args.align_audio, *args.offset_range, *args.scale_range)
    else:
        if args.offset is None or args.scale is None:
            parser.error('Provide --offset and --scale, or use --align-audio and --offset-range')
        offset, scale, strength = args.offset, args.scale, None
    results = {str(path): compare(ref, path, offset, scale, args.seconds, args.tolerance, args.start)
               for path in args.predictions}
    print(json.dumps({'start': args.start, 'seconds': args.seconds, 'offset': offset, 'scale': scale,
                      'audio_onset_strength': strength,
                      'tolerance': args.tolerance, 'results': results}, indent=2))


if __name__ == '__main__':
    main()

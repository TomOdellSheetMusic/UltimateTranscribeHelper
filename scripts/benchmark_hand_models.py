"""Run independent hand models on a shared MIDI and preserve its timing.

This research script expects external repositories and checkpoints under
dist/. It does not install or integrate third-party models into Omnizart.
"""
import argparse
from collections import Counter, defaultdict, deque
from fractions import Fraction
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dist' / 'benchmark-runtime-deps'))
sys.path.insert(0, str(ROOT / 'dist' / 'benchmark-midi2hands' / 'src'))
sys.path.insert(0, str(ROOT / 'dist' / 'benchmark-pm2s'))
sys.path.insert(0, str(ROOT / 'dist' / 'benchmark-bachprop'))


def records_from_hands(path):
    midi = mido.MidiFile(path)
    if [track.name for track in midi.tracks[-2:]] != ['Right hand', 'Left hand']:
        raise ValueError('Expected final right/left hand tracks.')
    records = []
    for hand, track in enumerate(midi.tracks[-2:], 1):
        active = defaultdict(deque)
        tick = 0
        for index, msg in enumerate(track):
            tick += msg.time
            if msg.type == 'note_on' and msg.velocity:
                active[msg.note].append((tick, index, msg.velocity))
            elif msg.type in ('note_on', 'note_off'):
                if not active[msg.note]:
                    raise ValueError('Unpaired note-off.')
                start, on_index, velocity = active[msg.note].popleft()
                records.append(dict(start=start, end=tick, pitch=msg.note, velocity=velocity,
                                    hand=hand, on=on_index, off=index))
        if any(active.values()):
            raise ValueError('Unpaired note-on.')
    records.sort(key=lambda n: (n['start'], n['end'], n['pitch'], n['hand'], n['on']))
    return midi, records


def reference_midi(source, destination, measure_limit=16):
    """Convert written piano staves to labeled MIDI for a limited benchmark.

    Staff is a proxy for hand, not verified fingering. Read the first measures
    in written order; repeat playback and ornament realization are not modeled.
    """
    root = ET.parse(source).getroot()
    parts = root.findall('part')
    if len(parts) != 1:
        raise ValueError('Reference must contain one piano part.')
    measures = parts[0].findall('measure')[:measure_limit]
    divisions, origin, tempo = 1, Fraction(0), 500000
    time_signature = None
    tempo_events, notes, ties = [], {1: [], 2: []}, {}
    natural = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    for measure in measures:
        cursor, maximum, chord_onset = Fraction(0), Fraction(0), Fraction(0)
        for element in measure:
            if element.tag == 'attributes':
                if element.findtext('divisions'):
                    divisions = int(element.findtext('divisions'))
                time = element.find('time')
                if time is not None and time_signature is None:
                    time_signature = (int(time.findtext('beats')), int(time.findtext('beat-type')))
            elif element.tag in ('backup', 'forward'):
                duration = Fraction(int(element.findtext('duration')), divisions)
                cursor += duration if element.tag == 'forward' else -duration
            elif element.tag == 'direction':
                sound = element.find('sound')
                if sound is not None and sound.get('tempo'):
                    offset = Fraction(int(element.findtext('offset', '0')), divisions)
                    tempo_events.append((origin + cursor + offset, mido.bpm2tempo(float(sound.get('tempo')))))
            elif element.tag == 'note':
                if element.find('grace') is not None:
                    continue
                duration = Fraction(int(element.findtext('duration', '0')), divisions)
                onset = chord_onset if element.find('chord') is not None else cursor
                if element.find('chord') is None:
                    chord_onset = onset
                    cursor += duration
                maximum = max(maximum, onset + duration, cursor)
                pitch = element.find('pitch')
                if pitch is None or duration <= 0:
                    continue
                staff = int(element.findtext('staff', '1'))
                if staff not in (1, 2):
                    raise ValueError('Reference has more than two staves.')
                number = (int(pitch.findtext('octave')) + 1) * 12 + natural[pitch.findtext('step')] + int(pitch.findtext('alter', '0'))
                voice = element.findtext('voice', '1')
                key = (staff, voice, number)
                kinds = {tie.get('type') for tie in element.findall('tie')}
                start = origin + onset
                end = start + duration
                if 'stop' in kinds:
                    if key not in ties:
                        raise ValueError(f'Unmatched tie in reference: {key}')
                    start = ties.pop(key)
                if 'start' in kinds:
                    if key in ties:
                        raise ValueError(f'Ambiguous tie in reference: {key}')
                    ties[key] = start
                else:
                    notes[staff].append((start, end, number))
        if maximum <= 0:
            raise ValueError('Reference contains an empty measure.')
        origin += maximum
    for (staff, _, pitch), start in ties.items():
        notes[staff].append((start, origin, pitch))
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    timing = [(0, mido.MetaMessage('set_tempo', tempo=tempo))]
    if time_signature:
        timing.append((0, mido.MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1])))
    timing += [(round(position * 480), mido.MetaMessage('set_tempo', tempo=value)) for position, value in tempo_events]
    track, last = mido.MidiTrack(), 0
    for tick, msg in sorted(timing, key=lambda item: item[0]):
        track.append(msg.copy(time=tick-last))
        last = tick
    midi.tracks.append(track)
    for staff, name in ((1, 'Right hand'), (2, 'Left hand')):
        events = [(0, -1, mido.MetaMessage('track_name', name=name))]
        for start, end, pitch in notes[staff]:
            events.append((round(start * 480), 1, mido.Message('note_on', note=pitch, velocity=80, channel=staff-1)))
            events.append((round(end * 480), 0, mido.Message('note_off', note=pitch, velocity=0, channel=staff-1)))
        track, last = mido.MidiTrack(), 0
        for tick, _, msg in sorted(events, key=lambda item: (item[0], item[1])):
            track.append(msg.copy(time=tick-last))
            last = tick
        midi.tracks.append(track)
    if not notes[1] or not notes[2]:
        raise ValueError('Reference requires notes on both piano staves.')
    if Path(destination).exists():
        raise FileExistsError(destination)
    midi.save(destination)
    print(f'Reference: {len(measures)} written measures, upper={len(notes[1])}, lower={len(notes[2])}.')


def prepare(source, folder):
    from omnizart.desktop.smoothing import _tempo_map
    midi, records = records_from_hands(source)
    at = _tempo_map(midi)
    for note in records:
        note['start_seconds'] = at(note['start'])
        note['end_seconds'] = at(note['end'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'notes.json').write_text(json.dumps(records), encoding='utf-8')
    # Models see a single anonymous note track, not the original hand labels.
    model_input = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    model_input.tracks.append(mido.MidiTrack(midi.tracks[0]))
    events = []
    for i, note in enumerate(records):
        events.append((note['start'], 1, i, mido.Message('note_on', note=note['pitch'], velocity=note['velocity'])))
        events.append((note['end'], 0, i, mido.Message('note_off', note=note['pitch'], velocity=0)))
    track, last = mido.MidiTrack(), 0
    for tick, _, _, msg in sorted(events, key=lambda item: item[:3]):
        track.append(msg.copy(time=tick-last))
        last = tick
    model_input.tracks.append(track)
    model_input.save(folder / 'model_input.mid')
    print(f'Prepared {len(records)} notes at {midi.ticks_per_beat} ticks per beat.')


def match_predictions(records, keys_and_labels):
    remaining = defaultdict(deque)
    for index, note in enumerate(records):
        remaining[(note['start'], note['pitch'])].append(index)
    labels = [None] * len(records)
    for start, pitch, label in keys_and_labels:
        queue = remaining[(start, pitch)]
        if not queue:
            raise ValueError(f'Prediction could not be mapped to source note: {(start, pitch)}')
        labels[queue.popleft()] = int(label)
    if any(value not in (1, 2) for value in labels):
        raise ValueError('Some notes were not assigned to a hand.')
    return labels


def predict_midi2hands(folder, records):
    from midiutils.midi_preprocessor import MidiPreprocessor
    from midi2hands.models.generative import GenerativeHandFormer
    from midi2hands.models.onnex.onnex_model import ONNXModel
    model = ONNXModel()
    events = MidiPreprocessor().get_midi_events(folder / 'model_input.mid')
    _, _, prediction = GenerativeHandFormer(model).inference(events, model.window_size, 'cpu')
    # The upstream preprocessor labels track 1 "right", but its bundled
    # training MIDI uses track 0 "Rechts" (right), track 1 "Links" (left).
    # Thus the checkpoint's learned 1/0 hand polarity is opposite the names
    # used by its inference helper. Keep feedback in native polarity and only
    # correct the output labels here.
    return match_predictions(records, ((event.start, event.note, 2 if label == 1 else 1)
                                      for event, label in zip(events, prediction)))


def pretty_notes(folder):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(folder / 'model_input.mid'))
    notes = sorted((note for instrument in midi.instruments if not instrument.is_drum
                    for note in instrument.notes), key=lambda note: note.start)
    return midi, notes


def predict_pm2s(folder, records):
    import torch
    from pm2s.features.hand_part import RNNHandPartProcessor
    from pm2s.io.midi_read import read_note_sequence
    midi, notes = pretty_notes(folder)
    sequence = read_note_sequence(str(folder / 'model_input.mid'))
    if len(notes) != len(sequence):
        raise ValueError('PM2S excluded some MIDI notes.')
    torch.set_num_threads(2)
    with torch.no_grad():
        prediction = RNNHandPartProcessor().process_note_seq(sequence)
    # PM2S's training code uses the instrument index as the hand label; its
    # comment describing that index conflicts with the published checkpoint on
    # a separate, hand-written calibration score. The held-out references also
    # confirm index 0 behaves as the upper/right staff for this checkpoint.
    return match_predictions(records, ((midi.time_to_tick(note.start), note.pitch, 2 if label == 1 else 1)
                                      for note, label in zip(notes, prediction)))


def predict_bachprop(folder, records):
    import keras
    midi, notes = pretty_notes(folder)
    model = keras.saving.load_model(ROOT / 'dist' / 'benchmark-bachprop' / 'best_model.keras', compile=False)
    features, previous = [], 0.0
    for note in notes:
        features.append([note.pitch / 128., note.end - note.start, note.start - previous, note.velocity / 127.])
        previous = note.start
    # The released model has fixed 50-note input, as used in training.
    windows = np.zeros(((len(features) + 49) // 50, 50, 4), dtype=np.float32)
    for i, feature in enumerate(features):
        windows[i // 50, i % 50] = feature
    outputs = keras.ops.convert_to_numpy(model(windows, training=False)).reshape(-1)[:len(notes)]
    return match_predictions(records, ((midi.time_to_tick(note.start), note.pitch, 2 if value > .5 else 1)
                                      for note, value in zip(notes, outputs)))


def predict_omnizart(folder, records):
    from omnizart.desktop.notation import _assign_piano_hands
    from omnizart.desktop.svsep import predict_staff
    midi = mido.MidiFile(folder / 'model_input.mid')
    time_signature = next((message for track in midi.tracks for message in track
                           if message.type == 'time_signature'), None)
    numerator = time_signature.numerator if time_signature else 4
    denominator = time_signature.denominator if time_signature else 4
    measure = numerator * 16 / denominator
    if measure != int(measure):
        raise ValueError('Unsupported reference time signature.')
    notes = [(round(n['start'] * 4 / midi.ticks_per_beat),
              max(round(n['start'] * 4 / midi.ticks_per_beat) + 1, round(n['end'] * 4 / midi.ticks_per_beat)),
              n['pitch'], n['velocity']) for n in records]
    preferences = predict_staff(notes, int(measure), denominator)
    (folder / 'svsep_raw_labels.json').write_text(json.dumps(preferences), encoding='utf-8')
    assigned = _assign_piano_hands(notes, int(measure), preferences)
    locations = defaultdict(deque)
    for index, note in enumerate(notes):
        locations[note].append(index)
    labels = [None] * len(notes)
    for hand, hand_notes in assigned.items():
        for note in hand_notes:
            labels[locations[note].popleft()] = hand
    return labels


def assemble(source, target, records, labels):
    midi, source_records = records_from_hands(source)
    if source_records != [{k: value for k, value in record.items() if not k.endswith('_seconds')} for record in records]:
        raise ValueError('Source MIDI differs from the prepared benchmark.')
    note_events = {1: [], 2: []}
    for note, staff in zip(records, labels):
        track = midi.tracks[-2 if note['hand'] == 1 else -1]
        channel = 0 if staff == 1 else 1
        note_events[staff].append((note['start'], 1, track[note['on']].copy(channel=channel)))
        note_events[staff].append((note['end'], 0, track[note['off']].copy(channel=channel)))
    for staff in (1, 2):
        original = midi.tracks[-2 if staff == 1 else -1]
        channel = next((msg.channel for msg in original if hasattr(msg, 'channel')), staff-1)
        events, tick = [], 0
        for msg in original:
            tick += msg.time
            if msg.type not in ('note_on', 'note_off', 'end_of_track'):
                events.append((tick, -1, msg))
        events.extend((tick, order, msg.copy(channel=channel)) for tick, order, msg in note_events[staff])
        events.sort(key=lambda item: (item[0], item[1]))
        out, last = mido.MidiTrack(), 0
        for tick, _, msg in events:
            out.append(msg.copy(time=tick-last))
            last = tick
        midi.tracks[-2 if staff == 1 else -1] = out
    if Path(target).exists():
        raise FileExistsError(target)
    midi.save(target)
    _, output_records = records_from_hands(target)
    signature = lambda items: Counter((n['start'], n['end'], n['pitch'], n['velocity']) for n in items)
    if signature(records) != signature(output_records):
        raise ValueError('Output changed MIDI notes or timing.')
    print(f'Saved {target}: right={labels.count(1)}, left={labels.count(2)}; note timing preserved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['reference', 'prepare', 'midi2hands', 'pm2s', 'bachprop', 'omnizart'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'reference':
        args.folder.mkdir(parents=True, exist_ok=True)
        reference_midi(args.source, args.folder / 'reference_hands.mid')
        return
    if args.mode == 'prepare':
        prepare(args.source, args.folder)
        return
    records = json.loads((args.folder / 'notes.json').read_text(encoding='utf-8'))
    labels = {'midi2hands': predict_midi2hands, 'pm2s': predict_pm2s, 'bachprop': predict_bachprop,
              'omnizart': predict_omnizart}[args.mode](args.folder, records)
    (args.folder / (args.mode + '_labels.json')).write_text(json.dumps(labels), encoding='utf-8')
    assemble(args.source, args.folder / (args.mode + '_hands.mid'), records, labels)


if __name__ == '__main__':
    main()

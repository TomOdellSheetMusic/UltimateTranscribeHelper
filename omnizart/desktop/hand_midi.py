"""Split one piano MIDI part into independent, unquantized hand tracks."""
from collections import defaultdict, deque
from pathlib import Path
import warnings

import mido


def split_piano_midi(source, destination, analysis=None, vocals_last=False):
    """Write right and left hand tracks, retaining original note timing and metadata.

    Hand assignment is an estimate. A unique piano part must be identifiable;
    other instruments, percussion and non-channel metadata are retained.
    """
    from omnizart.desktop.notation import _assign_piano_hands

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing MIDI: {destination}')
    midi = mido.MidiFile(str(source))
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('This MIDI timing format is not supported.')
    events = []
    pitched_parts = set()
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        track_events = []
        for index, message in enumerate(track):
            absolute += message.time
            track_events.append((absolute, index, message))
            if message.type == 'note_on' and message.velocity and message.channel != 9:
                pitched_parts.add((track_index, message.channel))
        events.append(track_events)
    piano_key = analysis.get('piano_key') if analysis is not None else None
    if piano_key is None and len(pitched_parts) == 1:
        piano_key = next(iter(pitched_parts))
    if piano_key not in pitched_parts:
        raise ValueError('Hand splitting requires exactly one identifiable pitched piano track/channel.')
    piano_track, piano_channel = piano_key
    used_channels = {message.channel for track_index, track in enumerate(events) for _, _, message in track
                     if hasattr(message, 'channel') and (track_index, message.channel) != piano_key}
    free_channels = [channel for channel in range(16) if channel != 9 and channel not in used_channels]
    if len(free_channels) < 2:
        raise ValueError('There are not enough unused MIDI channels to separate the piano hands.')
    channels = {1: free_channels[0], 2: free_channels[1]}
    pending = defaultdict(deque)
    notes = []
    for tick, index, message in events[piano_track]:
        if getattr(message, 'channel', None) != piano_channel:
            continue
        if message.type == 'note_on' and message.velocity:
            pending[message.note].append((tick, index, message.velocity))
        elif message.type == 'note_off' or (message.type == 'note_on' and not message.velocity):
            if not pending[message.note]:
                raise ValueError('The piano MIDI has a note-off without a matching note-on.')
            start, on_index, velocity = pending[message.note].popleft()
            entry = (start, max(tick, start + 1), message.note, velocity)
            notes.append((entry, on_index, index))
    if any(pending.values()):
        raise ValueError('The piano MIDI has notes without matching note-off events.')
    entries = [item[0] for item in notes]
    if analysis is not None:
        labels = analysis['hands']
        allowed = (1, 2, None) if analysis.get('simplified') else (1, 2)
        if len(labels) != len(entries) or any(label not in allowed for label in labels):
            raise ValueError('The score hand assignments do not match the MIDI notes.')
        assigned = {staff: [entry for entry, label in zip(entries, labels) if label == staff] for staff in (1, 2)}
    else:
        with warnings.catch_warnings(record=True) as review:
            warnings.simplefilter('always', UserWarning)
            assigned = _assign_piano_hands(entries, measure_length=midi.ticks_per_beat * 4,
                                           unit_scale=midi.ticks_per_beat / 4)
        if review:
            warnings.warn('Piano hand spans or held-note counts need review in some passages.', UserWarning)
    note_hands = {}
    note_positions = {}
    if analysis is not None:
        for (_, on_index, off_index), staff in zip(notes, labels):
            if staff is not None:
                note_hands[on_index] = note_hands[off_index] = staff
    else:
        occurrences = defaultdict(deque)
        for entry, on_index, off_index in notes:
            occurrences[entry].append((on_index, off_index))
        for staff in (1, 2):
            for entry in assigned[staff]:
                on_index, off_index = occurrences[entry].popleft()
                note_hands[on_index] = note_hands[off_index] = staff
    if analysis is not None and analysis.get('quantize'):
        if len(analysis.get('positions', [])) != len(notes):
            raise ValueError('The score timing does not match the MIDI notes.')
        for (_, on_index, off_index), position in zip(notes, analysis['positions']):
            if position is None:
                continue
            start, end = position
            note_positions[on_index] = round(start * midi.ticks_per_beat)
            note_positions[off_index] = round(end * midi.ticks_per_beat)

    output = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    hand_events = {1: [], 2: []}
    hand_positions = {1: [], 2: []}
    other_overrides = {}
    other_omitted = set()
    if analysis is not None and analysis.get('quantize'):
        for key, positions in analysis.get('other_positions', {}).items():
            track_index, channel = key
            active = defaultdict(deque)
            completed = []
            for _, index, message in events[track_index]:
                if getattr(message, 'channel', None) != channel:
                    continue
                if message.type == 'note_on' and message.velocity:
                    active[message.note].append(index)
                elif message.type == 'note_off' or (message.type == 'note_on' and not message.velocity):
                    if not active[message.note]:
                        raise ValueError('An instrument has a note-off without a matching note-on.')
                    completed.append((active[message.note].popleft(), index))
            if len(completed) != len(positions) or any(active.values()):
                raise ValueError('Score timing does not match an instrument MIDI track.')
            for (on_index, off_index), position in zip(completed, positions):
                if position is None:
                    other_omitted.update(((track_index, on_index), (track_index, off_index)))
                    continue
                start, end = position
                other_overrides[(track_index, on_index)] = round(start * midi.ticks_per_beat)
                other_overrides[(track_index, off_index)] = round(end * midi.ticks_per_beat)
    output_overrides = {}
    for track_index, track in enumerate(events):
        original = []
        for tick, index, message in track:
            if track_index == piano_track and getattr(message, 'channel', None) == piano_channel:
                if message.type in ('note_on', 'note_off'):
                    if index not in note_hands:
                        continue
                    staff = note_hands[index]
                    hand_events[staff].append((tick, message.copy(channel=channels[staff])))
                    hand_positions[staff].append(note_positions.get(index))
                else:
                    for staff in (1, 2):
                        hand_events[staff].append((tick, message.copy(channel=channels[staff])))
                        hand_positions[staff].append(None)
            else:
                if (track_index, index) in other_omitted:
                    continue
                if (track_index, index) in other_overrides:
                    output_overrides[(track_index, len(original))] = other_overrides[(track_index, index)]
                original.append((tick, message))
        output.tracks.append(_make_track(original))
    # Sustain can be stored in a separate conductor/control track while still
    # addressing the piano channel. The split hands use new channels, so copy
    # those pedal events to both hands as well.
    for track_index, track in enumerate(events):
        if track_index == piano_track:
            continue
        for tick, _, message in track:
            if (message.type == 'control_change' and message.channel == piano_channel and
                    message.control == 64):
                for staff in (1, 2):
                    hand_events[staff].append((tick, message.copy(channel=channels[staff])))
                    hand_positions[staff].append(None)
    for staff, name in ((1, 'Right hand'), (2, 'Left hand')):
        # Keep event overrides in the same order as the sorted hand track.
        ordered = sorted(zip(hand_events[staff], hand_positions[staff]), key=lambda item: item[0][0])
        hand_events[staff] = [event for event, _ in ordered]
        hand_positions[staff] = [position for _, position in ordered]
        output.tracks.append(_make_track([(0, mido.MetaMessage('track_name', name=name)), *hand_events[staff]]))
    if analysis is not None and analysis.get('grid') is not None:
        overrides = {(len(midi.tracks) + staff - 1, index + 1): position
                     for staff in (1, 2) for index, position in enumerate(hand_positions[staff]) if position is not None}
        overrides.update(output_overrides)
        output = _align_midi(output, analysis['grid'], analysis['meter'], overrides, analysis.get('key'),
                             analysis.get('pickup_offset', 0))
    elif analysis is not None and (analysis.get('key') or analysis.get('apply_meter')):
        output = _update_metadata(output, analysis.get('key'), analysis['meter'] if analysis.get('apply_meter') else None)
    if vocals_last:
        _order_vocal_tracks_last(output)
    with destination.open('xb') as stream:
        try:
            output.save(file=stream)
        except Exception:
            stream.close()
            destination.unlink(missing_ok=True)
            raise
    return len(assigned[1]), len(assigned[2])


def _order_vocal_tracks_last(midi):
    """Order generated hands low to high and put vocals last in tracks and channels."""
    vocals = [track for track in midi.tracks if track.name.casefold() == 'vocals']
    if not vocals:
        return
    left = [track for track in midi.tracks if track.name == 'Left hand']
    right = [track for track in midi.tracks if track.name == 'Right hand']
    selected = {id(track) for track in vocals + left + right}
    other = [track for track in midi.tracks if id(track) not in selected]
    midi.tracks[:] = other + left + right + vocals
    channels = []
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity and msg.channel != 9 and msg.channel not in channels:
                channels.append(msg.channel)
    for track in midi.tracks:
        for msg in track:
            if hasattr(msg, 'channel') and msg.channel != 9 and msg.channel not in channels:
                channels.append(msg.channel)
    mapping = dict(zip(channels, (channel for channel in range(16) if channel != 9)))
    for track in midi.tracks:
        for msg in track:
            if hasattr(msg, 'channel') and msg.channel != 9:
                msg.channel = mapping[msg.channel]


def _align_midi(midi, grid, meter, overrides=None, key=None, pickup_offset=0):
    """Map performance seconds onto the selected beat grid and tempo changes."""
    shift = round(pickup_offset * midi.ticks_per_beat)
    measure = round(meter[0] * 4 / meter[1] * midi.ticks_per_beat)
    if not 0 <= shift < measure:
        raise ValueError('Invalid pickup offset for MIDI bar alignment.')
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            events.append((tick, track_index, order, message))
    events.sort(key=lambda item: item[:3])
    tracks = [[] for _ in midi.tracks]
    seconds, last_tick, tempo = 0.0, 0, 500000
    for tick, track_index, order, message in events:
        seconds += (tick - last_tick) * tempo / (midi.ticks_per_beat * 1000000)
        last_tick = tick
        if message.type == 'set_tempo':
            tempo = message.tempo
            continue
        if message.type == 'time_signature':
            continue
        if key and message.type == 'key_signature':
            continue
        raw_tick = (overrides or {}).get((track_index, order), round(grid.position(seconds) * midi.ticks_per_beat))
        new_tick = max(0, raw_tick - shift)
        tracks[track_index].append((new_tick, message))
    output = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    period = grid.tempo_segments[0][1]
    timing = [(0, mido.MetaMessage('set_tempo', tempo=round(period * 1000000)))]
    if shift:
        pickup = measure - shift
        sixteenths = round(pickup * 4 / midi.ticks_per_beat)
        if sixteenths < 1 or abs(sixteenths * midi.ticks_per_beat / 4 - pickup) > 1:
            raise ValueError('The MIDI pickup cannot be represented safely.')
        denominator = 16
        while denominator > 4 and sixteenths % 2 == 0:
            sixteenths //= 2
            denominator //= 2
        timing.extend([(0, mido.MetaMessage('time_signature', numerator=sixteenths, denominator=denominator)),
                       (pickup, mido.MetaMessage('time_signature', numerator=meter[0], denominator=meter[1]))])
    else:
        timing.append((0, mido.MetaMessage('time_signature', numerator=meter[0], denominator=meter[1])))
    if key:
        timing.append((0, mido.MetaMessage('key_signature', key=key)))
    if grid.manual_tempo is None:
        for i, period in grid.tempo_segments[1:]:
            position = round(grid.position(grid.beats[i]) * midi.ticks_per_beat)
            tempo_message = mido.MetaMessage('set_tempo', tempo=round(period * 1000000))
            if position <= shift:
                timing[0] = (0, tempo_message)
            else:
                timing.append((position - shift, tempo_message))
    output.tracks.append(_make_track(sorted(timing, key=lambda item: item[0])))
    output.tracks.extend(_make_track(sorted(track, key=lambda item: item[0])) for track in tracks)
    return output


def _update_metadata(midi, key, meter):
    result = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    metadata = []
    if key:
        metadata.append((0, mido.MetaMessage('key_signature', key=key)))
    if meter:
        metadata.append((0, mido.MetaMessage('time_signature', numerator=meter[0], denominator=meter[1])))
    result.tracks.append(_make_track(metadata))
    for track in midi.tracks:
        tick, events = 0, []
        for message in track:
            tick += message.time
            if (key and message.type == 'key_signature') or (meter and message.type == 'time_signature'):
                continue
            events.append((tick, message))
        result.tracks.append(_make_track(events))
    return result


def _make_track(events):
    track = mido.MidiTrack()
    previous = 0
    for tick, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    return track

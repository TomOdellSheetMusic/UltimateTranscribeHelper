"""Split one piano MIDI part into independent, unquantized hand tracks."""
from collections import defaultdict, deque
from pathlib import Path
import warnings

import mido


def split_piano_midi(source, destination):
    """Write right and left hand tracks, retaining original note timing and metadata.

    Hand assignment is an estimate. Only one pitched MIDI track/channel can be
    interpreted as piano; percussion and all non-channel metadata are retained.
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
    if len(pitched_parts) != 1:
        raise ValueError('Hand splitting requires exactly one pitched piano track/channel. Other instruments cannot be assigned safely.')
    piano_track, piano_channel = next(iter(pitched_parts))
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
    with warnings.catch_warnings(record=True) as review:
        warnings.simplefilter('always', UserWarning)
        assigned = _assign_piano_hands(entries, measure_length=midi.ticks_per_beat * 4,
                                       unit_scale=midi.ticks_per_beat / 4)
    if review:
        warnings.warn('Piano hand spans or held-note counts need review in some passages.', UserWarning)
    occurrences = defaultdict(deque)
    for entry, on_index, off_index in notes:
        occurrences[entry].append((on_index, off_index))
    note_hands = {}
    for staff in (1, 2):
        for entry in assigned[staff]:
            on_index, off_index = occurrences[entry].popleft()
            note_hands[on_index] = note_hands[off_index] = staff

    output = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    hand_events = {1: [], 2: []}
    for track_index, track in enumerate(events):
        original = []
        for tick, index, message in track:
            if track_index == piano_track and getattr(message, 'channel', None) == piano_channel:
                if message.type in ('note_on', 'note_off'):
                    staff = note_hands[index]
                    hand_events[staff].append((tick, message.copy(channel=staff - 1)))
                else:
                    for staff in (1, 2):
                        hand_events[staff].append((tick, message.copy(channel=staff - 1)))
            else:
                original.append((tick, message))
        output.tracks.append(_make_track(original))
    for staff, name in ((1, 'Right hand'), (2, 'Left hand')):
        output.tracks.append(_make_track([(0, mido.MetaMessage('track_name', name=name)), *hand_events[staff]]))
    with destination.open('xb') as stream:
        try:
            output.save(file=stream)
        except Exception:
            stream.close()
            destination.unlink(missing_ok=True)
            raise
    return len(assigned[1]), len(assigned[2])


def _make_track(events):
    track = mido.MidiTrack()
    previous = 0
    for tick, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    return track

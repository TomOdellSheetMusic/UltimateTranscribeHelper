"""Conservative, opt-in MIDI fragment repair without changing the source MIDI."""
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import mido
import warnings


def _shorten_pedal_tails(midi):
    """Shorten likely baked-in pedal tails at the next attack, retaining CC64.

    Only notes ending at a pedal release are candidates. The true key release
    cannot be recovered from MIDI, so this is an optional estimate.
    """
    at = _tempo_map(midi)
    controls = defaultdict(list)
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'control_change' and msg.control == 64 and msg.channel != 9:
                controls[msg.channel].append((tick, msg.value >= 64))
    intervals = defaultdict(list)
    for channel, events in controls.items():
        down = None
        for tick, pressed in sorted(events):
            if pressed and down is None:
                down = tick
            elif not pressed and down is not None:
                intervals[channel].append((down, tick))
                down = None
    if not any(intervals.values()):
        warnings.warn('Pedal cleanup skipped: no complete sustain-pedal events were found in the MIDI.', UserWarning)
        return 0
    changed = 0
    for track in midi.tracks:
        if any(msg.type == 'track_name' and 'vocal' in msg.name.casefold() for msg in track):
            continue
        events, active, pairs, ambiguous, attacks = [], {}, [], set(), defaultdict(set)
        tick = 0
        for msg in track:
            tick += msg.time
            index = len(events)
            events.append((tick, msg))
            if msg.type not in ('note_on', 'note_off') or msg.channel == 9:
                continue
            key = (msg.channel, msg.note)
            if msg.type == 'note_on' and msg.velocity:
                attacks[msg.channel].add(tick)
                if key in active:
                    ambiguous.add(key)
                active[key] = (tick, index)
            elif key in active:
                start, on_index = active.pop(key)
                pairs.append((key, start, tick, on_index, index))
            else:
                ambiguous.add(key)
        ambiguous.update(active)
        attacks = {channel: sorted(values) for channel, values in attacks.items()}
        replacements = {}
        for (channel, pitch), start, end, _, off_index in pairs:
            if (channel, pitch) in ambiguous:
                continue
            possible = next(((down, up) for down, up in intervals[channel]
                             if down <= start < end <= up and at(up) - at(end) <= .02), None)
            if possible is None:
                continue
            next_index = bisect_right(attacks[channel], start)
            if next_index >= len(attacks[channel]):
                continue
            next_attack = attacks[channel][next_index]
            if next_attack < end and at(next_attack) - at(start) >= .1:
                replacements[off_index] = next_attack
                changed += 1
        if replacements:
            ordered = sorted(((replacements.get(i, tick), i, msg) for i, (tick, msg) in enumerate(events)),
                             key=lambda item: (item[0], 0 if item[1] in replacements else 1, item[1]))
            last = 0
            rewritten = mido.MidiTrack()
            for tick, _, msg in ordered:
                rewritten.append(msg.copy(time=tick - last))
                last = tick
            track[:] = rewritten
    return changed


def _tempo_map(midi):
    changes = [(0, 500000)]
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                events.append((tick, track_index, message.tempo))
    changes.extend((tick, tempo) for tick, _, tempo in sorted(events))
    ticks, seconds, tempos = [0], [0.0], [500000]
    for tick, tempo in changes[1:]:
        elapsed = (tick - ticks[-1]) * tempos[-1] / (midi.ticks_per_beat * 1_000_000)
        ticks.append(tick)
        seconds.append(seconds[-1] + elapsed)
        tempos.append(tempo)

    def at(tick):
        index = bisect_right(ticks, tick) - 1
        return seconds[index] + (tick - ticks[index]) * tempos[index] / (midi.ticks_per_beat * 1_000_000)

    return at


def _expression_guards(midi):
    """Track sustain and pitch-bend activity across all tracks for each channel."""
    events = defaultdict(list)
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "pitchwheel" or (message.type == "control_change" and message.control == 64):
                events[message.channel].append((tick, message))
    guards = {}
    for channel, values in events.items():
        values.sort(key=lambda item: item[0])
        ticks, states = [], []
        pedal, pitch = False, 0
        for tick, message in values:
            if message.type == "pitchwheel":
                pitch = message.pitch
            else:
                pedal = message.value >= 64
            ticks.append(tick)
            states.append(pedal or pitch != 0)
        guards[channel] = (ticks, states)

    def protected(channel, start, end):
        ticks, states = guards.get(channel, ([], []))
        start_pos = bisect_right(ticks, start)
        return ((start_pos > 0 and states[start_pos - 1]) or
                bisect_right(ticks, end) > start_pos)

    return protected


def smooth_midi(source, destination, max_gap_ms=30, min_note_ms=0, join_notes=True, quiet_velocity=0,
                shorten_pedal=False):
    """Merge likely broken short notes, preserving other events and the source file.

    Skip percussion, vocal tracks, and passages with pedal or pitch bends. These
    safeguards favor leaving an artifact over changing expressive performances.
    """
    if not 0 <= max_gap_ms <= 200:
        raise ValueError("Maximum smoothing gap must be between 0 and 200 ms.")
    if not 0 <= min_note_ms <= 200:
        raise ValueError("Minimum note length must be between 0 and 200 ms.")
    if not isinstance(quiet_velocity, int) or not 0 <= quiet_velocity <= 127:
        raise ValueError("Quiet note velocity must be between 0 and 127.")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("The cleaned MIDI must have a different filename.")
    midi = mido.MidiFile(str(source))
    if midi.type == 2:
        raise ValueError("Smoothing does not support independent-sequence MIDI files.")
    at = _tempo_map(midi)
    protected = _expression_guards(midi)
    total = 0
    for track in midi.tracks:
        if any(msg.type == "track_name" and "vocal" in msg.name.casefold() for msg in track):
            continue
        absolute, tick = [], 0
        for message in track:
            tick += message.time
            absolute.append((tick, message))
        active, notes, ambiguous = {}, defaultdict(list), set()
        for index, (tick, message) in enumerate(absolute):
            if message.type not in ("note_on", "note_off") or message.channel == 9:
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                if key in active:
                    ambiguous.add(key)
                active[key] = (index, tick)
            elif key in active:
                start_index, start_tick = active.pop(key)
                notes[key].append([start_index, index, start_tick, tick])
            else:
                ambiguous.add(key)
        ambiguous.update(active)
        removed = set()
        for key, pairs in notes.items():
            if key in ambiguous:
                continue
            previous = None
            groups = []
            for current in pairs:
                velocity = absolute[current[0]][1].velocity
                if quiet_velocity and velocity <= quiet_velocity:
                    # A soft same-pitch reattack immediately following a louder
                    # note often represents its tail. Hold the original attack
                    # through it only where expression controls make this safe.
                    if previous is not None:
                        gap_ms = (at(current[2]) - at(previous[3])) * 1000
                        if 0 <= gap_ms <= 30 and not protected(key[0], previous[2], current[3]):
                            removed.update((previous[1], current[0]))
                            previous[1], previous[3] = current[1], current[3]
                            total += 1
                            continue
                    removed.update((current[0], current[1]))
                    continue
                if previous is not None:
                    gap_ms = (at(current[2]) - at(previous[3])) * 1000
                    left_ms = (at(previous[3]) - at(previous[2])) * 1000
                    right_ms = (at(current[3]) - at(current[2])) * 1000
                    if (join_notes and 0 <= gap_ms <= max_gap_ms and min(left_ms, right_ms) <= 120 and
                            not protected(key[0], previous[2], current[3])):
                        removed.update((previous[1], current[0]))
                        previous[1], previous[3] = current[1], current[3]
                        total += 1
                        continue
                if previous is not None:
                    groups.append(previous)
                previous = current[:]
            if previous is not None:
                groups.append(previous)
            for group in groups:
                duration_ms = (at(group[3]) - at(group[2])) * 1000
                if min_note_ms and duration_ms < min_note_ms and not protected(key[0], group[2], group[3]):
                    removed.update((group[0], group[1]))
        if removed:
            rewritten = mido.MidiTrack()
            last_tick = 0
            for index, (tick, message) in enumerate(absolute):
                if index not in removed:
                    rewritten.append(message.copy(time=tick - last_tick))
                    last_tick = tick
            track[:] = rewritten
    if shorten_pedal:
        shortened = _shorten_pedal_tails(midi)
        warnings.warn(f'Pedal cleanup shortened {shortened} likely pedal-held note(s); review their durations.', UserWarning)
    try:
        with destination.open("xb") as stream:
            try:
                midi.save(file=stream)
            except Exception:
                stream.close()
                destination.unlink(missing_ok=True)
                raise
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing cleaned MIDI: {destination}") from exc
    return total

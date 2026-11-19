"""Create an editable MusicXML draft directly from MIDI, without an engraver.

Rhythm is quantized for notation only. Independent overlapping note groups are
allocated to separate voices so that a held note need not be tied at every
attack in another voice. The original MIDI is never modified.
"""
from collections import defaultdict, deque
from bisect import bisect_right
import math
import subprocess
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import mido


def _element(parent, name, value=None, **attributes):
    child = ET.SubElement(parent, name, **attributes)
    if value is not None:
        child.text = str(value)
    return child


def _read_midi(path):
    midi = mido.MidiFile(str(path))
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("This MIDI timing format cannot be converted to a score.")
    events = []
    for index, track in enumerate(midi.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            events.append((tick, index, order, message))
    events.sort(key=lambda x: x[:3])
    time, tick, tempo, initial_tempo = 0.0, 0, 500000, None
    signature = None
    programs, names = defaultdict(int), {}
    active, notes = defaultdict(deque), defaultdict(list)
    for position, track, _, message in events:
        time += (position - tick) * tempo / (midi.ticks_per_beat * 1000000)
        tick = position
        if message.type == 'set_tempo':
            tempo = message.tempo
            if initial_tempo is None:
                initial_tempo = tempo
        elif message.type == 'time_signature':
            current = (message.numerator, message.denominator)
            if signature is not None and signature != current:
                raise ValueError('Scores with time signature changes are not yet supported.')
            signature = current
        elif message.type == 'track_name':
            names[track] = message.name
        elif message.type == 'program_change':
            programs[(track, message.channel)] = message.program
        elif message.type == 'note_on' and message.velocity:
            active[(track, message.channel, message.note)].append((time, message.velocity))
        elif message.type in ('note_off', 'note_on'):
            key = (track, message.channel, message.note)
            if active[key]:
                start, velocity = active[key].popleft()
                if time > start:
                    notes[(track, message.channel)].append((start, time, message.note, velocity))
    if any(active.values()):
        raise ValueError('The MIDI contains notes without a matching note-off event.')
    if not notes:
        raise ValueError('The MIDI contains no notes to notate.')
    return notes, programs, names, signature or (4, 4), 60000000 / (initial_tempo or 500000)


def _estimate_tempo(notes, fallback):
    starts = sorted(note[0] for group in notes.values() for note in group)
    clusters = []
    for start in starts:
        if not clusters or start - clusters[-1] > .06:
            clusters.append(start)
    if len(clusters) < 12:
        return fallback
    best = (0, fallback)
    for bpm in range(70, 201):
        # Eighth-note periodicity distinguishes the beat from a generic
        # sixteenth grid. A free phase allows an initial pickup or silence.
        values = [2 * math.pi * t * bpm / 30 for t in clusters]
        coherence = math.hypot(sum(math.cos(v) for v in values),
                               sum(math.sin(v) for v in values)) / len(values)
        if (coherence > best[0] + .005 or
                (abs(coherence - best[0]) <= .005 and abs(bpm - fallback) < abs(best[1] - fallback))):
            best = coherence, bpm
    return best[1] if best[0] >= .25 else fallback


def _voices(notes):
    """Group equal-onset/equal-end notes as chords and color overlaps into voices."""
    chords = defaultdict(list)
    for start, end, pitch, velocity in notes:
        chords[(start, end)].append((pitch, velocity))
    lanes, ends = [], []
    for (start, end), pitches in sorted(chords.items(), key=lambda x: (x[0][0], -x[0][1])):
        available = [i for i, last in enumerate(ends) if last <= start]
        index = max(available, key=lambda i: ends[i]) if available else len(lanes)
        if index == len(lanes):
            lanes.append([])
            ends.append(0)
        lanes[index].append((start, end, sorted(pitches)))
        ends[index] = end
    return lanes





def _simplify_to_voices(notes, limit=4):
    """Keep louder simultaneous chords first, dropping only conflicting groups.

    This acts solely on the score draft. It groups notes with the same start
    and end as one chord; a discarded chord is reported to the caller.
    """
    chords = defaultdict(list)
    for start, end, pitch, velocity in notes:
        chords[(start, end)].append((start, end, pitch, velocity))
    active, kept, dropped = [], set(), set()
    for (start, end), members in sorted(chords.items(), key=lambda item: (item[0][0], -max(n[3] for n in item[1]), item[0][1])):
        active = [group for group in active if group[1] > start]
        group = (start, end)
        active.append(group)
        kept.add(group)
        if len(active) > limit:
            loser = min(active, key=lambda key: (sum(n[3] for n in chords[key]), -(key[1] - key[0]), -key[0]))
            active.remove(loser)
            kept.remove(loser)
            dropped.add(loser)
    output = [note for key in sorted(kept) for note in chords[key]]
    return output, sum(len(chords[key]) for key in dropped)

def _assign_piano_hands(entries, measure_length=16, preferences=None, unit_scale=1):
    """Partition each attack using notes still held in both hands.

    A held note stays on its original staff. The cost prefers compact hands,
    ordered pitches and no more than five simultaneously held notes per hand.
    Some MIDI passages cannot be played by two hands without editing notes.
    """
    attacks = defaultdict(list)
    if preferences is not None and len(preferences) != len(entries):
        raise ValueError('Piano staff predictions do not match the input notes.')
    for index, entry in enumerate(entries):
        attacks[entry[0]].append((entry, preferences[index] if preferences is not None else None))
    assigned = {1: [], 2: []}
    active = {1: [], 2: []}
    positions = {1: None, 2: None}
    last_onset = {1: None, 2: None}
    last_single_staff = None
    difficult_measures = set()
    for onset, attack in sorted(attacks.items()):
        attack.sort(key=lambda item: item[0][2])
        incoming = [item[0] for item in attack]
        for staff in active:
            active[staff] = [note for note in active[staff] if note[1] > onset]
        best = None
        for cut in range(len(incoming) + 1):
            left = active[2] + incoming[:cut]
            right = active[1] + incoming[cut:]
            left_pitches = [note[2] for note in left]
            right_pitches = [note[2] for note in right]
            left_span = max(left_pitches) - min(left_pitches) if left_pitches else 0
            right_span = max(right_pitches) - min(right_pitches) if right_pitches else 0
            # A large interval or excess simultaneous notes is expensive;
            # prefer the lower hand for low pitches when costs otherwise tie.
            excess = sum(max(0, span - 12) ** 2 for span in (left_span, right_span))
            fingers = sum(max(0, len(set(pitches)) - 5) ** 2 for pitches in (left_pitches, right_pitches))
            crossing = max(0, max(left_pitches) - min(right_pitches)) if left_pitches and right_pitches else 0
            misplaced = sum(max(0, n[2] - 60) for n in incoming[:cut]) + sum(max(0, 60 - n[2]) for n in incoming[cut:])
            disagreement = sum(preference is not None and preference != (2 if index < cut else 1)
                               for index, (_, preference) in enumerate(attack))
            movement = 0
            for staff, group in ((2, incoming[:cut]), (1, incoming[cut:])):
                if group and positions[staff] is not None:
                    center = sum(note[2] for note in group) / len(group)
                    movement += abs(center - positions[staff]) / max(1, (onset - last_onset[staff]) / unit_scale)
            continuity = 0
            if len(incoming) == 1 and last_single_staff is not None:
                other = 2 if last_single_staff == 1 else 1
                candidate_staff = 2 if cut else 1
                if (candidate_staff == other and not active[last_single_staff] and not active[other] and
                        onset - last_onset[last_single_staff] <= 8 * unit_scale and
                        abs(incoming[0][2] - positions[last_single_staff]) <= 7):
                    continuity = 3
            chord_split = (6 if 0 < cut < len(incoming) and not active[1] and not active[2] and
                           len({note[2] for note in incoming}) <= 5 and
                           len({note[1] for note in incoming}) == 1 and
                           incoming[-1][2] - incoming[0][2] <= 12 else 0)
            cost = 10 * excess + 20 * fingers + 3 * crossing + left_span + right_span + .1 * misplaced + 4 * disagreement + movement + continuity + chord_split
            if best is None or cost < best[0]:
                best = (cost, cut, left_span, right_span, len(set(left_pitches)), len(set(right_pitches)))
        _, cut, left_span, right_span, left_count, right_count = best
        last_single_staff = (2 if cut else 1) if len(incoming) == 1 else None
        if max(left_span, right_span) > 12 or max(left_count, right_count) > 5:
            difficult_measures.add(onset // measure_length + 1)
        for staff, group in ((2, incoming[:cut]), (1, incoming[cut:])):
            assigned[staff].extend(group)
            active[staff].extend(group)
            if group:
                positions[staff] = sum(note[2] for note in group) / len(group)
                last_onset[staff] = onset
    if difficult_measures:
        measures = sorted(difficult_measures)
        shown = ", ".join(str(number) for number in measures[:12])
        if len(measures) > 12:
            shown += ", ?"
        warnings.warn(f'Piano hand spans or held-note counts need review in measures {shown} ({len(measures)} measures total).', UserWarning)
    return assigned


def _prefer_right_hand_octave(staff_notes, move_left_chords=False):
    """Move nearby simultaneous left attacks when the right hand can hold them.

    This is an optional correction after the selected hand-assignment model.
    A multi-note left-hand attack is protected unless explicitly requested.
    The full overlap is checked for reach, finger count and duplicate pitches.
    """
    right = list(staff_notes[1])
    left = list(staff_notes[2])
    moved = []
    attacks = defaultdict(list)
    for note in left:
        attacks[note[0]].append(note)
    for onset, group in sorted(attacks.items()):
        anchors = [note[2] for note in right if note[0] == onset]
        if not anchors or (not move_left_chords and len({note[2] for note in group}) > 1):
            continue
        highest = max(anchors)
        for note in sorted(group, key=lambda value: value[2], reverse=True):
            start, end, pitch, _ = note
            if not 0 <= highest - pitch <= 12:
                continue
            overlapping = [other for other in right if other[0] < end and other[1] > start]
            if any(other[2] == pitch for other in overlapping):
                continue
            points = {start} | {other[0] for other in overlapping if start < other[0] < end}
            unsafe = False
            for point in points:
                pitches = {pitch} | {other[2] for other in overlapping if other[0] <= point < other[1]}
                if len(pitches) > 5 or max(pitches) - min(pitches) > 12:
                    unsafe = True
                    break
            if unsafe:
                continue
            left.remove(note)
            right.append(note)
            moved.append(note)
    return {1: right, 2: left}, moved


def _simplify_piano_hands(staff_notes, measure_length):
    """Shorten conflicting held notes or omit notes to fit basic hand limits.

    Preserve the top melody and lowest bass when possible. This is a heuristic,
    not a guarantee of playability; the input MIDI is never changed.
    """
    output = {staff: list(notes) for staff, notes in staff_notes.items()}
    omitted, shortened = [], []
    for staff in (1, 2):
        notes = output[staff]
        for onset in sorted({note[0] for note in notes}):
            while True:
                active = [note for note in notes if note[0] <= onset < note[1]]
                pitches = sorted({note[2] for note in active})
                if len(pitches) <= 5 and (not pitches or pitches[-1] - pitches[0] <= 12):
                    break
                protected = max(pitches) if staff == 1 else min(pitches)
                held = [note for note in active if note[0] < onset]
                if held:
                    def release_rank(note):
                        rest = list(active)
                        rest.remove(note)
                        remaining = sorted({other[2] for other in rest})
                        span = remaining[-1] - remaining[0] if remaining else 0
                        return (max(0, span - 12), max(0, len(remaining) - 5),
                                note[2] == protected, note[3], note[0])
                    released = min(held, key=release_rank)
                    notes.remove(released)
                    notes.append((released[0], onset, released[2], released[3]))
                    shortened.append((staff, released, onset))
                    continue
                candidates = [note for note in active if note[2] != protected]
                if not candidates:
                    break
                def rank(note):
                    remaining = [p for p in pitches if p != note[2]]
                    span = remaining[-1] - remaining[0] if remaining else 0
                    return (max(0, span - 12), max(0, len(remaining) - 5), note[3], note[1] - note[0])
                removed = min(candidates, key=rank)
                notes.remove(removed)
                omitted.append((staff, removed))
    if shortened:
        measures = sorted({onset // measure_length + 1 for _, _, onset in shortened})
        warnings.warn('Score simplification shortened %d held note(s) in measures %s to reduce hand reach; '
                      'the original MIDI is unchanged. Review the score and pedal use.' %
                      (len(shortened), ', '.join(map(str, measures))), UserWarning)
    if omitted:
        measures = sorted({note[0] // measure_length + 1 for _, note in omitted})
        warnings.warn('Score simplification omitted %d note(s) in measures %s; '
                      'the original MIDI is unchanged. Review the score for playability. '
                      'Omitted notes (measure, hand, MIDI pitch): %s%s' %
                      (len(omitted), ', '.join(map(str, measures)),
                       '; '.join('%d %s %d' % (note[0] // measure_length + 1,
                                                 'right' if staff == 1 else 'left', note[2])
                                 for staff, note in omitted[:30]),
                       ' ...' if len(omitted) > 30 else ''), UserWarning)
    return output


def _sliced_piano_voices(staff_notes):
    """Represent dense overlaps as tied chords when four voices are insufficient."""
    result = {}
    for staff, entries in staff_notes.items():
        boundaries = sorted({time for start, end, _, _ in entries for time in (start, end)})
        lane = []
        for left, right in zip(boundaries, boundaries[1:]):
            pitches = [(pitch, velocity, left > start, right < end)
                       for start, end, pitch, velocity in entries if start <= left and end >= right]
            if pitches:
                lane.append((left, right, sorted(pitches)))
        result[staff] = [lane] if lane else []
    return result


def _dense_piano_voices(staff_notes):
    """Limit chord slicing to overlap regions that exceed four voices."""
    output = {1: [], 2: []}
    sliced = False
    for staff, entries in staff_notes.items():
        components, current, last_end = [], [], -1
        for entry in sorted(entries):
            if current and entry[0] >= last_end:
                components.append(current)
                current = []
                last_end = -1
            current.append(entry)
            last_end = max(last_end, entry[1])
        if current:
            components.append(current)
        for component in components:
            lanes = _voices(component)
            if len(lanes) > 4:
                lanes = _sliced_piano_voices({staff: component})[staff]
                sliced = True
            while len(output[staff]) < len(lanes):
                output[staff].append([])
            for index, lane in enumerate(lanes):
                output[staff][index].extend(lane)
    if sliced:
        warnings.warn('Dense overlapping piano notes required extra ties; review the score for playability.', UserWarning)
    return output


def _piano_voices(staff_notes):
    """Keep hand assignments while limiting each staff to four voices."""
    groups = []
    for staff, entries in staff_notes.items():
        chords = defaultdict(list)
        for start, end, pitch, velocity in entries:
            chords[(start, end)].append((pitch, velocity))
        groups.extend((start, end, staff, sorted(pitches)) for (start, end), pitches in chords.items())
    groups.sort(key=lambda item: (item[0], -item[1], item[2]))
    lanes = {1: [], 2: []}
    ends = {1: [], 2: []}
    for start, end, preferred, pitches in groups:
        available = [i for i, last in enumerate(ends[preferred]) if last <= start]
        if available:
            index = max(available, key=lambda i: ends[preferred][i])
        elif len(lanes[preferred]) < 4:
            index = len(lanes[preferred])
            lanes[preferred].append([])
            ends[preferred].append(0)
        else:
            return _dense_piano_voices(staff_notes)
        lanes[preferred][index].append((start, end, pitches))
        ends[preferred][index] = end
    return lanes

def _duration_pieces(start, end, measure_length):
    # Sixteenth-note units. Splitting at bar boundaries and using conventional
    # dotted values avoids tying each chord tone at unrelated voice attacks.
    values = ((16, 'whole', False), (12, 'half', True), (8, 'half', False),
              (6, 'quarter', True), (4, 'quarter', False), (3, 'eighth', True),
              (2, 'eighth', False), (1, '16th', False))
    while start < end:
        limit = min(end, (start // measure_length + 1) * measure_length)
        for size, kind, dotted in values:
            if size <= limit - start and (size < 4 or start % 4 == 0):
                yield start, start + size, kind, dotted
                start += size
                break


def _pitch(note, pitch, fifths=0):
    # Spell diatonic notes according to the key signature, including E#, B#,
    # Cb and Fb. Chromatic notes still use a simple sharp/flat preference.
    names = [('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
             ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0)]
    if fifths < 0:
        for index, step in ((1, 'D'), (3, 'E'), (6, 'G'), (8, 'A'), (10, 'B')):
            names[index] = (step, -1)
    natural = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    altered = ('FCGDAEB'[:fifths] if fifths > 0 else 'BEADGCF'[:-fifths])
    for step, value in natural.items():
        alteration = (1 if fifths > 0 else -1) if step in altered else 0
        names[(value + alteration) % 12] = (step, alteration)
    step, alter = names[pitch % 12]
    p = _element(note, 'pitch')
    _element(p, 'step', step)
    if alter:
        _element(p, 'alter', alter)
    _element(p, 'octave', (pitch - natural[step] - alter) // 12 - 1)


def _write_note(measure, pitches, voice, staff, start, end, kind, dotted, tied_before, tied_after, fifths=0, triplet=False):
    first = None
    for index, pitch_info in enumerate(pitches):
        pitch, velocity = pitch_info[:2]
        note_tied_before = tied_before or (len(pitch_info) > 2 and pitch_info[2])
        note_tied_after = tied_after or (len(pitch_info) > 3 and pitch_info[3])
        note = _element(measure, 'note', dynamics=str(round(velocity * 100 / 90, 2)))
        if first is None:
            first = note
        if index:
            _element(note, 'chord')
        _pitch(note, pitch, fifths)
        _element(note, 'duration', end - start)
        if note_tied_before:
            _element(note, 'tie', type='stop')
        if note_tied_after:
            _element(note, 'tie', type='start')
        _element(note, 'voice', voice)
        _element(note, 'type', kind)
        if dotted:
            _element(note, 'dot')
        if triplet:
            modification = _element(note, 'time-modification')
            _element(modification, 'actual-notes', 3)
            _element(modification, 'normal-notes', 2)
        if staff is not None:
            _element(note, 'staff', staff)
        if note_tied_before or note_tied_after:
            notation = _element(note, 'notations')
            if note_tied_before:
                _element(notation, 'tied', type='stop')
            if note_tied_after:
                _element(notation, 'tied', type='start')
    return first



def _score_pieces(start, end, measure_length, unit_scale):
    if unit_scale == 1:
        for left, right, kind, dotted in _duration_pieces(start, end, measure_length):
            yield left, right, kind, dotted, False
    else:
        from omnizart.desktop.rhythm import pieces
        yield from pieces(start, end, measure_length)


def _write_rests(measure, start, end, measure_length, voice, staff, unit_scale=1):
    written = []
    for left, right, kind, dotted, triplet in _score_pieces(start, end, measure_length, unit_scale):
        note = _element(measure, 'note')
        _element(note, 'rest')
        _element(note, 'duration', right - left)
        _element(note, 'voice', voice)
        _element(note, 'type', kind)
        if dotted:
            _element(note, 'dot')
        if triplet:
            modification = _element(note, 'time-modification')
            _element(modification, 'actual-notes', 3)
            _element(modification, 'normal-notes', 2)
        if staff is not None:
            _element(note, 'staff', staff)
        written.append((note, left, right, kind, triplet))
    return written


def _mark_tuplets(events):
    """Bracket consecutive triplet pieces of the same written value."""
    groups = []
    current, key = [], None
    for note, left, right, kind, triplet in events:
        current_key = kind if triplet else None
        if current and (current_key != key or left != current[-1][2] or len(current) == 3):
            groups.append(current)
            current = []
        if triplet:
            current.append((note, left, right))
            key = current_key
        else:
            key = None
    if current:
        groups.append(current)
    for group in groups:
        for note, typ in ((group[0][0], 'start'), (group[-1][0], 'stop')):
            notations = note.find('notations')
            if notations is None:
                notations = _element(note, 'notations')
            _element(notations, 'tuplet', type=typ)

def export_score(source, destination, tempo=None, piano=True, simplify=False, neural_staff=False, audio=None, align_bars=False, downbeat=None, beats_per_bar=None, score_key='Auto', analysis_out=None, analysis_only=False, notate_pedal=False, clarify_pedal_melody=False, prefer_right_octave=False, move_left_chords=False):
    """Write a beat-aligned MusicXML draft with independent voices and stable clefs."""
    destination = Path(destination)
    if not analysis_only and destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing score: {destination}')
    notes, programs, names, signature, fallback = _read_midi(source)
    percussion = [key for key in notes if key[1] == 9]
    if percussion:
        notes = {key: value for key, value in notes.items() if key[1] != 9}
        if not notes:
            raise ValueError('The MIDI contains only percussion; pitched score export requires at least one pitched instrument.')
        warnings.warn('Percussion tracks are omitted from the MusicXML score; the original MIDI keeps all tracks.', UserWarning)
    piano_key = None
    if piano:
        candidates = [key for key in notes if programs[key] < 8 or 'piano' in names.get(key[0], '').lower()]
        if len(notes) == 1:
            piano_key = next(iter(notes))
        elif len(candidates) == 1:
            piano_key = candidates[0]
    if neural_staff and piano_key is None:
        raise ValueError('Neural piano staff assignment requires an identifiable piano instrument track.')
    from omnizart.desktop.score_key import KEYS, estimate_key
    if score_key == 'Auto':
        score_key = estimate_key(notes)
        if score_key:
            warnings.warn(f'Score key estimated as {score_key}; review the key signature and spelling.', UserWarning)
    elif score_key not in KEYS:
        raise ValueError('Unknown score key: ' + str(score_key))
    fifths, mode = KEYS[score_key] if score_key else (0, None)
    bpm = float(tempo) if tempo is not None else _estimate_tempo(notes, fallback)
    if not 30 <= bpm <= 300:
        raise ValueError('Score tempo must be between 30 and 300 BPM.')
    numerator, denominator = signature
    if beats_per_bar is not None:
        if beats_per_bar not in (3, 4):
            raise ValueError('Beats per bar must be 3 or 4.')
        numerator, denominator = beats_per_bar, 4
    grid = None
    alignment_warning = None
    downbeats = []
    if align_bars:
        if audio is None:
            raise ValueError('Audio is required for automatic bar alignment.')
        try:
            from omnizart.desktop.beat_alignment import detect
            from omnizart.desktop.beat_grid import BeatGrid
            if denominator != 4 or numerator not in (3, 4):
                raise ValueError('Automatic bar alignment currently supports 3/4 and 4/4 scores.')
            beats, downbeats = detect(audio)
            grid = BeatGrid(beats, downbeats, fallback_meter=numerator, tempo=tempo, downbeat=downbeat, meter=beats_per_bar)
            if beats_per_bar is None and grid.meter_detected and grid.meter != numerator:
                warnings.warn(f'Audio bar alignment inferred {grid.meter}/4 instead of the MIDI meter {numerator}/4. Review the score.', UserWarning)
            numerator, denominator = grid.meter, 4
            bpm = grid.bpm
            if beats_per_bar is None and not grid.meter_detected:
                warnings.warn(f'Audio bar alignment could not confidently infer the meter; using {numerator}/4. Review the score.', UserWarning)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if denominator == 4 and numerator in (3, 4):
                from omnizart.desktop.beat_grid import BeatGrid
                period = 60 / bpm
                detected = None
                for value in downbeats:
                    try:
                        candidate = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(candidate) and candidate >= 0:
                        detected = candidate
                        break
                anchor = downbeat if downbeat is not None else detected if detected is not None else 0
                origin = 'the specified downbeat' if downbeat is not None else 'an estimated downbeat' if detected is not None else 'an assumed start at zero'
                grid = BeatGrid([i * period for i in range(4)], [], fallback_meter=numerator,
                                tempo=bpm, downbeat=anchor, meter=beats_per_bar)
                numerator, denominator = grid.meter, 4
                alignment_warning = f'Audio bar alignment unavailable ({exc}); using a constant-tempo grid at {bpm:g} BPM with {origin}. Review the timing and bar lines.'
            else:
                alignment_warning = f'Audio bar alignment unavailable ({exc}); score uses a constant-tempo grid instead. Review the timing and bar lines.'
            warnings.warn(alignment_warning, UserWarning)
    elif downbeat is not None:
        if denominator != 4 or numerator not in (3, 4):
            raise ValueError('Manual bar alignment currently supports 3/4 and 4/4 scores.')
        from omnizart.desktop.beat_grid import BeatGrid
        period = 60 / bpm
        grid = BeatGrid([i * period for i in range(4)], [], fallback_meter=numerator if denominator == 4 else 4,
                        tempo=bpm, downbeat=downbeat)
        numerator, denominator = grid.meter, 4
    length = numerator * 16 / denominator
    if length != int(length) or not 1 <= length <= 64:
        raise ValueError('This time signature is not supported by the score grid.')
    length = int(length)
    # BeatGrid extrapolates the bar containing audio time zero.  If that bar
    # starts before the recording, represent only its audible tail as an
    # implicit pickup rather than inserting silence before time zero.
    if grid and grid.unit(0) >= length:
        # Rounding can put audio time zero on the following downbeat; do not
        # create a whole silent bar when that happens.
        grid.origin += grid.meter
    first_left = grid.unit(0) if grid else 0
    from omnizart.desktop.rhythm import triplet_beats, quantize, next_position
    def beat_position(time):
        return grid.position(time) if grid else time * bpm / 60
    triplets = (triplet_beats(beat_position(start) for entries in notes.values() for start, _, _, _ in entries)
                if denominator == 4 else set())
    unit_scale = 3 if triplets else 1
    original_length = length
    length *= unit_scale
    first_left *= unit_scale
    pedal_by_channel = defaultdict(list)
    pedal_positions = defaultdict(list)
    if notate_pedal or clarify_pedal_melody:
        from omnizart.desktop.smoothing import _tempo_map
        pedal_midi = mido.MidiFile(str(source))
        seconds_at = _tempo_map(pedal_midi)
        for track in pedal_midi.tracks:
            tick = 0
            for message in track:
                tick += message.time
                if message.type == 'control_change' and message.control == 64 and message.channel != 9:
                    position = max(0, beat_position(seconds_at(tick)) * 4 * unit_scale)
                    pedal_positions[message.channel].append((position, message.value >= 64))
                    pedal_by_channel[message.channel].append((round(position), message.value >= 64))
    score = ET.Element('score-partwise', version='4.0')
    work = _element(score, 'work')
    _element(work, 'work-title', Path(source).stem)
    parts = _element(score, 'part-list')
    groups = []
    for number, (key, entries) in enumerate(sorted(notes.items()), 1):
        is_piano = key == piano_key
        indexed = []
        for note_index, (start, end, pitch, velocity) in enumerate(entries):
            if triplets:
                onset = quantize(beat_position(start), triplets)
                offset = quantize(beat_position(end), triplets)
                if offset <= onset:
                    offset = next_position(onset, triplets)
            else:
                onset = grid.unit(start) if grid else max(0, round(start * bpm / 60 * 4))
                offset = max(onset + 1, grid.unit(end) if grid else round(end * bpm / 60 * 4))
            indexed.append(((onset, offset, pitch, velocity), note_index))
        if neural_staff:
            indexed.sort()
        quantized = [note for note, _ in indexed]
        if simplify and not is_piano and len(_voices(quantized)) > 4:
            quantized, dropped = _simplify_to_voices(quantized)
            if dropped:
                warnings.warn(f'Score simplification omitted {dropped} note(s) from instrument {number}; the original MIDI is unchanged.', UserWarning)
        preferences = None
        if neural_staff and is_piano:
            from omnizart.desktop.svsep import predict_staff
            prediction_notes = [(round(start / unit_scale), max(round(start / unit_scale) + 1, round(end / unit_scale)), pitch, velocity)
                                for start, end, pitch, velocity in quantized]
            preferences = predict_staff(prediction_notes, original_length, denominator)
        staff_notes = _assign_piano_hands(quantized, length, preferences, unit_scale) if is_piano else {1: quantized}
        if clarify_pedal_melody and is_piano:
            intervals, down = [], None
            for position, pressed in sorted(pedal_positions[key[1]], key=lambda event: event[0]):
                if pressed and down is None:
                    down = position
                elif not pressed and down is not None:
                    intervals.append((down, position))
                    down = None
            if not intervals:
                warnings.warn('Pedal melody cleanup skipped: no complete sustain-pedal events were found.', UserWarning)
            indices = defaultdict(deque)
            for note, index in indexed:
                indices[note].append(index)
            replacements = {}
            for staff, assigned in staff_notes.items():
                attacks = sorted({note[0] for note in assigned})
                chords = {(note[0], note[1]) for note in assigned}
                dense_points = [attack for attack in attacks
                                if sum(start <= attack < end for start, end in chords) > 4]
                revised = []
                for note in assigned:
                    index = indices[note].popleft()
                    start, end, pitch, velocity = note
                    next_index = bisect_right(attacks, start)
                    next_attack = attacks[next_index] if next_index < len(attacks) else None
                    if (next_attack is not None and next_attack < end and
                            any(next_attack <= point < end for point in dense_points) and
                            any(down <= next_attack < up and end <= up for down, up in intervals)):
                        note = (start, next_attack, pitch, velocity)
                        replacements[index] = note
                    revised.append(note)
                staff_notes[staff] = revised
            indexed = [(replacements.get(index, note), index) for note, index in indexed]
            if intervals:
                warnings.warn(f'Pedal melody cleanup shortened {len(replacements)} note(s); review the notation.', UserWarning)
        if prefer_right_octave and is_piano:
            staff_notes, moved = _prefer_right_hand_octave(staff_notes, move_left_chords=move_left_chords)
            warnings.warn(f'Octave hand adjustment moved {len(moved)} note(s) to the right hand; review the assignments.', UserWarning)
        if analysis_out is not None and is_piano:
            indices = defaultdict(deque)
            for note, index in indexed:
                indices[note].append(index)
            hands = [None] * len(entries)
            positions = [None] * len(entries)
            for note, index in indexed:
                positions[index] = (note[0] / (4 * unit_scale), note[1] / (4 * unit_scale))
            for staff, assigned in staff_notes.items():
                for note in assigned:
                    hands[indices[note].popleft()] = staff
            analysis_out.update(hands=hands, positions=positions, grid=grid, tempo=bpm, meter=(numerator, denominator),
                                pickup_offset=first_left / (4 * unit_scale), key=score_key, piano_key=key,
                                alignment_warning=alignment_warning)
        elif analysis_out is not None:
            positions = [None] * len(entries)
            remaining = defaultdict(deque)
            for note, index in indexed:
                remaining[note].append(index)
            for note in quantized:
                index = remaining[note].popleft()
                positions[index] = (note[0] / (4 * unit_scale), note[1] / (4 * unit_scale))
            analysis_out.setdefault('other_positions', {})[key] = positions
        if simplify and is_piano:
            staff_notes = _simplify_piano_hands(staff_notes, length)
            if analysis_out is not None:
                available = defaultdict(list)
                for note, index in indexed:
                    available[(hands[index], note[0], note[2], note[3])].append((note, index))
                simplified_hands = [None] * len(entries)
                simplified_positions = [None] * len(entries)
                for staff, assigned in staff_notes.items():
                    for note in assigned:
                        candidates = available[(staff, note[0], note[2], note[3])]
                        match = next((i for i, (original, _) in enumerate(candidates) if original == note), 0)
                        _, index = candidates.pop(match)
                        simplified_hands[index] = staff
                        simplified_positions[index] = (note[0] / (4 * unit_scale), note[1] / (4 * unit_scale))
                analysis_out.update(hands=simplified_hands, positions=simplified_positions, simplified=True)
        score_part = _element(parts, 'score-part', id=f'P{number}')
        _element(score_part, 'part-name', names.get(key[0]) or ('Piano' if is_piano else f'Instrument {number}'))
        inst = _element(score_part, 'score-instrument', id=f'I{number}')
        _element(inst, 'instrument-name', names.get(key[0]) or ('Piano' if is_piano else f'Instrument {number}'))
        midi_inst = _element(score_part, 'midi-instrument', id=f'I{number}')
        _element(midi_inst, 'midi-channel', key[1] + 1)
        _element(midi_inst, 'midi-program', programs[key] + 1)
        groups.append((number, staff_notes, is_piano, key))
    if analysis_only:
        if analysis_out is not None and 'hands' not in analysis_out:
            raise ValueError('Hand-track export requires one piano instrument track.')
        return bpm
    for number, staff_notes, is_piano, part_key in groups:
        part = _element(score, 'part', id=f'P{number}')
        lanes = _piano_voices(staff_notes) if is_piano else {staff: _voices(entries) for staff, entries in staff_notes.items()}
        if any(len(voices) > 4 for voices in lanes.values()):
            raise ValueError('This instrument requires more than four simultaneous voices. Enable Simplify score when needed and try again; the MIDI will be preserved.')
        last = max(end for entries in staff_notes.values() for _, end, _, _ in entries)
        pedal_events = []
        if is_piano and notate_pedal:
            state = False
            for position, pressed in sorted(pedal_by_channel[part_key[1]]):
                if pressed != state:
                    pedal_events.append((position, pressed))
                    state = pressed
            if pedal_events:
                last = max(last, pedal_events[-1][0] + 1)
        count = (last + length - 1) // length
        for measure_no in range(count):
            left, right = measure_no * length, (measure_no + 1) * length
            if measure_no == 0:
                left = first_left
            measure = _element(part, 'measure', number=str(measure_no + 1),
                               **({'implicit': 'yes'} if measure_no == 0 and first_left else {}))
            if measure_no == 0:
                attrs = _element(measure, 'attributes')
                _element(attrs, 'divisions', 4 * unit_scale)
                if mode:
                    key_signature = _element(attrs, 'key')
                    _element(key_signature, 'fifths', fifths)
                    _element(key_signature, 'mode', mode)
                time = _element(attrs, 'time')
                _element(time, 'beats', numerator)
                _element(time, 'beat-type', denominator)
                if is_piano:
                    _element(attrs, 'staves', 2)
                for staff in staff_notes:
                    clef = _element(attrs, 'clef', **({'number': str(staff)} if is_piano else {}))
                    _element(clef, 'sign', 'F' if staff == 2 else 'G')
                    _element(clef, 'line', 4 if staff == 2 else 2)
                direction = _element(measure, 'direction', placement='above')
                dt = _element(direction, 'direction-type')
                metro = _element(dt, 'metronome')
                _element(metro, 'beat-unit', 'quarter')
                initial_bpm = bpm
                if grid:
                    initial_marks = grid.tempo_marks(original_length, 0)
                    initial_bpm = next((value for offset, value in reversed(initial_marks) if offset * unit_scale <= first_left), bpm)
                _element(metro, 'per-minute', f'{initial_bpm:g}')
                _element(direction, 'sound', tempo=f'{initial_bpm:g}')
            if grid:
                for offset, current_bpm in grid.tempo_marks(original_length, measure_no):
                    offset *= unit_scale
                    if measure_no == 0:
                        if offset < first_left:
                            continue
                        offset -= first_left
                    direction = _element(measure, 'direction')
                    dtype = _element(direction, 'direction-type')
                    _element(dtype, 'words', '\u200b', **{'print-object': 'no'})
                    if offset:
                        _element(direction, 'offset', offset, sound='yes')
                    _element(direction, 'sound', tempo=f'{current_bpm:.6f}')
            for position, pressed in pedal_events:
                if left <= position < right:
                    direction = _element(measure, 'direction', placement='below')
                    dtype = _element(direction, 'direction-type')
                    _element(dtype, 'pedal', type='start' if pressed else 'stop', line='yes')
                    if position > left:
                        _element(direction, 'offset', position - left, sound='yes')
                    _element(direction, 'staff', 2)
                    _element(direction, 'sound', **{'damper-pedal': 'yes' if pressed else 'no'})
            streams = [(staff, idx, lane) for staff, ls in lanes.items() for idx, lane in enumerate(ls)
                       if any(start < right and end > left for start, end, _ in lane)]
            for staff in staff_notes:
                if not any(stream[0] == staff for stream in streams):
                    streams.append((staff, 0, []))
            streams.sort(key=lambda item: (item[0], item[1]))
            for stream_idx, (staff, lane_idx, lane) in enumerate(streams):
                if stream_idx:
                    backup = _element(measure, 'backup')
                    _element(backup, 'duration', right - left)
                voice = str((staff - 1) * 100 + lane_idx + 1)
                cursor = left
                written = []
                for start, end, pitches in lane:
                    if start >= right or end <= left:
                        continue
                    segment_start, segment_end = max(start, left), min(end, right)
                    if segment_start > cursor:
                        written.extend(_write_rests(measure, cursor, segment_start, length, voice,
                                                    staff if is_piano else None, unit_scale))
                    for a, b, kind, dotted, triplet in _score_pieces(segment_start, segment_end, length, unit_scale):
                        note = _write_note(measure, pitches, voice, staff if is_piano else None, a, b, kind, dotted,
                                           a > start, b < end, fifths, triplet)
                        written.append((note, a, b, kind, triplet))
                    cursor = segment_end
                if cursor < right:
                    written.extend(_write_rests(measure, cursor, right, length, voice,
                                                staff if is_piano else None, unit_scale))
                if triplets:
                    _mark_tuplets(written)
            from omnizart.desktop.score_export import _hide_covered_rests
            _hide_covered_rests(measure)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as output:
        ET.ElementTree(score).write(output, encoding='utf-8', xml_declaration=True)
    return bpm

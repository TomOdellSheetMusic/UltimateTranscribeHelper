"""Conservative beat-level detection and spelling of simple triplet rhythms."""
from collections import defaultdict
import math


def triplet_beats(onsets):
    """Find pairs of distinct third-beat attacks in the same two-beat group.

    A single off-grid attack is not enough evidence to change the rhythm. The
    threshold is expressed in beats, so it does not assume a particular tempo.
    """
    evidence = defaultdict(set)
    boundaries = set()
    for position in onsets:
        if not math.isfinite(position) or position < 0:
            continue
        beat = int(math.floor(position))
        phase = position - beat
        if phase <= .07:
            boundaries.add(beat)
        elif phase >= .93:
            boundaries.add(beat + 1)
        for third in (1, 2):
            if abs(phase - third / 3) <= .055 and abs(phase - round(phase * 4) / 4) >= .045:
                evidence[beat].add(third)
    selected = set()
    for beat, thirds in evidence.items():
        if beat in boundaries and thirds == {1, 2}:
            selected.add(beat)
        if beat in boundaries and 2 in thirds and 1 in evidence.get(beat + 1, set()):
            selected.update((beat, beat + 1))
    return selected


def quantize(position, triplets):
    """Return a score position in twelfths of a quarter note."""
    position = max(0.0, position)
    beat = int(math.floor(position))
    subdivisions = 3 if beat in triplets else 4
    return beat * 12 + round((position - beat) * subdivisions) * (12 // subdivisions)


def next_position(position, triplets):
    beat = position // 12
    step = 4 if beat in triplets else 3
    return min((beat + 1) * 12, (position // step + 1) * step)


def pieces(start, end, measure_length):
    """Spell a span on the mixed grid with the fewest conventional pieces.

    Each returned item is (start, end, type, dotted, triplet). Unrepresentable
    intervals fail explicitly rather than generating malformed MusicXML.
    """
    values = ((48, 'whole', False, False), (36, 'half', True, False),
              (24, 'half', False, False), (18, 'quarter', True, False),
              (12, 'quarter', False, False), (9, 'eighth', True, False),
              (8, 'quarter', False, True), (6, 'eighth', False, False),
              (4, 'eighth', False, True), (3, '16th', False, False))
    best = {end: (0, None)}
    for position in range(end - 1, start - 1, -1):
        bar_end = (position // measure_length + 1) * measure_length
        for size, kind, dotted, triplet in values:
            right = position + size
            if right > min(end, bar_end) or right not in best:
                continue
            if triplet:
                if position % 4:
                    continue
            elif position % 3 or (size >= 12 and position % 12):
                continue
            cost = best[right][0] + 1 + (.03 if triplet else 0)
            if position not in best or cost < best[position][0]:
                best[position] = (cost, (right, kind, dotted, triplet))
    if start not in best:
        raise ValueError('The detected mixed rhythm cannot be represented safely; review the beat alignment.')
    while start < end:
        right, kind, dotted, triplet = best[start][1]
        yield start, right, kind, dotted, triplet
        start = right

"""Conservative major/minor key estimates for score notation."""
import math

MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
KEYS = {'C': (0, 'major'), 'G': (1, 'major'), 'D': (2, 'major'), 'A': (3, 'major'),
        'E': (4, 'major'), 'B': (5, 'major'), 'F#': (6, 'major'), 'C#': (7, 'major'),
        'F': (-1, 'major'), 'Bb': (-2, 'major'), 'Eb': (-3, 'major'), 'Ab': (-4, 'major'),
        'Db': (-5, 'major'), 'Gb': (-6, 'major'), 'Cb': (-7, 'major'),
        'Am': (0, 'minor'), 'Em': (1, 'minor'), 'Bm': (2, 'minor'), 'F#m': (3, 'minor'),
        'C#m': (4, 'minor'), 'G#m': (5, 'minor'), 'D#m': (6, 'minor'), 'A#m': (7, 'minor'),
        'Dm': (-1, 'minor'), 'Gm': (-2, 'minor'), 'Cm': (-3, 'minor'), 'Fm': (-4, 'minor'),
        'Bbm': (-5, 'minor'), 'Ebm': (-6, 'minor'), 'Abm': (-7, 'minor')}
PITCHES = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
           'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
           'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}


def estimate_key(notes):
    """Return a key name only when sufficient evidence separates candidates."""
    histogram = [0.0] * 12
    count = 0
    for group in notes.values():
        for start, end, pitch, _ in group:
            histogram[pitch % 12] += min(max(end - start, 0.02), 2.0)
            count += 1
    if count < 8 or sum(value > 0 for value in histogram) < 4:
        return None
    mean = sum(histogram) / 12
    norm = math.sqrt(sum((x - mean) ** 2 for x in histogram))
    if norm == 0:
        return None
    candidates = []
    for name, (fifths, mode) in KEYS.items():
        root = PITCHES[name[:-1] if mode == 'minor' else name]
        profile = MAJOR if mode == 'major' else MINOR
        average = sum(profile) / 12
        denom = norm * math.sqrt(sum((x - average) ** 2 for x in profile))
        correlation = sum((histogram[(i + root) % 12] - mean) * (profile[i] - average)
                          for i in range(12)) / denom
        candidates.append((correlation, -abs(fifths), name))
    candidates.sort(reverse=True)
    best = candidates[0]
    # Enharmonic spellings represent the same tonal candidate.
    other = next((item for item in candidates[1:] if
                  (PITCHES[item[2][:-1] if KEYS[item[2]][1] == 'minor' else item[2]], KEYS[item[2]][1]) !=
                  (PITCHES[best[2][:-1] if KEYS[best[2]][1] == 'minor' else best[2]], KEYS[best[2]][1])), None)
    return best[2] if best[0] >= .5 and other and best[0] - other[0] >= .06 else None

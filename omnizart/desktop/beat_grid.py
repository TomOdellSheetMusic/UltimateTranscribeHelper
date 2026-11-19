"""Map audio beat and downbeat times to score positions and tempo marks."""
from bisect import bisect_right
from collections import Counter
import math
import statistics


class BeatGrid:
    """Piecewise linear map from audio seconds to beats, aligned to downbeats."""
    def __init__(self, beats, downbeats, fallback_meter=4, tempo=None, downbeat=None, meter=None):
        self.beats = [float(value) for value in beats]
        if len(self.beats) < 4 or any(not math.isfinite(t) or t < 0 for t in self.beats):
            raise ValueError('Too few valid beats for bar alignment.')
        steps = [b - a for a, b in zip(self.beats, self.beats[1:])]
        if any(step <= 0 or step < .2 or step > 2 for step in steps):
            raise ValueError('The detected beat sequence is irregular or out of range.')
        typical = statistics.median(steps)
        if sum(not .55 * typical <= step <= 1.8 * typical for step in steps) > len(steps) * .15:
            raise ValueError('The detected beat sequence is too inconsistent for reliable bar alignment.')
        matched = []
        for time in downbeats:
            time = float(time)
            if not math.isfinite(time) or time < 0:
                continue
            index = min(range(len(self.beats)), key=lambda i: abs(self.beats[i] - time))
            if abs(self.beats[index] - time) <= min(.1, typical * .25) and (not matched or index > matched[-1]):
                matched.append(index)
        if downbeat is None and not matched:
            raise ValueError('The tracker did not find a reliable downbeat.')
        spacings = [b - a for a, b in zip(matched, matched[1:])]
        candidate, count = Counter(spacings).most_common(1)[0] if spacings else (None, 0)
        self.meter_detected = candidate in (3, 4) and count / len(spacings) >= .7
        self.meter = int(meter) if meter is not None else (candidate if self.meter_detected else fallback_meter)
        if self.meter not in (3, 4):
            raise ValueError('Automatic bar alignment currently supports three or four beats per bar.')
        anchor = float(downbeat) if downbeat is not None else self.beats[matched[0]]
        if not math.isfinite(anchor) or anchor < 0:
            raise ValueError('First downbeat must be a non-negative time in seconds.')
        self.manual_tempo = float(tempo) if tempo is not None else None
        if self.manual_tempo is not None and not 30 <= self.manual_tempo <= 300:
            raise ValueError('Score tempo must be between 30 and 300 BPM.')
        self.anchor = anchor
        self.anchor_position = self._raw_position(anchor)
        self.origin = math.floor((self._raw_position(0) - self.anchor_position) / self.meter) * self.meter
        self.tempo_segments = self._simplify_tempos()

    def _simplify_tempos(self, tolerance=.035):
        """Approximate beat times within 35 ms while retaining expressive tempo changes.

        The detected beats still determine note and bar positions. Only the
        playback tempo map is simplified, using the same segments for both
        MusicXML and MIDI. Every detected beat is within tolerance of the map.
        """
        if self.manual_tempo is not None:
            return [(0, 60 / self.manual_tempo)]
        last = len(self.beats) - 1
        boundaries = {0, last}
        pending = [(0, last)]
        while pending:
            start, end = pending.pop()
            period = (self.beats[end] - self.beats[start]) / (end - start)
            deviations = ((abs(self.beats[i] - (self.beats[start] + (i - start) * period)), i)
                          for i in range(start + 1, end))
            error, index = max(deviations, default=(0, start))
            if error > tolerance:
                boundaries.add(index)
                pending.extend(((start, index), (index, end)))
        ordered = sorted(boundaries)
        return [(start, (self.beats[end] - self.beats[start]) / (end - start))
                for start, end in zip(ordered, ordered[1:])]

    def _raw_position(self, time):
        if self.manual_tempo is not None:
            return (time - self.anchor) * self.manual_tempo / 60
        index = max(0, min(len(self.beats) - 2, bisect_right(self.beats, time) - 1))
        left, right = self.beats[index:index + 2]
        return index + (time - left) / (right - left)

    def position(self, time):
        return self._raw_position(time) - self.anchor_position - self.origin

    def unit(self, time):
        return max(0, round(self.position(time) * 4))

    def tempo_marks(self, measure_length, measure_number):
        left = measure_number * measure_length
        right = left + measure_length
        if self.manual_tempo is not None:
            return [(0, self.manual_tempo)] if measure_number == 0 else []
        marks = []
        if measure_number == 0:
            marks.append((0, 60 / self.tempo_segments[0][1]))
        for index, period in self.tempo_segments:
            position = round((index - self.anchor_position - self.origin) * 4)
            if left <= position < right:
                marks.append((position - left, 60 / period))
        # Prefer the last mark if a downbeat is exactly at the beginning.
        return sorted(dict(marks).items())

    @property
    def bpm(self):
        return self.manual_tempo or 60 / statistics.median(b - a for a, b in zip(self.beats, self.beats[1:]))

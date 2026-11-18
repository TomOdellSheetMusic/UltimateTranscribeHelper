"""Compare two MIDI transcriptions in absolute playback time."""
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import mido
from omnizart.desktop.smoothing import _tempo_map


def _notes(path):
    midi = mido.MidiFile(str(path))
    if midi.type == 2:
        raise ValueError("Cannot compare independent-track MIDI files.")
    seconds = _tempo_map(midi)
    result = []
    for index, track in enumerate(midi.tracks, 1):
        label = next((msg.name for msg in track if msg.type == "track_name" and msg.name), f"Track {index}")
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity and msg.channel != 9:
                result.append((seconds(tick), msg.note, label))
    return sorted(result)


def write_comparison(source, reference, destination, tolerance=0.1):
    """Compare pitched onsets within 100 ms, without modifying either MIDI."""
    current, prior = _notes(source), _notes(reference)
    by_pitch = defaultdict(list)
    for index, note in enumerate(prior):
        by_pitch[note[1]].append((note[0], index))
    used, differences = set(), []
    for time, pitch, track in current:
        candidates = by_pitch[pitch]
        start = bisect_left(candidates, (time - tolerance, -1))
        end = bisect_left(candidates, (time + tolerance, len(prior)))
        possible = [(abs(candidates[i][0] - time), candidates[i][1])
                    for i in range(start, end) if candidates[i][1] not in used]
        if possible:
            used.add(min(possible)[1])
        else:
            differences.append((time, track, f"Only in new MIDI: note {pitch}"))
    for index, (time, pitch, track) in enumerate(prior):
        if index not in used:
            differences.append((time, track, f"Only in comparison MIDI: note {pitch}"))
    differences.sort()
    precision = len(used) / len(current) if current else 0.0
    recall = len(used) / len(prior) if prior else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lines = ["Omnizart MIDI comparison", "", f"New: {Path(source).name}", f"Comparison: {Path(reference).name}",
             "", "Same pitch and onset within 100 ms counts as a match. Track names and note lengths are not matched.",
             "Percussion channel is excluded. Audio must start at the same position in both files.",
             "A difference is not proof that either transcription is correct.", "",
             f"Matched: {len(used)}; differences: {len(differences)}.",
             "If the comparison MIDI is a verified reference, onset metrics are:",
             f"Precision: {precision:.3f}; recall: {recall:.3f}; F1: {f1:.3f}.",
             "These metrics do not assess note length, hand assignment, rhythm notation or playability.", ""]
    for timestamp, track, detail in differences[:1000]:
        minutes, seconds = divmod(timestamp, 60)
        lines.append(f"{int(minutes):02d}:{seconds:06.3f}  |  {track}  |  {detail}")
    if len(differences) > 1000:
        lines.append(f"... {len(differences) - 1000} further differences omitted.")
    with Path(destination).open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    return len(differences)

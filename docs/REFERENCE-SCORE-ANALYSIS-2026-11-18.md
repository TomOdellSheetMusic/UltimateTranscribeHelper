# Preliminary analysis using a hand-written reference score

This is an exploratory measurement, not a general model ranking. The reference
is the author's `Tom Odell - Best of Friends (piano only).mscz`, and the audio
is `Tom Odell - Best Of Friends.mp3` from their music library. The files were
read locally; neither the score nor the recording is included in this repository.
The score is a human arrangement, not necessarily a note-perfect transcription
of that exact recording. Its differences from the recording are not automatically
errors in either the score or a model.

## Reference notation

The score has two staves and 81 measures per staff. Its written tempo is 135
quarter notes per minute in 4/4. It contains 944 written note elements; this
count includes tied continuations and is not a count of performed attacks.
Counting the voice containers per staff-measure gives 79 with one voice, 77
with two and six with three. Thus overlapping voices are common in this score,
even though the passage may be played by just two hands. An exporter needs to
represent those voices without producing unnecessarily confusing rests and ties.

Across the local score collection, some files contain tuplets, including `Go
Tell Her now` and `You're gonna break my heart tonight`. The current fixed
sixteenth-note notation grid cannot faithfully express those rhythms. These
examples are useful regression candidates, subject to confirming the matching
recording and arrangement.

## Pilot transcription comparison

Both bundled Transkun and Basic Pitch were run with their default settings on
the first 30 seconds of the unseparated stereo MP3, on CPU. The comparison uses
the first 15 seconds of the written score at its marked tempo. The first chord
in the Transkun output occurs at approximately 1.056 seconds in the recording.
The written times were shifted by that amount and multiplied by 1.02 to roughly
match the recording. The scale was selected by inspecting Transkun matches on
this same excerpt, so the Transkun result has an alignment advantage and must
not be used as an unbiased model ranking.

Written notes were extracted directly from the score's MuseScore XML: both
staves and all voices, respecting written durations and dots, with 4/4 bars
and the initial 135 BPM. This simple extractor was checked against the first
measures only; it is not a general MuseScore parser. It does not interpret
repeats, tempo changes, tuplets, ties or later meter changes. Only written note
onsets in the first 15 seconds are included. A predicted note is counted once
if its pitch matches a reference note and its onset is within 100 ms of the
adjusted reference onset. Predicted notes are counted within the corresponding
15.3-second audio window. Matching is greedy and nearest-first, not an optimal
global alignment.

| Model | Reference onsets | Predicted onsets | Matches | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transkun | 96 | 104 | 67 | 0.644 | 0.698 | 0.670 |
| Basic Pitch | 96 | 125 | 67 | 0.536 | 0.698 | 0.606 |

Precision here means matched predicted onsets divided by predicted onsets;
recall means matched reference onsets divided by reference onsets. The values
are approximate: performance timing, arrangement differences, the vocal and
other instruments in the mix, extraction limitations and the chosen alignment
all affect them. They measure neither note duration nor score readability.
In particular, the 1.02 timing adjustment was chosen using the same Transkun
result being measured; it was then applied unchanged to Basic Pitch. No
separate validation excerpt, source separation, or human error review was used.

## Recommended next measurements

1. Verify which audio recording corresponds to each score, and distinguish
   piano-only parts from vocal lines and alternate arrangements.
2. Build a repeat-aware reference export and align score to audio with a method
   that supports changing tempo. Inspect alignments before calculating metrics.
3. Evaluate multiple songs and held-out passages, including separated piano and
   the original mix, using identical alignment for all models. Report onset and
   offset metrics and listen to representative disagreements.
4. Compare generated MusicXML against the reference's meter, rhythmic spelling,
   staff and voice assignment, rests and ties; inspect the rendered score and
   playability as well as numerical statistics. A simple symbol count cannot
   establish whether notation is correct or readable.

The generated pilot MIDI files and short audio excerpt are local build artifacts
under `dist/score-analysis`, not committed reference material.

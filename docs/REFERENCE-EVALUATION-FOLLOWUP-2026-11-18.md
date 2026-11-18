# Reference evaluation follow-up

Two short excerpts were compared against the author's hand-written scores.
Their recordings and scores remain on the local machine. This is a small pilot,
not a general model ranking or proof that the arrangements reproduce every
performed note.

The evaluation script `scripts/evaluate_reference.py` reads supported MuseScore
archives and compares note onsets with a shared recording alignment. It rejects
tuplets, repeats, unsupported spanners, position changes, meter changes and
other notation it cannot safely interpret. It currently supports 3/4 and 4/4,
constant tempo and a subset of MuseScore 2 and 3 files. Legacy ties are counted
as one attack. These restrictions are intentional; more reference coverage is
needed before treating the collection as a benchmark.

Unlike the earlier pilot, alignment here maximizes the recording's audio onset
strength at the written note times. It does not use any transcription model's
output. The search uses one constant time offset and one tempo scale over each
15-second excerpt. Search ranges were chosen after examining the recordings and
the Transkun predictions, so they are not a blind test. Repeated patterns and
non-piano instruments can produce incorrect alignment; visual or listening review
is still necessary. A constant tempo cannot represent rubato. The earlier
pilot's fitted values and a first-half-only fit demonstrate how drastically a
small tempo error can alter the resulting accuracy.

Notes are matched one-to-one by pitch within 100 ms of the adjusted reference
onset, maximizing the number of matches independently for each pitch. Precision
is matches divided by predicted onsets; recall is matches divided by reference
onsets. Durations, bar lines, voice assignment, playability and appearance are
not measured. The mixture includes non-piano sounds; a different arrangement,
missing reference notes, tempo changes and audio alignment errors all affect
the scores.

An exploratory run on `Long Way Down`, `Sense`, and `True Colors` produced
poor note matches. The `Long Way Down` alignment landed near the edge of its
search range, and the other score/recording pairs may differ in arrangement,
transposition, or alignment. Those results have not been validated and are not
included in the model comparison; they cannot be interpreted as model accuracy.

## First 15 seconds of Best of Friends

The reference is `Best of Friends (piano only).mscz`; the audio is the original
stereo MP3. Fitted offset: 1.04 s, scale: 1.022. There are 96 reference attacks.
Models ran on the first 30 seconds of audio with their bundled default settings.

| Input and model | Predicted | Matched | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original mix, Transkun | 104 | 67 | .644 | .698 | .670 |
| Original mix, Transkun V2 Aug | 106 | 67 | .632 | .698 | .663 |
| Original mix, Pedal Piano | 116 | 71 | .612 | .740 | .670 |
| Original mix, Basic Pitch | 125 | 67 | .536 | .698 | .606 |
| Six-stem isolated piano, Transkun | 101 | 64 | .634 | .667 | .650 |

The isolated piano result does not improve this excerpt under this metric. It
does not establish that isolation is worse on other songs or for listening.

## First 15 seconds of I Know

The reference is `I know.mscz`; the audio is the original stereo MP3. Fitted
offset: 4.00 s, scale: 1.000. There are 115 reference attacks.

| Model on original mix | Predicted | Matched | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transkun | 102 | 80 | .784 | .696 | .737 |
| Transkun V2 Aug | 111 | 83 | .748 | .722 | .735 |
| Pedal Piano | 119 | 83 | .697 | .722 | .709 |
| Basic Pitch | 126 | 76 | .603 | .661 | .631 |

## Notation findings

The first eight measures of the two-staff Best of Friends reference contain 88
note events in the simple extraction used for a preliminary staff check. Given
those already quantized reference notes, the current heuristic assigned all 88
to the same staves as the reference. This does not evaluate audio transcription
or establish accuracy on difficult hand crossings, pedal, or other pieces.

The reference collection includes C-sharp-major scores and several G-flat-major
scores. The exporter previously used only a global sharp/flat preference,
which could spell key-signature notes incorrectly. It now spells diatonic notes
according to the signature, including B-sharp, E-sharp, C-flat and F-flat, and
computes their octave without changing the sounding MIDI pitch. Chromatic
spelling still needs contextual harmonic analysis.

For a concrete example, `Tom Odell - Silhouette.mscz` is in a seven-sharp key.
Its stored spelling values include 72 B-sharp note elements and 249 E-sharp
note elements (including any tied continuations). The old exporter would have
written those sounding pitches as C and F natural. The change is covered by
tests over every MIDI pitch and key signature, plus explicit octave examples.

The reference scores also contain tuplets and overlapping voices. The exporter
now detects some clearly supported eighth- and quarter-note triplet patterns
in quarter-note meters while retaining straight sixteenths elsewhere. Its
conservative beat-level heuristic does not cover all tuplets or compound meters,
and alignment and expressive timing can still make its interpretation wrong.
When a beat has clear evidence of both straight subdivisions and triplets, the
exporter conservatively retains the straight grid; simultaneous polyrhythms
therefore still need manual correction.
This evaluation does not establish that generated scores are readable.

An automated example reproduces the three quarter-note triplet chords in
measure 28 of the Go Tell Her Now reference. Music21 independently parsed the
generated MusicXML as three quarter-note triplets, each lasting two thirds of a
quarter note. A measured Best of Friends passage that had initially produced
false triplets was added as a negative regression; the corrected detector does
not classify that passage as a triplet. These are small examples rather than a
general accuracy benchmark.

The reference archive contains twelve explicit 3:2 quarter-note tuplets in Go
Tell Her Now and 29 explicit 3:2 eighth-note tuplets in You're Gonna Break My
Heart Tonight. Those two common values are supported by the new spelling path;
this count alone does not verify that their detection succeeds from audio.
The measure 28 regression also includes its simultaneous octave bass and
upper-staff triads. Music21 independently read both staves as three 3:2
quarter-note triplets with two-thirds-of-a-beat durations, with two pitches
per bass chord and three per upper chord.

## DuET score experiment

[DuET](https://github.com/wei-zeng98/DuET) revision
`d68c7f219aa5f459673aa4b64f1d491c13d7eee4` and model
`duet-epoch83-v1` were run on CPU on the same 30-second Transkun MIDI from
Best of Friends. The model retained all 182 input note predictions. Its
readable MusicXML reported a 66 BPM quarter-note tempo and eight 4/4 measures;
the human reference marks 135 BPM. Omnizart with a manually specified 132.09
BPM and 1.04 s downbeat produced 17 measures, including an empty first bar.
Both outputs need review: DuET chose approximately half the reference beat
rate, and Omnizart extrapolated a whole opening bar before the first downbeat.
This is one short excerpt, not evidence about DuET on other music.

DuET's score contained 248 pitched note elements, 67 tie starts, seven rests
and two clefs; Omnizart's contained 222 pitched note elements, 40 tie starts,
193 rests and two clefs. These counts are affected by different meter, bar
placement and voice encoding; they do not establish which output is more
readable or accurate. DuET also reported one duplicate removed and one overlap
shortened in its listening MIDI. Its model weights are separately licensed under
CC BY-NC 4.0, and the experiment was local; it is not bundled in Omnizart.

The same model was also run on the 30-second `I Know` Transkun MIDI. Here it
reported 148 BPM, 4/4 and a six-flat key signature, matching the reference's
initial 148 BPM, 4/4 and key signature. Its readable output contained 16
measures and 261 pitched note elements. This checks initial metadata only: its
bar boundaries, notes, voices and visual readability have not been compared
measure by measure. These contrasting examples are a reason to evaluate DuET
further rather than accept or reject it based on one song.

The exporter now marks an extrapolated opening measure as a partial measure,
starting at recording time zero instead of inserting the part of the bar that
precedes the recording. A regression test checks the shorter MusicXML duration,
downbeat position and preservation of the input MIDI. Pickup-note handling is
also covered by the existing beat-grid tests. A MuseScore playback round trip of
this new case has not yet been verified. On the Best of Friends pilot the
partial opening measure now lasts approximately 1.022 seconds instead of a
full 1.817-second 4/4 bar; the original first detected attack is at 1.056
seconds. The remaining difference is partly due to sixteenth-note quantization.

## Reproduction

Run with the project runtime, supplying local paths to the score, recording and
already generated MIDI files. For example:

```powershell
.\.runtime311\python.exe scripts\evaluate_reference.py SCORE.mscz MODEL1.mid MODEL2.mid --seconds 15 --align-audio AUDIO.wav --offset-range 0 2
```

The offset range must cover the recording time where the score starts. For a
manually verified alignment, use `--offset` and `--scale` instead. The optional
`--start` selects a later written-score window. Keep the same alignment and
window for all models, and validate that the score and audio are the same
arrangement. Output and reference data are not included in the repository.

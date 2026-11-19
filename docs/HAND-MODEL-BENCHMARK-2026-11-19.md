# Hand-assignment pilot: Sirens and three written scores

This is a small exploratory benchmark, not a ranking across music or a claim
that a written staff always equals the hand actually used. No new model was
integrated into the app. Model-specific Sirens MIDIs are in the local ignored
`dist/hand-model-benchmark/sirens-exact/` directory; the source files and downloaded
model environments are also local and are not part of the repository.

**Listening and visual review:** The user compared the three alternative Sirens
hand MIDIs with the existing Omnizart output and reported that all three were
worse. The score-derived agreement figures below do not override that feedback
or establish performance on this transcription. Keep the existing model;
the alternatives have not been integrated.

## Inputs and fairness

All Sirens models received the exact 2,459-note
`Tom Odell - Sirens_2_hands.mid` supplied by the user. It already contains the
app's bar alignment, quantization and hand assignment. The alternative models
saw the same notes and tempo map
on a single anonymous MIDI track; their labels were applied back to the same
aligned MIDI. Every delivered MIDI was checked for identical note starts, ends,
pitches and velocities, and identical tempo, meter, pedal and other control
events. The three models' predictions are used directly, without Omnizart's
reach heuristic. Thus this is a comparison of hand assignments, not of their
respective timing or score-engraving systems.

For the numerical pilot, MuseScore exported the user's Best of Friends (piano
only), I know and Silhouette scores to MusicXML. The first 16 written measures
of each were extracted with ties joined and upper/lower staves treated as
right/left labels. These excerpts contain no grace notes, tuplets, time-signature
changes or repeat barlines. Timing follows the written score and its tempo;
velocity is uniformly 80. These are clean score-derived notes, not model
transcriptions or performances, and the assignment of a note to a written staff
does not prove the physical hand. No direct ground-truth labels were available
for the Sirens MIDI. The Best of Friends first-eight-measure extraction was
cross-checked against `scripts/evaluate_reference.py`: both found 88 attacks,
with onset differences limited to 0.1 ms from rounding.

## Agreement with written staves

Each entry is correctly assigned notes / all notes. The overall column is
weighted by note count. The Best of Friends excerpt was used to check output
label orientation, so its results are not an independent polarity test.

| Method | Best of Friends | I know | Silhouette | Overall |
| --- | ---: | ---: | ---: | ---: |
| Current app: Piano SVSep + heuristic | 135/176 (76.7%) | 230/233 (98.7%) | 138/138 (100.0%) | 503/547 (92.0%) |
| Piano SVSep raw staff labels | 127/176 (72.2%) | 227/233 (97.4%) | 136/138 (98.6%) | 490/547 (89.6%) |
| midi2hands, corrected polarity | 175/176 (99.4%) | 190/233 (81.5%) | 126/138 (91.3%) | 491/547 (89.8%) |
| PM2S hand model, calibrated polarity | 172/176 (97.7%) | 221/233 (94.8%) | 137/138 (99.3%) | 530/547 (96.9%) |
| BACHprop, 50-note chunks | 169/176 (96.0%) | 221/233 (94.8%) | 129/138 (93.5%) | 519/547 (94.9%) |

For the current app, score positions were converted to its sixteenth-note input
grid and the same Piano SVSep checkpoint and hand-reach heuristic were used.
The other models used their released input preprocessing, except for the
explicit adaptations below. The high figures here must not be extrapolated to
the noisy, pedal-heavy Sirens transcription or to hand crossings.

## Model provenance and adaptations

* [midi2hands](https://github.com/pianoviz/midi2hands), commit
  `78e546b520144b301de5e793ceb83be61356a65d`, ONNX SHA-256
  `6b7a7858a881e9e231bb3aeeecc1480c9b43a816e3ab1c2cac2182dc34b44e18dd`.
  The published autoregressive inference and MIDI preprocessing were used.
  Its preprocessor calls track 1 right hand, but the bundled training MIDI
  files inspected label track 0 `Rechts` and track 1 `Links`. The checkpoint's
  output polarity was inverted for readable right/left tracks; the unmodified
  polarity output from the first Sirens run is retained locally in the
  `sirens/` folder as `midi2hands_native_hands.mid`.
* [PM2S](https://github.com/cheriell/PM2S), commit
  `9586f91cd16aaa50dbb82f720f6a34a3e0186f47`, official hand checkpoint SHA-256
  `35d0b768fc68f9ec23f77b512ce3cada9b4e9dd816e3ab1c2a1714fa05bc5805`.
  Only its hand model was run; it did not set bar lines, tempo or note lengths.
  Its training code assigns labels from instrument order while a comment names
  the opposite polarity. Best of Friends was used to calibrate that polarity;
  I know and Silhouette are separate score excerpts for this polarity check.
  The native-polarity MIDI from the first Sirens run is retained locally in
  the `sirens/` folder as `pm2s_native_hands.mid`.
* [BACHprop](https://github.com/lemonadedw/BACHprop), commit
  `ec4672194effa90891bc83af4b748f7b7c045051`, checkpoint SHA-256
  `a5f6cfed61d4a68e849bf4fd70061022d0a7eb67a0232128f0f8c261f302e7ac`.
  The checkpoint has a fixed 50-note input even though its example prediction
  script submits a whole song. Inputs were split into the non-overlapping
  50-note windows used by its training loader, with the last window zero-padded;
  Keras 3.12 ran with the PyTorch backend. Boundary effects and padding can
  affect predictions.

`scripts/benchmark_hand_models.py` performs the local extraction, inference and
MIDI reconstruction. Its dependencies and checkpoints must be separately
downloaded; it refuses to overwrite MIDI outputs. User recordings and scores
are not checked in.

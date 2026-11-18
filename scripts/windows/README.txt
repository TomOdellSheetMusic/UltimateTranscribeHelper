Omnizart for Windows

Extract the complete archive to a folder, then double-click Omnizart.vbs.
Choose a transcription type and an input file, select an output folder, and
click Add to queue. Jobs run one at a time and retain the settings selected
when added. Drop multiple files onto the input area or choose Add multiple
files to queue them with the same settings. Select a job to see its details;
use its Open folder button to open its results. Waiting jobs can be removed. The queue is kept while the app
is open. Use the Input, Queue and Details buttons to navigate, and the tabs to
find separation, score, cleanup and model options.
Results for the same input name share a folder in your chosen
output folder; new runs use numbered file names and do not replace earlier ones.

Piano beat alignment, neural hand assignment, score simplification and note
cleanup are combined in the final `_hands.mid`; the original transcription
is also kept. Beat alignment writes a tempo map and quantizes final notes to
the same grid as the optional MusicXML score. A pickup may introduce leading
silence so the first detected bar line falls on a MIDI bar boundary. For a
hand-split MIDI without those options, select "Save MIDI with left and right
hand tracks" under MIDI cleanup. Hand assignments are estimates.
To process an existing file, choose "Split piano MIDI into hands" as the task.
This standalone mode can also use neural hand assignment; it preserves the
input timing, velocities, tempo changes and pedal controls. It requires one
pitched piano track/channel; percussion is retained. Since this mode has no
audio, it uses the input MIDI's timing instead of detecting audio beats.

Beat tracking accepts MIDI input. The other modes accept audio input. Vocal
pitch contour produces CSV and WAV files; the other modes produce MIDI and,
in some cases, CSV files. Transcription runs locally and can take several
minutes. The included models are trained on particular kinds of music and
their results may require editing.

Basic Pitch is an alternative model for individual instruments and voice. It
runs on the CPU and is included with the app. Advanced settings change with
the selected model; hover over a setting for an explanation.

Transkun is an additional CPU piano model. For music containing both voice
and piano, try "Reduce vocals"; the optional accompaniment WAV lets you hear
what the separator removed. Separation can also affect the piano, so compare
the results with and without it. Choose UVR MDX or the legacy Spleeter model
in the separation model selector; both run offline on CPU. Select "Add a separate vocal track to MIDI"
to transcribe the sung melody alongside the instruments in one MIDI file.
This takes longer and both tracks can require editing. Models & import lists the bundled choices,
links to their projects and lets you select a complete compatible Omnizart
music checkpoint folder. Multiple imported checkpoints remain available in
the model dropdown. Other model formats are not interchangeable.
"Manage separation models" offers optional CPU UVR Vocal FT and Instrumental
HQ4/HQ5 checkpoints for download or import. These require the sherpa-onnx versions
with metadata, and the app verifies each file before use.

Pedal Piano is an optional CPU model that predicts piano notes and sustain-pedal
events. In Models & import, download its checkpoint (about 164 MiB) or import
the official checkpoint file. The app verifies the model before loading it.
Transkun V2 Aug is another optional piano checkpoint. Select it in the model
dropdown, then open Models & import to download or import its official archive.

Optional GPU acceleration is available for UVR separation via DirectML on
compatible NVIDIA, AMD and Intel graphics. Enable "Use GPU for UVR separation";
CPU is the default. GPU index 0 is the default Windows adapter. If there is
more than one GPU, try index 1 for the other adapter. Transkun, vocal transcription
and the other transcription models still use CPU. A compatible DirectX 12
driver is required; select CPU if GPU initialization fails.

Optional "Compare with another MIDI" compares pitched note onsets against an
earlier transcription of the same audio and writes `_comparison.txt`.
Align both recordings at the same start; matching ignores track and duration.
Differences are not proof that either model is right. Drum notes are excluded.

Optional "Create formatted MusicXML score" creates an editable score from
MIDI without requiring MuseScore. Tempo is estimated automatically or can be
entered manually. Audio bar alignment uses Beat This! to estimate beats and
bar lines; its verified model downloads on first use. The first bar-line time
can be entered manually. Clear triplet patterns can be written as triplets;
other notes use a sixteenth grid. Choose MuseScore only if you also want a PDF. Score timing,
voices and spelling are estimates; review the resulting notation. The original
MIDI is not changed. Optional neural piano hand assignment installs the
Piano SVSep code, its model checkpoint and dependencies on first use. The code
is MIT-licensed; the checkpoint's redistribution terms need separate review.
First-time setup requires internet and can use around 1.6 GiB of disk space.
Hand assignment preserves the notes and cannot guarantee a playable score.
The optional simplify-score setting creates a separate simplified piano score.
It may shorten held notes or omit notes to reduce hand span and simultaneous
notes, with a warning.
The full score and original MIDI are preserved. For non-piano instruments it
may simplify the score when needed. Review all results for playability.


Optional "Create note review report" writes a separate `_review.txt` with
timestamps for very short, overlapping and stuck notes, plus possible vocal
octave errors on named vocal tracks. The report
contains suggestions, not confirmed errors. It does not change the MIDI and is
off by default. Percussion channel notes are excluded.

Optional "Smooth fragmented notes" writes a separate `_smoothed.mid` beside
the original. It is off by default. It joins nearby same-pitch notes when at
least one is short, using the chosen maximum gap in milliseconds; intentional
repeated notes can also be merged. Compare the two files. Percussion, named
vocal tracks and passages with sustain or pitch-bend activity are left alone.
Some files have no eligible fragments and will sound unchanged.
The separate "Remove very short notes" option is also off by default and
can remove genuine short notes. Both options write a separate cleaned MIDI.

Experimental "Separate stems (audio)" uses an optional HT-Demucs ONNX model
to create separate drums, bass, other, vocals, guitar and piano WAV files.
Choose Models & import to download or import the model first. GPU is optional;
the separated piano can be imperfect. Open a resulting WAV as input to
transcribe that stem. This mode does not automatically create MIDI files.

For an audio file that is already isolated piano, select "Input is already
piano only". This does not run separation. It also permits neural piano hand
assignment with YourMT3 or MuScriptor. When a model predicts multiple instrument
tracks from that input, the app saves its original multitrack MIDI and creates
a separate single-piano MIDI for the score; predicted notes may still be wrong.

The standalone stem mode and the "Separate a stem before transcribing" option can also use pymss. Choose
"pymss model" and "Browse pymss models" to install its optional CPU runtime,
search its catalog, download a model, or register local weights and a matching
configuration with the correct architecture. Catalog models may download on
first use. The catalog window lets you choose Hugging Face, ModelScope or the
HF mirror as the download source. Select one stem to transcribe immediately,
and enable "Keep separated WAV" only when you want an audio copy. In standalone
stem mode, select a stem to save, type another pymss stem name, or choose All.
The chosen model must produce the requested stem. Separation backends can still
calculate stems that are not saved. The
separate custom six-stem ONNX option supports only models compatible with the
built-in HT-Demucs mix/stems interface and output order, not arbitrary ONNX.
pymss runs on CPU in its isolated runtime; the existing DirectML option applies
to the built-in six-stem and UVR models, not pymss.

For the project licence see LICENSE. Third-party components retain their own
licences and notices in the runtime and model directories.
See MODEL-LICENSES.md for the model inventory and unresolved redistribution
rights. This build is intended for local evaluation until those are resolved.

The NNLS Chroma plugin is licensed under GPL version 2 or later; see
omnizart/resource/vamp/COPYING. The bundled madmom model files
are licensed for noncommercial use (CC BY-NC-SA 4.0). Commercial
distribution or use of those models requires permission from their authors.
Basic Pitch is distributed under the Apache License 2.0. Its licence and
notices are included in runtime/Lib/site-packages/basic_pitch-0.4.0.dist-info.
The UVR MDX inference implementation is adapted from python-audio-separator;
its MIT licence is in omnizart/desktop/UVR-NOTICES.txt.

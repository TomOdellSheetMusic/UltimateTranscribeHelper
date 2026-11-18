# Optional transcription models

The desktop application can use YourMT3 and MuScriptor as additional
transcribers. Their packages are installed automatically the first time each
model is selected. YourMT3 downloads its weights on first use. MuScriptor's
gated weights must be downloaded manually after accepting Hugging Face's terms.

## YourMT3

The first transcription downloads the YourMT3 checkpoint (about 536 MB) into
the backend's cache. YourMT3 targets multi-instrument recordings and can emit
several instrument tracks. The integration runs `mt3-infer` in a separate
Python environment. Its checkpoint download currently requires Git and Git LFS
on PATH. The weights are published under Apache-2.0 terms; review the upstream
project's per-component notices before redistributing a build.

When **Isolate piano before transcribing** is combined with YourMT3, the app
transcribes the separated piano stem and combines YourMT3's predicted notes
into one piano track. The original multi-instrument prediction is retained as
`_yourmt3_multitrack.mid`. Relabelling does not remove non-piano sounds or
guarantee that the notes are correct; the separator and model can make errors.

## MuScriptor

In **Models & import**, open the official [MuScriptor small model page](https://huggingface.co/MuScriptor/muscriptor-small),
accept its access terms, and download both `model.safetensors` and `config.json` from the Files tab. Save them in the
folder shown there using **Open MuScriptor folder**. The app loads these local
files; it does not download the gated weights itself. The weights are licensed
CC BY-NC 4.0, so they are limited to non-commercial use. Its software is MIT
licensed. The app selects MuScriptor's MIDI output; the score export can turn
that MIDI into MusicXML, with an optional PDF rendered through MuseScore.

Use Python 3.10–3.12 for MuScriptor. The model's `small` variant is intended
for CPU use; larger variants need more memory and benefit from a GPU. Consult
the official project documentation for model selection and Hugging Face
authentication steps.

## Score output

Select **Create formatted MusicXML score** to create an editable score directly
from the MIDI. No MuseScore installation is required. Omnizart estimates the
tempo from note attacks, quantizes a score-only copy to sixteenth notes, assigns
overlapping notes to separate voices, and uses a grand staff for piano models.
**Align bars to audio beats** uses [Beat This!](https://github.com/CPJKU/beat_this)
to estimate beats and downbeats from the original audio and follows its beat
sequence with a score tempo map. The first use installs the optional tracker
and downloads its verified model. The tracker currently supports automatic
bar alignment in 3/4 and 4/4; if it cannot get a reliable grid, the score uses
a constant-tempo fallback and displays a warning. When the meter is unclear,
the MIDI meter is retained and a warning is shown. Enter a BPM, beats per bar,
and/or the first bar-line time in seconds to correct detection, or disable
audio alignment.
The score may contain leading rests to preserve a pickup before the first
bar line. The original MIDI is not changed. A separately installed MuseScore 4
can optionally render a PDF.

**Score key** can be selected manually. Auto estimates major or minor keys
from note durations only when the evidence is sufficiently distinct; otherwise
no key is asserted. Flat keys use common flat spellings. Key detection and
enharmonic spelling are estimates and need review. Key changes are not detected.

The score is a draft: automatic tempo, beat alignment, hand assignment, pitch
spelling, durations and ties may need editing. Percussion tracks are omitted
from MusicXML but remain in the MIDI. Time-signature changes are not yet supported. Dense piano passages may need
extra ties, and passages that exceed a hand span or five held notes are flagged
for review. Score export does not alter the MIDI.


### Optional piano hand model

**Experimental: neural piano hand assignment** uses the
[Piano SVSep](https://github.com/CPJKU/piano_svsep) research model as a source
of staff suggestions. On first use, Omnizart installs dependencies into a
separate environment, including CPU PyTorch, and downloads a SHA-256-verified
model checkpoint. First-time setup requires internet and can use around 1.6 GiB
of disk space. The installation checks the model with a small inference before
marking it ready.
The code is MIT-licensed; the checkpoint's redistribution terms need separate review.
It requires internet for setup, runs on CPU, and can take time on long pieces.
The model's suggestions are balanced with hand-span checks; they are not a
guarantee of playable music. The assignment keeps the input notes; it does not
automatically remove notes that make a passage unplayable. Compare the result
with the default assignment and review difficult passages.
This option requires a single piano instrument track.

**Simplify score when needed** is off by default. For piano it creates a separate
`_simplified.musicxml` in addition to the full score and may omit
notes to reduce simultaneous hand spans and note counts, preserving the top
right-hand and bottom left-hand notes where possible. For other instruments it
may omit quieter conflicting groups when more than four voices overlap. It
reports omissions and never changes the original MIDI. This heuristic does not
guarantee playability; review the result and compare it with the full score.

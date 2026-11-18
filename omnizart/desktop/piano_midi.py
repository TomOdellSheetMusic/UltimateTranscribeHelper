"""Create a single piano track from a piano-stem transcription."""
from pathlib import Path
import warnings


def write_single_piano(source, destination):
    """Retain predicted notes while replacing model instrument labels.

    This does not verify whether a note actually belongs to a piano. The caller
    keeps the unmodified model output as a separate MIDI for comparison.
    """
    import pretty_midi
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing MIDI: {destination}')
    midi = pretty_midi.PrettyMIDI(str(source))
    piano = pretty_midi.Instrument(program=0, name='Piano')
    for instrument in midi.instruments:
        piano.notes.extend(instrument.notes)
        piano.control_changes.extend(instrument.control_changes)
        piano.pitch_bends.extend(instrument.pitch_bends)
    if not piano.notes:
        raise ValueError('The piano-stem transcription contains no notes.')
    midi.instruments = [piano]
    with destination.open('xb') as output:
        try:
            midi.write(output)
        except Exception:
            output.close()
            destination.unlink(missing_ok=True)
            raise
    warnings.warn('YourMT3 instrument labels were combined into one piano track. '
                  'The original multitrack MIDI is preserved; notes may still need correction.', UserWarning)

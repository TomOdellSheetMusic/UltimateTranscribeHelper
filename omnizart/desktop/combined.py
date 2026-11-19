"""Combine instrument transcription with a separated vocal melody."""
from pathlib import Path
from tempfile import TemporaryDirectory


def merge_tracks(instrument_midi, vocal_midi, destination):
    import pretty_midi

    result = pretty_midi.PrettyMIDI(str(instrument_midi))
    vocals = [track for track in vocal_midi.instruments if not track.is_drum and track.notes]
    if not vocals:
        raise ValueError("No pitched vocal notes were detected; try the instrumental-only option or another recording.")
    for track in vocals:
        track.name = "Vocals"
        track.program = 54  # General MIDI voice oohs
    result.instruments = vocals + result.instruments
    result.write(str(destination))


def add_vocals(input_audio, vocal_stem, instrument_midi, output, model='omnizart', device='cpu', gpu_index=0):
    if model == 'omnizart':
        from omnizart.vocal import app
        print("Transcribing the separated vocal melody with Omnizart Vocal on CPU...", flush=True)
        vocal_midi = app.transcribe(input_audio, output=None, vocal_audio=str(vocal_stem))
    elif model == 'rosvot':
        from omnizart.desktop.rosvot import transcribe
        vocal_midi = transcribe(vocal_stem, device=device, gpu_index=gpu_index)
    elif model == 'basic-pitch':
        import pretty_midi
        from omnizart.desktop.basic_pitch import transcribe
        with TemporaryDirectory(prefix='omnizart-vocal-midi-') as temporary:
            path = Path(temporary) / 'vocal.mid'
            transcribe(str(vocal_stem), str(path))
            vocal_midi = pretty_midi.PrettyMIDI(str(path))
    else:
        raise ValueError('Unknown vocal transcription model: ' + str(model))
    destination = Path(output)
    if destination.is_dir():
        destination = destination / (Path(input_audio).stem + ".mid")
    elif destination.suffix.lower() not in (".mid", ".midi"):
        destination = Path(str(destination) + ".mid")
    merge_tracks(instrument_midi, vocal_midi, destination)
    print(f"Combined MIDI written to {destination}", flush=True)

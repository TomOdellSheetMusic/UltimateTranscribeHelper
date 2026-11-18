# pylint: disable=C0303,W1401
import click
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess

from omnizart.cli import silence_tensorflow
from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS
from omnizart.utils import LazyLoader


music = LazyLoader("music", globals(), "omnizart.music")


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
@click.option("--onset-th", type=click.FloatRange(0, 20), default=None, help="Override note onset threshold.")
@click.option("--duration-th", type=click.FloatRange(0, 20), default=None, help="Override note duration threshold.")
@click.option("--instrument-th", type=click.FloatRange(0, 5), default=None, help="Override instrument confidence threshold.")
@click.option("--frame-th", type=click.FloatRange(0, 1), default=None, help="Basic Pitch frame threshold.")
@click.option("--minimum-note-ms", type=click.FloatRange(0, 2000), default=None, help="Basic Pitch minimum note length in milliseconds.")
@click.option("--remove-vocals", is_flag=True, help="Separate vocals before instrument transcription.")
@click.option("--piano-only", is_flag=True, help="Separate a piano stem before transcription (requires the six-stem model).")
@click.option("--input-is-piano", is_flag=True, help="Treat the input as already-isolated piano; do not run piano separation.")
@click.option("--piano-separator", type=click.Choice(['six-stem', 'pymss']), default='six-stem', help="Backend for piano isolation.")
@click.option("--pymss-model", default=None, help="pymss catalog or registered model for piano isolation.")
@click.option("--pymss-source", type=click.Choice(['huggingface', 'modelscope', 'hf-mirror']), default='huggingface')
@click.option("--custom-stem-model", type=click.Path(exists=True, dir_okay=False), default=None, help="Compatible six-stem ONNX model.")
@click.option("--save-accompaniment", is_flag=True, help="Keep the separated accompaniment WAV.")
@click.option("--include-vocals", is_flag=True, help="Add a separate vocal melody track to the MIDI.")
@click.option("--separator", type=click.Choice(["spleeter", "uvr", "vocal-ft", "inst-hq4", "inst-hq5"]), default="spleeter", help="Vocal separation model.")
@click.option("--uvr-device", type=click.Choice(["cpu", "directml"]), default="cpu", help="Device for audio separation.")
@click.option("--gpu-index", type=click.IntRange(min=0), default=0, help="DirectML adapter index.")
@click.option("--compare-midi", type=click.Path(exists=True, dir_okay=False), default=None, help="Compare note onsets with an existing MIDI.")
@click.option("--score-grid", type=click.Choice(["off", "8", "16", "32"]), default="off", help="Optional score-only rhythm grid.")
@click.option("--score", is_flag=True, help="Write a notation draft as MusicXML without MuseScore.")
@click.option("--align-bars", is_flag=True, help="Detect beats and downbeats from the source audio.")
@click.option("--score-meter", type=click.Choice(["3", "4"]), default=None, help="Override the beats per bar.")
@click.option("--score-downbeat", type=click.FloatRange(min=0), default=None, help="Override the first bar line in seconds.")
@click.option("--neural-staff", is_flag=True, help="Use experimental Piano SVSep staff suggestions for one piano track.")
@click.option("--simplify-score", is_flag=True, help="Omit conflicting notes from the score if necessary; preserve the MIDI.")
@click.option("--score-tempo", type=click.FloatRange(30, 300), default=None, help="Score tempo in BPM; omit to estimate it.")
@click.option("--score-key", default="Auto", help="Score key, for example Bb or F#m; Auto estimates a key.")
@click.option("--musescore", type=click.Path(exists=True, dir_okay=False), default=None, help="MuseScore executable for optional MusicXML export.")
@click.option("--lint-notes", is_flag=True, help="Save a non-destructive transcription review report.")
@click.option("--smooth-notes", is_flag=True, help="Also save a conservatively smoothed MIDI copy.")
@click.option("--smooth-gap-ms", type=click.FloatRange(0, 200), default=30, help="Maximum gap to join, in milliseconds.")
@click.option("--remove-short-ms", type=click.FloatRange(0, 200), default=0, help="Also remove notes shorter than this duration; 0 disables.")
def transcribe(input_audio, model_path, output, onset_th, duration_th, instrument_th, frame_th, minimum_note_ms,
               remove_vocals, piano_only, input_is_piano, piano_separator, pymss_model, pymss_source, custom_stem_model,
               save_accompaniment, include_vocals, separator, uvr_device, gpu_index,
               compare_midi, score_grid, score, align_bars, score_meter, score_downbeat, neural_staff, simplify_score, score_tempo, score_key, musescore, lint_notes, smooth_notes, smooth_gap_ms, remove_short_ms):
    """Transcribe a single audio and output as a MIDI file.

    This will output a MIDI file with the same name as the given audio, except the
    extension will be replaced with '.mid'.

    Supported modes are: Piano, Stream, Pop

    \b
    Example Usage
    $ omnizart music transcribe \\ 
        example.wav \\ 
        --model-path path/to/model \\ 
        --output example.mid
    """
    if score_grid != "off" and not musescore:
        raise click.UsageError("--score-grid requires --musescore.")
    if save_accompaniment and not remove_vocals:
        raise click.UsageError("--save-accompaniment requires --remove-vocals.")
    if piano_only and (remove_vocals or include_vocals):
        raise click.UsageError("--piano-only cannot be combined with --remove-vocals or --include-vocals.")
    if input_is_piano and (piano_only or remove_vocals or include_vocals):
        raise click.UsageError("--input-is-piano cannot be combined with audio separation or a vocal track.")
    if piano_separator == 'pymss' and piano_only and (not pymss_model or custom_stem_model or uvr_device != 'cpu'):
        raise click.UsageError('pymss piano isolation requires a model and uses its own CPU runtime.')
    if piano_only and model_path in ("Pop", "Stream"):
        click.echo("Piano-only mode works best with a piano model; Pop and Stream may still produce non-piano notes.", err=True)
    if uvr_device != "cpu" and not (piano_only or (separator == "uvr" and (remove_vocals or include_vocals))):
        raise click.UsageError("DirectML requires piano-only or UVR vocal separation.")
    if model_path in ("Transkun (piano)", "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor"):
        if any(value is not None for value in (onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)):
            raise click.UsageError("These thresholds are not available for the selected piano model.")
    elif model_path == "Basic Pitch":
        if onset_th is not None and onset_th > 1:
            raise click.BadParameter("Basic Pitch onset must be between 0 and 1", param_hint="--onset-th")
        if duration_th is not None or instrument_th is not None:
            raise click.UsageError("Duration and instrument thresholds apply to Omnizart models only.")
    elif frame_th is not None or minimum_note_ms is not None:
        raise click.UsageError("Frame threshold and minimum note length apply to Basic Pitch only.")
    if piano_only:
        with TemporaryDirectory(prefix="omnizart-piano-") as temporary:
            base = Path(temporary) / "separated"
            if piano_separator == 'pymss':
                from omnizart.desktop.pymss_backend import separate as separate_pymss
                separate_pymss(input_audio, base, pymss_model, selected_stems=['piano'], source_name=pymss_source)
            else:
                from omnizart.desktop.six_stems import separate as separate_stems
                options = dict(device=uvr_device, gpu_index=gpu_index)
                if custom_stem_model:
                    options['custom_model'] = custom_stem_model
                separate_stems(input_audio, base, **options)
            piano_audio = str(base.with_name(base.name + "_piano.wav"))
            if model_path in ("YourMT3", "MuScriptor"):
                _transcribe_piano_input(piano_audio, input_audio, model_path, output,
                                        onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
            else:
                _transcribe(piano_audio, model_path, str(_midi_target(output, input_audio)),
                            onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
    elif remove_vocals or include_vocals:
        from omnizart.desktop.separation import remove_vocals as separate, separate_stems
        with TemporaryDirectory(prefix="omnizart-separation-") as temporary:
            separated = Path(temporary) / (Path(input_audio).stem + ".wav")
            vocal_stem = Path(temporary) / "vocals.wav"
            if separator == "uvr":
                from omnizart.desktop.mdx import separate as separate_uvr
                if uvr_device == "cpu":
                    separate_uvr(input_audio, separated, vocal_stem if include_vocals else None)
                else:
                    separate_uvr(input_audio, separated, vocal_stem if include_vocals else None,
                                 device=uvr_device, gpu_index=gpu_index)
            elif separator in ("vocal-ft", "inst-hq4", "inst-hq5"):
                from omnizart.desktop.sherpa_uvr import separate as separate_sherpa_uvr
                separate_sherpa_uvr(separator, input_audio, separated, vocal_stem if include_vocals else None)
            elif include_vocals:
                separate_stems(input_audio, separated, vocal_stem)
            else:
                separate(input_audio, separated)
            if save_accompaniment:
                target = Path(output)
                folder = target if target.is_dir() else target.parent
                name = Path(input_audio).stem if target.is_dir() else (
                    target.stem if target.suffix.lower() in (".mid", ".midi") else target.name)
                saved = folder / (name + "_accompaniment.wav")
                try:
                    with separated.open("rb") as source_file, saved.open("xb") as destination_file:
                        shutil.copyfileobj(source_file, destination_file)
                except FileExistsError as exc:
                    raise click.ClickException(f"Refusing to overwrite existing accompaniment: {saved}") from exc
                click.echo(f"Accompaniment saved to {saved}")
            instrument_input = str(separated) if remove_vocals else input_audio
            if include_vocals:
                instrument_midi = Path(temporary) / "instruments.mid"
                _transcribe(instrument_input, model_path, str(instrument_midi), onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
                from omnizart.desktop.combined import add_vocals
                add_vocals(input_audio, vocal_stem, instrument_midi, output)
            else:
                _transcribe(instrument_input, model_path, output, onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
    else:
        if input_is_piano and model_path not in ("Piano", "PianoV2", "Transkun (piano)", "Transkun V2 Aug", "Pedal Piano"):
            _transcribe_piano_input(input_audio, input_audio, model_path, output,
                                    onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
        else:
            _transcribe(input_audio, model_path, output, onset_th, duration_th, instrument_th, frame_th, minimum_note_ms)
    if smooth_notes or remove_short_ms or lint_notes or score or musescore or compare_midi:
        target = Path(output)
        if input_is_piano and model_path not in ("Piano", "PianoV2", "Transkun (piano)", "Transkun V2 Aug", "Pedal Piano"):
            target = _midi_target(output, input_audio)
        elif target.is_dir():
            target = target / (Path(input_audio).stem + ".mid")
        elif model_path not in ("Basic Pitch", "Transkun (piano)", "Transkun V2 Aug", "Pedal Piano", "YourMT3", "MuScriptor") and not include_vocals:
            if not str(target).endswith(".mid"):
                target = Path(str(target) + ".mid")
        elif target.suffix.lower() not in (".mid", ".midi"):
            target = Path(str(target) + ".mid")
    if compare_midi:
        from omnizart.desktop.compare import write_comparison
        comparison = target.with_name(target.stem + "_comparison.txt")
        try:
            count = write_comparison(target, compare_midi, comparison)
        except (OSError, ValueError, EOFError) as exc:
            raise click.ClickException(f"Original MIDI was saved, but comparison failed: {exc}") from exc
        click.echo(f"Comparison saved to {comparison} ({count} differences).")
    if score:
        from omnizart.desktop.notation import export_score
        score_path = target.with_name(target.stem + ".musicxml")
        try:
            score_options = dict(tempo=score_tempo, score_key=score_key, neural_staff=neural_staff,
                               audio=input_audio, align_bars=align_bars, downbeat=score_downbeat,
                               beats_per_bar=int(score_meter) if score_meter else None,
                               piano=(input_is_piano or piano_only or model_path in
                                      ("Piano", "PianoV2", "Transkun (piano)",
                                       "Transkun V2 Aug", "Pedal Piano")))
            bpm = export_score(target, score_path, simplify=simplify_score and not score_options['piano'], **score_options)
            click.echo(f"MusicXML draft saved to {score_path} ({bpm:g} BPM). Review its notation.")
            if simplify_score and score_options['piano']:
                simplified_path = target.with_name(target.stem + "_simplified.musicxml")
                export_score(target, simplified_path, simplify=True, **score_options)
                click.echo(f"Simplified MusicXML saved to {simplified_path}. Review omitted notes and playability.")
            if musescore:
                from omnizart.desktop.score_export import _run_musescore
                pdf = target.with_name(target.stem + ".pdf")
                if pdf.exists():
                    raise FileExistsError(f"Refusing to overwrite existing PDF: {pdf}")
                _run_musescore(musescore, score_path, pdf)
                click.echo(f"PDF saved to {pdf}.")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            raise click.ClickException(f"Original MIDI was saved, but score export failed: {exc}") from exc
    elif musescore:
        from omnizart.desktop.score_export import export_formatted_score
        score_path = target.with_name(target.stem + ".musicxml")
        pdf = target.with_name(target.stem + ".pdf")
        try:
            export_formatted_score(target, score_path, pdf, musescore,
                                   0 if score_grid == "off" else int(score_grid),
                                   piano_grand_staff=(input_is_piano or piano_only or model_path in
                                                      ("Piano", "PianoV2", "Transkun (piano)",
                                                       "Transkun V2 Aug", "Pedal Piano")))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            raise click.ClickException(f"Original MIDI was saved, but MusicXML export failed: {exc}") from exc
        click.echo(f"Formatted MusicXML saved to {score_path}; print-ready PDF saved to {pdf}. Review the score before use.")
    if lint_notes:
        from omnizart.desktop.lint import write_report
        report = target.with_name(target.stem + "_review.txt")
        try:
            count = write_report(target, report)
        except (OSError, ValueError) as exc:
            raise click.ClickException(f"Original MIDI was saved, but review failed: {exc}") from exc
        click.echo(f"Review report saved to {report} ({count} suggestions).")
    if smooth_notes or remove_short_ms:
        from omnizart.desktop.smoothing import smooth_midi
        cleaned = target.with_name(target.stem + "_smoothed.mid")
        try:
            merged = smooth_midi(target, cleaned, smooth_gap_ms, remove_short_ms, join_notes=smooth_notes)
        except (OSError, ValueError) as exc:
            raise click.ClickException(f"Original MIDI was saved, but smoothing failed: {exc}") from exc
        click.echo(f"Cleaned MIDI saved to {cleaned} ({merged} fragments joined). Original preserved.")


def _midi_target(output, original_audio):
    target = Path(output)
    if target.is_dir():
        target = target / (Path(original_audio).stem + '.mid')
    elif target.suffix.lower() not in ('.mid', '.midi'):
        target = Path(str(target) + '.mid')
    return target


def _transcribe_piano_input(audio, original_audio, model_path, output, *thresholds):
    """Combine predicted instrument tracks only when the source is known to be piano."""
    from omnizart.desktop.piano_midi import write_single_piano
    target = _midi_target(output, original_audio)
    label = model_path.lower() if model_path in ('YourMT3', 'MuScriptor') else 'model'
    original = target.with_name(target.stem + '_' + label + '_multitrack.mid')
    if target.exists() or original.exists():
        raise click.ClickException('Refusing to overwrite an existing piano or multitrack MIDI.')
    with TemporaryDirectory(prefix='omnizart-piano-midi-') as temporary:
        raw = Path(temporary) / 'prediction.mid'
        _transcribe(audio, model_path, str(raw), *thresholds)
        with raw.open('rb') as source, original.open('xb') as preserved:
            shutil.copyfileobj(source, preserved)
    try:
        write_single_piano(original, target, model=model_path if model_path in ('YourMT3', 'MuScriptor') else 'Transcription model')
    except (OSError, ValueError) as exc:
        raise click.ClickException(f'Original model MIDI saved to {original}, but piano output failed: {exc}') from exc
    click.echo(f'Original model MIDI saved to {original}.')


def _transcribe(input_audio, model_path, output, onset_th, duration_th, instrument_th, frame_th, minimum_note_ms):
    if model_path == "Transkun V2 Aug":
        from omnizart.desktop.transkun_aug import transcribe as transkun_aug_transcribe
        transkun_aug_transcribe(input_audio, output)
        return
    if model_path == "Pedal Piano":
        from omnizart.desktop.pedal_piano import transcribe as pedal_piano_transcribe
        pedal_piano_transcribe(input_audio, output)
        return
    if model_path == "Transkun (piano)":
        from omnizart.desktop.transkun import transcribe as transkun_transcribe
        transkun_transcribe(input_audio, output)
        return
    if model_path == "Basic Pitch":
        from omnizart.desktop.basic_pitch import transcribe as basic_pitch_transcribe
        basic_pitch_transcribe(input_audio, output, onset_th, frame_th, minimum_note_ms)
        return
    if model_path == "YourMT3":
        from omnizart.desktop.yourmt3 import transcribe as yourmt3_transcribe
        yourmt3_transcribe(input_audio, output)
        return
    if model_path == "MuScriptor":
        from omnizart.desktop.muscriptor import transcribe as muscriptor_transcribe
        muscriptor_transcribe(input_audio, output)
        return
    silence_tensorflow()
    music.app.transcribe(input_audio, model_path, output=output, onset_th=onset_th,
                         duration_th=duration_th, instrument_th=instrument_th)


def process_doc():
    # Some dirty work for preserving and converting the docstring inside the decorated
    # function into .rst format.
    doc = transcribe.__doc__
    doc = doc.replace("\b", "").replace("    ", "").replace("--", "        --")

    code_block = "\n.. code-block:: bash\n\n"
    doc = doc.replace("$", f"{code_block}    $")

    return doc


__doc__ = process_doc()

"""Command for creating left- and right-hand piano MIDI tracks."""
import click
from pathlib import Path


@click.group()
def hands():
    """Split a piano MIDI into separate hand tracks."""


@hands.command()
@click.argument('input_midi', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', required=True, type=click.Path())
@click.option('--neural-staff', is_flag=True, help='Use neural hand suggestions on the input MIDI grid.')
def transcribe(input_midi, output, neural_staff):
    from omnizart.desktop.hand_midi import split_piano_midi
    target = Path(output)
    if target.suffix.lower() not in ('.mid', '.midi'):
        target = Path(str(target) + '.mid')
    try:
        analysis = None
        if neural_staff:
            from omnizart.desktop.notation import _read_midi, export_score
            analysis = {}
            original_tempo = _read_midi(input_midi)[-1]
            export_score(input_midi, target.with_suffix('.musicxml'), tempo=original_tempo,
                         neural_staff=True, analysis_out=analysis, analysis_only=True)
            analysis['key'] = None
        right, left = split_piano_midi(input_midi, target, analysis=analysis)
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'Hand tracks saved to {target} (right: {right}, left: {left}). Review the assignments.')

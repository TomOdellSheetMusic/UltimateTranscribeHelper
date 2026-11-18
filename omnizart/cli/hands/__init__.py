"""Command for creating left- and right-hand piano MIDI tracks."""
import click
from pathlib import Path


@click.group()
def hands():
    """Split a piano MIDI into separate hand tracks."""


@hands.command()
@click.argument('input_midi', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', required=True, type=click.Path())
def transcribe(input_midi, output):
    from omnizart.desktop.hand_midi import split_piano_midi
    target = Path(output)
    if target.suffix.lower() not in ('.mid', '.midi'):
        target = Path(str(target) + '.mid')
    try:
        right, left = split_piano_midi(input_midi, target)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'Hand tracks saved to {target} (right: {right}, left: {left}). Review the assignments.')

"""CLI entry point for optional six-stem audio separation."""
import click


@click.group()
def stems():
    """Separate an audio recording into six WAV stems."""


@stems.command()
@click.argument("input_audio", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(), help="Output basename, without extension.")
@click.option("--device", type=click.Choice(["cpu", "directml"]), default="cpu")
@click.option("--gpu-index", type=click.IntRange(0, 99), default=0)
@click.option("--stem", multiple=True, help="Save selected stems; repeat for more than one. Default: all.")
@click.option("--custom-model", type=click.Path(exists=True, dir_okay=False), default=None, help="User-supplied ONNX model compatible with the six-stem interface.")
@click.option("--backend", type=click.Choice(["six-stem", "pymss"]), default="six-stem")
@click.option("--pymss-model", default=None, help="pymss catalog or registered model name.")
@click.option("--pymss-source", type=click.Choice(['huggingface', 'modelscope', 'hf-mirror']), default='huggingface')
def transcribe(input_audio, output, device, gpu_index, stem, custom_model, backend, pymss_model, pymss_source):
    """Create drums, bass, other, vocals, guitar, and piano WAV files."""
    try:
        if backend == 'pymss':
            if device != 'cpu' or custom_model:
                raise ValueError('pymss uses its own CPU runtime and model registry.')
            from omnizart.desktop.pymss_backend import separate
            separate(input_audio, output, pymss_model, selected_stems=stem or None, source_name=pymss_source)
        else:
            from omnizart.desktop.six_stems import separate
            options = dict(device=device, gpu_index=gpu_index)
            if stem:
                options['selected_stems'] = stem
            if custom_model:
                options['custom_model'] = custom_model
            separate(input_audio, output, **options)
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

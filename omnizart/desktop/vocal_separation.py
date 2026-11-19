"""Route a vocal-separation request through a selected backend."""
from pathlib import Path


def separate(source, workdir, backend, model=None, source_name='huggingface',
             pymss_device='cpu', uvr_device='cpu', gpu_index=0,
             accompaniment=True, vocals=True):
    """Return accompaniment and vocals paths in workdir, when requested."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    instrumental = workdir / (Path(source).stem + '.wav')
    vocal = workdir / 'vocals.wav'
    if backend == 'pymss':
        if not model:
            raise ValueError('Choose a pymss vocal separation model.')
        from omnizart.desktop.pymss_backend import separate as pymss_separate, vocal_stems
        base = workdir / 'pymss'
        options = {'source_name': source_name}
        if pymss_device == 'cuda':
            options.update(device='cuda', gpu_index=gpu_index)
        outputs = pymss_separate(source, base, model, **options)
        vocal = vocal_stems(outputs, base, instrumental if accompaniment else None, source=source)
    elif backend == 'uvr':
        from omnizart.desktop.mdx import separate as uvr_separate
        options = {} if uvr_device == 'cpu' else {'device': uvr_device, 'gpu_index': gpu_index}
        uvr_separate(source, instrumental, vocal if vocals else None, **options)
    elif backend in ('vocal-ft', 'inst-hq4', 'inst-hq5'):
        from omnizart.desktop.sherpa_uvr import separate as sherpa_separate
        sherpa_separate(backend, source, instrumental, vocal if vocals else None)
    elif backend == 'spleeter':
        from omnizart.desktop.separation import separate_stems, remove_vocals
        if vocals:
            separate_stems(source, instrumental, vocal)
        else:
            remove_vocals(source, instrumental)
    else:
        raise ValueError('Unknown vocal separation backend: ' + str(backend))
    return instrumental if accompaniment else None, vocal if vocals else None

"""Decode ambiguous Beat This activations using the bundled madmom DBN."""
import json
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import omnizart  # Installs compatibility aliases required by madmom.
    import numpy as np
    import madmom.features.downbeats as downbeats

    class NumpyCompat:
        # madmom expects numpy to permit a ragged list of (path, score)
        # tuples. NumPy 1.24+ requires an explicit object dtype for that list.
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def asarray(value, *args, **kwargs):
            if isinstance(value, list) and value and isinstance(value[0], tuple):
                return np.asarray(value, dtype=object)
            return np.asarray(value, *args, **kwargs)

    source, target = map(Path, sys.argv[1:3])
    values = json.loads(source.read_text(encoding='utf-8'))
    beat = np.asarray(values['beat_probabilities'], dtype=np.float64)
    downbeat = np.asarray(values['downbeat_probabilities'], dtype=np.float64)
    if beat.ndim != 1 or len(beat) != len(downbeat) or not len(beat):
        raise ValueError('Invalid beat model probabilities.')
    epsilon = 1e-5
    beat = beat * (1 - epsilon) + epsilon / 2
    downbeat = downbeat * (1 - epsilon) + epsilon / 2
    activations = np.column_stack((np.maximum(beat - downbeat, epsilon / 2), downbeat))
    downbeats.np = NumpyCompat()
    processor = downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55,
                                                        max_bpm=215, fps=50, transition_lambda=100)
    result = processor(activations)
    target.write_text(json.dumps({'beats': result[:, 0].tolist(),
                                  'downbeats': result[result[:, 1] == 1, 0].tolist()}), encoding='utf-8')


if __name__ == '__main__':
    main()

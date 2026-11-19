"""Use Beat This! to detect beat and downbeat times from an audio file."""
import json
from pathlib import Path
import sys


def main():
    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio
    from beat_this.model.postprocessor import Postprocessor
    audio, checkpoint, output = sys.argv[1:4]
    signal, sample_rate = load_audio(audio)
    beat_logits, downbeat_logits = Audio2Frames(checkpoint_path=checkpoint, device='cpu')(signal, sample_rate)
    beats, downbeats = Postprocessor(type='minimal')(beat_logits, downbeat_logits)
    Path(output).write_text(json.dumps({'beats': beats.tolist(), 'downbeats': downbeats.tolist(),
                                      'beat_probabilities': beat_logits.sigmoid().tolist(),
                                      'downbeat_probabilities': downbeat_logits.sigmoid().tolist()}), encoding='utf-8')


if __name__ == '__main__':
    main()

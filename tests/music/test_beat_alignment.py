import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omnizart.desktop import beat_alignment


def test_regular_detection_does_not_run_decoder():
    regular = {'beats': [0, .5, 1, 1.5, 2], 'downbeats': [0, 2],
               'beat_probabilities': [.9] * 100, 'downbeat_probabilities': [.1] * 100}
    def run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(regular), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='')
    with patch.object(beat_alignment, 'ready', return_value=True), \
         patch.object(beat_alignment.subprocess, 'run', side_effect=run) as worker:
        assert beat_alignment.detect('song.wav') == (regular['beats'], regular['downbeats'])
    assert worker.call_count == 1


def test_irregular_detection_is_refined_without_rerunning_the_model():
    calls = []
    irregular = {'beats': [0, .3, .42, .6, .9, 1.2], 'downbeats': [0, .9],
                 'beat_probabilities': [.9] * 100, 'downbeat_probabilities': [.1] * 100}
    regular = {'beats': [0, .6, 1.2, 1.8, 2.4, 3], 'downbeats': [0, 1.8]}

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[-1])
        output.write_text(json.dumps(irregular if len(calls) == 1 else regular), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='')

    with patch.object(beat_alignment, 'ready', return_value=True), \
         patch.object(beat_alignment.subprocess, 'run', side_effect=run):
        beats, downbeats = beat_alignment.detect('song.wav')
    assert (beats, downbeats) == (regular['beats'], regular['downbeats'])
    assert len(calls) == 2


def test_failed_decoder_retains_candidates_for_explicit_fallback():
    calls = []
    irregular = {'beats': [0, .3, .42, .6, .9], 'downbeats': [0],
                 'beat_probabilities': [.9] * 100, 'downbeat_probabilities': [.1] * 100}

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            Path(command[-1]).write_text(json.dumps(irregular), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout='')
        return SimpleNamespace(returncode=1, stdout='decoder failed')

    with patch.object(beat_alignment, 'ready', return_value=True), \
         patch.object(beat_alignment.subprocess, 'run', side_effect=run):
        beats, downbeats = beat_alignment.detect('song.wav')
    assert (beats, downbeats) == (irregular['beats'], irregular['downbeats'])

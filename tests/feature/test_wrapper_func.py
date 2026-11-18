import h5py
import numpy as np
from pathlib import Path
import pytest

from omnizart.feature import wrapper_func as wfunc


def test_get_frame_by_time():
    assert wfunc.get_frame_by_time(150) == 25840
    assert wfunc.get_frame_by_time(77, 16000) == 4812
    assert wfunc.get_frame_by_time(2, 16000, 128) == 250


def test_extract_patch_cqt(mocker):
    fixture = Path(__file__).resolve().parents[1] / 'resource' / 'sample_feature.hdf'
    if not fixture.is_file():
        pytest.skip('Optional regression fixture sample_feature.hdf is absent; download the upstream resource archive.')
    with h5py.File(fixture, "r") as fin:
        mini_beat_arr = fin["mini_beat_arr"][:]
        cqt = fin["cqt"][:]
        patch_cqt = fin["patch_cqt"][:]

    mocked_cqt_loader = mocker.patch("omnizart.feature.wrapper_func.cqt")
    mocked_cqt_loader.extract_cqt = mocker.MagicMock(return_value=cqt)

    mocked_b4d_loader = mocker.patch("omnizart.feature.wrapper_func.b4d")
    mocked_b4d_loader.extract_mini_beat_from_audio_path = mocker.MagicMock(return_value=mini_beat_arr)

    extracted, _ = wfunc.extract_patch_cqt("audio/path", sampling_rate=44100, hop_size=256)

    assert extracted.shape == (1100, 120, 120)
    assert np.all(np.abs(patch_cqt-extracted) < 0.01)

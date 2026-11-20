"""Tests that run without the camera: the package imports, the DLL is found (if built), and EGrabberSource has the
shape of b4d's BlockSource contract (checked against b4d itself when b4d is installed)."""

import inspect
import os

import egrabber_camera
import numpy as np
import pytest
from egrabber_camera import FRAME_DTYPE, EGrabberSource, SourceInfo, SourceStats, find_dll

CONTRACT_METHODS = ("start", "wait_block", "release", "stop", "stats", "close", "block_array")


def test_source_has_contract_methods_and_constructor_convention():
    for m in CONTRACT_METHODS:
        assert callable(getattr(EGrabberSource, m)), m
    params = inspect.signature(EGrabberSource.__init__).parameters
    assert "frames_per_block" in params and "nblocks" in params and "buffers" in params


def test_frame_dtype_matches_b4d_contract():
    b4d_ingest = pytest.importorskip("b4d.ingest")
    assert FRAME_DTYPE == b4d_ingest.FRAME_DTYPE
    # the local SourceInfo / SourceStats mirrors carry the same fields as b4d's
    assert set(SourceInfo.__dataclass_fields__) == set(b4d_ingest.SourceInfo.__dataclass_fields__)
    assert set(SourceStats.__dataclass_fields__) == set(b4d_ingest.SourceStats.__dataclass_fields__)


def test_dll_discovery_points_into_bin():
    try:
        p = find_dll()
    except Exception:
        pytest.skip("egrabber_ingest.dll not built (python egrabber/build.py)")
    assert os.path.basename(p) == "egrabber_ingest.dll"


def test_camera_model_table():
    conv, offset, rn = egrabber_camera.CAMERA_MODELS["VPS-245MX2"]
    assert conv == pytest.approx(1 / 7.25) and offset == 2.05 and rn == 6.89


@pytest.mark.skipif(
    not os.environ.get("EGRABBER_CAMERA_TESTS"), reason="set EGRABBER_CAMERA_TESTS=1 with the camera attached"
)
def test_live_camera_short_grab():
    src = EGrabberSource(frames_per_block=4, nblocks=2)
    b4d_ingest = pytest.importorskip("b4d.ingest")
    b4d_ingest.validate_source(src)
    src.start(max_frames=8)
    got = 0
    while True:
        try:
            blk = src.wait_block(3000)
        except StopIteration:
            break
        if blk is None:
            continue
        assert blk.array.dtype == np.uint16 and blk.array.shape[1:] == tuple(src.info.shape)
        got += blk.nframes
        src.release(blk)
    src.close()
    assert got == 8 and src.stats().lost == 0

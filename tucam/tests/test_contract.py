"""Contract shape: TucamSource has b4d's BlockSource members and constructor convention (checked against b4d itself
when b4d is installed), and the package carries no dependency on another framework."""

import inspect
import os
import re
import sys

import numpy as np
import pytest

import tucam_camera
from tucam_camera import FRAME_DTYPE, SourceInfo, SourceStats, TucamSource

CONTRACT_METHODS = ("start", "wait_block", "release", "stop", "stats", "close", "block_array")
PKG = os.path.dirname(tucam_camera.__file__)
ROOT = os.path.dirname(os.path.dirname(PKG))


def test_source_has_contract_methods_and_constructor_convention():
    for m in CONTRACT_METHODS:
        assert callable(getattr(TucamSource, m)), m
    params = inspect.signature(TucamSource.__init__).parameters
    assert "frames_per_block" in params and "nblocks" in params and "buffers" in params


def test_frame_dtype_fields():
    assert FRAME_DTYPE.names == ("frame_id", "timestamp_ns", "host_ns", "size_filled", "incomplete", "slot")


def _b4d_protocol():
    """b4d.ingest.protocol, loaded from its file (it needs only numpy), so the check runs even where b4d's compiled
    extension does not import; skips when b4d is not installed"""
    import importlib.util

    try:
        spec = importlib.util.find_spec("b4d")
    except ImportError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("b4d not installed")
    path = os.path.join(list(spec.submodule_search_locations)[0], "ingest", "protocol.py")
    if not os.path.isfile(path):
        pytest.skip("b4d.ingest.protocol not found")
    pspec = importlib.util.spec_from_file_location("_b4d_protocol", path)
    mod = importlib.util.module_from_spec(pspec)
    sys.modules[pspec.name] = mod  # dataclasses look the module up while the class body runs
    pspec.loader.exec_module(mod)
    return mod


def test_matches_b4d_contract_when_installed(make_source):
    b4d_ingest = _b4d_protocol()
    assert FRAME_DTYPE == b4d_ingest.FRAME_DTYPE
    assert set(SourceInfo.__dataclass_fields__) == set(b4d_ingest.SourceInfo.__dataclass_fields__)
    assert set(SourceStats.__dataclass_fields__) == set(b4d_ingest.SourceStats.__dataclass_fields__)
    src = make_source()
    b4d_ingest.validate_source(src)
    assert isinstance(src, b4d_ingest.BlockSource)


def test_info_after_construction(fake, make_source):
    src = make_source()
    i = src.info
    assert i.shape == (32, 48) and np.dtype(i.dtype) == np.uint16 and i.frame_bytes == 32 * 48 * 2
    assert i.serial == "FAKE0001" and i.name == "Tucsen FakeCam 2048"
    assert i.fps == pytest.approx(297.0)  # AcquisitionFrameRate set to its maximum
    assert i.extra["chunk_mode"] and i.extra["required_size"] == 32 * 48 * 2 + 128
    assert i.extra["frame_stride"] % 64 == 0 and i.extra["frame_stride"] >= i.extra["required_size"]
    assert i.extra["contiguous"] is False and i.extra["copy"] is False
    assert fake.features["TriggerMode"][2] == "FreeRunning"
    assert fake.features["ChunkModeActive"][2] is True


def test_packed_format_refused(fake):
    fake.features["PixelFormat"][2] = "Mono12Packed"
    with pytest.raises(ValueError, match="packed"):
        TucamSource(frames_per_block=4, nblocks=2)
    assert not fake.open  # the camera is closed again


def test_unknown_serial(fake):
    with pytest.raises(RuntimeError, match="not found"):
        TucamSource("NOPE", frames_per_block=4, nblocks=2)


def test_device_buffers_raise(fake):
    with pytest.raises(NotImplementedError, match="GPUDirect RDMA is not available through the TUCam SDK"):
        TucamSource(frames_per_block=4, nblocks=2, buffers_kind="device")


def test_no_framework_dependency_or_mentions():
    pat = re.compile("|".join(["vo" + "xel", "v" + "xl", "ri" + "gup"]), re.I)  # split: this file must not match
    for d, _, files in os.walk(ROOT):
        if "__pycache__" in d or ".pytest_cache" in d or "egg-info" in d:
            continue
        for f in files:
            if f.endswith((".py", ".md", ".toml", ".cfg", ".txt")) or f in ("LICENSE", ".flake8"):
                text = open(os.path.join(d, f), encoding="utf-8").read()
                assert not pat.search(text), os.path.join(d, f)

"""Contract shape: DcamSource against b4d's BlockSource definition (loaded from the sibling b4d checkout's protocol.py
by file path when present, so the test does not need an importable b4d), the binding's struct sizes, constructor
checks."""

import ctypes
import importlib.util
import inspect
import os
import re
import sys

import numpy as np
import pytest

import dcam_camera
from dcam_camera import FRAME_DTYPE, DcamSource, SourceInfo, SourceStats, dcamapi

PROTOCOL = os.path.join(os.path.dirname(__file__), "..", "..", "b4d", "src", "b4d", "ingest", "protocol.py")
CONTRACT_METHODS = ("start", "wait_block", "release", "stop", "stats", "close", "block_array")


def _protocol():
    if not os.path.exists(PROTOCOL):
        pytest.skip("b4d/src/b4d/ingest/protocol.py not found next to this package")
    name = "b4d_protocol_for_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PROTOCOL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses look the module up while the class is built
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_source_has_contract_methods_and_constructor_convention():
    for m in CONTRACT_METHODS:
        assert callable(getattr(DcamSource, m)), m
    params = inspect.signature(DcamSource.__init__).parameters
    assert "frames_per_block" in params and "nblocks" in params and "buffers" in params


def test_frame_dtype_and_mirrors_match_b4d_contract():
    proto = _protocol()
    assert FRAME_DTYPE == proto.FRAME_DTYPE
    assert set(SourceInfo.__dataclass_fields__) == set(proto.SourceInfo.__dataclass_fields__)
    assert set(SourceStats.__dataclass_fields__) == set(proto.SourceStats.__dataclass_fields__)
    assert set(proto.FrameBlock.__slots__) <= set(dcam_camera.DcamBlock.__slots__)


def test_validate_source_accepts_dcam_source(make_source):
    proto = _protocol()
    src = make_source()
    info = proto.validate_source(src)
    assert isinstance(src, proto.BlockSource)
    assert info.shape == (32, 64) and np.dtype(info.dtype) == np.uint16
    assert info.frame_bytes == 32 * 64 * 2
    assert info.camera_model is None
    assert info.extra["zero_copy"] is True and info.extra["copy"] is False
    assert info.serial == "100001"


def test_package_does_not_import_b4d():
    root = os.path.dirname(dcam_camera.__file__)
    for name in os.listdir(root):
        if name.endswith(".py"):
            text = open(os.path.join(root, name), encoding="utf-8").read()
            assert not re.search(r"^\s*(import|from)\s+b4d", text, re.M), name


@pytest.mark.parametrize(
    ("struct", "size"),
    [
        # sizes from dcamapi4.h (#pragma pack(8), 64-bit), as checked by the earlier driver's binding tests
        (dcamapi.DCAMAPI_INIT, 32),
        (dcamapi.DCAMDEV_OPEN, 16),
        (dcamapi.DCAMDEV_STRING, 24),
        (dcamapi.DCAMPROP_ATTR, 88),
        (dcamapi.DCAMPROP_VALUETEXT, 32),
        (dcamapi.DCAMBUF_ATTACH, 24),
        (dcamapi.DCAMCAP_TRANSFERINFO, 16),
        (dcamapi.DCAMBUF_FRAME, 64),
        (dcamapi.DCAMWAIT_OPEN, 24),
        (dcamapi.DCAMWAIT_START, 16),
    ],
)
def test_struct_sizes(struct, size):
    assert ctypes.sizeof(struct) == size
    assert getattr(struct(), struct._fields_[0][0]) == size


def test_constructor_checks(make_source):
    with pytest.raises(ValueError, match="nblocks must be >= 2"):
        make_source(nblocks=1)
    with pytest.raises(ValueError, match="MONO12"):
        make_source(pixel_type="MONO12")
    with pytest.raises(ValueError, match="buffers has 1 entries"):
        make_source(buffers=[np.zeros((4, 32, 64), np.uint16)])


def test_serial_not_found(fake_dcam):
    with pytest.raises(RuntimeError, match="not found"):
        DcamSource("999999", frames_per_block=4, nblocks=2)


def test_fps_not_writable_in_area_mode(make_source):
    with pytest.raises(ValueError, match="INTERNAL FRAME RATE"):
        make_source(fps=10.0)

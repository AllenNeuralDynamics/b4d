"""Fixtures: the DCAM-API runtime replaced by dcam_fake.FakeDcamLib (one ORCA-Flash4.0 V3 model, S/N 100001)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from dcam_fake import FakeDcamLib  # noqa: E402

from dcam_camera import dcam, dcamapi  # noqa: E402


@pytest.fixture
def fake_dcam(monkeypatch):
    lib = FakeDcamLib()
    monkeypatch.setattr(dcamapi, "load_library", lambda _path=None: lib)
    monkeypatch.setattr(dcam, "_sdk", None)
    return lib


@pytest.fixture
def make_source(fake_dcam):
    """DcamSource on the fake with a small ROI (64 x 32 px) so frames are cheap; closes it after the test."""
    from dcam_camera import DcamSource

    made = []

    def make(**kw):
        kw.setdefault("serial", "100001")
        kw.setdefault("roi", (0, 0, 64, 32))
        kw.setdefault("frames_per_block", 4)
        kw.setdefault("nblocks", 3)
        src = DcamSource(**kw)
        made.append(src)
        return src

    yield make
    for s in made:
        s.close()

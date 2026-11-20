import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import fake_tucam  # noqa: E402

import tucam_camera  # noqa: E402


@pytest.fixture
def fake():
    """a fresh fake SDK installed as the loaded binding; returns its api (the fake camera)"""
    api = fake_tucam.FakeApi()
    tucam_camera.install(api, fake_tucam.D, root="fake-sdk")
    return api


@pytest.fixture
def make_source(fake):
    made = []

    def make(**kw):
        kw.setdefault("frames_per_block", 4)
        kw.setdefault("nblocks", 2)
        src = tucam_camera.TucamSource(**kw)
        made.append(src)
        return src

    yield make
    for s in made:
        s.close()

"""Tests that run without a camera: PvcamSource against a fake PVCAM runtime (tests/fake_pvcam.py) that writes frames
into the source's circular buffer with ctypes.memmove.

A live test runs with PVCAM_CAMERA_TESTS=1 and a camera attached.
"""

import importlib.util
import inspect
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import pvcam_camera  # noqa: E402
from fake_pvcam import FakePvcam  # noqa: E402
from pvcam_camera import FRAME_DTYPE, PvcamSource, SourceInfo, SourceStats  # noqa: E402
from pvcam_camera import pvcam as pv  # noqa: E402

CONTRACT_METHODS = ("start", "wait_block", "release", "stop", "stats", "close", "block_array")
SHAPE = (24, 32)  # (H, W)


def b4d_contract():
    """b4d.ingest if it imports; else b4d's protocol.py loaded by path from the sibling source tree (numpy only)"""
    try:
        import b4d.ingest

        return b4d.ingest
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(__file__), "..", "..", "b4d", "src", "b4d", "ingest", "protocol.py")
    if not os.path.isfile(path):
        pytest.skip("b4d not importable and b4d/src/b4d/ingest/protocol.py not found")
    spec = importlib.util.spec_from_file_location("b4d_protocol", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b4d_protocol"] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


def make(fake=None, **kw):
    fake = fake or FakePvcam(shape=SHAPE)
    kw.setdefault("frames_per_block", 4)
    kw.setdefault("nblocks", 2)
    kw.setdefault("poll_interval_s", 0.0005)
    return fake, PvcamSource(runtime=fake, **kw)


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("condition not met in time")
        time.sleep(0.002)


def next_block(src, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        blk = src.wait_block(100)
        if blk is not None:
            return blk
    raise AssertionError("no block in time")


def drain(src):
    out = []
    while True:
        try:
            blk = src.wait_block(200)
        except StopIteration:
            return out
        if blk is not None:
            out.append((blk.nframes, blk.partial, blk.frames.copy(), blk.array.copy()))
            src.release(blk)


# ---------------------------------------------------------------- contract shape
def test_source_has_contract_methods_and_constructor_convention():
    for m in CONTRACT_METHODS:
        assert callable(getattr(PvcamSource, m)), m
    params = inspect.signature(PvcamSource.__init__).parameters
    assert "frames_per_block" in params and "nblocks" in params and "buffers" in params
    assert "buffers_kind" in params


def test_frame_dtype_and_mirrors_match_b4d_contract():
    b4d_ingest = b4d_contract()
    assert FRAME_DTYPE == b4d_ingest.FRAME_DTYPE
    assert set(SourceInfo.__dataclass_fields__) == set(b4d_ingest.SourceInfo.__dataclass_fields__)
    assert set(SourceStats.__dataclass_fields__) == set(b4d_ingest.SourceStats.__dataclass_fields__)


def test_validate_source_against_b4d():
    b4d_ingest = b4d_contract()
    fake, src = make()
    try:
        info = b4d_ingest.validate_source(src)
        assert tuple(info.shape) == SHAPE
    finally:
        src.close()


def test_info_and_setup():
    fake, src = make()
    try:
        cam = fake.cams[src.hcam]
        assert src.info.shape == SHAPE and np.dtype(src.info.dtype) == np.uint16
        assert src.info.frame_bytes == SHAPE[0] * SHAPE[1] * 2  # metadata switched off: slot == H*W*bytes
        assert cam.params[pv.PARAM_METADATA_ENABLED] is False
        regions, exp_mode, exposure, buffer_mode = cam.setup
        assert buffer_mode == pv.CIRC_NO_OVERWRITE
        assert exp_mode == pv.EXT_TRIG_INTERNAL
        assert src.info.extra["exp_mode"] == pv.EXT_TRIG_INTERNAL and src.info.extra["exposure_us"] == 1.0
        assert src.info.fps == pytest.approx(1e6 / 11300)  # readout time 11.3 ms dominates
        assert src.info.serial == "A00F000001" and src.info.camera_model is None  # 12-bit port: no model
    finally:
        src.close()
    assert fake.refs == 0 and not fake.opened


def test_open_by_serial():
    fake, src = make(serial="A00F000002")
    try:
        assert src.info.serial == "A00F000002" and src.info.extra["camera_name"] == "PMPCIECam01"
        assert list(fake.opened) == [1]  # the non-matching camera was closed again
    finally:
        src.close()
    with pytest.raises(LookupError):
        make(serial="nope")


def test_metadata_that_cannot_be_switched_off_is_rejected():
    fake = FakePvcam(shape=SHAPE, metadata_writable=False)
    with pytest.raises(ValueError, match="metadata"):
        make(fake)
    assert fake.refs == 0 and not fake.opened


def test_kinetix_8bit_speed_port_camera_model():
    fake, src = make(readout_port=1)
    try:
        assert np.dtype(src.info.dtype) == np.uint8 and src.info.frame_bytes == SHAPE[0] * SHAPE[1]
        assert src.info.camera_model == (0.851, 6.0, 2.35)
        assert src.info.extra["readout_port_name"] == "Speed"
    finally:
        src.close()


def test_ring_larger_than_uns32_is_rejected():
    fake = FakePvcam(shape=(3200, 3200))
    with pytest.raises(ValueError, match="uns32"):
        PvcamSource(runtime=fake, frames_per_block=64, nblocks=4)  # 256 x 20.48 MB > 4 GiB
    assert fake.refs == 0


def test_device_buffers_raise():
    with pytest.raises(NotImplementedError, match="GPUDirect RDMA is not available through the PVCAM API"):
        PvcamSource(runtime=FakePvcam(shape=SHAPE), buffers_kind="device")


# ---------------------------------------------------------------- zero copy
def test_zero_copy_blocks_are_views_of_the_circular_buffer():
    fake, src = make()
    try:
        fake.allow(8)
        src.start()
        ring = src.ring
        assert fake.start_address == ring.ctypes.data and fake.start_size == ring.nbytes
        assert ring.ctypes.data % 4096 == 0 and src.info.extra["ring_page_aligned"]
        fb = src.info.frame_bytes
        for k in range(2):
            blk = next_block(src)
            assert blk.index == k and blk.seq == k
            assert np.shares_memory(blk.array, ring)
            assert blk.array.ctypes.data == ring.ctypes.data + k * 4 * fb
            assert blk.full_array.ctypes.data == src.block_array(k).ctypes.data
            for i in range(4):  # the fake wrote frame k*4+i+1 straight into this memory
                assert (blk.array[i] == k * 4 + i + 1).all()
            src.release(blk)
    finally:
        src.close()


def test_caller_buffer_single_array():
    h, w = SHAPE
    mem = np.zeros((8, h, w), np.uint16)
    fake, src = make(buffers=[mem])
    try:
        fake.allow(4)
        src.start(max_frames=4)
        blk = next_block(src)
        assert np.shares_memory(blk.array, mem) and fake.start_address == mem.ctypes.data
        assert (mem[:4, 0, 0] == [1, 2, 3, 4]).all()
        src.release(blk)
    finally:
        src.close()


def test_caller_buffers_per_block_must_be_one_allocation():
    h, w = SHAPE
    mem = np.zeros((8, h, w), np.uint16)
    fake, src = make(buffers=[mem[:4], mem[4:]])  # adjacent pieces of one allocation: accepted
    src.close()
    with pytest.raises(ValueError, match="one circular buffer"):
        make(buffers=[np.zeros((4, h, w), np.uint16), np.zeros((4, h, w), np.uint16)])


# ---------------------------------------------------------------- metadata, partial block
def test_metadata_rows_and_partial_final_block():
    fake, src = make()
    try:
        fake.allow(8)  # fill the ring; more frames before a release would be dropped (lost)
        src.start(max_frames=11)  # 2 full blocks of 4 + a partial block of 3
        b0 = next_block(src)
        got = [(b0.nframes, b0.partial, b0.frames.copy(), b0.array.copy())]
        src.release(b0)
        wait_for(lambda: fake.unlock_calls == 4)
        fake.allow(100)  # the source stops the camera after frame 11
        got += drain(src)
    finally:
        src.close()
    assert [(n, p) for n, p, _, _ in got] == [(4, False), (4, False), (3, True)]
    frames = np.concatenate([f for _, _, f, _ in got])
    assert frames.dtype == FRAME_DTYPE
    assert list(frames["frame_id"]) == list(range(1, 12))
    assert list(frames["timestamp_ns"]) == [k * 113 * 100_000 for k in range(1, 12)]  # 0.1 ms counts -> ns
    assert list(frames["slot"]) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert (frames["size_filled"] == SHAPE[0] * SHAPE[1] * 2).all() and (frames["incomplete"] == 0).all()
    assert (np.diff(frames["host_ns"].astype(np.int64)) >= 0).all()
    assert got[2][3].shape == (3,) + SHAPE and (got[2][3][:, 0, 0] == [9, 10, 11]).all()
    s = src.stats()
    assert s.delivered == 11 and s.lost == 0 and s.blocks_completed == 3 and s.blocks_released == 3
    assert s.fps_measured == pytest.approx(1 / 0.0113)


def test_frame_info_kept_per_block():
    fake, src = make()
    try:
        fake.allow(4)
        src.start(max_frames=4)
        blk = next_block(src)
        assert list(blk.frame_info["frame_nr"]) == [1, 2, 3, 4]
        assert list(blk.frame_info["timestamp_bof"]) == [k * 113 - 113 for k in range(1, 5)]
        src.release(blk)
    finally:
        src.close()


# ---------------------------------------------------------------- back-pressure and losses
def test_held_blocks_back_pressure_and_loss_counting():
    fake, src = make()
    try:
        fake.allow(8)
        src.start()
        held = [next_block(src), next_block(src)]  # the whole ring is held by the consumer
        assert fake.unlock_calls == 0  # frames stay locked while their block is held
        fake.allow(5)
        wait_for(lambda: fake.produced == 13)
        assert fake.dropped == 5 and fake.overwritten_locked == 0
        for k, blk in enumerate(held):  # no-overwrite honoured: held memory intact
            assert (blk.array[:, 0, 0] == np.arange(1, 5) + 4 * k).all()
        for blk in held:
            src.release(blk)
        wait_for(lambda: fake.unlock_calls == 8)
        fake.allow(4)
        blk = next_block(src)
        assert list(blk.frames["frame_id"]) == [14, 15, 16, 17]  # the gap 9..13 is the loss
        src.release(blk)
        s = src.stats()
        assert s.lost == 5 and s.incomplete == 0 and s.delivered == 12
    finally:
        src.close()


def test_release_out_of_order_unlocks_in_fifo_order():
    fake, src = make(nblocks=3)
    try:
        fake.allow(8)
        src.start()
        b0, b1 = next_block(src), next_block(src)
        src.release(b1)
        time.sleep(0.05)
        assert fake.unlock_calls == 0  # block 1 waits for block 0: PVCAM unlocks the oldest frame
        src.release(b0)
        wait_for(lambda: fake.unlock_calls == 8)
    finally:
        src.close()


def test_overwrite_of_held_block_is_detected():
    fake = FakePvcam(shape=SHAPE, overwrite_locked=True)
    fake, src = make(fake)
    try:
        fake.allow(8)
        src.start()
        held = [next_block(src), next_block(src)]
        fake.allow(1)  # frame 9 lands in slot 0, inside held block 0
        wait_for(lambda: src.stats().extra["error"] != "")
        s = src.stats()
        assert "overwritten" in s.extra["error"] and s.incomplete >= 1 and s.extra["overwritten_held_frames"] == 1
        for blk in held:
            src.release(blk)
        with pytest.raises(StopIteration):
            while True:
                src.wait_block(200)
    finally:
        src.close()


def test_one_locked_frame_runtime_falls_back_to_unlock_on_retrieve():
    fake = FakePvcam(shape=SHAPE, require_unlock=True)
    fake, src = make(fake)
    try:
        fake.allow(8)
        src.start(max_frames=8)
        got = drain(src)
    finally:
        src.close()
    assert [n for n, _, _, _ in got] == [4, 4]
    assert list(np.concatenate([f for _, _, f, _ in got])["frame_id"]) == list(range(1, 9))
    s = src.stats()
    assert s.extra["lock_mode"] == "unlock_on_retrieve" and s.lost == 0 and fake.unlock_calls == 8


def test_stop_delivers_partial_block_and_stopiteration():
    fake, src = make()
    try:
        fake.allow(6)
        src.start()
        blk = next_block(src)
        src.release(blk)
        wait_for(lambda: src.stats().delivered == 6)
        src.stop()
        got = drain(src)
        assert [(n, p) for n, p, _, _ in got] == [(2, True)]
        assert not fake.running
    finally:
        src.close()
    assert fake.refs == 0


def test_package_exports():
    assert pvcam_camera.GPU_STATUS.startswith("GPUDirect RDMA is not available through the PVCAM API")
    assert pvcam_camera.CAMERA_MODELS[("kinetix", 1, 8)] == (0.851, 6.0, 2.35)


@pytest.mark.skipif(not os.environ.get("PVCAM_CAMERA_TESTS"), reason="set PVCAM_CAMERA_TESTS=1 with a camera attached")
def test_live_camera_short_grab():
    src = PvcamSource(frames_per_block=4, nblocks=3)
    try:
        src.start(max_frames=10)
        got = drain(src)
    finally:
        src.close()
    assert sum(n for n, _, _, _ in got) == 10 and src.stats().lost == 0

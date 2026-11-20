"""DcamSource on the fake DCAM runtime: zero copy, metadata, overrun detection, missed frames, partial final block,
caller buffers, device buffers, MONO12 copy path, lockframe fallback.

Most tests set ``fake.frames_per_wait = 0`` and play the camera with ``fake.produce(n, skip=...)``, then call
``wait_block(0)``, so the frame count at every step is known exactly.
"""

import numpy as np
import pytest
from dcam_fake import E, pattern

from dcam_camera import GPU_STATUS, DcamSource, aligned_empty

H, W = 32, 64


def manual(src, fake, max_frames=None):
    src.start(max_frames=max_frames)
    fake.frames_per_wait = 0
    return src


def drain(src, timeout_ms=1000):
    out = []
    while True:
        try:
            blk = src.wait_block(timeout_ms)
        except StopIteration:
            return out
        if blk is None:
            continue
        out.append((blk.nframes, blk.partial, blk.frames.copy(), blk.array.copy()))
        src.release(blk)


# ---------------------------------------------------------------- zero copy
def test_block_is_the_attached_memory(make_source, fake_dcam):
    src = manual(make_source(nblocks=3, frames_per_block=4), fake_dcam)
    assert fake_dcam.attached is not None and len(fake_dcam.attached) == 12
    fake_dcam.produce(4)
    blk = src.wait_block(0)
    assert blk is not None and blk.index == 0 and blk.nframes == 4 and not blk.partial
    pool = src.block_array(0)
    assert np.shares_memory(blk.array, pool)
    assert pool.ctypes.data % 4096 == 0  # page-aligned block
    for s in range(4):  # DCAM's pointer for frame s is frame s of the returned array
        assert blk.array[s].ctypes.data == fake_dcam.attached[s]
        np.testing.assert_array_equal(blk.array[s], pattern(s, H, W))
    # every pool block is attached frame by frame, block after block
    for b in range(3):
        for s in range(4):
            assert fake_dcam.attached[b * 4 + s] == src.block_array(b)[s].ctypes.data
    src.release(blk)
    assert src.stats().extra["zero_copy"] is True and src.stats().extra["copy"] is False


def test_aligned_empty():
    a = aligned_empty(10_000)
    assert a.nbytes == 10_000 and a.ctypes.data % 4096 == 0


# ---------------------------------------------------------------- metadata
def test_frame_metadata_rows(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(4)
    blk = src.wait_block(0)
    f = blk.frames
    assert f.dtype.names == ("frame_id", "timestamp_ns", "host_ns", "size_filled", "incomplete", "slot")
    np.testing.assert_array_equal(f["frame_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(f["timestamp_ns"], [(1000 + k) * 1_000_000_000 for k in range(4)])
    assert (f["host_ns"] > 0).all() and (np.diff(f["host_ns"].astype(np.int64)) >= 0).all()
    np.testing.assert_array_equal(f["size_filled"], [H * W * 2] * 4)
    np.testing.assert_array_equal(f["incomplete"], [0] * 4)
    np.testing.assert_array_equal(f["slot"], [0, 1, 2, 3])
    src.release(blk)
    fake_dcam.produce(4)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [4, 5, 6, 7])
    assert blk.seq == 1 and blk.gen == 1 and blk.index == 1
    src.release(blk)
    s = src.stats()
    assert s.delivered == 8 and s.lost == 0 and s.blocks_completed == 2 and s.blocks_released == 2
    assert s.fps_measured == pytest.approx(1.0)  # the fake's timestamps advance 1 s per frame


def test_incomplete_block_is_not_delivered(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(3)
    assert src.wait_block(0) is None
    assert src.stats().ready_blocks == 0
    fake_dcam.produce(1)
    assert src.stats().ready_blocks == 1
    assert src.wait_block(0) is not None


# ---------------------------------------------------------------- overrun (DCAM has no back-pressure)
def test_overrun_of_a_held_block_is_detected(make_source, fake_dcam):
    src = manual(make_source(nblocks=2, frames_per_block=4), fake_dcam)
    fake_dcam.produce(4)
    blk0 = src.wait_block(0)
    assert src.overwritten(blk0) == 0
    fake_dcam.produce(3)  # count 7: frame 7 (block 1) in flight, block 0 intact
    assert src.overwritten(blk0) == 0
    fake_dcam.produce(1)  # count 8: frame 8 is being written into block 0 slot 0
    assert src.overwritten(blk0) == 1
    blk1 = src.wait_block(0)  # block 1 (frames 4..7) is intact and is delivered
    assert blk1.gen == 1 and blk1.index == 1
    np.testing.assert_array_equal(blk1.array[0], pattern(4, H, W))
    fake_dcam.produce(2)  # count 10: frames 8, 9 written, 10 in flight -> 3 slots of block 0 gone
    np.testing.assert_array_equal(blk0.array[0], pattern(8, H, W))  # the memory really changed under the consumer
    assert src.overwritten(blk0) == 3
    src.release(blk0)
    s = src.stats()
    assert s.lost == 3 and s.delivered == 8 - 3
    assert s.extra["overrun"] is True and s.extra["overrun_seqs"] == [0] and s.extra["lost_overwritten"] == 3
    src.release(blk1)


def test_blocks_overwritten_before_delivery_are_lost_not_returned(make_source, fake_dcam):
    src = manual(make_source(nblocks=2, frames_per_block=4), fake_dcam)
    fake_dcam.produce(12)  # frames 0..11 captured, nothing consumed; frame 12 in flight into block 1
    blk = src.wait_block(0)
    assert blk.gen == 2 and blk.seq == 0 and blk.index == 0
    np.testing.assert_array_equal(blk.frames["frame_id"], [8, 9, 10, 11])
    for s in range(4):
        np.testing.assert_array_equal(blk.array[s], pattern(8 + s, H, W))
    src.release(blk)
    st = src.stats()
    assert st.lost == 8 and st.extra["lost_blocks_before_delivery"] == 2 and st.delivered == 4


def test_held_pool_block_blocks_delivery_of_its_next_use(make_source, fake_dcam):
    src = manual(make_source(nblocks=2, frames_per_block=2), fake_dcam)
    fake_dcam.produce(2)
    b0 = src.wait_block(0)
    fake_dcam.produce(2)
    b1 = src.wait_block(0)
    src.release(b1)
    fake_dcam.produce(2)  # gen 2 (pool block 0) is complete while the consumer still holds gen 0
    assert src.wait_block(0) is None
    assert src.stats().free_blocks == 1
    src.release(b0)
    assert src.stats().extra["overrun"] is True


# ---------------------------------------------------------------- missed frames (framestamp gaps)
def test_missed_frames_from_framestamp_gaps_with_startup_offset(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(1, skip=14)  # camera ran 14 frames before DCAM received one: not a loss
    fake_dcam.produce(3)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [14, 15, 16, 17])
    src.release(blk)
    assert src.stats().lost == 0 and src.stats().extra["stamp0"] == 14
    fake_dcam.produce(2)
    fake_dcam.produce(2, skip=3)  # 3 frames lost before DCAM
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [18, 19, 23, 24])
    src.release(blk)
    s = src.stats()
    assert s.lost == 3 and s.extra["missed"] == 3 and s.extra["lost_overwritten"] == 0


# stamps="attach" (metadata arrays attached next to the frames) is the default; the
# real ORCA-Flash4.0 V3 refuses lockframe with attached frames but fills these arrays.
def test_attached_metadata_gives_camera_stamps_without_lockframe(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4), fake_dcam)
    assert set(fake_dcam.meta) == {3, 4} and all(n == 12 for _, n in fake_dcam.meta.values())
    fake_dcam.lockframe_attach_error = E.NOTREADY  # as on the real camera: must not matter
    fake_dcam.produce(4, skip=5)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [5, 6, 7, 8])
    np.testing.assert_array_equal(blk.frames["timestamp_ns"], [(1000 + k) * 1_000_000_000 for k in (5, 6, 7, 8)])
    src.release(blk)
    fake_dcam.produce(2)
    fake_dcam.produce(2, skip=2)  # 2 frames lost before DCAM: seen in the attached framestamps
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [9, 10, 13, 14])
    src.release(blk)
    s = src.stats()
    assert fake_dcam.lockframe_calls == 0
    assert s.extra["stamps"] == "attach" and s.extra["stamps_error"] is None and s.extra["stamps_unwritten"] == 0
    assert s.extra["stamp0"] == 5 and s.extra["missed"] == 2 and s.lost == 2
    src.stop()
    drain(src)
    assert fake_dcam.meta == {} and fake_dcam.attached is None  # metadata and frames released after the drain


def test_attached_metadata_wraps_with_the_pool(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4, nblocks=2), fake_dcam)
    ids = []
    for _ in range(5):  # 20 frames through an 8-frame pool: the metadata arrays wrap like the frames
        fake_dcam.produce(4)
        blk = src.wait_block(0)
        ids.extend(blk.frames["frame_id"].tolist())
        np.testing.assert_array_equal(blk.array[0], pattern(int(blk.frames["frame_id"][0]), H, W))
        src.release(blk)
    assert ids == list(range(20)) and src.stats().lost == 0


def test_metadata_attach_refused_falls_back_to_lockframe(make_source, fake_dcam):
    fake_dcam.meta_attach_error = E.NOTSUPPORT
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(4, skip=5)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [5, 6, 7, 8])
    src.release(blk)
    s = src.stats()
    assert s.extra["stamps"] == "lockframe" and "metadata attach refused" in s.extra["stamps_error"]
    assert fake_dcam.lockframe_calls == 4


def test_metadata_and_lockframe_refused_fall_back_to_host_time(make_source, fake_dcam):
    fake_dcam.meta_attach_error = E.NOTSUPPORT
    fake_dcam.lockframe_attach_error = E.NOTREADY
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(4, skip=5)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(blk.frames["timestamp_ns"], blk.frames["host_ns"])
    src.release(blk)
    s = src.stats()
    assert s.extra["stamps"] == "none"
    assert "metadata attach refused" in s.extra["stamps_error"] and "NOTREADY" in s.extra["stamps_error"]


def test_lockframe_refused_falls_back_to_dcam_index(make_source, fake_dcam):
    fake_dcam.lockframe_attach_error = E.NOTSUPPORT
    src = manual(make_source(frames_per_block=4, stamps="lockframe"), fake_dcam)
    assert fake_dcam.meta == {}
    fake_dcam.produce(4, skip=5)
    blk = src.wait_block(0)
    np.testing.assert_array_equal(blk.frames["frame_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(blk.frames["timestamp_ns"], blk.frames["host_ns"])
    np.testing.assert_array_equal(blk.array[0], pattern(5, H, W))  # the data is still zero copy
    src.release(blk)
    s = src.stats()
    assert s.extra["stamps"] == "none" and "NOTSUPPORT" in s.extra["stamps_error"]


# ---------------------------------------------------------------- final partial block
def test_max_frames_delivers_final_partial_block(make_source, fake_dcam):
    src = make_source(frames_per_block=4)
    src.start(max_frames=10)  # the fake produces one frame per wait
    got = drain(src)
    assert [(n, p) for n, p, _, _ in got] == [(4, False), (4, False), (2, True)]
    ids = np.concatenate([f["frame_id"] for _, _, f, _ in got])
    np.testing.assert_array_equal(ids, np.arange(10))
    np.testing.assert_array_equal(got[2][3][1], pattern(9, H, W))
    s = src.stats()
    assert s.delivered == 10 and s.lost == 0 and s.extra["captured"] == 10
    assert fake_dcam.attached is None  # released after the drain


def test_stop_delivers_final_partial_block(make_source, fake_dcam):
    src = manual(make_source(frames_per_block=4), fake_dcam)
    fake_dcam.produce(6)
    src.stop()
    got = drain(src, timeout_ms=0)
    assert [(n, p) for n, p, _, _ in got] == [(4, False), (2, True)]
    with pytest.raises(StopIteration):
        src.wait_block(0)


def test_restart_after_drain(make_source, fake_dcam):
    src = make_source(frames_per_block=2)
    src.start(max_frames=4)
    assert sum(n for n, *_ in drain(src)) == 4
    src.start(max_frames=3)
    got = drain(src)
    assert [(n, p) for n, p, _, _ in got] == [(2, False), (1, True)]
    assert src.stats().delivered == 3


# ---------------------------------------------------------------- caller buffers, device buffers
def test_caller_buffers_receive_the_frames(make_source, fake_dcam):
    bufs = [np.zeros((5, H, W), np.uint16) for _ in range(3)]  # >= frames_per_block frames each
    src = manual(make_source(nblocks=3, frames_per_block=4, buffers=bufs), fake_dcam)
    for b in range(3):
        for s in range(4):
            assert fake_dcam.attached[b * 4 + s] == bufs[b][s].ctypes.data
    fake_dcam.produce(8)
    blk = src.wait_block(0)
    assert np.shares_memory(blk.array, bufs[0])
    np.testing.assert_array_equal(bufs[0][3], pattern(3, H, W))
    src.release(blk)
    blk = src.wait_block(0)
    assert np.shares_memory(blk.array, bufs[1])
    src.release(blk)
    assert (bufs[0][4] == 0).all()  # the extra frame of a caller buffer is not used


def test_caller_buffers_are_checked(make_source):
    with pytest.raises(ValueError, match="must be uint16"):
        make_source(nblocks=2, buffers=[np.zeros((4, H, W), np.uint8)] * 2)
    with pytest.raises(ValueError, match="must be uint16"):
        make_source(nblocks=2, frames_per_block=4, buffers=[np.zeros((3, H, W), np.uint16)] * 2)
    with pytest.raises(ValueError, match="C-contiguous"):
        make_source(nblocks=2, frames_per_block=4, buffers=[np.zeros((4, H, 2 * W), np.uint16)[:, :, ::2]] * 2)


def test_device_buffers_raise(fake_dcam):
    with pytest.raises(NotImplementedError, match="GPUDirect RDMA is not available through the DCAM-API"):
        DcamSource("100001", frames_per_block=4, nblocks=2, buffers=[object(), object()], buffers_kind="device")
    assert "ATTACHKIND_FRAME" in GPU_STATUS


# ---------------------------------------------------------------- pixel types
def test_mono8_is_zero_copy(make_source, fake_dcam):
    src = manual(make_source(pixel_type="MONO8"), fake_dcam)
    assert np.dtype(src.info.dtype) == np.uint8 and src.info.frame_bytes == H * W
    fake_dcam.produce(4)
    blk = src.wait_block(0)
    assert blk.array.dtype == np.uint8 and blk.array[0].ctypes.data == fake_dcam.attached[0]
    np.testing.assert_array_equal(blk.array[2], pattern(2, H, W) & 0xFF)
    src.release(blk)


def test_mono12_copy_path(make_source, fake_dcam):
    src = manual(make_source(pixel_type="MONO12", mono12="copy"), fake_dcam)
    assert src.info.extra["copy"] is True and src.info.extra["zero_copy"] is False
    assert np.dtype(src.info.dtype) == np.uint16 and src.info.frame_bytes == H * W * 2
    assert src.info.extra["buffer_rowbytes"] == W * 3 // 2
    fake_dcam.produce(4)
    blk = src.wait_block(0)
    for s in range(4):
        np.testing.assert_array_equal(blk.array[s], pattern(s, H, W))
        assert blk.array[s].ctypes.data != fake_dcam.attached[s]  # unpacked into the pool, not the DCAM frame
    assert src.overwritten(blk) == 0
    src.release(blk)
    assert src.stats().extra["copy"] is True

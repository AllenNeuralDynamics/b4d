"""TucamSource against the fake SDK: zero copy into the announced slots, back-pressure loss counting, per-frame
metadata, the final partial block, caller-owned buffers, and announce/revoke bookkeeping."""

import numpy as np
import pytest

from tucam_camera import TucamSource

FPB, NB = 4, 2


def take(src, timeout_ms=10):
    blk = src.wait_block(timeout_ms)
    assert blk is not None
    return blk


def test_zero_copy_into_announced_slots(fake, make_source):
    src = make_source()
    src.start()
    assert len(fake.announced) == FPB * NB and fake.policy == 1  # one SDK buffer per frame slot, MANUAL queue
    stride = src.info.extra["frame_stride"]
    addrs = sorted(a for a, _ in fake.announced.values())
    fake.produce(FPB)
    blk = take(src)
    a = blk.array
    assert a.shape == (FPB, 32, 48) and a.dtype == np.uint16
    assert a.strides == (stride, 48 * 2, 2) and not a.flags.c_contiguous  # chunk data: frames not contiguous
    full = src.block_array(blk.index)
    assert np.shares_memory(a, full)
    for j in range(FPB):  # each frame's pixels sit at the address the SDK was given for that slot
        assert a[j].ctypes.data in addrs
        assert a[j].ctypes.data == full.ctypes.data + j * stride
        assert a[j].flags.c_contiguous
        np.testing.assert_array_equal(a[j].ravel(), fake.pattern(j))
    # the chunk trailer follows the pixels inside the stride and is not part of the view
    tail = np.frombuffer((np.ctypeslib.ctypes.c_uint8 * 128).from_address(a[0].ctypes.data + a[0].nbytes), np.uint8)
    assert (tail == 0xCC).all()
    # memory written after delivery shows through the view: no copy was made
    a[0, 0, 0] = 4095
    assert full[0, 0, 0] == 4095
    src.release(blk)
    assert src.stats().delivered == FPB and src.stats().lost == 0


def test_chunk_mode_off_gives_contiguous_blocks(fake, make_source):
    src = make_source(chunk_mode=False)
    assert src.info.extra["contiguous"] and not src.info.extra["loss_counting"]
    src.start()
    fake.produce(FPB)
    blk = take(src)
    assert blk.array.flags.c_contiguous and blk.array.strides[0] == 32 * 48 * 2
    np.testing.assert_array_equal(blk.array[2].ravel(), fake.pattern(2))


def test_blocks_in_order_and_requeued_on_release(fake, make_source):
    src = make_source()
    src.start()
    fake.produce(FPB * NB)
    b0, b1 = take(src), take(src)
    assert (b0.seq, b1.seq) == (0, 1) and b0.index != b1.index
    assert len(fake.fifo) == 0  # every slot is held by the consumer
    src.release(b0)
    assert len(fake.fifo) == FPB  # release re-queues exactly that block's slots
    fake.produce(FPB)
    b2 = take(src)
    assert b2.index == b0.index and b2.seq == 2
    assert list(b2.frames["frame_id"]) == [8, 9, 10, 11]


def test_back_pressure_losses_counted_while_blocks_held(fake, make_source):
    src = make_source()
    src.start()
    fake.produce(FPB * NB)
    held = [take(src), take(src)]  # the consumer holds every block: no slot is queued
    assert fake.produce(5) == 0  # the camera keeps running; these 5 frames find no buffer
    src.release(held[0])
    fake.produce(FPB)
    blk = take(src)
    s = src.stats()
    assert fake.lost == 5
    assert s.lost == 5 and s.extra["lost_overflow"] == 5 and s.extra["lost_before_host"] == 0
    assert list(blk.frames["frame_id"]) == [13, 14, 15, 16]  # the counter jumped over the lost frames
    assert s.extra["pool_empty_events"] >= 1


def test_losses_before_host_counted_separately(fake, make_source):
    src = make_source(nblocks=3)
    src.start()
    fake.produce(2)
    fake.skip(3)  # lost between sensor and host although buffers were queued
    fake.produce(2)
    blk = take(src)
    s = src.stats()
    assert s.lost == 3 and s.extra["lost_before_host"] == 3 and s.extra["lost_overflow"] == 0
    assert list(blk.frames["frame_id"]) == [0, 1, 5, 6]


def test_frame_metadata_rows(fake, make_source):
    src = make_source()
    src.start()
    fake.incomplete_next = 0
    fake.produce(2)
    fake.incomplete_next = 1
    fake.produce(2)
    blk = take(src)
    f = blk.frames
    assert f.dtype.names == ("frame_id", "timestamp_ns", "host_ns", "size_filled", "incomplete", "slot")
    assert list(f["frame_id"]) == [0, 1, 2, 3]
    assert list(f["timestamp_ns"]) == [(10_000 + i * 3367) * 1000 for i in range(4)]  # camera us -> ns
    assert (f["host_ns"] > 0).all() and (np.diff(f["host_ns"].astype(np.int64)) >= 0).all()
    assert list(f["slot"]) == [0, 1, 2, 3]
    assert list(f["incomplete"]) == [0, 0, 1, 0]
    assert list(f["size_filled"]) == [32 * 48 * 2, 32 * 48 * 2, 0, 32 * 48 * 2]
    s = src.stats()
    assert s.incomplete == 1 and s.delivered == 4
    assert s.fps_measured == pytest.approx(1e6 / 3367)
    rows = f.copy()
    src.release(blk)
    np.testing.assert_array_equal(blk.frames, rows)  # the table is the consumer's: release does not clear it


def test_layout_mismatch_flagged(fake, make_source):
    src = make_source()
    src.start()
    fake.offset_next = 64  # the SDK reports the pixels at another offset than the view assumes
    fake.produce(FPB)
    blk = take(src)
    assert blk.frames["incomplete"][0] == 1 and src.stats().extra["layout_mismatch"] == 1


def test_partial_final_block_after_max_frames(fake, make_source):
    src = make_source(nblocks=3)
    src.start(max_frames=10)
    assert fake.frame_count == 10
    blocks = []
    fake.produce(FPB * 2)
    blocks += [take(src), take(src)]
    for b in blocks:
        src.release(b)
    fake.produce(10)  # the camera stops by itself after 10 frames
    last = take(src)
    assert last.partial and last.nframes == 2 and last.array.shape == (2, 32, 48) and last.frames.shape == (2,)
    assert last.full_array.shape == (FPB, 32, 48)
    np.testing.assert_array_equal(last.array[1].ravel(), fake.pattern(9))
    src.release(last)
    with pytest.raises(StopIteration):
        src.wait_block(10)
    assert src.stats().delivered == 10


def test_partial_final_block_after_stop(fake, make_source):
    src = make_source()
    src.start()
    fake.produce(FPB + 3)
    full = take(src)
    assert src.wait_block(10) is None  # the second block is not complete yet: timeout
    src.stop()
    part = take(src)
    assert not full.partial and part.partial and part.nframes == 3 and part.seq == 1
    assert list(part.frames["frame_id"]) == [4, 5, 6]
    src.release(full)
    src.release(part)
    with pytest.raises(StopIteration):
        src.wait_block(10)


def test_caller_buffers_receive_the_frames(fake, make_source):
    stride = make_source().info.extra["frame_stride"]
    bufs = [np.zeros(FPB * stride, np.uint8) for _ in range(NB)]
    src = make_source(buffers=bufs)
    assert src.info.extra["buffers"] == "caller"
    src.start()
    fake.produce(FPB)
    blk = take(src)
    assert any(np.shares_memory(blk.array, b) for b in bufs)
    owner = next(b for b in bufs if np.shares_memory(blk.array, b))
    np.testing.assert_array_equal(owner[stride : stride + 32 * 48 * 2].view(np.uint16), fake.pattern(1))


def test_caller_buffers_validated(fake, make_source):
    stride = make_source().info.extra["frame_stride"]
    with pytest.raises(ValueError, match="frame_stride"):
        TucamSource(frames_per_block=FPB, nblocks=NB, buffers=[np.zeros((FPB, 32, 48), np.uint16)] * NB)
    with pytest.raises(ValueError, match="nblocks"):
        TucamSource(frames_per_block=FPB, nblocks=NB, buffers=[np.zeros(FPB * stride, np.uint8)])
    # with chunk mode off a plain (frames_per_block, H, W) array is exactly one block
    src = make_source(chunk_mode=False, buffers=[np.zeros((FPB, 32, 48), np.uint16) for _ in range(NB)])
    src.start()
    fake.produce(FPB)
    assert take(src).array.flags.c_contiguous


def test_close_revokes_and_closes(fake):
    src = TucamSource(frames_per_block=FPB, nblocks=NB)
    src.start()
    fake.produce(3)
    src.close()
    assert fake.announced == {} and fake.callback is None and not fake.open and fake.policy == 0
    src.close()  # idempotent


# the real SDK crashes the process at exit unless TucTerminate is called
def test_terminate_closes_open_cameras_and_terminates_the_sdk(fake):
    from tucam_camera import tucam as T

    cam = T.TUCam(fake.serial)
    assert fake.open and cam in T._open_cams
    assert T.terminate() == 0
    assert not fake.open and cam.handle is None and fake.terminated == 1
    assert T._sdk is None and T.terminate() is None  # nothing loaded any more: a no-op
    T.install(fake, __import__("fake_tucam").D)  # the next sdk() call would load (here: install) it again
    cam2 = T.TUCam(fake.serial)
    cam2.close()
    assert cam2 not in T._open_cams


def test_settings_change_before_start_refused(fake, make_source):
    src = make_source()
    fake.features["Width"][2] = 64
    with pytest.raises(RuntimeError, match="frame layout"):
        src.start()


def test_release_twice_refused(fake, make_source):
    src = make_source()
    src.start()
    fake.produce(FPB)
    blk = take(src)
    src.release(blk)
    with pytest.raises(ValueError):
        src.release(blk)


# ---------------------------------------------------------------- nodes= and packed="copy"
def test_unpack_mono12packed_bit_layout():
    from fake_tucam import pack_mono12packed

    from tucam_camera.source import unpack_mono12packed

    # GigE Mono12Packed, as the Aries 6504 delivers it: p0 = b0 << 4 | b1 & 0xF, p1 = b2 << 4 | b1 >> 4
    raw = np.array([0x06, 0x35, 0x06, 0xAB, 0xC1, 0xFF], np.uint8)
    out = np.zeros(4, np.uint16)
    unpack_mono12packed(raw, out)
    assert out.tolist() == [0x065, 0x063, 0xAB1, 0xFFC]
    img = np.random.default_rng(1).integers(0, 4096, (8, 16)).astype(np.uint16)
    back = np.zeros_like(img)
    unpack_mono12packed(pack_mono12packed(img), back)
    np.testing.assert_array_equal(back, img)


def test_nodes_written_in_order_before_the_layout_is_read(fake, make_source):
    fake.mode_formats = {"Dynamic": "Mono16", "Speed": "Mono12Packed"}
    src = make_source(nodes={"SensorOperationMode": "Dynamic", "Dynamic": "LowGain", "Width": 32, "Height": 16})
    assert src.info.shape == (16, 32) and src.info.extra["pixel_format"] == "Mono16"
    assert src.info.extra["gain_mode"] == "LowGain" and src.info.extra["nodes"]["Width"] == 32
    with pytest.raises(ValueError, match="nodes: cannot set PixelFormat"):  # read-only node: refused, camera closed
        make_source(nodes={"PixelFormat": "Mono8"})


def test_packed_refused_by_default_names_the_copy_option(fake):
    fake.mode_formats = {"Speed": "Mono12Packed"}
    with pytest.raises(ValueError, match="packed='copy'"):
        TucamSource(frames_per_block=FPB, nblocks=NB, nodes={"SensorOperationMode": "Speed"})
    assert not fake.open


@pytest.mark.parametrize("threads", [1, 3])
def test_packed_copy_unpacks_into_contiguous_pool(fake, make_source, threads):
    fake.mode_formats = {"Speed": "Mono12Packed"}
    src = make_source(nodes={"SensorOperationMode": "Speed", "Speed": "HighGain"}, packed="copy",
                      unpack_threads=threads)
    x = src.info.extra
    assert x["copy"] and not x["zero_copy"] and x["pixel_format"] == "Mono12Packed" and x["contiguous"]
    assert src.info.dtype == np.uint16 and src.info.frame_bytes == 32 * 48 * 2
    assert x["staging_stride"] >= 32 * 48 * 3 // 2 + 128
    src.start()
    # the SDK is given the packed staging slots, not the pool
    pool = [src.block_array(i) for i in range(NB)]
    for a, _ in fake.announced.values():
        assert not any(p.ctypes.data <= a < p.ctypes.data + p.nbytes for p in pool)
    fake.produce(FPB)
    blk = take(src)
    assert blk.array.flags.c_contiguous and blk.array.shape == (FPB, 32, 48)
    for j in range(FPB):
        np.testing.assert_array_equal(blk.array[j].ravel(), fake.pattern(j))
    src.release(blk)
    s = src.stats()
    assert s.extra["copy"] and s.extra["unpacked"] == FPB and s.extra["unpack_ms_max"] > 0
    assert s.delivered == FPB and s.lost == 0 and s.incomplete == 0


def test_packed_copy_keeps_back_pressure_loss_counting(fake, make_source):
    fake.mode_formats = {"Speed": "Mono12Packed"}
    src = make_source(nodes={"SensorOperationMode": "Speed"}, packed="copy")
    src.start()
    fake.produce(FPB * NB)
    held = [take(src), take(src)]
    assert fake.produce(5) == 0  # every staging slot belongs to a held block: nothing is queued
    src.release(held[0])
    fake.produce(FPB)
    blk = take(src)
    assert list(blk.frames["frame_id"]) == [13, 14, 15, 16]
    np.testing.assert_array_equal(blk.array[0].ravel(), fake.pattern(13))
    s = src.stats()
    assert s.lost == 5 and s.extra["lost_overflow"] == 5

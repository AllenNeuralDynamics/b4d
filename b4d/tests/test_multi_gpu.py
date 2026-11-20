"""MultiGpuEncoder: split="rows" (every block shared out as bands of tile rows, one per device) gives the chunk bytes of the
whole-block mode and of the C encoder, for edge tiles, partial blocks, both upload paths and concurrent calls."""
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import b4d

CAM = b4d.Camera(0.5, 100.0, 1.6)
CASES = [((128, 300, 330), (64, 64, 64), (0, 0, 0), (128, 300, 330)),       # 5 tile rows (odd), x edge
         ((64, 512, 200), (64, 64, 64), (0, 0, 0), (64, 512, 200)),
         ((128, 256, 320), (64, 64, 64), (64, 0, 0), (160, 200, 320)),       # partial block (z and y clipped by the array)
         ((16, 64, 128), (16, 64, 64), (0, 0, 0), (16, 64, 128))]            # one tile row: the second device gets nothing
IDS = ["odd_rows_edges", "even_rows", "partial_block", "one_row"]


def _image(shape, seed=7, electrons=200.0):
    rng = np.random.default_rng(seed)
    e = rng.poisson(electrons, shape) + rng.normal(0, CAM.read_noise, shape)
    return np.round(CAM.offset + CAM.conversion * e).clip(0, 65535).astype(np.uint16)


def _order(shape, chunks):
    cz, cy, cx = chunks
    return [(z, y, x) for z in range(0, shape[0], cz) for y in range(0, shape[1], cy) for x in range(0, shape[2], cx)]


def _ref(blk, order, chunks, origin, ashape):
    ez, ey, ex = (min(n, a - o) for n, a, o in zip(blk.shape, ashape, origin)); cz, cy, cx = chunks; out = []
    for z, y, x in order:
        if z >= ez or y >= ey or x >= ex:
            out.append(None); continue
        t = np.zeros(chunks, blk.dtype); v = blk[z:min(z + cz, ez), y:min(y + cy, ey), x:min(x + cx, ex)]
        t[:v.shape[0], :v.shape[1], :v.shape[2]] = v
        out.append(b4d.compress(t, q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice"))
    return out


def _blobs(r):
    if isinstance(r, tuple):
        hbuf, off, ln = r
        return [hbuf[o:o + n].tobytes() if n >= 0 else None for o, n in zip(off.tolist(), ln.tolist())]
    return [None if b is None else bytes(b) for b in r]


def _cuda_devices():
    cp = pytest.importorskip("cupy")
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception:
        n = 0
    if n < 2:
        pytest.skip("needs two CUDA devices")


def test_bands_are_whole_tile_rows():
    from b4d.cuda_encoder import MultiGpuEncoder
    enc = MultiGpuEncoder.__new__(MultiGpuEncoder); enc.encs = [None, None]      # the band arithmetic needs no device
    assert enc._bands(300, 64) == [(0, 128), (128, 300)]
    assert enc._bands(256, 64) == [(0, 128), (128, 256)]
    assert enc._bands(64, 64) == [(0, 0), (0, 64)]
    with pytest.raises(ValueError):
        MultiGpuEncoder(CAM, devices=(0, 1), split="frames", q=1.0, mode=1, backend="emu")


@pytest.mark.parametrize("shape,chunks,origin,ashape", CASES, ids=IDS)
def test_rows_split_emulated_matches_c(shape, chunks, origin, ashape):
    if shutil.which("gcc") is None:
        pytest.skip("the emulated kernel is built with gcc")
    from b4d.cuda_encoder import MultiGpuEncoder
    blk = _image(shape, seed=3); order = _order(shape, chunks)
    enc = MultiGpuEncoder(CAM, devices=(0, 1), split="rows", q=1.0, mode=1, backend="emu", predictor="above3", entropy="rice", check=False)
    assert _blobs(enc.encode_tiles(blk, origin, order, chunks, ashape)) == _ref(blk, order, chunks, origin, ashape)
    enc.close()


@pytest.mark.parametrize("sliced", [True, False], ids=["sliced", "whole"])
@pytest.mark.parametrize("shape,chunks,origin,ashape", CASES, ids=IDS)
def test_rows_split_matches_whole_block_mode_and_c(shape, chunks, origin, ashape, sliced):
    _cuda_devices()
    from b4d.cuda_encoder import MultiGpuEncoder
    blk = _image(shape, seed=5); order = _order(shape, chunks)
    kw = dict(q=1.0, mode=1, backend="cuda", slots=1, predictor="above3", entropy="rice", check=False, batch_tiles=3)
    split = MultiGpuEncoder(CAM, devices=(0, 1), split="rows", **kw)
    whole = MultiGpuEncoder(CAM, devices=(0, 1), **kw)
    for e in split.encs + whole.encs:
        e.overlap_upload = sliced; e.upload_slices = 5
    got = _blobs(split.encode_tiles(blk, origin, order, chunks, ashape))
    assert all(("sliced" in e.stats["upload_mode"]) == sliced for e in split.encs if e.stats["blocks"])
    assert got == _blobs(whole.encode_tiles(blk, origin, order, chunks, ashape))
    assert got == _ref(blk, order, chunks, origin, ashape)
    split.close(); whole.close()


def test_rows_split_concurrent_calls_and_reused_buffers():
    """several writer threads at once (two slots per device), blocks in one reused host buffer each"""
    _cuda_devices()
    from b4d.cuda_encoder import MultiGpuEncoder
    shape, chunks = (64, 384, 256), (64, 64, 64); order = _order(shape, chunks)
    enc = MultiGpuEncoder(CAM, devices=(0, 1), split="rows", q=1.0, mode=1, backend="cuda", slots=2, predictor="above3",
                          entropy="rice", check=False, batch_tiles=4)
    enc.prepare(shape, chunks)
    bufs = [np.empty(shape, np.uint16) for _ in range(3)]
    for rnd in range(2):
        blks = [_image(shape, seed=10 * rnd + i) for i in range(3)]
        for b, x in zip(bufs, blks):
            b[...] = x
        with ThreadPoolExecutor(3) as ex:
            outs = list(ex.map(lambda b: _blobs(enc.encode_tiles(b, (0, 0, 0), order, chunks, shape)), bufs))
        for blk, got in zip(blks, outs):
            assert got == _ref(blk, order, chunks, (0, 0, 0), shape)
    assert enc.stats["blocks_per_device"] == [6, 6]
    enc.close()


def test_register_page_locks_each_buffer_once_and_bytes_are_unchanged():
    """register() page-locks the writer's buffers up front (each distinct buffer once, the same buffer twice counts once);
    encoding from them afterwards gives the C bytes in both modes"""
    _cuda_devices()
    from b4d.cuda_encoder import MultiGpuEncoder
    shape, chunks = (64, 256, 192), (64, 64, 64); order = _order(shape, chunks)
    bufs = [_image(shape, seed=40 + i) for i in range(2)]
    for split in ("blocks", "rows"):
        enc = MultiGpuEncoder(CAM, devices=(0, 1), split=split, q=1.0, mode=1, backend="cuda", slots=1, predictor="above3",
                              entropy="rice", check=False, batch_tiles=4)
        assert enc.register(bufs + [bufs[0]]) == 2
        assert enc.register(bufs) == 0
        for b in bufs:
            assert _blobs(enc.encode_tiles(b, (0, 0, 0), order, chunks, shape)) == _ref(b, order, chunks, (0, 0, 0), shape)
        enc.close()

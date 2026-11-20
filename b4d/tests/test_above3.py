"""The above3 predictor: an encoder setting over the mean3d stream (format v4, plane height 0), on every write path."""
import dataclasses

import numpy as np
import pytest

import b4d

CAM = b4d.Camera(0.5, 100.0, 1.6)


def _image(shape=(8, 96, 128), seed=7, electrons=200.0):
    rng = np.random.default_rng(seed)
    e = rng.poisson(electrons, shape) + rng.normal(0, CAM.read_noise, shape)
    return np.round(CAM.offset + CAM.conversion * e).clip(0, 65535).astype(np.uint16)


def _tv(a, b):
    m = int(max(a.max(), b.max())) + 1
    return 0.5 * np.abs(np.bincount(a.ravel(), minlength=m) - np.bincount(b.ravel(), minlength=m)).sum() / a.size


def _transform(x):
    k = CAM.read_noise ** 2 + 0.375
    e = (x.astype(float) - CAM.offset) / CAM.conversion
    return np.where(e >= 0, 2 * np.sqrt(np.maximum(e, 0) + k), 2 * np.sqrt(k) + e / np.sqrt(k))


@pytest.mark.parametrize("entropy", ["rice", "huffman", "ans"])
@pytest.mark.parametrize("mode,q", [(0, 0.0), (1, 1.0), (1, 2.0), (2, 1.0)])
def test_same_bytes_as_mean3d_without_planes(entropy, mode, q):
    a = _image()
    kw = dict(q=q, mode=mode, camera=CAM, entropy=entropy)
    assert b4d.compress(a, predictor="above3", **kw) == b4d.compress(a, predictor="mean3d", row_width=a.shape[-1], **kw)


@pytest.mark.parametrize("entropy", ["rice", "huffman", "ans"])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
def test_round_trip_keeps_the_bound(entropy, q):
    a = _image()
    d = b4d.decompress(b4d.compress(a, q=q, mode=1, camera=CAM, predictor="above3", entropy=entropy))
    assert d.shape == a.shape
    assert np.abs(_transform(a) - _transform(d)).max() <= q / 2 + 0.2      # + DN rounding on inversion


def test_lossless():
    a = _image()
    assert np.array_equal(b4d.decompress(b4d.compress(a, q=0.0, mode=0, camera=CAM, predictor="above3", entropy="rice")), a)


def test_reproduces_the_histogram_where_above_does_not():
    a = _image()
    tv = lambda p: _tv(a, b4d.decompress(b4d.compress(a, q=1.0, mode=1, camera=CAM, predictor=p, entropy="rice")))
    assert tv("above3") < 0.1 < tv("above")


def test_tiles_match_single_compress():
    a = _image(shape=(16, 64, 64))
    blobs = b4d.compress_tiles(a, (8, 32, 32), q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice", threads=2)
    assert len(blobs) == 8
    assert np.array_equal(b4d.decompress(blobs[0]), b4d.decompress(b4d.compress(a[:8, :32, :32], q=1.0, mode=1, camera=CAM,
                                                                                predictor="above3", entropy="rice")))


def test_thread_count_does_not_change_the_bytes():
    a = _image()
    kw = dict(q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice")
    assert b4d.compress(a, threads=1, **kw) == b4d.compress(a, threads=4, **kw)


def test_native_zarr_writer_matches_single_compress(tmp_path):
    import struct

    import zarr

    from b4d.zarr3 import B4D
    from b4d import zarr_store

    img = _image(shape=(16, 128, 128))
    codec = B4D(q=1, mode=1, camera=CAM, predictor="above3", entropy="rice")
    path = str(tmp_path / "a.zarr")
    arr = zarr_store.create(path, img.shape, dtype=img.dtype, shards=(16, 128, 128), chunks=(16, 64, 64), codec=codec)
    arr[0:16].write(img).result()
    arr.close()
    assert zarr.open_array(path, mode="r").shape == img.shape
    shard = open(f"{path}/c/0/0/0", "rb").read()
    off, size = struct.unpack("<QQ", shard[-(4 * 16 + 4):-4][:16])
    assert shard[off:off + size] == b4d.compress(img[:16, :64, :64], q=1, mode=1, camera=CAM, predictor="above3", entropy="rice")


@pytest.mark.parametrize("entropy", ["rice", "ans"])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("cam", [CAM, b4d.Camera(1 / 7.25, 2.05, 6.8)], ids=["bright", "below-offset"])
def test_cuda_encoder_matches_c_encoder(entropy, q, cam):
    """the CUDA encoder's mode-1 above3 (b4d_resid_m1_tiles + the Rice kernel or the host ANS coder) gives the C encoder's
    bytes, including partial edge tiles and samples below the offset (the transform's linear branch)"""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    rng = np.random.default_rng(11)
    shape, chunks = (40, 100, 150), (16, 64, 64)
    e = rng.poisson(rng.uniform(0, 300, shape)) + rng.normal(0, cam.read_noise, shape)
    blk = np.round(cam.offset + cam.conversion * e).clip(0, 65535).astype(np.uint16)
    kw = dict(q=q, mode=1, backend="cuda", slots=1, predictor="above3", entropy=entropy, check=False, batch_tiles=4)
    if entropy == "ans" and b4d._vst_of(cam) is not None:   # the exact VST (default) runs only with the device Rice coder
        with pytest.raises(ValueError, match="exact VST"):
            CudaTileEncoder(cam, **kw)
        cam = dataclasses.replace(cam, exact_vst=False)     # the Anscombe path still matches the C encoder with ANS
    enc = CudaTileEncoder(cam, **kw)
    order = [(z, y, x) for z in range(0, 40, 16) for y in range(0, 100, 64) for x in range(0, 150, 64)]
    hbuf, off, ln = enc.encode_tiles(blk, (0, 0, 0), order, chunks, shape)
    for k, (z, y, x) in enumerate(order):
        tile = np.zeros(chunks, np.uint16)                       # zarr chunks are full size: edge tiles are zero padded
        sub = blk[z:z + 16, y:y + 64, x:x + 64]; tile[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        ref = b4d.compress(tile, q=q, mode=1, camera=cam, predictor="above3", entropy=entropy)
        assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, (entropy, q, k)


@pytest.mark.parametrize("sm_limit", [0, 1, 7])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_cuda_encoder_uint8_and_sm_limit(dtype, sm_limit):
    """uint8 blocks (widened on the device, typesize 1 headers) and persistent launches on a subset of the SMs give the
    C encoder's bytes, for above3 (mode 1) and above (mode 2)"""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    rng = np.random.default_rng(5)
    shape, chunks = (40, 100, 150), (16, 64, 64)
    cam = b4d.Camera(0.03, 2.0, 11.0) if dtype == "uint8" else CAM
    top = 255 if dtype == "uint8" else 65535
    e = rng.poisson(rng.uniform(0, 4000 if dtype == "uint8" else 300, shape)) + rng.normal(0, cam.read_noise, shape)
    blk = np.round(cam.offset + cam.conversion * e).clip(0, top).astype(dtype)
    blk[:16, :64, :64] = 0                                      # an all-zero tile: finished on the host
    order = [(z, y, x) for z in range(0, 40, 16) for y in range(0, 100, 64) for x in range(0, 150, 64)]
    for pred, mode in (("above3", 1), ("above", 2)):
        enc = CudaTileEncoder(cam, q=1.0, mode=mode, backend="cuda", slots=1, predictor=pred, entropy="rice", check=False,
                              batch_tiles=4, sm_limit=sm_limit)
        hbuf, off, ln = enc.encode_tiles(blk, (0, 0, 0), order, chunks, shape)
        for k, (z, y, x) in enumerate(order):
            tile = np.zeros(chunks, blk.dtype)
            sub = blk[z:z + 16, y:y + 64, x:x + 64]; tile[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
            ref = b4d.compress(tile, q=1.0, mode=mode, camera=cam, predictor=pred, entropy="rice")
            assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, (pred, k)


def test_cuda_encoder_two_slots_concurrent_sm_limit():
    """two pipeline slots encoding at once with persistent launches (sm_limit > 0): each stream has its own work counter,
    so neither block's kernels take the other's tiles (a shared counter gave wrong blobs and offsets on 8-bit cameras)"""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from concurrent.futures import ThreadPoolExecutor
    from b4d.cuda_encoder import CudaTileEncoder
    rng = np.random.default_rng(9)
    shape, chunks = (32, 256, 320), (16, 64, 64)
    blks = [_image(shape=shape, seed=s) for s in (1, 2, 3, 4)]
    order = [(z, y, x) for z in range(0, 32, 16) for y in range(0, 256, 64) for x in range(0, 320, 64)]
    enc = CudaTileEncoder(CAM, q=1.0, mode=1, backend="cuda", slots=2, predictor="above3", entropy="rice", check=False,
                          batch_tiles=4, sm_limit=int(rng.integers(2, 8)))
    with ThreadPoolExecutor(2) as ex:
        outs = list(ex.map(lambda b: enc.encode_tiles(b, (0, 0, 0), order, chunks, shape), blks))
    for blk, (hbuf, off, ln) in zip(blks, outs):
        for k, (z, y, x) in enumerate(order):
            ref = b4d.compress(np.ascontiguousarray(blk[z:z + 16, y:y + 64, x:x + 64]), q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice")
            assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, k


@pytest.mark.parametrize("shape,chunks,origin,ashape", [((128, 300, 330), (64, 64, 64), (0, 0, 0), (128, 300, 330)),
                                                        ((64, 512, 200), (64, 64, 64), (0, 0, 0), (64, 512, 200)),
                                                        ((128, 256, 320), (64, 64, 64), (64, 0, 0), (160, 200, 320))],
                         ids=["two_slabs_edges", "one_slab_bands", "partial_block"])
def test_cuda_encoder_sliced_upload_matches_c(shape, chunks, origin, ashape):
    """the sliced upload (copy stream + per-slice events, batches start when their rows have arrived) gives the C
    encoder's bytes for every tile, with many slices and batches, edge tiles and a partial block"""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    blk = _image(shape=shape, seed=5)
    cz, cy, cx = chunks
    order = [(z, y, x) for z in range(0, shape[0], cz) for y in range(0, shape[1], cy) for x in range(0, shape[2], cx)]
    enc = CudaTileEncoder(CAM, q=1.0, mode=1, backend="cuda", slots=1, predictor="above3", entropy="rice", check=False, batch_tiles=3)
    enc.overlap_upload = True; enc.upload_slices = 7
    hbuf, off, ln = enc.encode_tiles(blk, origin, order, chunks, ashape)
    assert "sliced" in enc.stats["upload_mode"]
    ez, ey, ex = (min(n, a - o) for n, a, o in zip(shape, ashape, origin))
    for k, (z, y, x) in enumerate(order):
        if z >= ez or y >= ey or x >= ex:
            assert off[k] < 0; continue
        t = np.zeros(chunks, blk.dtype); v = blk[z:min(z + cz, ez), y:min(y + cy, ey), x:min(x + cx, ex)]
        t[:v.shape[0], :v.shape[1], :v.shape[2]] = v
        ref = b4d.compress(t, q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice")
        assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, k
    enc.close()


def test_cuda_encoder_refuses_above3_outside_mode_1():
    from b4d.cuda_encoder import CudaTileEncoder
    with pytest.raises(ValueError):
        CudaTileEncoder(CAM, q=1, mode=2, backend="emu", predictor="above3")


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("mode,pred,dither", [(1, "above", False), (1, "above", True), (1, "above3", True), (2, "above", True)],
                         ids=["m1_above", "m1_above_dither", "m1_above3_dither", "m2_above_dither"])
def test_cuda_encoder_mode1_above_and_dither(mode, pred, dither, dtype):
    """mode-1 'above' and the seeded dither (format v8: mode 1 above/above3, mode 2 above) give the C encoder's bytes,
    including partial edge tiles and uint8"""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    rng = np.random.default_rng(13)
    shape, chunks = (40, 100, 150), (16, 64, 64)
    cam = b4d.Camera(0.03, 2.0, 11.0) if dtype == "uint8" else CAM
    top = 255 if dtype == "uint8" else 65535
    e = rng.poisson(rng.uniform(0, 4000 if dtype == "uint8" else 300, shape)) + rng.normal(0, cam.read_noise, shape)
    blk = np.round(cam.offset + cam.conversion * e).clip(0, top).astype(dtype)
    order = [(z, y, x) for z in range(0, 40, 16) for y in range(0, 100, 64) for x in range(0, 150, 64)]
    enc = CudaTileEncoder(cam, q=1.0, mode=mode, backend="cuda", slots=1, predictor=pred, entropy="rice", check=False,
                          batch_tiles=4, dither=dither)
    hbuf, off, ln = enc.encode_tiles(blk, (0, 0, 0), order, chunks, shape)
    for k, (z, y, x) in enumerate(order):
        tile = np.zeros(chunks, blk.dtype)
        sub = blk[z:z + 16, y:y + 64, x:x + 64]; tile[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        ref = b4d.compress(tile, q=1.0, mode=mode, camera=cam, predictor=pred, entropy="rice", dither=dither)
        assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, (k,)


@pytest.mark.parametrize("mode,pred,dither", [(1, "above3", False), (2, "above", True)])
def test_native_zarr_writer_one_tile_bands(tmp_path, mode, pred, dither):
    """64^3 uint16 chunks are gathered one per band and encoded from the band without a tile copy: every chunk, including
    the zero-padded edge chunks, is still the bytes of b4d.compress on that chunk"""
    import struct

    from b4d.zarr3 import B4D
    from b4d import zarr_store

    img = _image(shape=(64, 150, 200))
    codec = B4D(q=1, mode=mode, camera=CAM, predictor=pred, entropy="rice", dither=dither)
    path = str(tmp_path / "a.zarr")
    arr = zarr_store.create(path, img.shape, dtype=img.dtype, shards=(64, 192, 256), chunks=(64, 64, 64), codec=codec, threads=4)
    arr[0:64].write(img).result()
    arr.close()
    shard = open(f"{path}/c/0/0/0", "rb").read()
    n = 3 * 4
    index = struct.unpack(f"<{2 * n}Q", shard[-(16 * n + 4):-4])
    for k in range(n):
        ky, kx = divmod(k, 4)
        off, size = index[2 * k], index[2 * k + 1]
        if ky * 64 >= 150 or kx * 64 >= 200:
            assert size == 2 ** 64 - 1                         # chunk outside the array: absent
            continue
        tile = np.zeros((64, 64, 64), np.uint16)
        sub = img[:, ky * 64:ky * 64 + 64, kx * 64:kx * 64 + 64]; tile[:, :sub.shape[1], :sub.shape[2]] = sub
        ref = b4d.compress(tile, q=1, mode=mode, camera=CAM, predictor=pred, entropy="rice", dither=dither)
        assert shard[off:off + size] == ref, k

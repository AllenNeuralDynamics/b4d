import numpy as np
import pytest

import b4d

CAM = b4d.Camera(conversion=2.0, offset=100.0, read_noise=3.0)
RNG = np.random.default_rng(1234)
OLD = dict(predictor="left_above", entropy="huffman")     # the older defaults these tests were written against
LOSSLESS = dict(q=0, **OLD)


def synth(shape, dtype=np.uint16, cam=CAM):
    """Smooth background + Gaussian spots + Poisson noise, in DN."""
    y = np.indices(shape[-2:], dtype=np.float64)
    truth = 40 + 30 * y[1] / shape[-1]
    for _ in range(max(1, int(np.prod(shape[-2:])) // 4000)):
        cy, cx = RNG.integers(0, shape[-2]), RNG.integers(0, shape[-1])
        truth = truth + RNG.uniform(200, 2000) * np.exp(-((y[0] - cy) ** 2 + (y[1] - cx) ** 2) / 8.0)
    truth = np.broadcast_to(truth, shape)
    e = RNG.poisson(truth) + RNG.normal(0, cam.read_noise, shape)
    dn = cam.offset + cam.conversion * e
    return np.clip(np.round(dn), 0, np.iinfo(dtype).max).astype(dtype)


def assert_within_bound(img, out, q, cam=CAM):
    """Per-pixel |err| <= q/2 sigma (+1 DN rounding slack)."""
    sig = cam.sigma(img)
    err = np.abs(img.astype(np.float64) - out.astype(np.float64))
    # left+above adds one extra half-step of rounding; sigma is evaluated at the
    # original value, so allow 10% on top of q/2 sigma plus 1 DN
    ratio = err / (0.55 * q * sig + 1.0)
    assert ratio.max() <= 1.0, f"max err ratio {ratio.max():.2f}"
    # mean error of a uniform quantiser with step q*sigma is q*sigma/4
    assert np.mean(err / sig) <= 0.3 * q + 0.05


# ------------------------------------------------------------------ lossless
@pytest.mark.parametrize("shape", [(1,), (17,), (1, 1), (3, 5), (64, 64), (7, 33, 41), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("predictor", ["left_above", "above"])
@pytest.mark.parametrize("entropy", ["huffman", "rice", "ans"])
def test_lossless_roundtrip(shape, dtype, predictor, entropy):
    img = synth(shape if len(shape) >= 2 else (1,) + shape, dtype).reshape(shape)
    blob = b4d.compress(img, q=0, predictor=predictor, entropy=entropy)
    out = b4d.decompress(blob)
    assert out.dtype == img.dtype and out.shape == img.shape
    assert np.array_equal(img, out)
    assert out.flags.writeable


def test_lossless_extremes_and_random():
    for img in [np.zeros((16, 16), np.uint16), np.full((16, 16), 65535, np.uint16),
                RNG.integers(0, 65536, (64, 64), dtype=np.uint16),       # incompressible
                RNG.integers(0, 256, (64, 64), dtype=np.uint8)]:
        assert np.array_equal(img, b4d.decompress(b4d.compress(img, **LOSSLESS)))


def test_non_contiguous_and_views():
    img = synth((64, 128))
    view = img[::2, 5:100]
    assert np.array_equal(view, b4d.decompress(b4d.compress(view, **LOSSLESS)))
    assert np.array_equal(img.T, b4d.decompress(b4d.compress(img.T, **LOSSLESS)))


# ------------------------------------------------------------------ lossy
@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("q", [0.25, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("predictor", ["left_above", "above"])
@pytest.mark.parametrize("entropy", ["huffman", "rice", "ans"])
def test_lossy_error_bound(mode, q, predictor, entropy):
    img = synth((96, 160))
    blob = b4d.compress(img, q=q, mode=mode, camera=CAM, predictor=predictor, entropy=entropy)
    out = b4d.decompress(blob)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert_within_bound(img, out, q)
    assert len(blob) < len(b4d.compress(img, **LOSSLESS))          # lossy smaller than lossless


def test_lossy_uint8():
    cam = b4d.Camera(conversion=0.25, offset=10, read_noise=2.0)
    img = synth((64, 64), np.uint8, cam)
    out = b4d.decompress(b4d.compress(img, q=1.0, mode=1, camera=cam, **OLD))
    assert_within_bound(img, out, 1.0, cam)


def test_ratio_increases_with_q():
    img = synth((128, 256))
    sizes = [len(b4d.compress(img, q=q, mode=1, camera=CAM, **OLD)) for q in (0, 0.5, 1, 2, 4)]
    assert sizes == sorted(sizes, reverse=True)


def test_q_zero_is_lossless_regardless_of_mode():
    img = synth((32, 32))
    for mode in (1, 2):
        assert np.array_equal(img, b4d.decompress(b4d.compress(img, q=0, mode=mode, camera=CAM, **OLD)))


# ------------------------------------------------------------------ plumbing
def test_chunk_splitting():
    img = synth((40, 64))
    small = b4d.compress(img, max_chunk_bytes=64 * 2 * 7, **LOSSLESS)      # 7 rows per chunk -> 6 chunks
    assert np.array_equal(img, b4d.decompress(small))
    # 3-D input splits across slices too
    vol = synth((5, 20, 16))
    assert np.array_equal(vol, b4d.decompress(b4d.compress(vol, max_chunk_bytes=16 * 2 * 9, **LOSSLESS)))


def test_threads_do_not_change_result():
    img = synth((256, 256))
    for bs in (None, 512 * 2 * 8, 512 * 2 * 100 + 6):        # incl. rows straddling blocks
        ref = None
        for nt in (1, 2, 4):
            blob = b4d.compress(img, q=1, mode=1, camera=CAM, threads=nt, blocksize=bs, **OLD)
            out = b4d.decompress(blob, threads=nt)
            assert_within_bound(img, out, 1.0)
            if ref is None:
                ref = (blob, out)
            # Blosc2 appends blocks in thread completion order, so the byte layout
            # of the chunk may differ between thread counts; the size and the
            # decoded data must not.
            assert len(blob) == len(ref[0])
            assert np.array_equal(out, ref[1])
            assert np.array_equal(b4d.decompress(blob, threads=1), ref[1])


def test_blocksize_lossless_identical_output():
    img = synth((256, 256))
    for bs in (None, 512 * 2 * 8, 512 * 2 * 100 + 6, 0):
        assert np.array_equal(img, b4d.decompress(b4d.compress(img, blocksize=bs, **LOSSLESS)))


def test_out_parameter():
    img = synth((32, 48))
    blob = b4d.compress(img, **LOSSLESS)
    out = np.empty_like(img)
    res = b4d.decompress(blob, out=out)
    assert res is out and np.array_equal(out, img)
    with pytest.raises(ValueError):
        b4d.decompress(blob, out=np.empty((48, 32), np.uint16))


def test_save_load(tmp_path):
    img = synth((30, 70))
    p = tmp_path / "x.b4d"
    n = b4d.save(p, img, q=1, mode=1, camera=CAM, **OLD)
    assert p.stat().st_size == n
    assert_within_bound(img, b4d.load(p), 1.0)


def test_errors():
    with pytest.raises(TypeError):
        b4d.compress(np.zeros((4, 4), np.float32))
    with pytest.raises(TypeError):
        b4d.compress(np.zeros((4, 4), np.int16))
    with pytest.raises(ValueError):
        b4d.compress(np.zeros((4, 4), np.uint16), q=1, mode=3)
    with pytest.raises(ValueError):
        b4d.compress(np.zeros((4, 4), np.uint16), q=0.3, mode=1)
    with pytest.raises(ValueError):
        b4d.compress(np.zeros((4, 4), np.uint16), predictor="diagonal")
    with pytest.raises(ValueError):
        b4d.decompress(b"nonsense")
    blob = b4d.compress(np.zeros((4, 4), np.uint16), **LOSSLESS)
    with pytest.raises(Exception):
        b4d.decompress(blob[:-5])
    with pytest.raises(TypeError):                          # codec options are keyword-only
        b4d.compress(np.zeros((4, 4), np.uint16), 1.0)
    with pytest.raises(TypeError):
        b4d.decompress(blob, 2)


def test_defaults_are_m1_a3():
    img = synth((16, 64, 64))
    m1_a3 = b4d.compress(img, q=1.0, mode=1, camera=CAM, predictor="above3", entropy="rice")
    assert b4d.compress(img, camera=CAM) == m1_a3
    assert b4d.compress_tiles(img, (16, 32, 32), camera=CAM) == b4d.compress_tiles(img, (16, 32, 32), q=1.0, mode=1, camera=CAM,
                                                                                 predictor="above3", entropy="rice")


def test_camera_sigma():
    cam = b4d.Camera(2.0, 100.0, 3.0)
    assert cam.sigma(100) == pytest.approx(6.0)               # read noise only: 2 * 3
    assert cam.sigma(100 + 2 * 91) == pytest.approx(2 * 10.0)  # 91 e- + 9 = 100 -> 10 e-


# ------------------------------------------------------------------ zarr / numcodecs
def test_numcodecs_roundtrip():
    numcodecs = pytest.importorskip("numcodecs")
    codec = b4d.B4DCodec(q=1.0, mode=1, camera=CAM, predictor="above", entropy="rice")
    img = synth((64, 64))
    assert_within_bound(img, codec.decode(codec.encode(img)), 1.0)
    cfg = codec.get_config()
    codec2 = numcodecs.get_codec(cfg)
    assert isinstance(codec2, b4d.B4DCodec) and codec2.get_config() == cfg


def test_numcodecs_config_keeps_exact_vst_and_reads_nthreads():
    pytest.importorskip("numcodecs")
    cam = b4d.Camera(2.0, 100.0, 3.0, exact_vst=True)
    codec = b4d.B4DCodec(camera=cam, threads=2)
    cfg = codec.get_config()
    assert cfg["camera"]["exact_vst"] is True and cfg["threads"] == 2 and "nthreads" not in cfg
    back = b4d.B4DCodec.from_config(cfg)
    assert back.camera == cam and back.get_config() == cfg
    img = synth((32, 64))
    assert back.encode(img) == codec.encode(img) == b4d.compress(img, camera=cam)
    old = dict(cfg); old.pop("threads"); old["nthreads"] = 3           # an older-format .zarray
    assert b4d.B4DCodec.from_config(old).threads == 3


def test_zarr_array():
    zarr = pytest.importorskip("zarr")
    img = synth((4, 200, 300))
    codec = b4d.B4DCodec(q=1.0, mode=1, camera=CAM, **OLD)
    try:
        z = zarr.create_array(store={}, shape=img.shape, chunks=(1, 100, 300), dtype=img.dtype,
                              compressors=[codec], zarr_format=2)
    except (TypeError, AttributeError):                        # zarr 2.x API
        z = zarr.zeros(img.shape, chunks=(1, 100, 300), dtype=img.dtype, compressor=codec)
    z[:] = img
    assert_within_bound(img, z[:], 1.0)
    assert z.nbytes_stored() < img.nbytes / 3 if hasattr(z, "nbytes_stored") else True


def test_zarr3_sharded_array(tmp_path):
    zarr = pytest.importorskip("zarr", minversion="3")
    from b4d.zarr3 import B4D
    img = synth((64, 96, 128))
    path = tmp_path / "s.zarr"
    for q, mode in ((0, 1), (1.0, 1), (1.0, 2)):
        z = zarr.create_array(store=path, shape=img.shape, chunks=(32, 32, 32), shards=img.shape, dtype=img.dtype,
                              serializer=B4D(q=q, mode=mode, camera=CAM, predictor="above", entropy="rice"),
                              compressors=None, zarr_format=3, overwrite=True)
        z[:] = img
        back = zarr.open_array(path, mode="r")            # re-opened from metadata -> codec resolved by name
        inner = back.metadata.codecs[0].to_dict()["configuration"]["codecs"]   # inside sharding_indexed
        assert inner[0]["name"] == "b4d"
        out = back[:]
        if q == 0:
            assert np.array_equal(img, out)
        else:
            assert_within_bound(img, out, q)
        assert sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) < img.nbytes / 2
    with pytest.raises(TypeError):
        zarr.create_array(store={}, shape=(4, 4), chunks=(4, 4), dtype="float32", serializer=B4D(), compressors=None)


def test_concurrent_compress_with_different_parameters():
    """Parameters are per call: threads racing with different cameras/options must
    each get the same bytes and decode as their single-threaded run."""
    from concurrent.futures import ThreadPoolExecutor
    img = synth((32, 64, 64))
    configs = [dict(q=q, mode=m, camera=b4d.Camera(c, 100.0, r), predictor=p, entropy=e)
               for q in (0, 1.0) for m in (1, 2) for c in (1.0, 2.0) for r in (1.0, 3.0)
               for p in ("left_above", "above") for e in ("huffman", "rice", "ans")]
    ref = [b4d.compress(img, **cfg) for cfg in configs]
    with ThreadPoolExecutor(8) as ex:
        for _ in range(3):
            got = list(ex.map(lambda cfg: b4d.compress(img, **cfg), configs))
            assert got == ref
            outs = list(ex.map(b4d.decompress, got))
    for cfg, o in zip(configs, outs):
        if cfg["q"] == 0:
            assert np.array_equal(o, img)
        else:
            assert_within_bound(img, o, cfg["q"], cfg["camera"])


def test_ans_smaller_than_huffman_on_sparse_residuals():
    img = synth((32, 256, 256))
    for q in (1.0, 2.0):
        h = len(b4d.compress(img, q=q, mode=2, camera=CAM, predictor="left_above", entropy="huffman"))
        a = len(b4d.compress(img, q=q, mode=2, camera=CAM, predictor="left_above", entropy="ans"))
        assert a < h * 0.99, (q, a, h)
        assert np.array_equal(b4d.decompress(b4d.compress(img, q=q, mode=2, camera=CAM, predictor="left_above", entropy="ans")),
                              b4d.decompress(b4d.compress(img, q=q, mode=2, camera=CAM, predictor="left_above",
                                                          entropy="huffman")))   # same lattice


def test_zarr3_ans_codec(tmp_path):
    zarr = pytest.importorskip("zarr", minversion="3")
    from b4d.zarr3 import B4D
    img = synth((16, 64, 64))
    z = zarr.create_array(store=tmp_path / "a.zarr", shape=img.shape, chunks=(8, 32, 64), dtype=img.dtype,
                          serializer=B4D(q=1.0, mode=2, camera=CAM, predictor="left_above", entropy="ans"), compressors=None, zarr_format=3)
    z[:] = img
    back = zarr.open_array(tmp_path / "a.zarr", mode="r")
    assert back.metadata.codecs[0].to_dict()["configuration"]["entropy"] == "ans"
    assert_within_bound(img, back[:], 1.0)


def test_compress_tiles_identical_to_per_tile_compress():
    cam = CAM
    img = synth((32, 300, 700))
    for tile in ((32, 128, 128), (16, 64, 64), (32, 300, 700), (8, 96, 200)):
        for kw in (dict(q=0, mode=0, **OLD), dict(q=1, mode=2, camera=cam, predictor="above", entropy="rice"),
                   dict(q=2, mode=2, camera=cam, predictor="left_above", entropy="ans")):
            blobs = b4d.compress_tiles(img, tile, **kw)
            tz, ty, tx = tile; k = 0
            for z0 in range(0, img.shape[0], tz):
                for y0 in range(0, img.shape[1], ty):
                    for x0 in range(0, img.shape[2], tx):
                        sub = img[z0:z0 + tz, y0:y0 + ty, x0:x0 + tx]
                        assert blobs[k] == b4d.compress(sub, **kw), (tile, kw, (z0, y0, x0))
                        k += 1
            assert k == len(blobs)


def test_compress_is_history_independent():
    """Blosc2 keeps block-size state in reused contexts; blobs must not depend on earlier calls."""
    img = synth((32, 128, 128)); small = img[:, :40, :]
    ref = b4d.compress(img, q=1, mode=2, camera=CAM, **OLD)
    b4d.compress(small, q=1, mode=2, camera=CAM, **OLD)                # a smaller chunk in between
    assert b4d.compress(img, q=1, mode=2, camera=CAM, **OLD) == ref
    b4d.compress(img, q=1, mode=2, camera=CAM, blocksize=0, **OLD)     # auto block size in between
    assert b4d.compress(img, q=1, mode=2, camera=CAM, **OLD) == ref

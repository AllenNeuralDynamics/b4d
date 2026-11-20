"""Seeded subtractive dither (format v6, mode 2): opt-in, decodable, bounded, and unbiased at low ADU."""
import json
import struct

import numpy as np
import pytest

import b4d

CAM = b4d.Camera(0.5, 100.0, 1.6, exact_vst=False)   # Anscombe: format v10 (the exact VST is v11, test_exact_vst)
LOW = b4d.Camera(1 / 7.25, 2.05, 6.89)   # IMX811-like: read noise about one DN


def _image(cam=CAM, electrons=200.0, shape=(8, 96, 128), seed=7):
    rng = np.random.default_rng(seed)
    e = rng.poisson(electrons, shape) + rng.normal(0, cam.read_noise, shape)
    return np.round(cam.offset + cam.conversion * e).clip(0, 65535).astype(np.uint16)


def _block_versions(blob):
    n = 7 + 8 * 3
    npieces = int.from_bytes(blob[n:n + 4], "little")
    sizes = [int.from_bytes(blob[n + 4 + 4 * i:n + 8 + 4 * i], "little") for i in range(npieces)]
    off = n + 4 + 4 * npieces; vers = set()
    for s in sizes:
        chunk = blob[off:off + s]; off += s
        bstart = int.from_bytes(chunk[32:36], "little")
        vers.add(chunk[bstart + 4])
    return vers


def _sigma_error(raw, dec, cam):
    t = lambda x: 2 * np.sqrt(np.maximum((x.astype(float) - cam.offset) / cam.conversion + cam.read_noise ** 2 + 0.375, 0))
    return np.abs(t(raw) - t(dec))


@pytest.mark.parametrize("predictor", ["above", "left_above", "mean3d"])
@pytest.mark.parametrize("entropy", ["rice", "huffman", "ans"])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
def test_round_trip_keeps_the_bound(predictor, entropy, q):
    raw = _image()
    blob = b4d.compress(raw, q=q, mode=2, camera=CAM, predictor=predictor, entropy=entropy, dither=True)
    dec = b4d.decompress(blob)
    assert dec.dtype == raw.dtype and dec.shape == raw.shape
    # q/2 in the transform plus half a DN of integer rounding, away from the linear extension below the offset
    slope = 1 / np.sqrt(CAM.conversion * (raw.astype(float) - CAM.offset) + CAM.conversion ** 2 * (CAM.read_noise ** 2 + 0.375))
    assert (_sigma_error(raw, dec, CAM) <= q / 2 + 0.5 * slope + 1e-2).all()


def test_uint8():
    raw = (_image(electrons=50.0) // 2).astype(np.uint8)
    cam = b4d.Camera(0.25, 50.0, 1.6)
    dec = b4d.decompress(b4d.compress(raw, q=1.0, mode=2, camera=cam, predictor="above", entropy="rice", dither=True))
    assert dec.dtype == np.uint8 and abs(float(dec.mean()) - float(raw.mean())) < 0.05


@pytest.mark.parametrize("electrons", [0, 4, 16])
def test_removes_the_low_adu_bias(electrons):
    raw = _image(LOW, electrons, (16, 256, 256))
    kw = dict(q=2.0, mode=2, camera=LOW, predictor="above", entropy="rice")
    plain = b4d.decompress(b4d.compress(raw, **kw))
    dith = b4d.decompress(b4d.compress(raw, dither=True, **kw))
    bias_plain = abs(float(plain.mean()) - float(raw.mean()))
    bias_dith = abs(float(dith.mean()) - float(raw.mean()))
    assert bias_dith < 0.05
    assert bias_dith < bias_plain or bias_plain < 0.05


def test_deterministic_across_threads_and_varies_between_blocks():
    raw = _image(shape=(32, 256, 256))
    kw = dict(q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True, blocksize=256 * 2 * 64)
    one = b4d.decompress(b4d.compress(raw, threads=1, **kw))
    assert np.array_equal(one, b4d.decompress(b4d.compress(raw, threads=4, **kw), threads=4))
    # identical pixel values in different blocks get different errors, so the error does not repeat between chunks
    a = _image(shape=(1, 64, 256), seed=1); b = a.copy(); b[0, 0, 0] += 1
    da = b4d.decompress(b4d.compress(a, q=2.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True))
    db = b4d.decompress(b4d.compress(b, q=2.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True))
    assert (da != db).mean() > 0.1


def test_off_by_default_leaves_the_bytes_alone():
    raw = _image()
    kw = dict(q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice")
    assert b4d.compress(raw, **kw) != b4d.compress(raw, dither=True, **kw)
    assert b4d.compress(raw, **kw) == b4d.compress(raw, dither=False, **kw)
    assert _block_versions(b4d.compress(raw, **kw)) == {10}                # Rice: v10 (8 streams, carries every option)
    assert _block_versions(b4d.compress(raw, dither=True, **kw)) == {10}   # (v8 / v6 dithered blocks still decode)
    assert _block_versions(b4d.compress(raw, **{**kw, "entropy": "huffman"})) == {3}


def test_tiles():
    raw = _image()
    kw = dict(q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True)
    tiles = b4d.compress_tiles(raw, (8, 32, 64), **kw)
    assert np.array_equal(b4d.decompress(tiles[0]), b4d.decompress(b4d.compress(raw[:8, :32, :64], row_width=64, **kw)))


@pytest.mark.parametrize("kw", [dict(mode=1, q=0.0), dict(mode=2, q=0.0), dict(mode=2, q=1.0, predictor="b3d")])
def test_refused(kw):
    with pytest.raises(ValueError, match="dither"):
        b4d.compress(_image(), camera=CAM, dither=True, **kw)


def test_listed_needs_a_calibration():
    with pytest.raises(ValueError, match="calibration"):
        b4d.compress(_image(), q=1.0, mode=1, camera=CAM, predictor="above3", dither="listed")


def test_numcodecs_config():
    c = b4d.B4DCodec(q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True)
    assert c.get_config()["dither"] is True
    assert b4d.B4DCodec.from_config(c.get_config()).dither is True
    assert "dither" not in b4d.B4DCodec(q=1.0, mode=2, camera=CAM).get_config()


def test_zarr_store(tmp_path):
    from b4d.zarr3 import B4D
    from b4d import zarr_store

    assert "dither" not in B4D(q=1.0, mode=2, camera=CAM).to_dict()["configuration"]
    codec = B4D(q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice", dither=True)
    assert B4D.from_dict(codec.to_dict()).dither is True
    raw = _image(shape=(16, 64, 64))
    path = str(tmp_path / "a.zarr")
    arr = zarr_store.create(path, raw.shape, dtype="uint16", shards=(16, 64, 64), chunks=(8, 32, 32), codec=codec)
    arr[0:16].write(raw).result()
    arr.close()
    meta = json.load(open(path + "/zarr.json"))
    assert "dither" in json.dumps(meta)
    import zarr
    dec = zarr.open_array(path, mode="r")[:]
    assert (_sigma_error(raw, dec, CAM) <= 0.5 + 0.1).all()
    assert not np.array_equal(dec, b4d.decompress(b4d.compress(raw, q=1.0, mode=2, camera=CAM, predictor="above", entropy="rice")))


@pytest.mark.parametrize("pred,dither", [("above3", False), ("above3", True), ("above", True)])
def test_mode1_flat_decodes_without_bias(pred, dither):
    """mode 1 decodes a flat scene to the input's mean: the closed loop's error is spread over one step, so the decoder takes
    off the inverse transform's curvature (conv q^2 / 48 high otherwise), and with dither also the -1/2 fixed-point unit
    mean of the dithered remainder (about +0.2 DN at 7500 DN otherwise)"""
    cam = b4d.Camera(1.3718879, 34.8625, 2.6578)
    rng = np.random.default_rng(3)
    for level in (1000.0, 7500.0):
        e = rng.poisson((level - cam.offset) / cam.conversion, (64, 256, 256)) + rng.normal(0, cam.read_noise, (64, 256, 256))
        img = np.round(cam.offset + cam.conversion * e).clip(0, 65535).astype(np.uint16)
        bl = b4d.compress_tiles(img, (64, 64, 64), q=1.0, mode=1, camera=cam, predictor=pred, entropy="rice", dither=dither)
        dec = np.concatenate([np.frombuffer(b4d.decompress(b), np.uint16) for b in bl]).astype(np.float64)
        bias = dec.mean() - img.mean(dtype=np.float64)
        sd = np.sqrt(((cam.conversion * np.sqrt(level / cam.conversion)) ** 2) / 12 / img.size)   # step in DN ~ sigma
        assert abs(bias) < max(5 * sd, 0.01), (level, bias)


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8])
@pytest.mark.parametrize("q", [0.25, 1.0, 3.0])
def test_mode1_streamed_rows_match_the_stored_path(q, dtype):
    """the Rice coder's streamed mode-1 rows (vectorised for above / above3 after the first row) decode to the same data as the
    stored path the Huffman coder uses, for row widths that are not multiples of the vector width and residuals of every sign"""
    rng = np.random.default_rng(11)
    for w in (61, 200):
        img = (rng.poisson(60, (16, 23, w)) + rng.integers(0, 3, (16, 23, w)) * 15).clip(0, np.iinfo(dtype).max).astype(dtype)
        for pred in ("above", "above3"):
            for d in (False, True):
                kw = dict(q=q, mode=1, camera=CAM, predictor=pred, dither=d)
                a = b4d.decompress(b4d.compress(img, entropy="rice", **kw))
                b = b4d.decompress(b4d.compress(img, entropy="huffman", **kw))
                assert np.array_equal(a, b), (w, pred, d)
                assert q < 1 or not np.array_equal(a, img)     # lossy: neither path fell back to storing the block raw (q < 1: step under 1 DN)

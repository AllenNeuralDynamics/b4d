"""The exact variance-stabilising transform (format v9): Camera(..., exact_vst=True) codes through a curve that is one sigma per
unit for the camera model at every signal, and every chunk carries the curve's description."""
import struct

import numpy as np
import pytest

import b4d

CAM = b4d.Camera(1.3733, 34.0928, 2.6469, exact_vst=False)  # beta3 IMX411: 1.37 DN/e-, offset 34 DN, 2.65 e-
CAMX = b4d.Camera(1.3733, 34.0928, 2.6469, exact_vst=True)            # the variance objective
CAMS = b4d.Camera(1.3733, 34.0928, 2.6469, exact_vst="std")           # the classic one


def _image(shape=(16, 96, 128), seed=7):
    rng = np.random.default_rng(seed)
    lam = np.where(rng.random(shape) < 0.6, 0.0, rng.uniform(0, 80, shape))   # mostly dark, as camera data
    e = rng.poisson(lam) + rng.normal(0, CAM.read_noise, shape)
    return np.round(CAM.offset + CAM.conversion * e).clip(0, 65535).astype(np.uint16)


def _block_header(blob):
    """the B4D block header of a one-piece, one-block blob: B4DP container, Blosc2 header (32), bstart, csize"""
    nd = blob[6]
    at = 7 + 8 * nd + 4 + 4                                      # shape, npieces, piece size
    return blob[at + 32 + 4 + 4:]


def test_the_std_curve_is_one_sigma_per_unit():
    info = b4d.exact_vst(CAMS)
    assert info["max_err"] < 1e-3
    assert np.abs(info["std_exact"] - 1).max() < 1e-3
    assert np.abs(info["std_anscombe"] - 1).max() > 0.05          # what it fixes: Anscombe is 7.5 % off at the offset here
    assert np.all(np.diff(info["table"]) > 0)


def test_anscombe_is_v10_and_exact_is_v11():
    a = _image()
    kw = dict(q=1.0, mode=1, predictor="above3", entropy="rice", blocksize=a.nbytes)
    ha, hx = _block_header(b4d.compress(a, camera=CAM, **kw)), _block_header(b4d.compress(a, camera=CAMX, **kw))
    assert ha[0] == 10 and hx[0] == 11                            # Rice: v10 (8 streams); with the exact VST v11 (v9 + 8 streams)
    nk = hx[44]; lam, = struct.unpack("<f", hx[48:52])
    g = struct.unpack(f"<{nk}f", hx[52:52 + 4 * nk])
    info = b4d.exact_vst(CAMX)
    assert lam == np.float32(info["lam_max"]) and np.array_equal(np.float32(g), np.float32(info["g"]))


@pytest.mark.parametrize("mode,pred,entropy,dither", [(1, "above3", "rice", False), (1, "above3", "ans", False), (1, "left_above", "huffman", False),
                                                      (1, "above3", "rice", True), (2, "above", "rice", False), (2, "above", "rice", True),
                                                      (2, "above", "ans", False)])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
def test_round_trip_keeps_the_bound_in_the_exact_curve(mode, pred, entropy, dither, q):
    a = _image()
    blob = b4d.compress(a, q=q, mode=mode, camera=CAMX, predictor=pred, entropy=entropy, dither=dither, threads=4)
    d = b4d.decompress(blob)
    T = b4d.exact_vst(CAMX)["table"]
    # each pixel within half a step of the exact curve (+ DN rounding on inversion, + the dither's mean correction)
    assert np.abs(T[a] - T[d]).max() <= q / 2 + 0.35
    assert b4d.decompress(b4d.compress(a, q=q, mode=mode, camera=CAMX, predictor=pred, entropy=entropy, dither=dither, threads=1)).tobytes() == d.tobytes()


def test_objectives_are_distinct_and_validated():
    s, v = b4d.exact_vst(CAMS), b4d.exact_vst(CAMX)
    assert v["max_err"] < 1e-3 and not np.array_equal(s["g"], v["g"])
    assert 1.03 < v["std_exact"][0] < 1.10                       # the variance fit is not one sigma per unit at the offset
    with pytest.raises(ValueError):
        b4d.compress(_image(), q=1.0, mode=1, camera=b4d.Camera(1.0, 0.0, 2.0, exact_vst="bogus"), predictor="above3", entropy="rice")


@pytest.mark.parametrize("cam,lo,hi", [(CAMX, 0.075, 0.092), (CAMS, 0.092, 0.104)])
def test_dark_added_variance_per_objective(cam, lo, hi):
    """the variance objective puts a dark pixel's added variance at ~q^2/12 of its own (1/12 = 0.083, plus the predictor's
    share); the std one at ~0.098; Anscombe ~0.110"""
    rng = np.random.default_rng(1)
    a = np.round(CAM.offset + CAM.conversion * rng.normal(0, CAM.read_noise, (32, 256, 256))).astype(np.uint16)
    d = b4d.decompress(b4d.compress(a, q=1.0, mode=1, camera=cam, predictor="above3", entropy="rice", threads=8))
    ex = np.var(d.astype(float)) / np.var(a.astype(float)) - 1
    assert lo < ex < hi, ex


def test_dark_noise_is_kept_closer_to_q2_over_12():
    """the point of it: in the dark the decoded variance exceeds the raw one by about q^2/12 of it, as in the light
    (Anscombe's step there is ~8 % wider than one sigma, so it adds ~11 %)"""
    rng = np.random.default_rng(1)
    a = np.round(CAM.offset + CAM.conversion * rng.normal(0, CAM.read_noise, (32, 256, 256))).astype(np.uint16)
    kw = dict(q=1.0, mode=1, predictor="above3", entropy="rice", threads=8)
    excess = {k: np.var(b4d.decompress(b4d.compress(a, camera=c, **kw)).astype(float)) / np.var(a.astype(float)) - 1 for k, c in (("a", CAM), ("x", CAMX))}
    assert excess["x"] < excess["a"] - 0.005, excess


def test_tiles_match_single_compress():
    a = _image(shape=(16, 64, 64))
    blobs = b4d.compress_tiles(a, (8, 32, 32), q=1.0, mode=1, camera=CAMX, predictor="above3", entropy="rice", threads=2)
    one = b4d.compress(np.ascontiguousarray(a[:8, :32, :32]), q=1.0, mode=1, camera=CAMX, predictor="above3", entropy="rice")
    assert np.array_equal(b4d.decompress(blobs[0]), b4d.decompress(one))


def test_lossless_ignores_it():
    a = _image()
    assert b4d.compress(a, q=0.0, mode=0, camera=CAMX, predictor="above3", entropy="rice") == \
        b4d.compress(a, q=0.0, mode=0, camera=CAM, predictor="above3", entropy="rice")


def test_a_damaged_description_is_refused():
    a = _image()
    blob = bytearray(b4d.compress(a, q=1.0, mode=1, camera=CAMX, predictor="above3", entropy="rice", blocksize=a.nbytes))
    hdr_at = len(blob) - len(_block_header(bytes(blob)))
    blob[hdr_at + 44] = 1                                         # nk = 1: not a curve
    with pytest.raises(Exception):
        b4d.decompress(bytes(blob))


def test_native_zarr_writer(tmp_path):
    zarr = pytest.importorskip("zarr")
    import b4d.zarr_store as zs
    from b4d.zarr3 import B4D
    img = _image(shape=(16, 128, 128))
    path = str(tmp_path / "x.zarr")
    arr = zs.create(path, img.shape, dtype=img.dtype, shards=(16, 128, 128), chunks=(16, 64, 64),
                    codec=B4D(q=1.0, mode=1, camera=CAMX, predictor="above3", entropy="rice"), threads=2)
    arr[0:16].write(img).result(); arr.close()
    back = zarr.open_array(path, mode="r")
    assert back.metadata.to_dict()["codecs"][0]["configuration"]["codecs"][0]["configuration"]["exact_vst"] is True
    ref = np.concatenate([np.concatenate([b4d.decompress(b4d.compress(np.ascontiguousarray(img[:, y:y + 64, x:x + 64]), q=1.0, mode=1, camera=CAMX,
                                                                   predictor="above3", entropy="rice")) for x in (0, 64)], axis=2) for y in (0, 64)], axis=1)
    assert np.array_equal(back[:], ref)


@pytest.mark.parametrize("mode,pred,dither", [(1, "above3", False), (2, "above", False), (2, "above", True), (1, "above3", True)])
def test_cuda_encoder_gives_the_c_bytes(mode, pred, dither):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    shape, chunks = (40, 100, 150), (16, 64, 64)
    blk = _image(shape=shape, seed=3)
    enc = CudaTileEncoder(CAMX, q=1.0, mode=mode, backend="cuda", slots=1, predictor=pred, entropy="rice", check=False, batch_tiles=4, dither=dither)
    order = [(z, y, x) for z in range(0, 40, 16) for y in range(0, 100, 64) for x in range(0, 150, 64)]
    hbuf, off, ln = enc.encode_tiles(blk, (0, 0, 0), order, chunks, shape)
    for k, (z, y, x) in enumerate(order):
        tile = np.zeros(chunks, np.uint16)
        sub = blk[z:z + 16, y:y + 64, x:x + 64]; tile[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        ref = b4d.compress(tile, q=1.0, mode=mode, camera=CAMX, predictor=pred, entropy="rice", dither=dither)
        assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, (k, mode, dither)


def test_an_unattainable_fit_falls_back_to_anscombe():
    """sub-ADU noise (gsprint: 0.03 DN/e-, rn 11 e- = 0.33 ADU): the variance fit is left ~0.75 off and coding through it
    grew the data 30 %, so it is not used: Anscombe bytes, with a warning"""
    rng = np.random.default_rng(3)
    a = np.clip(np.round(2.0 + 0.03 * (rng.poisson(20.0, (8, 64, 64)) + rng.normal(0, 11.0, (8, 64, 64)))), 0, 255).astype(np.uint8)
    cam = b4d.Camera(0.03, 2.0, 11.0, exact_vst=False)
    kw = dict(q=1.0, mode=1, predictor="above3", entropy="rice")
    b4d._VST_WARNED.clear()                                  # warned once per process: an earlier test may have used this model
    with pytest.warns(RuntimeWarning, match="no exact VST"):
        bx = b4d.compress(a, camera=b4d.Camera(0.03, 2.0, 11.0, exact_vst=True), **kw)
    assert bx == b4d.compress(a, camera=cam, **kw)

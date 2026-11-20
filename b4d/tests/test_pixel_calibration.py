"""Per-pixel calibration: noisy pixels get a coarser step, the list travels in the chunk (format v7), decoding needs nothing else."""
import numpy as np
import pytest

import b4d

CAM = b4d.Camera(1 / 0.77, 34.0, 3.48)
C = CAM.read_noise ** 2 + 0.375
H, W = 96, 128


def _sensor(seed=3, noisy=0.04):
    r = np.random.default_rng(seed)
    rn = CAM.read_noise * np.exp(r.normal(-0.2, 0.1, (H, W)))
    tail = r.random((H, W)) < noisy
    rn[tail] *= r.uniform(1.5, 8, tail.sum())
    return rn * CAM.conversion, tail                                     # DN, listed-to-be


def _image(rn_dn, planes=32, electrons=30.0, seed=7, offset=None, shape=None):
    r = np.random.default_rng(seed)
    h, w = shape or rn_dn.shape
    e = r.poisson(electrons, (planes, h, w)) + r.normal(0, 1, (planes, h, w)) * rn_dn[:h, :w] / CAM.conversion
    base = CAM.offset if offset is None else offset[:h, :w]
    return np.rint(base + CAM.conversion * e).clip(0, 65535).astype(np.uint16)


def _t(x):
    e = (np.asarray(x, dtype=float) - CAM.offset) / CAM.conversion
    return np.where(e >= 0, 2 * np.sqrt(np.maximum(e, 0) + C), 2 * np.sqrt(C) + e / np.sqrt(C))


def _kw(q=1.0, entropy="rice"):
    return dict(q=q, mode=1, camera=CAM, predictor="above3", entropy=entropy)


def test_no_listed_pixels_gives_the_same_bytes():
    quiet = np.full((H, W), CAM.read_noise * CAM.conversion * 0.5)
    a = _image(quiet, planes=8)
    cal = b4d.PixelCalibration(quiet, CAM)
    assert cal.listed == 0
    for ent in ("rice", "ans", "huffman"):
        assert b4d.compress(a, calibration=cal, **_kw(entropy=ent)) == b4d.compress(a, **_kw(entropy=ent))


@pytest.mark.parametrize("entropy", ["rice", "ans", "huffman"])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 4.0])
def test_bound_is_q_over_2_of_each_pixels_own_noise(entropy, q):
    rn, _ = _sensor(); a = _image(rn); cal = b4d.PixelCalibration(rn, CAM)
    d = b4d.decompress(b4d.compress(a, calibration=cal, **_kw(q, entropy)))
    err = np.abs(_t(a) - _t(d))
    listed = cal.tier != 0
    assert err[:, ~listed].max() <= q / 2 + 0.2                          # + DN rounding on inversion, as without a map
    # a listed pixel's step is its tier's noise at the predicted level, at most the tier's value at zero signal
    s0 = cal.TIER_RATIO ** 0.5 * cal.min_raise * cal.TIER_RATIO ** (cal.tier[listed].astype(float) - 2)
    assert (err[:, listed].max(0) <= q / 2 * s0 + 0.2).all()
    assert listed.any() and err[:, listed].max() > q / 2 + 0.2           # and the list really was used


def test_ratio_rises_and_quiet_pixels_keep_their_fidelity():
    rn, _ = _sensor(); a = _image(rn, planes=64); cal = b4d.PixelCalibration(rn, CAM)
    listed = cal.tier != 0
    for q in (1.0, 2.0, 4.0):
        p, c = b4d.compress(a, **_kw(q)), b4d.compress(a, calibration=cal, **_kw(q))
        assert len(c) < len(p)
        dp, dc = b4d.decompress(p), b4d.decompress(c)
        mse = lambda d, m: float(((_t(d) - _t(a))[:, m] ** 2).mean())
        assert mse(dc, ~listed) <= mse(dp, ~listed) * 1.02


def test_noise_of_listed_pixels_is_kept_not_removed():
    rn, _ = _sensor(); a = _image(rn, planes=128); cal = b4d.PixelCalibration(rn, CAM)
    listed = cal.tier >= 3
    d = b4d.decompress(b4d.compress(a, calibration=cal, **_kw(1.0)))
    ratio = _t(d)[:, listed].var(0).mean() / _t(a)[:, listed].var(0).mean()
    assert 0.95 < ratio < 1.2


def test_origin_selects_the_pixels():
    rn = np.full((H, W), CAM.read_noise * CAM.conversion * 0.5); rn[40, 50] *= 40
    a = _image(rn, planes=64, shape=(32, 64), electrons=5.0)
    cal = b4d.PixelCalibration(rn, CAM, min_raise=1.5)
    # the same noisy pixel, seen by a window of the sensor
    w = _image(rn[30:62, 20:84], planes=64, electrons=5.0)
    d = b4d.decompress(b4d.compress(w, calibration=cal, origin=(30, 20), **_kw(1.0)))
    err = np.abs(_t(w) - _t(d)).max(0)
    assert err[10, 30] > 1.0 and np.delete(err.ravel(), 10 * 64 + 30).max() <= 0.7
    # a window without it has nothing to list
    assert b4d.compress(a, calibration=cal, origin=(0, 0), **_kw(1.0)) == b4d.compress(a, **_kw(1.0))
    with pytest.raises(ValueError):
        b4d.compress(w, calibration=cal, origin=(70, 0), **_kw(1.0))


def test_a_list_that_costs_too_much_is_dropped():
    rn, _ = _sensor(noisy=0.3); a = _image(rn, planes=1); cal = b4d.PixelCalibration(rn, CAM)
    assert b4d.compress(a, calibration=cal, **_kw(1.0)) == b4d.compress(a, **_kw(1.0))


def test_hot_pixel_offsets_are_restored_and_save_bits():
    r = np.random.default_rng(11); rn = np.full((H, W), CAM.read_noise * CAM.conversion * 0.8); off = CAM.offset + r.normal(0, 0.1, (H, W))
    hot = r.random((H, W)) < 0.005; off[hot] += r.uniform(30, 200, hot.sum())
    a = _image(rn, planes=64, electrons=20.0, offset=off)
    with_off, without = b4d.PixelCalibration(rn, CAM, offset=off), b4d.PixelCalibration(rn, CAM)
    assert (with_off.tier != 0)[hot].mean() > 0.95 and with_off.listed < 0.02
    c = b4d.compress(a, calibration=with_off, **_kw(1.0))
    assert len(c) < 0.98 * len(b4d.compress(a, calibration=without, **_kw(1.0)))
    d = b4d.decompress(c)
    assert np.abs(_t(a) - _t(d)).max() <= 0.5 + 0.2                       # hot pixels stay where they were, within the bound


def test_flatten_decodes_to_the_corrected_image_with_an_ordinary_stream():
    r = np.random.default_rng(12); rn = np.full((H, W), CAM.read_noise * CAM.conversion * 0.8)
    off = CAM.offset + r.normal(0, 1.5, (H, W)); resp = 1 + r.normal(0, 0.02, (H, W))
    x = r.poisson(3000 * resp, (64, H, W)) + r.normal(0, 1, (64, H, W)) * rn / CAM.conversion
    a = np.rint(off + CAM.conversion * x).clip(0, 65535).astype(np.uint16)
    cal = b4d.PixelCalibration(rn, CAM, offset=off, response=resp, flatten=True)
    plain = b4d.compress(a, **_kw(1.0)); flat = b4d.compress(a, calibration=cal, **_kw(1.0))
    assert len(flat) < 0.95 * len(plain)
    assert b4d.decompress(flat).mean(0).std() < 0.5 * b4d.decompress(plain).mean(0).std()


def test_tiles_match_single_compress():
    rn, _ = _sensor(); a = _image(rn, planes=32); cal = b4d.PixelCalibration(rn, CAM)
    blobs = b4d.compress_tiles(a, (32, 48, 64), calibration=cal, **_kw(1.0))
    assert len(blobs) == 4
    assert blobs[3] == b4d.compress(a[:, 48:96, 64:128], calibration=cal, origin=(48, 64), **_kw(1.0))
    sub = b4d.compress_tiles(a[:, 48:, 64:], (32, 48, 64), calibration=cal, origin=(48, 64), **_kw(1.0))
    assert sub[0] == blobs[3]


def test_native_writer_matches_single_compress_and_reads_back(tmp_path):
    import struct

    import zarr

    from b4d import zarr_store
    from b4d.zarr3 import B4D

    rn, _ = _sensor(); a = _image(rn, planes=32); cal = b4d.PixelCalibration(rn, CAM)
    codec = B4D(q=1, mode=1, camera=CAM, predictor="above3", entropy="rice")
    out = {}
    for threads in (1, 4):
        path = str(tmp_path / f"a{threads}.zarr")
        arr = zarr_store.create(path, a.shape, dtype=a.dtype, shards=(32, 96, 128), chunks=(32, 48, 64), codec=codec, threads=threads, calibration=cal)
        arr[0:32].write(a).result(); arr.close()
        out[threads] = open(f"{path}/c/0/0/0", "rb").read()
    assert out[1] == out[4]
    index = out[1][-(4 * 16 + 4):-4]
    off, size = struct.unpack("<QQ", index[48:64])                          # chunk (1, 1)
    assert out[1][off:off + size] == b4d.compress(a[:, 48:96, 64:128], calibration=cal, origin=(48, 64), **_kw(1.0))
    d = zarr.open_array(str(tmp_path / "a1.zarr"), mode="r")[:]
    assert np.array_equal(d[:, 48:96, 64:128], b4d.decompress(b4d.compress(a[:, 48:96, 64:128], calibration=cal, origin=(48, 64), **_kw(1.0))))
    assert np.abs(_t(a) - _t(d))[:, cal.tier == 0].max() <= 0.5 + 0.2


def test_multiscale_lists_level_0_only(tmp_path):
    import zarr

    from b4d import zarr_store
    from b4d.zarr3 import B4D

    rn, _ = _sensor(); a = _image(rn, planes=32); cal = b4d.PixelCalibration(rn, CAM)
    codec = B4D(q=1, mode=1, camera=CAM, predictor="above3", entropy="rice")
    def write(name, **kw):
        ms = zarr_store.create_multiscale(str(tmp_path / name), a.shape, dtype=a.dtype, shards=(32, 96, 128), chunks=(32, 48, 64), codec=codec, levels=1, threads=2, **kw)
        ms[0:32].write(a).result(); ms.close()
    write("with", calibration=cal); write("plain")
    assert open(tmp_path / "with/1/c/0/0/0", "rb").read() == open(tmp_path / "plain/1/c/0/0/0", "rb").read()
    assert len(open(tmp_path / "with/0/c/0/0/0", "rb").read()) < len(open(tmp_path / "plain/0/c/0/0/0", "rb").read())
    d = zarr.open_array(str(tmp_path / "with/0"), mode="r")[:]
    assert np.abs(_t(a) - _t(d))[:, cal.tier == 0].max() <= 0.5 + 0.2


def test_uint8_round_trip():
    rn, _ = _sensor(); a = np.minimum(_image(rn, planes=32), 255).astype(np.uint8); cal = b4d.PixelCalibration(rn, CAM)
    d = b4d.decompress(b4d.compress(a, calibration=cal, **_kw(1.0)))
    assert d.dtype == np.uint8 and np.abs(_t(a) - _t(d))[:, cal.tier == 0].max() <= 0.5 + 0.2


def test_needs_mode_1_above3_and_the_same_camera():
    rn, _ = _sensor(); a = _image(rn, planes=4); cal = b4d.PixelCalibration(rn, CAM)
    with pytest.raises(ValueError):
        b4d.compress(a, q=1, mode=2, camera=CAM, predictor="above3", calibration=cal)
    with pytest.raises(ValueError):
        b4d.compress(a, q=1, mode=1, camera=CAM, predictor="above", calibration=cal)
    with pytest.raises(ValueError):
        b4d.compress(a, q=1, mode=1, camera=b4d.Camera(1.0, 34.0, 3.48), predictor="above3", calibration=cal)


def test_load_from_calibration_file(tmp_path):
    rn, _ = _sensor(); p = tmp_path / "cal.npz"
    np.savez_compressed(p, read_noise_dn=rn.astype(np.float32), offset=np.full(rn.shape, CAM.offset, np.float32), response=np.ones(rn.shape, np.float32))
    cal = b4d.PixelCalibration.load(p, CAM)
    assert 0 < cal.listed < 0.1
    assert b4d.compress(_image(rn, planes=16), calibration=cal, **_kw(1.0))


def test_corrupt_blocks_fail_cleanly():
    rn, _ = _sensor(); a = _image(rn, planes=16, shape=(64, 64)); cal = b4d.PixelCalibration(rn, CAM)
    blob = bytearray(b4d.compress(a, calibration=cal, **_kw(1.0)))
    r = np.random.default_rng(1); start = len(blob) - int(len(blob) * 0.97)
    list_end = min(len(blob), start + 700)
    for _ in range(300):                                                    # the headers and the list; must never crash
        b = bytearray(blob); i = int(r.integers(start, list_end)); b[i] = int(r.integers(0, 256))
        try:
            b4d.decompress(bytes(b))
        except (RuntimeError, ValueError):
            pass


def _list_at(blob, plane_rows, entries):
    """offset of the list inside a block: its header starts with these two numbers"""
    import struct
    i = bytes(blob).find(struct.pack("<II", plane_rows, entries))
    assert i >= 0, "the list is not in the block"
    return i


def _small_case():
    r = np.random.default_rng(5); h = w = 64
    rn = np.full((h, w), CAM.read_noise * CAM.conversion * 0.8)
    rn[r.random((h, w)) < 0.05] *= 5
    a = np.rint(CAM.offset + CAM.conversion * (r.poisson(30.0, (64, h, w)) + r.normal(0, 1, (64, h, w)) * rn / CAM.conversion)).clip(0, 65535).astype(np.uint16)
    off = np.full((h, w), CAM.offset); hot = r.random((h, w)) < 0.01; off[hot] += 60
    cal = b4d.PixelCalibration(rn, CAM, offset=off)
    entries = int((cal.tier > 0).sum())
    blob = bytes(b4d.compress(a, calibration=cal, **_kw(1.0)))
    return a, cal, blob, _list_at(blob, h, entries), entries


def test_step_follows_the_noise_formula():
    """the bound test alone passes with too fine a step: the worst error must also reach it, at a known signal level"""
    rn = np.full((H, W), CAM.read_noise * CAM.conversion * 0.8)
    noisy = np.zeros((H, W), bool); noisy[::8, ::8] = True                  # apart, so each one's prediction comes from quiet pixels
    extra, level = 400.0, 50.0                                              # e-^2 above the camera's, electrons
    rn[noisy] = np.sqrt(C - 0.375 + extra) * CAM.conversion
    cal = b4d.PixelCalibration(rn, CAM, skip=0)
    a = _image(rn, planes=64, electrons=level)
    d = b4d.decompress(b4d.compress(a, calibration=cal, **_kw(1.0)))
    worst = np.abs(_t(d) - _t(a))[1:, noisy].max()
    expected = 0.5 * np.sqrt(1 + extra / (level + C))                       # the tier's middle and the prediction's spread move it a little
    assert 0.9 * expected < worst < 1.2 * expected


def test_listed_residual_overflow_is_not_written():
    """a huge T range (tiny conversion) and q = 0.25 send one listed pixel past 16 bits; the block must not wrap"""
    cam = b4d.Camera(0.0005, 34.0, 3.48)
    rn = np.full((H, W), cam.read_noise * cam.conversion * 0.5); rn[10, 20] = 6.0 * cam.conversion
    a = np.full((8, H, W), 40, np.uint16); a[1::2, 10, 20] = 60000
    cal = b4d.PixelCalibration(rn, cam, skip=0)
    assert (cal.tier > 0).sum() == 1
    d = b4d.decompress(b4d.compress(a, calibration=cal, q=0.25, mode=1, camera=cam, predictor="above3", entropy="huffman"))
    assert abs(int(d[1, 10, 20]) - 60000) < 200


def test_corrupt_list_fails_cleanly_everywhere():
    a, cal, blob, at, entries = _small_case()
    good = np.asarray(b4d.decompress(blob))
    end = at + 20 + 4 * blob[at + 16] + entries
    r = np.random.default_rng(1); rejected = changed = 0
    for _ in range(3000):
        b = bytearray(blob)
        for _ in range(int(r.integers(1, 4))):
            b[int(r.integers(at, end))] = int(r.integers(0, 256))
        try:
            changed += not np.array_equal(np.asarray(b4d.decompress(bytes(b))), good)
        except (RuntimeError, ValueError):
            rejected += 1
    assert rejected > 300 and changed > 300                                 # the damage reached the list


@pytest.mark.parametrize("value", [1e30, -1.0, float("nan"), float("inf")])
def test_absurd_noise_value_is_rejected(value):
    import struct
    a, cal, blob, at, entries = _small_case()
    b = bytearray(blob); struct.pack_into("<f", b, at + 20, value)
    with pytest.raises((RuntimeError, ValueError)):
        b4d.decompress(bytes(b))


def test_list_position_past_the_block_is_rejected():
    a, cal, blob, at, entries = _small_case()
    first = at + 20 + 4 * blob[at + 16]
    b = bytearray(blob)
    b[first] = (31 << 3) | (blob[first] & 7)                                # a long gap ...
    b[first + 1:first + 5] = b"\xff\xff\xff\x7f"                            # ... far past the end of the plane
    with pytest.raises((RuntimeError, ValueError)):
        b4d.decompress(bytes(b))

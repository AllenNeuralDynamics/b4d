"""Bit-exact regression tests for the single-core speed work (AVX2 Rice emit, position/table Rice reader,
gathered exact-VST decode table, gathered mode-1 LUT, per-thread compress_tiles buffers). None of it may change a byte:

* GOLDEN: sha256 of the compressed bytes and of the decoded array for fixed inputs. A deliberate format change has to
  update these (and say so): format v10 / v11 (8 Rice streams, 2026-11-15) changed every compressed hash and no decoded one;
  GOLDEN_V9 keeps the hashes of the 4-stream build before it.
* data/golden_*.b4d: blobs written by that older build; the current decoder must reproduce the stored decode hash,
  independently of what the current encoder writes.

Inputs come from an integer hash (not numpy's Generator), so they do not change with the numpy version.
Cases target the changed code: escapes / long codes (spikes), high k (16-bit noise, lossless), zero groups (flat
areas), row lengths that are not a multiple of 8 (gather tails), uint8, exact VST on/off, contiguous and strided tiles,
repeated compress_tiles calls (buffer reuse) and several threads.
"""
import hashlib
import os

import numpy as np
import pytest

import b4d

DATA = os.path.join(os.path.dirname(__file__), "data")
CAM = b4d.Camera(conversion=0.44, offset=100.0, read_noise=2.1, exact_vst=False)   # golden bytes recorded with the Anscombe transform
CAMX = b4d.Camera(conversion=0.44, offset=100.0, read_noise=2.1, exact_vst="variance")


def hashed(shape, seed):
    """uint64 hash of the flat index (splitmix64 finaliser), reproducible everywhere."""
    with np.errstate(over="ignore"):                               # uint64 wraparound is intended
        x = np.arange(int(np.prod(shape)), dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x.reshape(shape)


def image(kind, shape, dtype=np.uint16, seed=1):
    h = hashed(shape, seed)
    hi = np.iinfo(dtype).max
    if kind == "noise":                                            # 13-bit noise: k ~ 12-13 (full 16-bit would be stored raw)
        return (h % np.uint64(min(hi + 1, 1 << 13))).astype(dtype)
    zz, yy, xx = np.indices(shape, dtype=np.float64)
    level = 300.0 + 2000.0 * np.exp(-((yy - shape[1] / 2) ** 2 + (xx - shape[2] / 3) ** 2) / (0.1 * shape[1] * shape[2] + 1))
    u = (h >> np.uint64(11)).astype(np.float64) / 2.0 ** 53       # uniform [0, 1)
    v = ((h * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    g = np.sqrt(-2.0 * np.log(np.maximum(u, 1e-300))) * np.cos(2.0 * np.pi * v)   # Box-Muller normal
    a = np.round(100.0 + level * (1 + 0.1 * zz / shape[0]) + g * np.sqrt(0.44 * level + 4.4))
    if kind == "spikes":                                           # escapes / codes longer than the table
        a[(h % np.uint64(997)) == 0] = hi
    elif kind == "flat":                                           # zero groups
        a[..., : shape[2] // 2] = 100
    if dtype == np.uint8:
        a = a / 16
    return np.clip(a, 0, hi).astype(dtype)


CASES = {
    # name: (image kind, shape, dtype, tile or None, keyword arguments)
    "m1_a3_xv_tiles64": ("smooth", (128, 64, 64), np.uint16, (64, 64, 64), dict(mode=1, predictor="above3", q=1.0, camera=CAMX)),
    "m1_a3_xv_strided": ("smooth", (32, 96, 160), np.uint16, (16, 48, 64), dict(mode=1, predictor="above3", q=1.0, camera=CAMX)),
    "m1_a3_spikes": ("spikes", (8, 100, 255), np.uint16, None, dict(mode=1, predictor="above3", q=1.0, camera=CAM)),
    "m1_a3_xv_spikes": ("spikes", (8, 100, 255), np.uint16, None, dict(mode=1, predictor="above3", q=2.0, camera=CAMX)),
    "m1_a3_flat": ("flat", (4, 64, 128), np.uint16, None, dict(mode=1, predictor="above3", q=1.0, camera=CAMX)),
    "m1_odd_widths": ("smooth", (3, 17, 17), np.uint16, None, dict(mode=1, predictor="above3", q=0.5, camera=CAMX)),
    "m1_width1": ("smooth", (5, 9, 1), np.uint16, None, dict(mode=1, predictor="above3", q=1.0, camera=CAMX)),
    "m1_left_above": ("smooth", (4, 64, 100), np.uint16, None, dict(mode=1, predictor="left_above", q=1.0, camera=CAMX)),
    "m1_lossless_q0": ("smooth", (4, 64, 100), np.uint16, None, dict(mode=1, predictor="above3", q=0.0, camera=CAMX)),
    "m0_noise13": ("noise", (2, 64, 129), np.uint16, None, dict(mode=0, predictor="above3", q=0.0, camera=CAM)),
    "m0_spikes": ("spikes", (4, 64, 128), np.uint16, None, dict(mode=0, predictor="above3", q=0.0, camera=CAM)),
    "m1_uint8": ("smooth", (4, 64, 77), np.uint8, None, dict(mode=1, predictor="above3", q=1.0, camera=CAM)),
    "m2_a3": ("smooth", (4, 64, 128), np.uint16, None, dict(mode=2, predictor="above3", q=1.0, camera=CAMX)),
}

GOLDEN = {   # (compressed bytes, decoded array) sha256[:16]: format v10 / v11 (8 Rice streams, 2026-11-15; decoded arrays as in v9)
    "m1_a3_xv_tiles64": ('d07414fabdf0d877', '54cb524c70ec805b'),
    "m1_a3_xv_strided": ('53940f8c1089ef01', '9511abf50d3ee59c'),
    "m1_a3_spikes": ('ee652ee310e2fa9e', '489895a601856028'),
    "m1_a3_xv_spikes": ('0e52bf0f2e44734b', '4733a3703c586612'),
    "m1_a3_flat": ('283fd06fb2dd5d01', '0da11535906aa54f'),
    "m1_odd_widths": ('37cdb75bcfde2b1d', '527a33b5df57b222'),
    "m1_width1": ('cfb406d5a02be8f2', '568f9b5b28a3d8b0'),
    "m1_left_above": ('e2fdac75d4061050', 'ee9733f8498da5f9'),
    "m1_lossless_q0": ('aa0028d27803c1ab', '0f20e33bb9139f92'),
    "m0_noise13": ('3a3e00a73828f1e3', 'c284b6c0d5a1ad1d'),
    "m0_spikes": ('ed1d1431777e52ba', 'a7d648bb215da5b9'),
    "m1_uint8": ('71ef7408db8963e3', '5b444d68c8c5d61b'),
    "m2_a3": ('1a805fba44c61ef2', 'a3faf3ddb0193229'),
}

GOLDEN_V9 = {   # the same, written by the 4-stream build before format v10 (2026-11-15); data/golden_*.b4d are its blobs
    "m1_a3_xv_tiles64": ('48f7af1946534688', '54cb524c70ec805b'),
    "m1_a3_xv_strided": ('3fcf10ad9035af7e', '9511abf50d3ee59c'),
    "m1_a3_spikes": ('1c0026ee657f553b', '489895a601856028'),
    "m1_a3_xv_spikes": ('c529d28dfd908721', '4733a3703c586612'),
    "m1_a3_flat": ('c4c0940c3065530e', '0da11535906aa54f'),
    "m1_odd_widths": ('adea8c9483d1ad5f', '527a33b5df57b222'),
    "m1_width1": ('cfb406d5a02be8f2', '568f9b5b28a3d8b0'),
    "m1_left_above": ('863ddc9d84674244', 'ee9733f8498da5f9'),
    "m1_lossless_q0": ('3b480df092e128e4', '0f20e33bb9139f92'),
    "m0_noise13": ('eadbc8df444a7878', 'c284b6c0d5a1ad1d'),
    "m0_spikes": ('d142aadbea0b965e', 'a7d648bb215da5b9'),
    "m1_uint8": ('3c7e176e9884f8f4', '5b444d68c8c5d61b'),
    "m2_a3": ('b3b3d4d7643e69e0', 'a3faf3ddb0193229'),
}

BLOBS = ["m1_a3_xv_spikes", "m1_a3_flat", "m0_noise13"]           # stored under data/golden_<name>.b4d


def encode(name, threads=1):
    kind, shape, dtype, tile, kw = CASES[name]
    a = image(kind, shape, dtype)
    if tile is None:
        return a, [b4d.compress(a, threads=threads, entropy="rice", **kw)]
    return a, b4d.compress_tiles(a, tile, threads=threads, entropy="rice", **kw)


def digest(blobs):
    hb, hd = hashlib.sha256(), hashlib.sha256()
    for b in blobs:
        hb.update(b)
        hd.update(np.ascontiguousarray(b4d.decompress(b)).tobytes())
    return hb.hexdigest()[:16], hd.hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bytes_and_decode(name):
    a, blobs = encode(name)
    assert digest(blobs) == GOLDEN[name]


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c[4]["q"] == 0.0 and c[3] is None))
def test_lossless_cases_roundtrip_exactly(name):
    a, blobs = encode(name)
    np.testing.assert_array_equal(b4d.decompress(blobs[0]), a)


@pytest.mark.parametrize("name", ["m1_a3_xv_tiles64", "m1_a3_xv_strided"])
def test_compress_tiles_repeat_threads_and_single_calls_agree(name):
    """per-thread buffers kept between calls must not leak state; threads and per-tile compress() give the same bytes"""
    kind, shape, dtype, tile, kw = CASES[name]
    a, first = encode(name)
    _, again = encode(name)
    _, threaded = encode(name, threads=4)
    assert again == first and threaded == first
    singles = [b4d.compress(np.ascontiguousarray(a[z:z + tile[0], y:y + tile[1], x:x + tile[2]]), threads=1, entropy="rice", **kw)
               for z in range(0, shape[0], tile[0]) for y in range(0, shape[1], tile[1]) for x in range(0, shape[2], tile[2])]
    assert singles == first
    other = b4d.compress_tiles(image("spikes", shape, dtype, seed=7), tile, threads=1, entropy="rice", **kw)   # different data in between
    assert encode(name)[1] == first and other != first


@pytest.mark.parametrize("name", BLOBS)
def test_decode_stored_blob_from_older_build(name):
    with open(os.path.join(DATA, f"golden_{name}.b4d"), "rb") as f:
        blob = f.read()
    assert hashlib.sha256(np.ascontiguousarray(b4d.decompress(blob)).tobytes()).hexdigest()[:16] == GOLDEN_V9[name][1]
    assert GOLDEN_V9[name][1] == GOLDEN[name][1]                     # v10 / v11 decode to the same arrays

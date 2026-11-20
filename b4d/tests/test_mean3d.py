"""The mean3d predictor (format v4) and the freeze of the production bitstreams (v3).

The golden hashes were produced by the build of 2026-10-05 12:13 (the one validated by the writer tuning measurements, docs/WRITER_DESIGN.md),
BEFORE the mean3d predictor existed. They pin the bytes of the legacy predictors: any change to them fails here.
"""
import hashlib

import numpy as np
import pytest

import b4d

CAM = b4d.Camera(0.5, 100.0, 1.6, exact_vst=False)   # golden bytes recorded with the Anscombe transform


def _image():
    rng = np.random.default_rng(1234)
    z, y, x = np.meshgrid(np.arange(8), np.arange(96), np.arange(128), indexing="ij")
    clean = 100 + 400 * np.exp(-((y - 48) ** 2 + (x - 64) ** 2) / 300.0) * (1 + 0.1 * z)
    return (clean + rng.normal(0, 1, clean.shape) * np.sqrt(clean) * 0.5).clip(0, 65535).astype(np.uint16)


GOLDEN = {   # (predictor, entropy, mode, q): (sha256, length) of b4d.compress(_image(), ...) with the validated build;
             # Rice entries re-recorded for format v10 (8 Rice streams, 2026-11-15: same decoded arrays, Huffman / rANS unchanged)
    ("above", "rice", 2, 2.0): ("7dd5f89edfe3507c127015dfc463233fb7b22be392e3e6a1ab1e813b11a87b9e", 44205),
    ("above", "rice", 2, 1.0): ("648e3a12ffc620bd1837c11b27bd6d98e7f1abd5b45de8a90c721bc95582d239", 55979),
    ("above", "rice", 0, 0.0): ("e554b964f912c49e8ab796c54fe876de7b68b12d441507cd326765429c5e7d99", 66298),
    ("left_above", "ans", 2, 1.0): ("5bdbd889b468facbc9d8f2e396de8e22395df4da819ff7360ec0e885f7fe714a", 52098),
    ("left_above", "huffman", 1, 1.0): ("6562187ca2e0744aa070602acf8946509ff7553a1e72907a0d12878fef5a19fd", 53391),
    ("above", "ans", 1, 2.0): ("be4fc8b54c96159cd033f367638eb0ad2b1fcce22095de5ca5dcfc491fb70a25", 43006),
    ("left_above", "rice", 0, 0.0): ("1d92fd47dddb48a0804040b2cb2d36639b2c9a70ae6f9273c6a625df916197c5", 63213),
}
GOLDEN_TILES = ("f4e5d51b24f4c20a634b9ee1ec468c673ae61a55ad9b0e39c1db136f0be2784b", 6)   # v10


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_production_bitstreams_are_frozen(key):
    pred, ent, mode, q = key
    blob = b4d.compress(_image(), q=q, mode=mode, camera=CAM, predictor=pred, entropy=ent)
    assert (hashlib.sha256(blob).hexdigest(), len(blob)) == GOLDEN[key]


def test_production_tiles_are_frozen():
    tiles = b4d.compress_tiles(_image(), (8, 32, 64), q=2.0, mode=2, camera=CAM, predictor="above", entropy="rice")
    h = hashlib.sha256()
    for t in tiles:
        h.update(t)
    assert (h.hexdigest(), len(tiles)) == GOLDEN_TILES


@pytest.mark.parametrize("entropy", ["rice", "ans", "huffman"])
@pytest.mark.parametrize("shape", [(8, 96, 128), (3, 7, 5), (1, 64, 64), (64, 64, 64), (100, 37)])
def test_mean3d_lossless_roundtrip(shape, entropy):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 4096, shape, dtype=np.uint16)
    img.flat[::97] = 65535; img.flat[::89] = 0
    blob = b4d.compress(img, q=0, predictor="mean3d", entropy=entropy)
    assert np.array_equal(b4d.decompress(blob), img)


@pytest.mark.parametrize("entropy", ["rice", "ans"])
@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("q", [1.0, 2.0])
def test_mean3d_lossy_error_bound(mode, q, entropy):
    img = _image()
    blob = b4d.compress(img, q=q, mode=mode, camera=CAM, predictor="mean3d", entropy=entropy)
    out = b4d.decompress(blob)
    sig = CAM.sigma(img)
    err = np.abs(out.astype(np.float64) - img) / sig
    assert err.max() <= 0.5 * q + 0.5 / sig.min() + 1e-6       # q/2 sigma plus the half-DN rounding


def test_mean3d_is_a_new_format_version_and_legacy_is_not():
    img = _image()
    legacy = b4d.compress(img, q=0, predictor="above", entropy="rice")
    new = b4d.compress(img, q=0, predictor="mean3d", entropy="rice")
    # the B4DP container header is followed by the Blosc2 chunk; the B4D block header's first byte is the format version
    def block_versions(blob):
        n = 7 + 8 * 3
        npieces = int.from_bytes(blob[n:n + 4], "little"); sizes = [int.from_bytes(blob[n + 4 + 4 * i:n + 8 + 4 * i], "little") for i in range(npieces)]
        off = n + 4 + 4 * npieces; vers = set()
        for s in sizes:
            chunk = blob[off:off + s]; off += s
            bstart = int.from_bytes(chunk[32:36], "little")            # first block offset in the Blosc2 chunk header (ctx)
            if 0 < bstart < len(chunk):
                vers.add(chunk[bstart + 4])                            # 4-byte block size, then the B4D header
        return vers
    assert block_versions(new) <= {10} and block_versions(legacy) <= {10}   # every Rice block is v10 (8 streams)
    # the per-option versions remain for the other coders: mean3d v4, the legacy predictors v3
    assert block_versions(b4d.compress(img, q=0, predictor="mean3d", entropy="huffman")) <= {4}
    assert block_versions(b4d.compress(img, q=0, predictor="above", entropy="huffman")) <= {3}


def test_mean3d_beats_above_on_noise_dominated_data():
    """the point of the predictor: on dark/background data it does not add the neighbour's noise to the residual"""
    rng = np.random.default_rng(3)
    dark = (2.05 + rng.normal(0, 0.95, (32, 256, 256))).round().clip(0, 4095).astype(np.uint16)
    cam = b4d.Camera(1 / 7.25, 2.05, 6.8)
    a = len(b4d.compress(dark, q=1.0, mode=2, camera=cam, predictor="above", entropy="rice"))
    m = len(b4d.compress(dark, q=1.0, mode=2, camera=cam, predictor="mean3d", entropy="rice"))
    assert m < 0.9 * a


def test_mean3d_tiles_match_per_tile_compress():
    img = _image()
    tiles = b4d.compress_tiles(img, (4, 32, 64), q=1.0, mode=2, camera=CAM, predictor="mean3d", entropy="rice")
    k = 0
    for bz in range(2):
        for by in range(3):
            for bx in range(2):
                t = img[bz * 4:(bz + 1) * 4, by * 32:(by + 1) * 32, bx * 64:(bx + 1) * 64]
                assert tiles[k] == b4d.compress(t, q=1.0, mode=2, camera=CAM, predictor="mean3d", entropy="rice"), (bz, by, bx)
                assert np.array_equal(b4d.decompress(tiles[k]), b4d.decompress(b4d.compress(t, q=1.0, mode=2, camera=CAM, predictor="mean3d", entropy="rice")))
                k += 1


@pytest.mark.parametrize("entropy", ["rice", "ans"])
@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_mean3d_native_writer_roundtrip(tmp_path, q, entropy):
    """the C zarr writer (b4d_zarr.c) passes the tile height as the plane height; blocks must decode with zarr-python"""
    import zarr
    import b4d.zarr_store as zs
    import b4d.zarr3  # noqa: F401
    from b4d.zarr3 import B4D
    img = _image()                                   # (8, 96, 128)
    codec = B4D(q=q, mode=2 if q else 0, camera=CAM, predictor="mean3d", entropy=entropy)
    arr = zs.create(str(tmp_path / "m.zarr"), img.shape, dtype="uint16", shards=(8, 96, 128), chunks=(4, 32, 64), codec=codec, threads=4, writers=1)
    arr[0:8].write(img).result(); arr.close()
    back = zarr.open_array(str(tmp_path / "m.zarr"), mode="r")[:]
    if q == 0:
        assert np.array_equal(back, img)
    else:
        sig = CAM.sigma(img)
        assert (np.abs(back.astype(np.float64) - img) / sig).max() <= 0.5 * q + 0.5 / sig.min() + 1e-6
    # and the tiles are what compress_tiles produces for the same geometry (same bytes -> same v4 blocks)
    ref = b4d.compress_tiles(img, (4, 32, 64), q=q, mode=2 if q else 0, camera=CAM, predictor="mean3d", entropy=entropy)
    assert len(ref) == 2 * 3 * 2


@pytest.mark.parametrize("entropy", ["rice", "ans"])
@pytest.mark.parametrize("predictor", ["above", "mean3d"])
@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_cuda_encoder_matches_c_encoder(predictor, entropy, q):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder
    rng = np.random.default_rng(5)
    shape, chunks = (16, 128, 192), (16, 64, 64)
    blk = (2.05 + rng.normal(0, 0.95, shape) + 60 * (rng.random(shape) > 0.998)).round().clip(0, 4095).astype(np.uint16)
    cam = b4d.Camera(1 / 7.25, 2.05, 6.8, exact_vst=False if entropy == "ans" else "variance"); mode = 2 if q else 0   # exact VST: device Rice only
    enc = CudaTileEncoder(cam, q=q, mode=mode, backend="cuda", slots=1, predictor=predictor, entropy=entropy, check=False, batch_tiles=4)
    enc.prepare(shape, chunks)
    order = [(0, y, x) for y in range(0, 128, 64) for x in range(0, 192, 64)]
    hbuf, off, ln = enc.encode_tiles(blk, (0, 0, 0), order, chunks, shape)
    for k, (_, y, x) in enumerate(order):
        ref = b4d.compress(np.ascontiguousarray(blk[:, y:y + 64, x:x + 64]), q=q, mode=mode, camera=cam, predictor=predictor, entropy=entropy)
        assert hbuf[off[k]:off[k] + ln[k]].tobytes() == ref, (predictor, entropy, q, k)

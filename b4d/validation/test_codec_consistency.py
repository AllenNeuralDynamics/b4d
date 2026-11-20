"""Validation of the four codec options against the frozen reference bitstreams.

    python -m pytest validation -q                 # from b4d/
    B4D_UPDATE_GOLDEN=1 python -m pytest validation -q -k frozen    # re-record golden_bitstreams.json (only after an
                                                                    # intentional, documented format change)

The four options are predictor x entropy coder: above/rice (the production codec, format v3), above/ans,
mean3d/rice and mean3d/ans (format v4). For every option, mode 0 (lossless) and modes 1 and 2 at q = 1 and 2, on a
camera-model phantom (IMX811 noise model: Poisson + 6.89 e- read noise, offset 2.05 ADU, gain 7.25 e-/ADU) this file
checks that

1. the C encoder's bytes are identical to the recorded reference (sha256 + length in golden_bitstreams.json);
   the reference was recorded with the build whose above/rice output is pinned by tests/test_mean3d.py to the
   build of 2026-10-05 validated by the writer tuning measurements, so every option is anchored to the validated production bitstream;
2. the decoder inverts them: exactly for mode 0, within q/2 sigma (+ the half-ADU rounding) for modes 1 and 2;
3. the tile path used by the native writer (compress_tiles) produces the same bytes as whole-tile compress();
4. the native sharded-zarr writer stores exactly those tiles (file size = sum of tiles + shard index) and the
   stored data decode with zarr-python to the same array as the C decoder;
5. the CUDA encoder (if a GPU is present) produces the same bytes as the C encoder for all four options.

A failure in 1 with 2-5 passing means the bitstream changed (a format change: re-record the golden file and bump
the format version); a failure in 2-5 means an implementation bug.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import b4d

HERE = Path(__file__).resolve().parent
GOLDEN_PATH = HERE / "golden_bitstreams.json"
OPTIONS = [("above", "rice"), ("above", "ans"), ("mean3d", "rice"), ("mean3d", "ans"), ("b3d", "rlhuff")]
ENT_ARG = {"rlhuff": "rice"}  # the original pipeline has its own entropy coder; the entropy argument is ignored


def original_bound_ok(img, out, q) -> bool:
    """The original codec's guarantee, in its own VST units T(x) = 2 sqrt(e + rn^2) - 2 rn with e clamped at 0:
    the reconstruction lies within q/2 of T(x) and the stored integer is round(T^-1(.)), so [T(x) - q/2, T(x) + q/2]
    must meet [T(x_hat - 1/2), T(x_hat + 1/2)]."""

    def T(x):
        e = np.maximum(x - CAM.offset, 0) / CAM.conversion
        return 2 * np.sqrt(e + CAM.read_noise**2) - 2 * CAM.read_noise

    x = img.astype(np.float64)
    xh = out.astype(np.float64)
    return bool(np.all(T(xh - 0.5) <= T(x) + 0.5 * q + 1e-3) and np.all(T(xh + 0.5) >= T(x) - 0.5 * q - 1e-3))


SETTINGS = [(0, 0.0), (1, 1.0), (2, 1.0), (2, 2.0)]  # (mode, q)
CAM = b4d.Camera(1 / 7.25, 2.05, 6.89, exact_vst=False)   # golden bytes recorded with the Anscombe transform
SHAPE = (16, 192, 256)
TILE = (8, 64, 64)


def phantom(shape=SHAPE, peak_e=400.0, seed=2026) -> np.ndarray:
    """spheres on a dark background observed through the camera model (same construction as the tuning-measurement phantom)"""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    clean = np.full(shape, 5.0)
    for _ in range(40):
        cz, cy, cx = rng.uniform(0, nz), rng.uniform(0, ny), rng.uniform(0, nx)
        r = rng.uniform(2, 12)
        clean += rng.uniform(0.2, 1.0) * peak_e * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))
    e = rng.poisson(clean) + rng.normal(0, 6.89, shape)
    return np.clip(np.rint(2.05 + e / 7.25), 0, 4095).astype(np.uint16)


IMG = phantom()


def key(pred, ent, mode, q) -> str:
    return f"{pred}/{ent}/mode{mode}/q{q:g}"


def digest(blob: bytes):
    return [hashlib.sha256(blob).hexdigest(), len(blob)]


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}


def error_bound_ok(img, out, q) -> bool:
    if q == 0:
        return np.array_equal(out, img)
    sig = CAM.sigma(img)
    return float((np.abs(out.astype(np.float64) - img) / sig).max()) <= 0.5 * q + 0.5 / float(sig.min()) + 1e-6


CASES = [(p, e, m, q) for p, e in OPTIONS for m, q in SETTINGS]


@pytest.mark.parametrize("pred,ent,mode,q", CASES, ids=[key(*c) for c in CASES])
def test_c_encoder_bitstream_is_frozen(pred, ent, mode, q):
    blob = b4d.compress(IMG, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent))
    k = key(pred, ent, mode, q)
    golden = load_golden()
    if os.environ.get("B4D_UPDATE_GOLDEN"):
        golden[k] = digest(blob)
        golden["_image"] = dict(
            shape=list(SHAPE),
            sha256=hashlib.sha256(IMG.tobytes()).hexdigest(),
            build=str(getattr(b4d._b4d, "COMPILER", "?")),
        )
        GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True))
        pytest.skip("golden updated")
    assert k in golden, f"{k} not in {GOLDEN_PATH.name}; record it with B4D_UPDATE_GOLDEN=1 after reviewing the change"
    assert digest(blob) == golden[k], f"bitstream changed for {k}"


def test_phantom_is_the_golden_phantom():
    g = load_golden()
    if "_image" in g:
        assert (
            g["_image"]["sha256"] == hashlib.sha256(IMG.tobytes()).hexdigest()
        ), "the validation phantom changed; golden hashes are void"


@pytest.mark.parametrize("pred,ent,mode,q", CASES, ids=[key(*c) for c in CASES])
def test_decoder_inverts_within_bound(pred, ent, mode, q):
    blob = b4d.compress(IMG, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent))
    out = b4d.decompress(blob)
    assert out.shape == IMG.shape and out.dtype == IMG.dtype
    if pred == "b3d":
        assert q == 0 and np.array_equal(out, IMG) or (q > 0 and original_bound_ok(IMG, out, q))
    else:
        assert error_bound_ok(IMG, out, q)


@pytest.mark.parametrize("pred,ent", OPTIONS, ids=[f"{p}/{e}" for p, e in OPTIONS])
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_tile_path_equals_whole_tile_compress(pred, ent, q):
    mode = 2 if q else 0
    tiles = b4d.compress_tiles(IMG, TILE, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent))
    nz, ny, nx = (s // t for s, t in zip(SHAPE, TILE))
    assert len(tiles) == nz * ny * nx
    k = 0
    for bz in range(nz):
        for by in range(ny):
            for bx in range(nx):
                t = np.ascontiguousarray(
                    IMG[
                        bz * TILE[0] : (bz + 1) * TILE[0],
                        by * TILE[1] : (by + 1) * TILE[1],
                        bx * TILE[2] : (bx + 1) * TILE[2],
                    ]
                )
                assert tiles[k] == b4d.compress(
                    t, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent)
                ), (
                    pred,
                    ent,
                    q,
                    bz,
                    by,
                    bx,
                )
                k += 1


@pytest.mark.parametrize("pred,ent", OPTIONS, ids=[f"{p}/{e}" for p, e in OPTIONS])
@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_native_writer_stores_exactly_the_c_encoder_tiles(tmp_path, pred, ent, q):
    zarr = pytest.importorskip("zarr")
    import b4d.zarr3  # noqa: F401
    import b4d.zarr_store as zs
    from b4d.zarr3 import B4D

    mode = 2 if q else 0
    path = tmp_path / "v.zarr"
    arr = zs.create(
        str(path),
        SHAPE,
        dtype="uint16",
        shards=(16, 192, 256),
        chunks=TILE,
        codec=B4D(q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent)),
        threads=4,
        writers=1,
    )
    written = arr[0:16].write(IMG).result()
    arr.close()
    tiles = b4d.compress_tiles(IMG, TILE, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent))
    ntiles = len(tiles)
    assert written == sum(len(t) for t in tiles) + 16 * ntiles + 4, "shard = tiles + (offset, length) index + crc32c"
    shard = next(p for p in path.rglob("*") if p.is_file() and p.name not in ("zarr.json",))
    data = shard.read_bytes()
    assert len(data) == written
    for t in tiles:  # every tile is present verbatim in the shard file
        assert t in data
    back = zarr.open_array(str(path), mode="r")[:]
    assert np.array_equal(
        back,
        b4d.decompress(b4d.compress(IMG, q=q, mode=mode, camera=CAM, predictor=pred, entropy=ENT_ARG.get(ent, ent))),
    )


@pytest.mark.parametrize("pred,ent", OPTIONS, ids=[f"{p}/{e}" for p, e in OPTIONS])
@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_cuda_encoder_matches_c_encoder(pred, ent, q):
    if pred == "b3d":
        pytest.skip("the original pipeline runs on the C encoder only")
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from b4d.cuda_encoder import CudaTileEncoder

    mode = 2 if q else 0
    chunks = (16, 64, 64)
    enc = CudaTileEncoder(
        CAM, q=q, mode=mode, backend="cuda", slots=1, predictor=pred, entropy=ent, check=False, batch_tiles=4
    )
    enc.prepare(SHAPE, chunks)
    order = [(0, y, x) for y in range(0, SHAPE[1], 64) for x in range(0, SHAPE[2], 64)]
    hbuf, off, ln = enc.encode_tiles(IMG, (0, 0, 0), order, chunks, SHAPE)
    for k, (_, y, x) in enumerate(order):
        ref = b4d.compress(
            np.ascontiguousarray(IMG[:, y : y + 64, x : x + 64]),
            q=q,
            mode=mode,
            camera=CAM,
            predictor=pred,
            entropy=ent,
        )
        assert hbuf[off[k] : off[k] + ln[k]].tobytes() == ref, (pred, ent, q, k)

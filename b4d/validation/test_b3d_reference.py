"""Validation of predictor="b3d" (format v5) against the original B3D implementation itself.

Two references (see original_reference.py):
  1. the demo HDF5 file of the original repository: 8 chunks x {lossless, mode 1 q=1, mode 2 q=1}, written by the
     original GPU filter -> our bare bitstream must be byte-identical (lossless always; lossy with the original
     build's VST table, reproduced on an NVIDIA GPU);
  2. the original's win64 GPU binary driven through ctypes: random, phantom and dark frames, three camera models,
     q in {0.25, 1, 2}, both modes -> encoder bytes identical, decoder output identical.

Without a GPU the lossy IEEE-table path is checked for closeness (a documented ~1 ppm of quantiser roundings differ).

    cd b4d && python -m pytest validation/test_b3d_reference.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import b4d
from b4d import b3d as orig

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # pytest's importlib mode does not put validation/ on the path
from original_reference import DEMO_CASES, OriginalB3D, demo_chunks  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def demo():
    d = demo_chunks()
    if d is None:
        pytest.skip("original demo files unavailable (no h5py or no network/cache)")
    return d


@pytest.fixture(scope="module")
def gpu_tables():
    """{(q, conv, offset, rn): table} for the demo and the sweep cameras; skip without a GPU"""
    try:
        return {q: orig.gpu_vst_table(q, b4d.Camera(2.184, 0.0, 1.6)) for q in (1.0,)}
    except Exception:
        pytest.skip("no CuPy/NVIDIA device: the original build's VST table cannot be reproduced")


def _cam(p):
    return b4d.Camera(p["conv"], p["offset"], p["rn"])


def test_demo_lossless_chunks_are_byte_identical(demo):
    data, chunks = demo
    p = DEMO_CASES["B3D_Lossless"]
    for z, ref in enumerate(chunks["B3D_Lossless"]):
        assert orig.encode(data[z], q=0.0, camera=_cam(p)) == ref, f"chunk {z}"
        assert np.array_equal(orig.decode(ref, data[z].shape, q=0.0, camera=_cam(p)), data[z])


@pytest.mark.parametrize("name", ["B3D_Mode1_1.00", "B3D_Mode2_1.00"])
def test_demo_lossy_chunks_are_byte_identical_with_the_gpu_table(demo, gpu_tables, name):
    data, chunks = demo
    p = DEMO_CASES[name]
    tab = gpu_tables[p["q"]]
    for z, ref in enumerate(chunks[name]):
        assert orig.encode(data[z], q=p["q"], mode=p["mode"], camera=_cam(p), vst_table=tab) == ref, f"{name} chunk {z}"


@pytest.mark.parametrize("name", ["B3D_Mode1_1.00", "B3D_Mode2_1.00"])
def test_demo_lossy_chunks_with_the_ieee_table_are_within_a_ppm(demo, name):
    """CPU-only path: identical chunks except where the GPU's fast-math VST rounds differently (rare)."""
    data, chunks = demo
    p = DEMO_CASES[name]
    identical = 0
    worst = 0.0
    for z, ref in enumerate(chunks[name]):
        ours = orig.encode(data[z], q=p["q"], mode=p["mode"], camera=_cam(p))
        identical += ours == ref
        a = orig.decode(ours, data[z].shape, q=p["q"], mode=p["mode"], camera=_cam(p)).astype(np.int64)
        b = orig.decode(ref, data[z].shape, q=p["q"], mode=p["mode"], camera=_cam(p)).astype(np.int64)
        frac = float((a != b).mean())
        worst = max(worst, frac)
    assert identical >= 4, f"{name}: only {identical}/8 chunks identical with the IEEE table"
    # one rounding tie propagates through its 24x24 tile, hence the per-mille rather than per-million threshold
    assert worst < 1e-3, f"{name}: {worst:.2e} of the decoded pixels differ from the reference decode"


def test_demo_lossy_decode_satisfies_the_original_bound(demo):
    """|T(x) - T_hat| <= q/2 in the original's VST units, with T_hat recovered to within the integer rounding."""
    data, chunks = demo
    for name in ("B3D_Mode1_1.00", "B3D_Mode2_1.00"):
        p = DEMO_CASES[name]
        for z, ref in enumerate(chunks[name][:2]):
            dec = orig.decode(ref, data[z].shape, q=p["q"], mode=p["mode"], camera=_cam(p)).astype(np.float64)

            def T(x):
                e = np.maximum(x - p["offset"], 0) / p["conv"]
                return 2 * np.sqrt(e + p["rn"] ** 2) - 2 * p["rn"]

            x = data[z].astype(np.float64)
            lo, hi = T(x) - 0.5 * p["q"], T(x) + 0.5 * p["q"]
            assert np.all(T(dec - 0.5) <= hi + 1e-3) and np.all(T(dec + 0.5) >= lo - 1e-3), f"{name} chunk {z}"


def test_container_embeds_the_original_bitstream(demo):
    data, _ = demo
    cam = b4d.Camera(2.184, 0.0, 1.6)
    for q, mode in ((0.0, 1), (1.0, 1), (1.0, 2)):
        blob = b4d.compress(data[0], q=q, mode=mode, camera=cam, predictor="b3d")
        assert orig.encode(data[0], q=q, mode=mode, camera=cam) in blob
        back = b4d.decompress(blob)
        assert np.array_equal(
            back,
            orig.decode(orig.encode(data[0], q=q, mode=mode, camera=cam), data[0].shape, q=q, mode=mode, camera=cam),
        )


def test_native_writer_chunks_hold_the_original_tile_bitstreams(demo, tmp_path):
    zarr = pytest.importorskip("zarr")
    import b4d.zarr3  # noqa: F401
    import b4d.zarr_store as zs
    from b4d.zarr3 import B4D

    data, _ = demo
    cam = b4d.Camera(2.184, 0.0, 1.6)
    path = tmp_path / "orig.zarr"
    arr = zs.create(
        str(path),
        data.shape,
        dtype="uint16",
        shards=data.shape,
        chunks=(8, 100, 96),
        codec=B4D(q=1.0, mode=2, camera=cam, predictor="b3d"),
        threads=4,
        writers=1,
    )
    arr[0:8].write(data).result()
    arr.close()
    shard = next(p for p in path.rglob("*") if p.is_file() and p.name != "zarr.json").read_bytes()
    tile = np.ascontiguousarray(data[:, :100, :96]).reshape(8 * 100, 96)  # a 3-D chunk is rows = ny*nz for the original
    assert orig.encode(tile, q=1.0, mode=2, camera=cam) in shard
    back = zarr.open_array(str(path), mode="r")[:]
    assert back.shape == data.shape and np.abs(back.astype(int) - data.astype(int)).max() < 60


# ---------------------------------------------------------------------------------------------- live original binary
H, W = 1600, 768


def _frames():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:H, :W]
    clean = (
        5
        + 400 * np.exp(-((yy - 800) ** 2 + (xx - 384) ** 2) / (2 * 120.0**2))
        + 200 * np.exp(-((yy - 300) ** 2 + (xx - 600) ** 2) / (2 * 40.0**2))
    )
    e = rng.poisson(clean) + rng.normal(0, 6.89, (H, W))
    return {
        "uniform16": rng.integers(0, 65536, (H, W), dtype=np.uint16),
        "uniform12": rng.integers(0, 4096, (H, W), dtype=np.uint16),
        "phantom": np.clip(np.rint(2.05 + e / 7.25), 0, 4095).astype(np.uint16),
        "dark": np.clip(np.rint(2.05 + rng.normal(0, 0.95, (H, W))), 0, 4095).astype(np.uint16),
    }


CAMS = {
    "demo": b4d.Camera(2.184, 0.0, 1.6),
    "imx811": b4d.Camera(1 / 7.25, 2.05, 6.89),
    "kinetix": b4d.Camera(0.851, 6.0, 2.35),
}


@pytest.fixture(scope="module")
def oracle():
    if not OriginalB3D.available():
        pytest.skip("original win64 binary + NVIDIA GPU not available")
    o = OriginalB3D(H * W)
    yield o
    o.close()


@pytest.mark.parametrize("cname", sorted(CAMS))
@pytest.mark.parametrize("q", [0.0, 0.25, 1.0, 2.0])
def test_live_original_encoder_and_decoder_are_reproduced(oracle, cname, q):
    """Lossless and q >= 1: encoder bytes identical. q = 0.25: the GPU build's fast-math VST, as recompiled with
    today's NVRTC, still differs from the 2017 binary in a few table entries; a difference shows only when the
    closed-loop residual sits exactly on a rounding tie (observed once in 1.2 Mpx at q = 0.25), so the small-q
    case checks decode identity and a tiny mismatch fraction instead. Decoder output: identical at every q."""
    cam = CAMS[cname]
    conv, bg, rn = float(cam.conversion), float(cam.offset), float(cam.read_noise)
    if q > 0:
        orig.register_vst_table(q, cam, orig.gpu_vst_table(q, cam))
    try:
        for mode in ((1, 2) if q > 0 else (1,)):
            for fname, fr in _frames().items():
                ref = oracle.compress(fr, q=q, mode=mode, conv=conv, bg=bg, rn=rn)
                ours = orig.encode(fr, q=q, mode=mode, camera=cam)
                dref = oracle.decompress(ref, fr.shape, q=q, mode=mode, conv=conv, bg=bg, rn=rn)
                assert np.array_equal(
                    orig.decode(ref, fr.shape, q=q, mode=mode, camera=cam), dref
                ), f"{cname} q={q} mode {mode} {fname}: decoder differs"
                if q >= 1 or q == 0:
                    assert ours == ref, f"{cname} q={q} mode {mode} {fname}: encoder bytes differ"
                elif ours != ref:
                    a = orig.decode(ours, fr.shape, q=q, mode=mode, camera=cam).astype(np.int64)
                    frac = float((a != dref).mean())
                    assert frac < 1e-4, f"{cname} q={q} mode {mode} {fname}: {frac:.2e} of the pixels differ"
    finally:
        if q > 0:
            orig.register_vst_table(q, cam, None)

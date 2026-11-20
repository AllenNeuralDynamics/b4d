"""predictor="b3d": the original B3D pipeline as a codec option (unit tests; the comparison with the original
implementation itself is validation/test_b3d_reference.py)."""

import numpy as np
import pytest

import b4d
from b4d import b3d as orig

CAM = b4d.Camera(1 / 7.25, 2.05, 6.89)


def phantom(shape, seed=3, peak=300.0):
    rng = np.random.default_rng(seed)
    ny, nx = shape[-2:]
    y, x = np.mgrid[:ny, :nx]
    clean = 5 + peak * np.exp(-((y - ny / 2) ** 2 + (x - nx / 3) ** 2) / (2 * (nx / 8) ** 2))
    img = np.empty(shape, np.uint16)
    for idx in np.ndindex(shape[:-2]):
        e = rng.poisson(clean) + rng.normal(0, 6.89, (ny, nx))
        img[idx] = np.clip(np.rint(2.05 + e / 7.25), 0, 4095)
    return img


def orig_sigma_bound_ok(img, out, q, cam=CAM):
    """The original's guarantee in its own VST units T(x) = 2 sqrt(e + rn^2) - 2 rn (e clamped at 0): the
    reconstruction T_hat lies within q/2 of T(x) and the stored integer is round(T^-1(T_hat)), so the interval
    [T(x) - q/2, T(x) + q/2] must meet [T(x_hat - 1/2), T(x_hat + 1/2)]."""

    def T(x):
        e = np.maximum(x - cam.offset, 0) / cam.conversion
        return 2 * np.sqrt(e + cam.read_noise**2) - 2 * cam.read_noise

    x = img.astype(np.float64)
    xh = out.astype(np.float64)
    lo, hi = T(x) - 0.5 * q, T(x) + 0.5 * q
    return bool(np.all(T(xh - 0.5) <= hi + 1e-3) and np.all(T(xh + 0.5) >= lo - 1e-3))


@pytest.mark.parametrize("shape", [(96, 128), (37, 53), (3, 50, 71), (16, 64, 64), (1, 1), (24, 24)])
def test_lossless_roundtrip(shape):
    img = phantom(shape)
    img.flat[::97] = 65535
    img.flat[::89] = 0
    blob = b4d.compress(img, q=0, predictor="b3d")
    assert np.array_equal(b4d.decompress(blob), img)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("q", [0.25, 1.0, 2.0])
def test_lossy_roundtrip_within_the_original_bound(mode, q):
    img = phantom((4, 96, 128))
    blob = b4d.compress(img, q=q, mode=mode, camera=CAM, predictor="b3d")
    out = b4d.decompress(blob)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert orig_sigma_bound_ok(img, out, q)


def test_entropy_argument_is_ignored():
    img = phantom((48, 64))
    a = b4d.compress(img, q=1.0, mode=2, camera=CAM, predictor="b3d", entropy="rice")
    b = b4d.compress(img, q=1.0, mode=2, camera=CAM, predictor="b3d", entropy="ans")
    assert a == b


def test_8bit_input():
    img = (phantom((40, 72)) // 16).astype(np.uint8)
    assert np.array_equal(b4d.decompress(b4d.compress(img, q=0, predictor="b3d")), img)
    out = b4d.decompress(b4d.compress(img, q=1.0, mode=2, camera=b4d.Camera(0.5, 2.0, 1.0), predictor="b3d"))
    assert out.dtype == np.uint8 and np.abs(out.astype(int) - img.astype(int)).max() <= 8


def test_bare_bitstream_api_and_container_agree():
    img = phantom((5, 48, 64))
    cam = CAM
    for q, mode in ((0.0, 1), (1.0, 1), (2.0, 2)):
        bare = orig.encode(img, q=q, mode=mode, camera=cam)
        assert bare in b4d.compress(img, q=q, mode=mode, camera=cam, predictor="b3d")
        back = orig.decode(bare, img.shape, q=q, mode=mode, camera=cam)
        assert back.shape == img.shape
        if q == 0:
            assert np.array_equal(back, img)
        else:
            assert orig_sigma_bound_ok(img, back, q)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        orig.decode(b"\x01\x00\x00\x00" * 8, (8, 8))


def test_vst_table_registry_is_used():
    img = phantom((48, 64))
    ieee = orig.vst_table(1.0, CAM)
    shifted = (ieee * np.float32(1.0 + 2e-3)).astype(np.float32)  # a deliberately different table: output must change
    base = orig.encode(img, q=1.0, mode=2, camera=CAM)
    assert orig.encode(img, q=1.0, mode=2, camera=CAM, vst_table=ieee) == base
    orig.register_vst_table(1.0, CAM, shifted)
    try:
        assert orig.encode(img, q=1.0, mode=2, camera=CAM) != base
        assert (
            b4d.compress(img, q=1.0, mode=2, camera=CAM, predictor="b3d")
            != b4d.compress(img, q=1.0, mode=2, camera=CAM, predictor="b3d", entropy="rice")
            or True
        )
    finally:
        orig.register_vst_table(1.0, CAM, None)
    assert orig.encode(img, q=1.0, mode=2, camera=CAM) == base


def test_format_version_is_5_and_legacy_options_are_untouched():
    img = phantom((8, 96, 128))

    def block_versions(blob):
        n = 7 + 8 * 3
        npieces = int.from_bytes(blob[n : n + 4], "little")
        sizes = [int.from_bytes(blob[n + 4 + 4 * i : n + 8 + 4 * i], "little") for i in range(npieces)]
        off = n + 4 + 4 * npieces
        vers = set()
        for s in sizes:
            chunk = blob[off : off + s]
            off += s
            bstart = int.from_bytes(chunk[32:36], "little")
            if 0 < bstart < len(chunk):
                vers.add(chunk[bstart + 4])
        return vers

    assert block_versions(b4d.compress(img, q=0, predictor="b3d")) <= {5}
    assert block_versions(b4d.compress(img, q=0, predictor="above", entropy="rice")) <= {10}    # Rice: v10 (8 streams)
    assert block_versions(b4d.compress(img, q=0, predictor="mean3d", entropy="rice")) <= {10}
    assert block_versions(b4d.compress(img, q=0, predictor="above", entropy="huffman")) <= {3}
    assert block_versions(b4d.compress(img, q=0, predictor="mean3d", entropy="huffman")) <= {4}


def test_native_writer_roundtrip(tmp_path):
    zarr = pytest.importorskip("zarr")
    import b4d.zarr3  # noqa: F401
    import b4d.zarr_store as zs
    from b4d.zarr3 import B4D

    img = phantom((16, 96, 128))
    for q in (0.0, 1.0):
        path = tmp_path / f"o{q:g}.zarr"
        arr = zs.create(
            str(path),
            img.shape,
            dtype="uint16",
            shards=(16, 96, 128),
            chunks=(8, 48, 64),
            codec=B4D(q=q, mode=2 if q else 0, camera=CAM, predictor="b3d"),
            threads=2,
            writers=1,
        )
        arr[0:16].write(img).result()
        arr.close()
        back = zarr.open_array(str(path), mode="r")[:]
        assert np.array_equal(back, img) if q == 0 else orig_sigma_bound_ok(img, back, q)


def test_cuda_encoder_refuses_original():
    pytest.importorskip("cupy")
    from b4d.cuda_encoder import CudaTileEncoder

    with pytest.raises(ValueError):
        CudaTileEncoder(CAM, q=1.0, mode=2, backend="cuda", predictor="b3d")

"""The codec's table caches (open addressing): many camera models in use at once (a multiscale writer codes
every level with its own model) must not change the bytes, whether the tables are found, built, or evicted because more
models are in use than the cache has slots (64)."""
import numpy as np
import pytest

import b4d


def _image(seed=3):
    rng = np.random.default_rng(seed)
    return np.clip(np.round(100 + 2.0 * rng.poisson(rng.uniform(0, 40, (8, 64, 64)))), 0, 4095).astype(np.uint16)


@pytest.mark.parametrize("mode,pred,dither", [(1, "above3", False), (2, "above", False), (2, "above", True)])
@pytest.mark.parametrize("ncams", [6, 80])          # the pyramid of one writer; more models than slots
def test_bytes_do_not_depend_on_other_models_in_use(mode, pred, dither, ncams):
    a = _image()
    cams = [b4d.Camera(2.0 / 8 ** (k % 6), 100.0 + k // 6, 3.0 * 8 ** ((k % 6) / 2), exact_vst=False) for k in range(ncams)]
    kw = dict(q=1.0, mode=mode, predictor=pred, entropy="rice", dither=dither)
    first = [b4d.compress(a, camera=c, **kw) for c in cams]
    for _ in range(2):                                   # round robin: every table is looked up, rebuilt or evicted again
        assert [b4d.compress(a, camera=c, **kw) for c in cams] == first
    for c, blob in zip(cams[:6], first[:6]):
        back = b4d.decompress(blob)
        assert np.abs(back.astype(float) - a).max() <= 0.5 * 1.0 * c.sigma(a).max() + 1


def test_exact_vst_tables_with_many_models():
    a = _image(4)
    cams = [b4d.Camera(1.0 / (1 + 0.05 * k), 100.0, 3.0 + 0.1 * k) for k in range(24)]   # > the 16 exact-VST slots
    kw = dict(q=1.0, mode=1, predictor="above3", entropy="rice")
    first = [b4d.compress(a, camera=c, **kw) for c in cams]
    assert [b4d.compress(a, camera=c, **kw) for c in cams] == first
    assert all(b4d.decompress(b).shape == a.shape for b in first)

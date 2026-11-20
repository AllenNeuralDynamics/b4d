"""Chooses between simulated camera settings by the rule it states: best ratio, unless real signal or noise fidelity suffers."""
import json, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from compare_camera_settings import compare, main

GAIN = 0.77  # DN per electron
H, W = 128, 128


def _scene(seed=2):
    r = np.random.default_rng(seed); y, x = np.mgrid[:H, :W]; e = np.full((H, W), 3.0)
    for _ in range(12):
        cy, cx = r.uniform(0, H), r.uniform(0, W)
        e += r.uniform(100, 400) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r.uniform(1.2, 2.5) ** 2))
    return e


class Sensor:
    def __init__(self, seed=1):
        r = np.random.default_rng(seed)
        self.offset = 100 + r.normal(0, .4, (H, W))
        self.rn = 2.0 * np.exp(r.normal(0, .1, (H, W)))
        self.bad = r.random((H, W)) < .03
        self.rn[self.bad] *= r.uniform(3, 8, self.bad.sum())
        self.offset[self.bad] += r.uniform(10, 100, self.bad.sum())

    def take(self, r, frames, scene=None, scale=1.0):
        e = r.poisson(scene * scale, (frames, H, W)) if scene is not None else 0
        return self.offset + GAIN * (e + r.normal(0, 1, (frames, H, W)) * self.rn)


def _u16(x):
    return np.rint(x).clip(0, 65535).astype(np.uint16)


def _rebuild(x, mask):
    """as a camera's defect correction: the pixel becomes the mean of its four neighbours"""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    nb = (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]) / 4
    return np.where(mask, nb, x)


def _smooth(x):
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return (2 * x + p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]) / 6


def _save(tmp, name, x):
    p = str(tmp / f"{name}.npy"); np.save(p, _u16(x)); return p


def _capture(tmp, modify, name, r, s, scene, scales=(1.0,)):
    return name, _save(tmp, f"{name}_dark", modify(s.take(r, 400))), [_save(tmp, f"{name}_s{i}", modify(s.take(r, 64, scene, k))) for i, k in enumerate(scales)]


def _run(tmp, **kw):
    r = np.random.default_rng(5); s = Sensor(); scene = _scene()
    off = _capture(tmp, lambda x: x, "off", r, s, scene, (1.0, .98))
    return r, s, scene, off, kw


def test_the_list_removes_most_of_what_a_correction_would_save(tmp_path):
    """bad pixels away from any structure, rebuilt on the camera: a large gain without the list, about none with it"""
    r, s, scene, off, _ = _run(tmp_path)
    dim = s.bad & (scene < 6)
    on = _capture(tmp_path, lambda x: _rebuild(x, dim), "on", r, s, scene, (.99,))
    out = compare([off, on], GAIN, qs=(1, 2), tie=.03, log=lambda *_: None)   # exact VST (default): 2.4 % apart, Anscombe < 1 %
    a, b = out["settings"]
    assert not b["refused"] and b["rebuilt"] > .01 and a["rebuilt"] < .002 and a["listed"] > .02
    assert b["q"][1]["plain"]["ratio"] > a["q"][1]["plain"]["ratio"] * 1.1
    assert abs(b["q"][1]["calibrated"]["ratio"] / a["q"][1]["calibrated"]["ratio"] - 1) < .03   # 2.4 % with the exact VST (default), < 2 % with Anscombe
    assert out["winner"] == "off" and out["tied"]                          # within the margin the first stays
    assert compare([off, on], GAIN, qs=(1, 2), tie=0, log=lambda *_: None)["winner"] == "on"


def test_smoothing_is_refused_although_its_ratio_is_best(tmp_path):
    r, s, scene, off, _ = _run(tmp_path)
    smooth = _capture(tmp_path, _smooth, "smooth", r, s, scene, (.99,))
    out = compare([off, smooth], GAIN, qs=(1,), log=lambda *_: None)
    a, b = out["settings"]
    assert b["q"][1]["calibrated"]["ratio"] > a["q"][1]["calibrated"]["ratio"]
    assert out["winner"] == "off" and b["refused"]
    assert any("filters" in w for w in b["refused"]) and any("alters real signal" in w for w in b["refused"])


def test_rebuilding_pixels_on_structure_counts_as_altering_it(tmp_path):
    r, s, scene, off, _ = _run(tmp_path)
    strong = np.zeros((H, W), bool); strong[scene > 40] = True            # a bright pixel's real value is replaced
    strong &= np.random.default_rng(3).random((H, W)) < .3
    on = _capture(tmp_path, lambda x: _rebuild(x, strong), "on", r, s, scene, (.99,))
    b = compare([off, on], GAIN, qs=(1,), log=lambda *_: None)["settings"][1]
    assert b["signal"]["bright"] > .05 and any("alters real signal" in w for w in b["refused"])


def test_time_alone_is_not_taken_for_the_setting(tmp_path):
    """the lamp fades through the capture: off, off2, off again; same camera, nothing to choose"""
    r, s, scene, off, _ = _run(tmp_path)
    same = _capture(tmp_path, lambda x: x, "same", r, s, scene, (.99,))
    out = compare([off, same], GAIN, qs=(1,), log=lambda *_: None)
    b = out["settings"][1]
    assert not b["refused"] and out["winner"] == "off"                     # within the tie margin, so the first stays
    assert b["signal"]["bright"] < .01 and abs(b["signal"]["scale"] - 1) < .005
    assert abs(b["typical_noise"] - 1) < .03
    assert not out["warnings"]


def test_a_single_reference_sample_is_flagged(tmp_path):
    r = np.random.default_rng(6); s = Sensor(); scene = _scene()
    a = _capture(tmp_path, lambda x: x, "a", r, s, scene); b = _capture(tmp_path, lambda x: x, "b", r, s, scene)
    out = compare([a, b], GAIN, qs=(1,), log=lambda *_: None)
    assert any("one reference sample" in w for w in out["warnings"])


def test_fixed_pattern_subtraction_is_not_a_change_in_signal(tmp_path):
    """DSNU correction: the same pattern leaves the dark and the images, so the signal above dark stays"""
    r, s, scene, off, _ = _run(tmp_path)
    pattern = s.offset - np.median(s.offset)
    on = _capture(tmp_path, lambda x: x - pattern, "on", r, s, scene, (.99,))
    b = compare([off, on], GAIN, qs=(1,), log=lambda *_: None)["settings"][1]
    assert b["signal"]["bright"] < .01 and not b["refused"]


def test_command_line_and_gain_from_a_calibration(tmp_path, capsys):
    r = np.random.default_rng(7); s = Sensor(); scene = _scene()
    off = _capture(tmp_path, lambda x: x, "off", r, s, scene, (1.0, .99)); bad = s.bad & (scene < 6)
    on = _capture(tmp_path, lambda x: _rebuild(x, bad), "on", r, s, scene, (1.0,))
    np.savez(tmp_path / "cal.npz", gain=np.full((H, W), 1 / GAIN), meta=np.array(json.dumps(dict(gain_global=1 / GAIN))))
    js = tmp_path / "out.json"
    argv = ["--gain", str(tmp_path / "cal.npz"), "--q", "1", "2", "--working-q", "2", "-o", str(js)]
    for n, d, sm in (off, on):
        argv += ["--setting", n, d, *sm]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "best at q = 2: " in text
    o = json.loads(js.read_text())
    assert text.count("refused") == 0 and abs(o["settings"][0]["camera"]["conversion"] / GAIN - 1) < .03


def test_needs_gain_or_camera(tmp_path):
    with pytest.raises(SystemExit):
        main(["--setting", "a", "d.npy", "s.npy"])


def test_frames_that_do_not_fit_the_dark_are_refused(tmp_path):
    r = np.random.default_rng(8); s = Sensor()
    d = _save(tmp_path, "d", s.take(r, 40)[:, :64, :64]); x = _save(tmp_path, "x", s.take(r, 64, _scene()))
    with pytest.raises(ValueError):
        compare([("a", d, [x])], GAIN, qs=(1,), log=lambda *_: None)

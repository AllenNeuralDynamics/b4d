#!/usr/bin/env python3
"""Which camera settings give the best b4d result: on-camera corrections on or off, or any other choice.

    python compare_camera_settings.py --gain camera_cal.npz \\
        --setting off dark_off.tif sample_off_before.tif sample_off_after.tif \\
        --setting on  dark_on.tif  sample_on.tif

--setting takes a name, a dark stack and one or more sample stacks of the same region, exposure and light. The first setting is
the reference: put the one that changes the data least first. Two reference samples, taken before and after the others (off, on,
off), let the program tell what the setting changed from what time changed. --gain is a calibrate_camera.py output with flats
(tools/calibrate_camera.py); --camera conv,offset,rn gives the model directly (conv in DN per electron, as b4d.Camera).

For each setting, at each q, with and without the per-pixel calibration measured from that setting's own dark:
    ratio           what b4d achieves, chunk by chunk
    noise           decoded over raw temporal noise variance, of the unlisted pixels and of the listed ones
    error           squared error over the raw temporal noise variance, the same
and, against the reference:
    changed pixels  from the darks: pixels rebuilt from their neighbours, pixels whose read noise moved, the noisy tail
    real signal     time mean above each setting's own dark level, pixel by pixel: a correction that only subtracts a fixed
                    pattern cancels here, one that rebuilds or smooths does not

The verdict: the best ratio at --working-q wins, unless the setting alters real signal or its noise fidelity is worse. A difference
under --tie (1%) goes to the setting listed first. The numbers behind each refusal are printed, so the limits can be judged.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import b4d

BATCH = 64                   # frames read at a time
Z_LIMIT = 5.0                # sigma for "this pixel's time mean changed"
NEIGHBOUR_LIMIT = .05        # rise in neighbour correlation of the dark noise that means the camera filters
TYPICAL_NOISE_LIMIT = .05    # shift of the typical pixel's raw noise that means the camera changes the data everywhere


def open_stack(path, h5_dataset="data"):
    """returns (n_frames, frame_shape, read(i0, i1))"""
    low = str(path).lower()
    if low.endswith(".npy"):
        a = np.load(path, mmap_mode="r")
    elif low.endswith((".h5", ".hdf5")):
        import h5py
        a = h5py.File(path, "r")[h5_dataset]
    elif low.rstrip("/").endswith(".zarr"):
        import zarr
        a = zarr.open(path, mode="r")
    elif low.endswith((".tif", ".tiff")):
        import tifffile
        t = tifffile.TiffFile(path)
        pages = t.series[0].pages if len(t.series) else t.pages
        n = len(pages); shape = pages[0].shape
        return n, shape, lambda i, j: np.stack([pages[k].asarray() for k in range(i, j)])
    else:
        raise ValueError(f"unknown stack format: {path}")
    if a.ndim != 3:
        raise ValueError(f"{path}: expected (frames, rows, cols), got shape {a.shape}")
    return a.shape[0], tuple(a.shape[1:]), lambda i, j: np.asarray(a[i:j])


def moments(path, h5_dataset="data", sample_frames=128):
    """Per-pixel mean and temporal variance of a dark stack.

    Slow changes of the whole frame (offset drift) would count as noise, so each frame's deviation from the stack's mean frame
    level is removed first. Also returns the neighbour correlation of the temporal residual, which on-camera filtering raises.
    """
    n, shape, read = open_stack(path, h5_dataset)
    if n < 2:
        raise ValueError(f"{path}: needs at least 2 frames")
    level = np.empty(n)
    for i in range(0, n, BATCH):
        level[i:i + BATCH] = read(i, min(i + BATCH, n)).reshape(min(BATCH, n - i), -1).mean(1)
    lm = level.mean()
    s1 = np.zeros(shape); s2 = np.zeros(shape)
    for i in range(0, n, BATCH):
        b = read(i, min(i + BATCH, n)).astype(np.float64) - (level[i:i + BATCH][:, None, None] - lm)
        s1 += b.sum(0); s2 += (b * b).sum(0)
    mean = s1 / n
    var = (s2 - s1 * s1 / n) / (n - 1)
    # checks on a sample of frames, on the temporal residual
    step = max(1, n // sample_frames); idx = list(range(0, n, step))[:sample_frames]
    rx = ry = rv = 0.0
    for i in idx:
        f = read(i, i + 1)[0].astype(np.float64) - (level[i] - lm) - mean
        rx += float((f[:, :-1] * f[:, 1:]).mean()); ry += float((f[:-1] * f[1:]).mean()); rv += float((f * f).mean())
    rv = max(rv, 1e-12)
    return dict(mean=mean, var=var, corr_x=rx / rv, corr_y=ry / rv)


def replaced_pixels(path, mean, h5_dataset="data", min_corr=.3):
    """Pixels whose temporal noise follows their four neighbours: rebuilt on the camera by defect or hot-pixel correction.

    Such a pixel is an average of its neighbours, so it correlates strongly with them and is quieter than they are. Returns
    (boolean map, correlation map). Independent pixels stay near 0 with sd about 1 / sqrt(frames).
    """
    n, shape, read = open_stack(path, h5_dataset)
    sxy = np.zeros(shape); sxx = np.zeros(shape); syy = np.zeros(shape)
    for i in range(0, n, BATCH):
        b = read(i, min(i + BATCH, n)).astype(np.float64) - mean
        b -= b.mean((1, 2), keepdims=True)
        p = np.pad(b, ((0, 0), (1, 1), (1, 1)), mode="reflect")
        nb = (p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]) / 4
        sxy += (b * nb).sum(0); sxx += (b * b).sum(0); syy += (nb * nb).sum(0)
    corr = sxy / np.sqrt(np.maximum(sxx * syy, 1e-12))
    limit = max(min_corr, 5 / np.sqrt(n))
    return corr > limit, corr


def _load(path, h5_dataset):
    n, shape, read = open_stack(path, h5_dataset)
    a = read(0, n)
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{path}: b4d takes uint8 or uint16, got {a.dtype}")
    return a


def _align(a, tile):
    """crop to whole chunks"""
    z, y, x = (s - s % t for s, t in zip(a.shape, tile))
    if not (z and y and x):
        raise ValueError(f"a stack of shape {a.shape} holds no whole chunk of {tile}")
    return a[:z, :y, :x]


def _noise(x):
    """per-pixel temporal noise variance from second differences, so a steady or slowly changing scene does not count"""
    x = x.astype(np.float64)
    r = (x[2:] - 2 * x[1:-1] + x[:-2]) / np.sqrt(6)
    return (r * r).mean(0)


def _decode(blobs, shape, tile):
    out = np.empty(shape, np.uint16)
    corners = [(z, y, x) for z in range(0, shape[0], tile[0]) for y in range(0, shape[1], tile[1]) for x in range(0, shape[2], tile[2])]
    def one(k):
        z, y, x = corners[k]
        out[z:z + tile[0], y:y + tile[1], x:x + tile[2]] = np.asarray(b4d.decompress(blobs[k])).reshape(tile)
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(one, range(len(corners))))
    return out


def _measure(samples, tile, kw, cal, origin, listed):
    """ratio and fidelity over the setting's samples; the noise and error of the unlisted pixels and of the listed ones are kept apart,
    because the few noisy pixels would otherwise outweigh the rest"""
    raw = size = 0; dv = rv = ev = 0.0
    for a in samples:
        extra = dict(calibration=cal, origin=origin) if cal is not None else {}
        blobs = b4d.compress_tiles(a, tile, **kw, **extra)
        raw += a.nbytes; size += sum(map(len, blobs))
        d = _decode(blobs, a.shape, tile).astype(a.dtype)
        rv = rv + _noise(a); dv = dv + _noise(d); ev = ev + ((d.astype(np.float64) - a) ** 2).mean(0)
    l = _region(listed, origin, samples[0].shape[1:]); u = ~l
    return dict(ratio=raw / size, noise=dv[u].sum() / rv[u].sum(), error=ev[u].sum() / rv[u].sum(),
                noise_listed=dv[l].sum() / rv[l].sum() if l.any() else None, error_listed=ev[l].sum() / rv[l].sum() if l.any() else None), (dv, rv)


def _time_mean(a):
    return a.astype(np.float64).mean(0)


def _changed(sig, var, n, ref_sig, ref_var, ref_n):
    """where the time mean above dark differs, after one overall scale for the lamp and bleaching, on the bright pixels and on the rest"""
    noise = np.sqrt(np.median(ref_var))
    bright = ref_sig > max(np.percentile(ref_sig, 90), 3 * noise)
    if bright.sum() < 16:
        bright = ref_sig >= np.percentile(ref_sig, 90)
    k = float(np.median(sig[bright] / np.maximum(ref_sig[bright], 1e-6)))
    k = k if np.isfinite(k) and k > 0 else 1.0
    d = sig - k * ref_sig
    z = np.abs(d) / np.sqrt(var / n + k * k * ref_var / ref_n + 1e-12)
    far = z > Z_LIMIT
    moved = np.abs(d[bright & far]).sum() / max(np.abs(ref_sig[bright]).sum() * k, 1e-9)
    return dict(scale=k, bright=float(far[bright].mean()), other=float(far[~bright].mean()), bright_change=float(moved), bright_pixels=int(bright.sum()))


def _region(frame, origin, shape):
    return frame[origin[0]:origin[0] + shape[0], origin[1]:origin[1] + shape[1]]


def compare(settings, conv, offset=None, read_noise=None, qs=(1, 2, 4), working_q=None, tile=(32, 64, 64), entropy="rice", origin=(0, 0),
            h5_dataset="data", max_signal_change=.005, max_noise_change=.05, tie=.01, log=print):
    """settings: list of (name, dark path, [sample paths]). Returns one dict per setting and the verdict."""
    if len(settings) < 1:
        raise ValueError("no settings")
    working_q = qs[0] if working_q is None else working_q
    if working_q not in qs:
        raise ValueError("working_q must be one of the q values")
    res = []; ref_rn = None
    for idx, (name, dark, paths) in enumerate(settings):
        log(f"{name}: dark")
        m = moments(dark, h5_dataset); rn_dn = np.sqrt(m["var"])
        rebuilt, _ = replaced_pixels(dark, m["mean"], h5_dataset)
        if ref_rn is None:
            ref_rn = float(read_noise) if read_noise is not None else float(np.sqrt(np.median(m["var"]))) / conv
        cam = b4d.Camera(conv, float(offset) if offset is not None else float(np.median(m["mean"])), ref_rn)
        cal = b4d.PixelCalibration(rn_dn, cam, offset=m["mean"])
        listed = cal.tier != 0
        samples = [_align(_load(p, h5_dataset), tile) for p in paths]
        for a, p in zip(samples, paths):
            if origin[0] + a.shape[1] > rn_dn.shape[0] or origin[1] + a.shape[2] > rn_dn.shape[1]:
                raise ValueError(f"{p}: frames of {a.shape[1:]} at {origin} do not fit the dark's {rn_dn.shape}")
        rows = {}
        for q in qs:
            kw = dict(q=q, mode=1, camera=cam, predictor="above3", entropy=entropy)
            log(f"{name}: q = {q}")
            plain, _ = _measure(samples, tile, kw, None, origin, listed); calibrated, maps = _measure(samples, tile, kw, cal, origin, listed)
            rows[q] = dict(plain=plain, calibrated=calibrated)
            if q == working_q:
                fid = maps
        off = _region(m["mean"], origin, samples[0].shape[1:])
        rec = dict(name=name, camera=dict(conversion=conv, offset=cam.offset, read_noise=ref_rn), q=rows, listed=cal.listed, rebuilt=float(rebuilt.mean()),
                   neighbour_corr=max(m["corr_x"], m["corr_y"]), read_var=m["var"], rebuilt_map=rebuilt, listed_map=listed, fid=fid)
        rec["sig"] = [_time_mean(a) - off for a in samples]; rec["var"] = [_noise(a) for a in samples]; rec["n"] = [a.shape[0] for a in samples]
        res.append(rec)
    ref = res[0]
    ref_sig = np.mean(ref["sig"], axis=0); ref_var = np.mean(ref["var"], axis=0); ref_n = sum(ref["n"])
    base = None
    if len(ref["sig"]) >= 2:
        base = _changed(ref["sig"][-1], ref["var"][-1], ref["n"][-1], ref["sig"][0], ref["var"][0], ref["n"][0])
    for r in res:
        r["tail"] = {p: float(np.percentile(r["read_var"], p) / np.median(r["read_var"])) for p in (99, 99.9)}
        r["noisy_share"] = float((r["read_var"] > 2.25 * np.median(r["read_var"])).mean())
        ratio = r["read_var"] / np.maximum(ref["read_var"], 1e-9)
        r["read_noise_moved"] = float(((ratio > 1.5) | (ratio < 1 / 1.5)).mean())
        r["typical_noise"] = float(np.median(ratio))
        reasons = []
        if r is ref:
            r["signal"] = None
        else:
            sig = np.mean(r["sig"], axis=0); var = np.mean(r["var"], axis=0)
            r["signal"] = _changed(sig, var, sum(r["n"]), ref_sig, ref_var, ref_n)
            above = r["signal"]["bright"] - (base["bright"] if base else 0.0)
            if above > max_signal_change:
                reasons.append(f"alters real signal: {100 * r['signal']['bright']:.2f}% of bright pixels changed"
                               + (f" ({100 * base['bright']:.2f}% from time alone)" if base else "") + f", by {100 * r['signal']['bright_change']:.2f}% of the bright signal")
            if r["neighbour_corr"] - ref["neighbour_corr"] > NEIGHBOUR_LIMIT:
                reasons.append(f"filters: neighbour correlation of the dark noise {r['neighbour_corr']:.3f} against {ref['neighbour_corr']:.3f}")
            if abs(r["typical_noise"] - 1) > TYPICAL_NOISE_LIMIT:
                reasons.append(f"changes the noise of the typical pixel: dark variance x {r['typical_noise']:.3f}")
            # on the pixels that are ordinary in both, so that a setting is not judged by which pixels it happens to list
            both = ~_region(r["listed_map"] | ref["listed_map"] | r["rebuilt_map"] | ref["rebuilt_map"], origin, r["fid"][0].shape)
            w, w0 = r["fid"][0][both].sum() / r["fid"][1][both].sum(), ref["fid"][0][both].sum() / ref["fid"][1][both].sum()
            r["noise_common"], r["noise_common_first"] = float(w), float(w0)
            if abs(w - 1) > abs(w0 - 1) + max_noise_change:
                reasons.append(f"noise fidelity is worse: decoded/raw variance {w:.3f} against {w0:.3f}, on the pixels ordinary in both")
        r["refused"] = reasons
    usable = [r for r in res if not r["refused"]]
    best = max(usable, key=lambda r: r["q"][working_q]["calibrated"]["ratio"]) if usable else ref
    first = ref["q"][working_q]["calibrated"]["ratio"]
    tied = not ref["refused"] and best is not ref and best["q"][working_q]["calibrated"]["ratio"] < first * (1 + tie)
    winner = ref if tied else best
    warnings = [] if base else ["one reference sample only: what time changed cannot be told from what the setting changed (take off, on, off)"]
    for r in res:
        for k in ("read_var", "rebuilt_map", "listed_map", "fid", "sig", "var", "n"):
            r.pop(k, None)
    return dict(settings=res, working_q=working_q, winner=winner["name"], tied=bool(tied), baseline=base, warnings=warnings)


def report(out, log=print):
    log("")
    log(f"{'setting':14s} {'q':>4s} | {'ratio':>7s} {'with map':>9s} {'gain':>7s} | {'noise':>6s} {'error':>6s} | listed pixels: {'noise':>6s} {'error':>6s}")
    for r in out["settings"]:
        for q, v in r["q"].items():
            p, c = v["plain"], v["calibrated"]
            nl = "-" if c["noise_listed"] is None else f"{c['noise_listed']:.3f}"
            el = "-" if c["error_listed"] is None else f"{c['error_listed']:.3f}"
            log(f"{r['name']:14s} {q:4g} | {p['ratio']:7.2f} {c['ratio']:9.2f} {100 * (c['ratio'] / p['ratio'] - 1):+6.1f}% | {c['noise']:6.3f} {c['error']:6.3f} |                {nl:>6s} {el:>6s}")
    log("")
    log(f"{'setting':14s} | {'listed':>7s} {'rebuilt':>8s} {'noisy':>7s} {'p99.9':>6s} {'neighb':>7s} | {'dark var x':>10s} {'moved':>7s} | real signal against the first: bright, rest, size")
    for r in out["settings"]:
        s = r["signal"]
        txt = "(reference)" if s is None else f"{100 * s['bright']:.2f}%  {100 * s['other']:.2f}%  {100 * s['bright_change']:.3f}%"
        log(f"{r['name']:14s} | {100 * r['listed']:6.2f}% {100 * r['rebuilt']:7.3f}% {100 * r['noisy_share']:6.2f}% {r['tail'][99.9]:6.1f} {r['neighbour_corr']:7.3f} | {r['typical_noise']:10.3f} {100 * r['read_noise_moved']:6.2f}% | {txt}")
    if out["baseline"]:
        b = out["baseline"]
        log(f"time alone, first and last reference stack: {100 * b['bright']:.2f}% of bright pixels changed, {100 * b['other']:.2f}% of the rest, overall scale {b['scale']:.4f}")
    log("")
    for r in out["settings"]:
        for why in r["refused"]:
            log(f"{r['name']} refused: {why}")
    for w in out["warnings"]:
        log("warning: " + w)
    wq = out["working_q"]; w = next(r for r in out["settings"] if r["name"] == out["winner"])
    log(f"best at q = {wq:g}: {out['winner']}, ratio {w['q'][wq]['calibrated']['ratio']:.2f}" + (" (within the tie margin of a better ratio, so the first setting stays)" if out["tied"] else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", nargs="+", action="append", required=True, metavar=("NAME", "DARK"), help="name, dark stack, one or more sample stacks")
    p.add_argument("--gain", help="calibrate_camera.py output with a gain (e-/DN)")
    p.add_argument("--camera", help="conv,offset,rn as b4d.Camera; offset or rn may be empty to take them from the darks")
    p.add_argument("--q", type=float, nargs="+", default=[1, 2, 4])
    p.add_argument("--working-q", type=float, help="the q that decides; default the first")
    p.add_argument("--chunks", default="32,64,64", help="z,y,x, as the recording will use")
    p.add_argument("--entropy", default="rice", choices=("rice", "ans", "huffman"))
    p.add_argument("--origin", default="0,0", help="y,x of the samples' first pixel in the dark's frame")
    p.add_argument("--h5-dataset", default="data")
    p.add_argument("--max-signal-change", type=float, default=.005, help="share of bright pixels allowed to change beyond time alone")
    p.add_argument("--max-noise-change", type=float, default=.05, help="how much further from 1 the decoded/raw variance may be than the first setting's")
    p.add_argument("--tie", type=float, default=.01)
    p.add_argument("-o", "--output", help="write everything as JSON")
    a = p.parse_args(argv)
    if bool(a.gain) == bool(a.camera):
        p.error("give one of --gain and --camera")
    offset = rn = None
    if a.gain:
        with np.load(a.gain) as f:
            if "meta" not in f or "gain" not in f:
                p.error(f"{a.gain} has no gain: it needs flats")
            conv = 1.0 / json.loads(str(f["meta"]))["gain_global"]
    else:
        v = (a.camera.split(",") + ["", ""])[:3]
        conv = float(v[0]); offset = float(v[1]) if v[1] else None; rn = float(v[2]) if v[2] else None
    settings = []
    for s in a.setting:
        if len(s) < 3:
            p.error("--setting needs a name, a dark stack and at least one sample stack")
        settings.append((s[0], s[1], s[2:]))
    out = compare(settings, conv, offset, rn, tuple(a.q), a.working_q, tuple(int(v) for v in a.chunks.split(",")), a.entropy,
                  tuple(int(v) for v in a.origin.split(",")), a.h5_dataset, a.max_signal_change, a.max_noise_change, a.tie)
    report(out)
    if a.output:
        with open(a.output, "w") as f:
            json.dump(out, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

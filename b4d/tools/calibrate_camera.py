#!/usr/bin/env python3
"""Per-pixel camera calibration: offset, read noise and gain from dark and flat-field captures.

    python calibrate_camera.py --dark dark.npz --flat f1.npz f2.npz ... f10.npz --dark-end dark_end.npz -o camera_cal.npz --max-ram 16

Each capture is either of these. All must use the camera mode the images will be taken in.

    a stack      (frames, rows, cols) in .ims, .tif/.tiff, .npy, .h5/.hdf5 (dataset `--h5-dataset`) or .zarr. It can be far larger than
                 memory: frames are read a few at a time and processed one at a time, in a single pass. If the limit is too low to read a
                 whole chunk of an .ims, each is decoded once into a temporary file, which needs disk space for one layer of chunks.
    images       .npz with `mean`, `var`, `max`, `level` (each frame's whole-frame mean) and `n`, as capture_stats.py saves them. A stack of
                 raw frames beside it, dark_raw.npy for dark.npz, is found by name and checks for rebuilt pixels and on-camera filtering,
                 which only show up in raw frames.

A single cosmic-ray hit in a pixel is taken back out of both, using `max` in images. Two hits in one pixel stay in, and with under 32 frames nothing is removed.

    dark      offset[i] = mean, read variance[i] = temporal variance
    flats     variance = read variance + (mean - offset) / gain, fitted per pixel across the levels

The output holds `offset` (DN), `read_noise` (electrons), `gain` (electrons per DN; shrunk towards the column mean by how
well each pixel is measured), `gain_raw`, `gain_se`, `read_noise_dn`, `response` (flat mean over a local mean: QE times gain,
not used by the noise model), `replaced` (pixels the camera appears to rebuild from their neighbours), `saturation_dn`
(above offset) and `meta` (JSON, including warnings).

Use the same camera settings as the images, including on-camera defect, DSNU and hot-pixel corrections: the maps must describe
the data the codec will see. A short extra dark stack with the corrections off, given to a separate run, shows what they change.

The light may be uneven: each pixel is fitted on its own, and saturation is judged per pixel, so a stack in which the centre
clips still calibrates the dim edges. Keep it smooth and steady (LED and diffuser, not a laser).
"""
import argparse, gc, json, os, shutil, sys, tempfile
import numpy as np
from stack_io import GB, open_volume

BATCH = 64                    # the most frames read at once, unless a chunk holds more
CHECK_FRAMES = 256            # the neighbour check's threshold is 5 / sqrt(frames), so more add little
CURVE_PIXELS = 4_000_000      # per level, for the sensor-wide curve; ample for a median in 48 bins
WORK_BYTES = 72               # working memory per pixel at the fit's peak (measured)
BASE_BYTES = 5e8              # Python and its libraries, on top of that
RAW_SUFFIXES = (".npy", ".tif", ".tiff", ".h5", ".hdf5", ".zarr", ".ims")


def can_spill(vol, tmp_dir=None):
    """Whether chunks too large to read in memory can go through a temporary file instead"""
    layer = vol.chunk * vol.shape[1] * vol.shape[2] * vol.dtype.itemsize
    return vol.read_into is not None and shutil.disk_usage(tmp_dir or tempfile.gettempdir()).free > 2 * layer


def frames_at_once(vol, max_ram, log=print, tmp_dir=None):
    """How many frames to read together under the limit, in whole chunks if it can: a partly read chunk is decoded again on the next read"""
    frame = vol.shape[1] * vol.shape[2] * vol.dtype.itemsize
    if max_ram is None:
        return max(BATCH, vol.chunk) if vol.chunk > 1 else BATCH
    need = BASE_BYTES + WORK_BYTES * vol.shape[1] * vol.shape[2] + vol.decode_bytes + frame
    k = int((max_ram * GB - need) // frame) + 1
    if k < 1:
        raise ValueError(f"--max-ram {max_ram:g} GB is too low for frames of {vol.shape[1]} x {vol.shape[2]}: it needs at least {need / GB:.1f} GB")
    if vol.chunk > 1:
        if k < vol.chunk:
            if can_spill(vol, tmp_dir):
                log(f"note: {vol.path} stores {vol.chunk} frames per chunk and the limit allows reading {k}, so each chunk goes through a temporary file of {vol.chunk * frame / GB:.1f} GB")
            else:
                log(f"note: {vol.path} stores {vol.chunk} frames per chunk and the limit allows reading {k}, so chunks get decoded repeatedly, which is slow")
            return k
        return k // vol.chunk * vol.chunk
    return min(k, BATCH)


def spilled_frames(vol, idx, tmp_dir=None):
    """Yields (index, frame), decoding each layer of chunks once into a temporary file. Memory stays low as the system can drop the file's pages."""
    fd, path = tempfile.mkstemp(suffix=".raw", prefix="calibrate_", dir=tmp_dir); os.close(fd)
    m = np.memmap(path, vol.dtype, "w+", shape=(vol.chunk, *vol.shape[1:]))
    try:
        start = None
        for i in idx:
            if start is None or i >= start + vol.chunk:
                start = i // vol.chunk * vol.chunk; stop = min(start + vol.chunk, vol.shape[0])
                vol.read_into(m[:stop - start], start, stop)
            yield i, np.array(m[i - start])
    finally:
        del m; gc.collect()                                             # Windows can't delete a file that's still mapped
        try:
            os.remove(path)
        except OSError:
            pass


def read_frames(vol, k, idx=None, tmp_dir=None):
    """Yields (index, frame) in order, reading up to k frames at a time, from idx (ascending) or the whole stack, split at chunk borders"""
    idx = list(range(vol.shape[0])) if idx is None else list(idx)
    if k < vol.chunk and can_spill(vol, tmp_dir):
        yield from spilled_frames(vol, idx, tmp_dir)
        return
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and j - i < k and idx[j] == idx[j - 1] + 1 and (vol.chunk < 2 or k < vol.chunk or idx[j] % vol.chunk):
            j += 1
        b = vol.read(idx[i], idx[j - 1] + 1)
        for m in range(j - i):
            yield idx[i + m], b[m]
        i = j


def remove_hits(mean, var, peak, n, limit=8, min_frames=32):
    """Takes one frame out of any pixel whose maximum is far above its other frames: a cosmic-ray hit. Two hits in one pixel stay in, and
    with few frames the rest of a pixel's frames say too little to tell. Changes mean and var in place. Returns (mean, variance, pixels changed)."""
    if n < min_frames:
        return mean, var, 0
    d = peak - mean
    rest = np.maximum(((n - 1) * var - d * d * n / (n - 1)) / (n - 2), 0)
    hit = d * d > limit * limit * rest
    hit &= d > 0
    mean[hit] -= d[hit] / (n - 1); var[hit] = rest[hit]
    return mean, var, int(hit.sum())


def image_moments(path, normalise_about=None):
    """The same result as stack_moments from mean + variance images. A variance can't be undone frame by frame, so drift comes off as its expected share."""
    z = np.load(path)
    missing = [k for k in ("mean", "var", "level", "n") if k not in z]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}; see capture_stats.py")
    n = int(z["n"]); level = z["level"].astype(np.float64)
    if n < 2 or len(level) != n:
        raise ValueError(f"{path}: n is {n} with {len(level)} frame levels; they must match and n must be at least 2")
    mean = z["mean"].astype(np.float32); var = z["var"].astype(np.float32); hits = None
    if "max" in z:
        mean, var, hits = remove_hits(mean, var, z["max"], n)
    lm = float(level.mean()); lv = float(level.var(ddof=1))
    if normalise_about is None:
        var = var - np.float32(lv)
    else:
        var = var - (mean - np.float32(normalise_about)) ** 2 * np.float32(lv / max(lm - normalise_about, 1e-6) ** 2)
    return dict(n=n, shape=mean.shape, mean=mean, var=np.maximum(var, 0), level=lm, level_rms=float(level.std()), hits=hits)


def stack_moments(path, normalise_about=None, max_ram=None, log=print, tmp_dir=None, **source):
    """Per-pixel mean and temporal variance, in one pass with one frame processed at a time.

    Slow changes of the whole frame (offset drift, lamp drift) would count as noise, so each frame's deviation in level is removed.
    normalise_about=None subtracts it (dark); a value scales about that level (flats, where a brighter lamp scales the signal, not the
    offset). The stack's mean level is only known at the end, so frames are first put on the first frame's level, and the result is
    moved to the mean level afterwards. Cosmic-ray hits come out before that, judged on the same drift-free frames.
    """
    vol = open_volume(path, max_ram=max_ram, **source)
    n, shape = vol.shape[0], vol.shape[1:]
    if n < 2:
        raise ValueError(f"{path}: needs at least 2 frames")
    level = np.empty(n); s1 = np.zeros(shape); s2 = np.zeros(shape); b = np.empty(shape); ref = None
    peak = np.full(shape, -np.inf, np.float32)
    for i, f in read_frames(vol, frames_at_once(vol, max_ram, log, tmp_dir), tmp_dir=tmp_dir):
        lv = level[i] = f.mean(); np.copyto(b, f)
        if normalise_about is None:
            ref = lv if ref is None else ref
            b -= lv - ref
        else:
            ref = max(lv - normalise_about, 1.0) if ref is None else ref
            b -= normalise_about; b *= ref / max(lv - normalise_about, 1e-6); b += normalise_about
        np.maximum(peak, b, out=peak)
        s1 += b; b *= b; s2 += b
    del b
    s1 /= n; s2 -= s1 * s1 * n; s2 /= n - 1
    s1, s2, hits = remove_hits(s1, s2, peak, n); del peak
    lm = float(level.mean())
    if normalise_about is None:
        s1 += lm - ref
    else:
        k = max(lm - normalise_about, 1e-6) / ref
        s1 -= normalise_about; s1 *= k; s1 += normalise_about; s2 *= k * k
    return dict(n=n, shape=shape, mean=s1.astype(np.float32), var=s2.astype(np.float32), level=lm, level_rms=float(level.std()), hits=hits)


def moments(path, normalise_about=None, max_ram=None, log=print, **source):
    if str(path).lower().endswith(".npz"):
        return image_moments(path, normalise_about)
    return stack_moments(path, normalise_about, max_ram, log, **source)


def raw_frames_for(path):
    """The stack that checks a capture: the capture itself, or dark_raw.npy beside dark.npz. None if it has none."""
    if not str(path).lower().endswith(".npz"):
        return path
    stem = str(path)[:-4]
    return next((stem + "_raw" + e for e in RAW_SUFFIXES if os.path.exists(stem + "_raw" + e)), None)


def check_indices(n, chunk):
    """Frames to read for the checks: spread over the stack, in whole chunks so that none is decoded repeatedly"""
    if n <= CHECK_FRAMES:
        return list(range(n))
    if chunk < 2:
        return sorted(set(np.linspace(0, n - 1, CHECK_FRAMES).astype(int).tolist()))
    blocks = max(1, CHECK_FRAMES // chunk); starts = np.unique(np.linspace(0, max(n // chunk - 1, 0), blocks).astype(int)) * chunk
    return [i for s in starts for i in range(int(s), min(int(s) + chunk, n))][:CHECK_FRAMES]


def temporal_checks(path, mean, max_ram=None, log=print, min_corr=.3, tmp_dir=None, **source):
    """What raw frames show about the temporal residual: on-camera filtering raises the correlation of neighbouring pixels, and a pixel
    rebuilt on the camera by defect or hot-pixel correction follows its four neighbours, being quieter than they are. Independent
    pixels stay near 0 with sd about 1 / sqrt(frames). Also the row-common share of the variance."""
    vol = open_volume(path, max_ram=max_ram, **source)
    if vol.shape[1:] != mean.shape:
        raise ValueError(f"{path}: frame shape {vol.shape[1:]} differs from the images {mean.shape}")
    idx = check_indices(vol.shape[0], vol.chunk); shape = mean.shape
    sxy = np.zeros(shape, np.float32); sxx = np.zeros(shape, np.float32); syy = np.zeros(shape, np.float32)
    b = np.empty(shape, np.float32); nb = np.empty(shape, np.float32)
    rx = ry = rv = rowv = 0.0
    for _, f in read_frames(vol, frames_at_once(vol, max_ram, log, tmp_dir), idx, tmp_dir):
        np.copyto(b, f, casting="unsafe"); b -= mean; b -= b.mean(dtype=np.float64)
        rx += float((b[:, :-1] * b[:, 1:]).mean(dtype=np.float64)); ry += float((b[:-1] * b[1:]).mean(dtype=np.float64))
        rowv += float(b.mean(1).var())
        nb[:] = 0; nb[1:] += b[:-1]; nb[:-1] += b[1:]; nb[0] += b[1]; nb[-1] += b[-2]     # the edges reflect
        nb[:, 1:] += b[:, :-1]; nb[:, :-1] += b[:, 1:]; nb[:, 0] += b[:, 1]; nb[:, -1] += b[:, -2]
        nb *= .25
        sxy += b * nb; sxx += b * b; syy += nb * nb
        rv += float((b * b).mean(dtype=np.float64))
    corr = sxy / np.sqrt(np.maximum(sxx * syy, 1e-12))
    n = len(idx); rv = max(rv, 1e-12)                                   # a fully saturated stack has no variance
    return dict(frames=n, replaced=corr > max(min_corr, 5 / np.sqrt(n)), corr=corr, corr_x=rx / rv, corr_y=ry / rv, row_share=rowv / rv)


def pixel_sample(shape, seed=0):
    """Flat indices of a fixed sample of pixels, the same at every level; None if the sensor is small enough to use them all"""
    size = int(np.prod(shape))
    return None if size <= CURVE_PIXELS else np.sort(np.random.default_rng(seed).choice(size, CURVE_PIXELS, replace=False))


def sensor_curve(sig, exc, nbins=48):
    """Variance against mean for the whole sensor, from a sample of pixels at every level, binned by the pixel's own mean.

    Under uneven light a stack's average says little about saturation: the centre can clip while the average looks safe.
    Returns (bin centres above offset in DN, median excess variance, pixel-level count).
    """
    keep = sig > 1
    sig, exc = sig[keep], exc[keep]
    edges = np.geomspace(max(sig.min(), 1.0), sig.max() * 1.001, nbins + 1)
    idx = np.digitize(sig, edges) - 1
    cen, med, cnt = [], [], []
    for k in range(nbins):
        m = idx == k
        if m.sum() >= 50:
            cen.append(float(np.median(sig[m]))); med.append(float(np.median(exc[m]))); cnt.append(int(m.sum()))
    return np.array(cen), np.array(med), np.array(cnt)


def find_saturation(cen, med):
    """Signal (DN above offset) where the variance stops rising with the mean, or None"""
    if len(cen) < 4:
        return None
    peak = np.maximum.accumulate(med)
    for k in range(3, len(cen)):
        # past the highest variance so far, and well below the straight line through the points before it
        slope = np.median(med[max(0, k - 6):k] / np.maximum(cen[max(0, k - 6):k], 1e-9))
        if med[k] < .7 * peak[k] or med[k] < .6 * slope * cen[k]:
            return float(cen[k - 1])
    return None


def fit_gain(levels, offset, read_var, slope_all, sat_dn=None):
    """Weighted fit per pixel of (variance - read variance) = (mean - offset) / gain, through the origin. levels yields one flat at a time.

    A variance from N frames has relative sd sqrt(2 / (N - 1)), so each level is weighted by (N - 1) / (2 var^2), with var the variance
    the sensor-wide line predicts (slope_all, DN per electron): a pixel's own noisy variance would favour levels where it happens to read low.
    Returns gain, its standard error, and the number of levels each pixel used.
    """
    num = np.zeros_like(offset); den = np.zeros_like(offset); used = np.zeros(offset.shape, np.int16)
    for f in levels:
        sig = f["mean"] - offset; exc = f["var"] - read_var
        ok = sig > 0
        if sat_dn is not None:
            ok &= sig < .9 * sat_dn                                     # per pixel: a clipped centre drops out, its dim edges stay
        w = (f["n"] - 1) / (2 * np.maximum(read_var + np.maximum(sig, 0) * np.float32(slope_all), 1e-3) ** 2)
        w[~ok] = 0
        sw = w * sig
        num += sw * exc; den += sw * sig; used += ok
        del sig, exc, ok, w, sw
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(den > 0, num / den, np.nan)                    # DN per electron
        se_slope = np.where(den > 0, 1 / np.sqrt(den), np.nan)
        del num, den
        gain = 1 / slope
        se = se_slope * gain * gain
    return gain, se, used


def shrink_gain(gain, se):
    """Empirical Bayes: each pixel moves towards its column's gain by how far its own error exceeds the real spread.

    Column amplifiers are the usual structure in sCMOS gain, so the column median is the prior. Returns (shrunk gain,
    real pixel spread as a fraction of the gain, median per-pixel error as a fraction).
    """
    good = np.isfinite(gain) & (gain > 0) & np.isfinite(se)
    g = np.where(good, gain, np.nan)
    col = np.concatenate([np.nanmedian(g[:, i:i + 512], axis=0) for i in range(0, g.shape[1], 512)])[None]    # in parts: a whole-sensor call copies a lot
    col = np.where(np.isfinite(col), col, np.nanmedian(g))
    d = g - col
    # robust spread: a few wild fits must not set the prior
    total = (1.4826 * np.nanmedian(np.abs(d))) ** 2
    noise = np.nanmedian(np.where(good, se, np.nan)) ** 2
    tau2 = max(total - noise, 0.0)
    w = tau2 / (tau2 + np.where(good, se * se, np.inf))
    out = col + w * np.where(good, d, 0.0)
    ref = float(np.nanmedian(g))
    return out, float(np.sqrt(tau2) / ref), float(np.sqrt(noise) / ref)


def local_mean(a, k=15):
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, k, mode="nearest")


def calibrate(dark, flats=(), dark_end=None, h5_dataset="data", log=print, max_ram=16.0, tmp_dir=None, channel=0, timepoint=0):
    warn = []; hits = {}
    source = dict(max_ram=max_ram, log=log, tmp_dir=tmp_dir, h5_dataset=h5_dataset, channel=channel, timepoint=timepoint)
    D = moments(dark, **source)
    offset = D["mean"]; rv = D["var"]; shape = D["shape"]; hits["dark"] = D["hits"]; rv_mean = float(rv.mean(dtype=np.float64))
    sd = np.sqrt(rv)
    log(f"dark: {D['n']} frames, {shape[0]} x {shape[1]}, offset median {np.median(offset):.2f} DN, "
        f"read sd median {np.median(sd):.3f} DN, rms {np.sqrt(rv_mean):.3f} DN")
    raw = raw_frames_for(dark)
    meta = dict(dark_frames=D["n"], shape=list(shape), dark_level_rms_dn=D["level_rms"], max_ram_gb=max_ram)
    if raw is None:
        replaced = np.zeros(shape, bool)
        warn.append(f"no raw dark frames beside {dark} (expected {str(dark)[:-4]}_raw.npy): rebuilt pixels and on-camera filtering went unchecked")
        meta.update(replaced_pixels=None, corr_x=None, corr_y=None, row_common_var_dn2=None)
    else:
        C = temporal_checks(raw, offset, **source); replaced = C["replaced"]
        if abs(C["corr_x"]) > .02 or abs(C["corr_y"]) > .02:
            warn.append(f"neighbouring pixels are correlated in the dark (x {C['corr_x']:+.3f}, y {C['corr_y']:+.3f}): an on-camera "
                        "noise filter may be on; it changes the noise this calibration measures, so take the images with the same setting")
        log(f"pixels following their neighbours in the dark: {int(replaced.sum())} ({100 * replaced.mean():.3f}%), from {C['frames']} frames")
        if C["frames"] < 32:
            warn.append(f"only {C['frames']} raw dark frames: rebuilt pixels can't be told from noise with fewer than about 32")
        if replaced.any():
            warn.append(f"{int(replaced.sum())} pixels ({100 * replaced.mean():.3f}%) follow their neighbours in time: the camera is rebuilding them "
                        "(defect or hot-pixel correction); take the images with the same setting (map: `replaced`)")
        meta.update(replaced_pixels=int(replaced.sum()), corr_x=C["corr_x"], corr_y=C["corr_y"], check_frames=C["frames"],
                    row_common_var_dn2=max(C["row_share"] * rv_mean - rv_mean / shape[1], 0.0))
        del C
    if D["n"] < 1000:
        warn.append(f"{D['n']} dark frames give read variance only to +/-{100 * np.sqrt(2 / (D['n'] - 1)):.0f}% per pixel")
    if dark_end:
        E = moments(dark_end, **source); drift = E["mean"] - offset; hits["dark_end"] = E["hits"]
        if E["shape"] != shape:
            raise ValueError(f"{dark_end}: frame shape {E['shape']} differs from the dark {shape}")
        median_drift = float(np.median(drift)); rms = float(drift.std(dtype=np.float64))
        se = float(np.sqrt(np.median(rv) * (1 / D["n"] + 1 / E["n"])))
        meta.update(offset_drift_dn=median_drift, offset_drift_rms_dn=rms, offset_drift_expected_rms_dn=se)
        log(f"offset drift over the session: {median_drift:+.3f} DN median, rms {rms:.3f} DN (statistics alone: {se:.3f})")
        if abs(median_drift) > .25:
            warn.append(f"offset moved {median_drift:+.2f} DN during the session: let the camera warm up and repeat")
        del E, drift
    out = dict(offset=offset, read_noise_dn=sd, replaced=replaced)
    if flats:
        ref = float(np.median(offset))
        idx = pixel_sample(shape)
        take = (lambda m: m.ravel()) if idx is None else (lambda m: m.ravel()[idx])
        off_s = take(offset); rv_s = take(rv)
        F = []; sig_s = []; exc_s = []
        # each level's maps wait on disk, so memory doesn't grow with the number of levels; plain files, as Windows can't delete a mapped one
        with tempfile.TemporaryDirectory(dir=tmp_dir, prefix="calibrate_") as work:
            names = lambda i: (os.path.join(work, f"{i}_mean.npy"), os.path.join(work, f"{i}_var.npy"))
            for i, p in enumerate(flats):
                f = moments(p, normalise_about=ref, **source)
                if f["shape"] != shape:
                    raise ValueError(f"{p}: frame shape {f['shape']} differs from the dark stack {shape}")
                hits[str(p)] = f["hits"]
                s = take(f["mean"]) - off_s
                # the median of a variance from N frames sits below its mean by this factor, which the sensor-wide line must not inherit
                sig_s.append(s); exc_s.append(take(f["var"]) / (1 - 2 / (9 * (f["n"] - 1))) ** 3 - rv_s)
                F.append(dict(n=f["n"], mu=float(f["mean"].mean(dtype=np.float64)), va=float(f["var"].mean(dtype=np.float64)), level_rms=f["level_rms"]))
                a, b = names(i); np.save(a, f["mean"]); np.save(b, f["var"])
                del f, s
            levels = lambda: ({"mean": np.load(names(i)[0]), "var": np.load(names(i)[1]), "n": F[i]["n"]} for i in range(len(F)))
            log("flats: mean level (DN above offset), p1 to p99 across the sensor, frames")
            for k in sorted(range(len(F)), key=lambda k: F[k]["mu"]):
                lo, hi = np.percentile(sig_s[k], [1, 99])
                log(f"   {F[k]['mu'] - ref:10.1f}  {lo:10.1f} to {hi:10.1f}  {F[k]['n']:8d}")
            cen, med, cnt = sensor_curve(np.concatenate(sig_s), np.concatenate(exc_s))
            del exc_s
            sat = find_saturation(cen, med)
            if sat is not None:
                warn.append(f"variance stops following the mean above about {sat:.0f} DN over offset (saturation or a gain change); pixels are not used above that")
            lim = .9 * sat if sat is not None else np.inf
            use = (cen < lim) & (med > 0)
            if use.sum() < 3:
                warn.append(f"only {int(use.sum())} usable points on the sensor-wide curve")
            # one number per sensor; the shot-noise-dominated points split in two check for a dual-gain knee
            w = cnt[use] / (med[use] + rv_mean) ** 2                    # relative error of a variance is constant, so weight by 1 / variance^2
            gglobal = float((w * cen[use] ** 2).sum() / (w * cen[use] * med[use]).sum()) if use.any() else float("nan")
            bright = cen[use] > max(8 * rv_mean * max(gglobal, 1e-6), 0)
            if bright.sum() >= 6:
                c = cen[use][bright]; v = med[use][bright]; h = len(c) // 2
                lo = float(np.median(c[:h] / v[:h])); hi = float(np.median(c[h:] / v[h:]))
                meta.update(gain_low_levels=lo, gain_high_levels=hi)
                if abs(lo / hi - 1) > .1:
                    warn.append(f"gain differs between low ({lo:.3f}) and high ({hi:.3f}) signal: the camera may be in a dual-gain mode, and one gain per pixel will not fit")
            log(f"sensor-wide gain {gglobal:.4f} e-/DN")
            lowest_top = float(np.percentile(np.max(sig_s, axis=0), 1)) * (gglobal if np.isfinite(gglobal) else 1)
            log(f"dimmest pixels (p1) reach {lowest_top:.0f} e- at their brightest level")
            if lowest_top < 100:
                warn.append(f"the dimmest pixels only reach about {lowest_top:.0f} e-: add a brighter stack, even if the centre saturates in it")
            # the brightest stack where little of the sensor is clipped
            ok_stacks = [k for k, s in enumerate(sig_s) if sat is None or np.mean(s >= .9 * sat) < .01] or list(range(len(F)))
            top = max(ok_stacks, key=lambda k: F[k]["mu"])
            del sig_s, off_s, rv_s
            slope_all = 1 / gglobal if np.isfinite(gglobal) and gglobal > 0 else float(np.median(med[use] / cen[use])) if use.any() else 1.0
            gain_raw, se, used = fit_gain(levels(), offset, rv, slope_all, sat)
            gain, tau, per_pixel = shrink_gain(gain_raw, se)
            log(f"gain per pixel: measured to +/-{100 * per_pixel:.2f}%, real spread between pixels {100 * tau:.2f}%")
            if tau < per_pixel:
                warn.append("per-pixel gain is mostly measurement error at this frame count; the map is close to the column gains")
            sig = np.load(names(top)[0]) - offset
        response = sig / np.maximum(local_mean(sig), 1e-6); del sig
        out.update(gain=gain.astype(np.float32), gain_raw=gain_raw.astype(np.float32), gain_se=se.astype(np.float32),
                   response=response.astype(np.float32), levels_used=used)
        out["read_noise"] = (sd * gain).astype(np.float32)
        meta.update(flat_frames=[f["n"] for f in F], flat_levels_dn=[f["mu"] - ref for f in F], flat_variance_dn2=[f["va"] for f in F],
                    curve_signal_dn=cen.tolist(), curve_excess_variance_dn2=med.tolist(), dimmest_top_electrons=lowest_top, gain_global=gglobal, gain_spread=tau, gain_measurement_error=per_pixel, flat_level_rms_dn=[f["level_rms"] for f in F])
        out["saturation_dn"] = np.float32(sat if sat is not None else np.nan)      # above offset
        rne = out["read_noise"]
    else:
        warn.append("no flats: read noise stays in DN (`read_noise_dn`); gain and read_noise in electrons are not available")
        rne = None
    if rne is not None:
        q = np.percentile(rne, [50, 90, 99, 99.9]); rms = float(np.sqrt((rne.astype(np.float64) ** 2).mean()))
        log(f"read noise (e-): median {q[0]:.2f}, p90 {q[1]:.2f}, p99 {q[2]:.2f}, p99.9 {q[3]:.2f}, rms {rms:.2f}")
        meta.update(read_noise_median=float(q[0]), read_noise_rms=rms)
    if any(v is not None for v in hits.values()):
        total = sum(v or 0 for v in hits.values()); meta["cosmic_hits_removed"] = total
        log(f"pixels with a cosmic-ray hit removed: {total}")
    missing = [p for p, v in hits.items() if v is None and str({"dark": dark, "dark_end": dark_end}.get(p, p)).lower().endswith(".npz")]
    if missing:
        warn.append(f"images without `max`, so cosmic-ray hits stay in: {', '.join(missing)}")
    for w in warn:
        log("WARNING: " + w)
    meta["warnings"] = warn
    out["meta"] = np.array(json.dumps(meta))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", required=True, help="dark stack or mean + variance images: light fully blocked")
    ap.add_argument("--flat", nargs="*", default=[], help="flat-field stacks or images, one per light level")
    ap.add_argument("--dark-end", help="short dark capture from the end of the session, to check for drift")
    ap.add_argument("--max-ram", type=float, default=16.0, help="GB this may use; stops early if that can't work (default 16)")
    ap.add_argument("--tmp", help="folder for each flat level's maps while fitting, about 1 GB per level at 150 megapixels (default: beside the output)")
    ap.add_argument("--h5-dataset", default="data")
    ap.add_argument("--channel", type=int, default=0, help=".ims channel (default 0)")
    ap.add_argument("--timepoint", type=int, default=0, help=".ims timepoint (default 0)")
    ap.add_argument("-o", "--output", default="camera_cal.npz")
    a = ap.parse_args(argv)
    tmp = a.tmp or os.path.dirname(os.path.abspath(a.output)); os.makedirs(tmp, exist_ok=True)
    try:
        r = calibrate(a.dark, a.flat, a.dark_end, a.h5_dataset, max_ram=a.max_ram, tmp_dir=tmp, channel=a.channel, timepoint=a.timepoint)
    except ValueError as e:
        raise SystemExit(str(e))
    np.savez_compressed(a.output, **r)
    print(f"wrote {a.output}: {', '.join(k for k in r if k != 'meta')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

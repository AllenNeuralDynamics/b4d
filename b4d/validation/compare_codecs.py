"""Accuracy and speed of above3 against the original B3D pipeline and seeded dither, on real camera chunks.

    python compare_codecs.py chunks.npz --conversion 1.2987 --offset 34 --read-noise 3.48

chunks.npz holds uint16 arrays of one shape (z, y, x). Results are the numbers in docs/COMPARISON.md: per-chunk medians
for accuracy; for speed, all chunks as one array through compress_tiles (independent tiles, as the writer does), best of 3.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import b4d


def box(a, r=2):
    p = a.astype(float)
    for ax in range(3):
        c = np.cumsum(np.pad(p, [(r + 1, r) if i == ax else (0, 0) for i in range(3)], mode="edge"), axis=ax)
        n = p.shape[ax]
        p = (np.take(c, range(2 * r + 1, n + 2 * r + 1), axis=ax) - np.take(c, range(0, n), axis=ax)) / (2 * r + 1)
    return p


def otsu(a):
    h, e = np.histogram(a, bins=512)
    c = (e[:-1] + e[1:]) / 2
    w, m = np.cumsum(h), np.cumsum(h * c)
    return c[np.argmax((m[-1] / w[-1] * w - m) ** 2 / np.maximum(w * (w[-1] - w), 1))]


def corr(e, ax):
    e = e - e.mean()
    return (np.take(e, range(e.shape[ax] - 1), axis=ax) * np.take(e, range(1, e.shape[ax]), axis=ax)).mean() / e.var()


def accuracy(raw, dec, blob_size):
    err = dec.astype(float) - raw
    s = raw.shape
    bz = s[1] // 8 * 8, s[2] // 8 * 8                                     # local bias: 8x8 patch means, binned by raw DN
    fold = lambda a: a[:, :bz[0], :bz[1]].astype(float).reshape(s[0], bz[0] // 8, 8, bz[1] // 8, 8).mean((2, 4)).ravel()
    idx = np.floor(fold(raw)).astype(int)
    cnt, tot = np.bincount(idx), np.bincount(idx, weights=fold(err))
    ok = cnt >= 500
    m = int(max(raw.max(), dec.max())) + 1
    hr, hd = box(raw), box(dec)
    return dict(ratio=raw.nbytes / blob_size, bias=abs(err.mean()), maxbias=float(np.abs(tot[ok] / cnt[ok]).max()),
                rms=float(np.sqrt((err ** 2).mean())),
                hist=0.5 * np.abs(np.bincount(raw.ravel(), minlength=m) - np.bincount(dec.ravel(), minlength=m)).sum() / raw.size,
                errz=corr(err, 0), erry=corr(err, 1), errx=corr(err, 2),
                noisey=corr(dec - hd, 1) - corr(raw - hr, 1), noisex=corr(dec - hd, 2) - corr(raw - hr, 2))


def dice(raw, dec):
    r, d = box(raw), box(dec)
    t = otsu(r)
    f = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())
    return f(r > t, d > otsu(d)), f(r > t, d > t)                         # each image's own threshold; the raw image's for both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("--conversion", type=float, required=True)
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--read-noise", type=float, required=True)
    ap.add_argument("--dice-min-p99", type=float, default=100.0, help="Dice on chunks whose p99 is at least this (DN)")
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()
    cam = b4d.Camera(args.conversion, args.offset, args.read_noise)
    data = np.load(args.chunks)
    methods = {
        "above3 m1 rice": dict(mode=1, predictor="above3", entropy="rice"),
        "above3 m1 ans": dict(mode=1, predictor="above3", entropy="ans"),
        "dither m2 rice": dict(mode=2, predictor="above", entropy="rice", dither=True),
        "B3D m1": dict(mode=1, predictor="b3d"),
        "B3D m2": dict(mode=2, predictor="b3d"),
        "above m2 rice": dict(mode=2, predictor="above", entropy="rice"),
    }
    print("accuracy: median over", len(data.files), "chunks")
    for q in (1.0, 2.0, 4.0):
        for name, kw in methods.items():
            rows, dc = [], []
            for key in data.files:
                raw = data[key]
                blob = b4d.compress(raw, q=q, camera=cam, **kw)
                dec = b4d.decompress(blob)
                rows.append(accuracy(raw, dec, len(blob)))
                if np.percentile(raw, 99) >= args.dice_min_p99:
                    dc.append(dice(raw, dec))
            med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
            print(f"q={q:.0f} {name:15s} " + " ".join(f"{k} {v:.3f}" for k, v in med.items()) +
                  f" dice own {np.min(dc, 0)[0]:.4f}/{np.median(dc, 0)[0]:.4f} raw-thr {np.min(dc, 0)[1]:.4f}/{np.median(dc, 0)[1]:.4f} (min/med, n={len(dc)})", flush=True)

    if args.skip_speed:
        return

    all_ = np.concatenate([data[k] for k in data.files])
    tile = data[data.files[0]].shape
    mb = all_.nbytes / 1e6
    print("speed: MB/s,", mb, "MB through compress_tiles")
    for th in (1, 8):
        for q in (1.0, 2.0):
            for name, kw in methods.items():
                enc = dec = 0.0
                for _ in range(3):
                    t = time.perf_counter()
                    blobs = b4d.compress_tiles(all_, tile, q=q, camera=cam, threads=th, **kw)
                    enc = max(enc, mb / (time.perf_counter() - t))
                    with ThreadPoolExecutor(th) as ex:
                        t = time.perf_counter()
                        list(ex.map(b4d.decompress, blobs))
                        dec = max(dec, mb / (time.perf_counter() - t))
                print(f"threads={th} q={q:.0f} {name:15s} encode {enc:7.0f} decode {dec:7.0f} ratio {all_.nbytes / sum(map(len, blobs)):.2f}", flush=True)


if __name__ == "__main__":
    main()

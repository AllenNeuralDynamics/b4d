"""Figure 1 of README.md (figs/scatter_encdec_doc.png): encoding (a, b) and decoding (c, d) speed against compression
ratio, on 1 core (MB/s) and on 128 cores (total GB/s), for b4d, B3D, SZ3, ZFP, JPEG-LS, JPEG-XL, JPEG 2000 and Blosc2.
Reads the codec speed table data/codec_speed.csv (described in data/README.md); measures nothing.

    python make_scatter_fig.py [--csv FILE] [--out FILE.png] [--theme dark]

Each point is one codec setting at one chunk size, averaged over the three stacks (dark_pre, flat_3.1, flat_50).
Only settings equivalent to b4d at q = 1 are shown (see data/README.md). Color = codec family, marker = chunk size
(64^3 circle, 128^3 square, 256^3 triangle); the three chunk sizes of a setting are joined by a thin line. The lossless
and the lossy settings are each enclosed by a shaded zone: the minimum-area ellipse holding all of the zone's points
(x linear, y in decades, scaled so the fit is isotropic on screen), padded so the markers sit inside it. Codec names are
placed where they overlap no marker, no other label and no zone label.

Style: Arial 12 pt, 1 pt lines, outward major and minor ticks, no top / right spines, faint major gridlines, white paper;
1600 px wide. Layout: four panels, each half the page wide; the chunk-size legend and the notes below."""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.lines import Line2D
from matplotlib.path import Path

FS = 12                     # all text (pt)
WPX = 1600                  # saved width (px)
STYLE = {"font.family": "Arial", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
         "ytick.labelsize": FS, "legend.fontsize": FS, "figure.titlesize": FS,
         "axes.linewidth": 1.0, "lines.linewidth": 1.0, "patch.linewidth": 1.0, "grid.linewidth": 1.0,
         "lines.markeredgewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 1.0,
         "ytick.minor.width": 1.0, "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 3, "ytick.minor.size": 3,
         "xtick.direction": "out", "ytick.direction": "out", "xtick.minor.visible": True, "ytick.minor.visible": True,
         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 150,
         "figure.dpi": 100, "axes.unicode_minus": True, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
         "mathtext.default": "regular", "legend.handlelength": 1.2, "legend.borderaxespad": 0, "legend.columnspacing": 0.8}
# theme: --theme dark (or B4D_GUIDE_THEME=dark) draws the figure for a dark page into figs/dark, in the page colours of
# make_m1_a3_guide.py (GitHub's dark page)
THEME = "dark" if "--theme=dark" in sys.argv or ("--theme" in sys.argv and sys.argv[sys.argv.index("--theme") + 1] == "dark") \
    or os.environ.get("B4D_GUIDE_THEME") == "dark" else "light"
if THEME == "dark":
    PAPER, INK, MUTED = "#0d1117", "#e6edf3", "#8b949e"
    # zone fills: the same neutral tints, each with a lighter text colour of the same hue
    ZONE = {"Lossless": ("#c9b98f", "#d8c89c"), "Lossy": ("#a9bccf", "#b4c8dc")}
else:
    PAPER, INK, MUTED = "#ffffff", "#0b0b0b", "#8c8b87"
    # zone fills: neutral tints that are none of the series colours, each with a darker text colour of the same hue
    ZONE = {"Lossless": ("#c9b98f", "#7a6a40"), "Lossy": ("#a9bccf", "#4f6478")}
STYLE.update({"savefig.facecolor": PAPER, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
              "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
plt.rcParams.update(STYLE)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILIES = [
    ("b4d", lambda s: s.startswith("m1_a3")),
    ("B3D", lambda s: s.startswith("b3d")),
    ("SZ3", lambda s: s.startswith("sz3")),
    ("ZFP", lambda s: s.startswith("zfp")),
    ("JPEG-LS", lambda s: s.startswith("jls")),
    ("JPEG-XL", lambda s: s.startswith("jxl")),
    ("JPEG 2000", lambda s: s.startswith("j2k")),
    ("Blosc2", lambda s: s.startswith("blosc2")),
]
# the q = 1 equivalent of every codec: b4d and B3D at q = 1; SZ3, ZFP and JPEG-LS in noise units with error bound 0.5
# (= q/2 at q = 1); JPEG-XL and JPEG 2000 at the settings tuned to b4d's q = 1 ratio; Blosc2 (lossless). Plain LZ4 and
# Zstd are left out: Blosc2 with byte shuffle beats each on ratio and encode speed.
KEEP = {"m1_a3_xv", "b3d_m1", "sz3_nu0.5", "zfp_nu0.5", "jls_nu0.5", "jxl_a", "j2k_a",
        "blosc2_lz4", "blosc2_zstd1", "blosc2_zstd3"}
LOSSLESS = {"blosc2_lz4", "blosc2_zstd1", "blosc2_zstd3"}
# the lossless settings are labelled one by one, so the different settings are visible
LL_NAME = {"blosc2_lz4": "Blosc2 LZ4", "blosc2_zstd1": "Blosc2 Zstd 1", "blosc2_zstd3": "Blosc2 Zstd 3"}
LEVELS = ("dark_pre", "flat_3.1", "flat_50")     # stacks averaged
SIZE_MARK = {64: "o", 128: "s", 256: "^"}
MS = 46                     # marker area (pt^2)
ALPHA = 0.9
XK = 1.0                    # x units per y decade on screen, so the ellipse fit and its padding are isotropic
ASPECT = 0.62               # panel height / width


def family(s):
    return next(i for i, (_, f) in enumerate(FAMILIES) if f(s))


def load(path):
    """{(setting, chunk size, cores): mean ratio, codec_enc, codec_dec over LEVELS}"""
    acc = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["setting"] not in KEEP or r["level"] not in LEVELS:
                continue
            acc[(r["setting"], int(r["size"]), int(r["n"]))].append(r)
    return {k: {f: float(np.mean([float(x[f]) for x in rows])) for f in ("ratio", "codec_enc", "codec_dec")}
            for k, rows in acc.items()}


def ellipse(pts, pad=0.12, n=240):
    """the minimum-area ellipse holding all points (Khachiyan; x linear scaled by XK, y in log10), grown by pad in x and
    y so the markers sit inside it (pad added to both semi-axes); returns data x, y"""
    p = np.asarray(pts, float).copy(); p[:, 1] = np.log10(p[:, 1])
    p[:, 0] = p[:, 0] / XK; N = len(p); Q = np.vstack([p.T, np.ones(N)]); u = np.full(N, 1 / N)
    for _ in range(5000):
        X = Q @ np.diag(u) @ Q.T; M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv(X), Q)
        j = np.argmax(M); step = (M[j] - 3) / (3 * (M[j] - 1)); nu = (1 - step) * u; nu[j] += step
        if np.linalg.norm(nu - u) < 1e-9:
            u = nu; break
        u = nu
    c = p.T @ u; A = np.linalg.inv(p.T @ np.diag(u) @ p - np.outer(c, c)) / 2
    w, V = np.linalg.eigh(A); t = np.linspace(0, 2 * np.pi, n)
    e = c + (V @ ((1 / np.sqrt(w) + pad)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T
    return e[:, 0] * XK, 10 ** e[:, 1]


class Placer:
    """greedy label placement in display pixels: a label's box may not touch any marker (as a disc), other label or the
    axes' edge"""

    def __init__(self, ax, fig, pts, rad, lines=(), lrad=3.0, owner=None):
        self.ax, self.fig = ax, fig
        self.r = fig.canvas.get_renderer()
        P = [ax.transData.transform(np.asarray(pts))]; R = [np.full(len(pts), rad)]
        self.M = P[0]; self.owner = np.asarray(owner if owner is not None else [None] * len(pts), dtype=object)
        E = [np.zeros(len(pts), bool)]
        for ln in lines:                                       # every drawn line, sampled every ~2 px
            xs, ys = ln[0], ln[1]; ell = len(ln) > 2 and ln[2] == "ellipse"
            q = ax.transData.transform(np.column_stack([xs, ys]))
            for a, b in zip(q[:-1], q[1:]):
                k = max(2, int(np.hypot(*(b - a)) / 2)); s_ = a + (b - a) * np.linspace(0, 1, k)[:, None]
                P.append(s_); R.append(np.full(k, lrad)); E.append(np.full(k, ell))
        self.P = np.vstack(P); self.R = np.concatenate(R); self.E = np.concatenate(E); self.boxes = []
        self.relaxed = False                                   # True: the zone outlines may be crossed

    def _free(self, bb):
        ab = self.ax.get_window_extent(self.r)
        if bb.x0 < ab.x0 + 2 or bb.x1 > ab.x1 - 2 or bb.y0 < ab.y0 + 2 or bb.y1 > ab.y1 - 2:
            return False
        if any(bb.overlaps(b) for b in self.boxes):
            return False
        P, R = (self.P[~self.E], self.R[~self.E]) if self.relaxed else (self.P, self.R)
        cx = np.clip(P[:, 0], bb.x0, bb.x1); cy = np.clip(P[:, 1], bb.y0, bb.y1)
        return not np.any((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2 < R ** 2)

    def place(self, text, anchors, angles=None, leader=False, who=None, avoid=(), inside=None, maxdist=280, **kw):
        """every candidate spot (distances x anchors x angles) whose box is free; the best scores lowest:
        distance to its anchor + a penalty for other codecs' markers within 20 px of the box (so a label reads as
        belonging to its own points). A thin leader line to the anchor when the label sits more than 20 px away."""
        angs = np.deg2rad(np.arange(0, 360, 22.5) if angles is None else np.asarray(angles))
        t = self.ax.text(0, 0, text, **kw); e = t.get_window_extent(self.r); t.remove()
        w, h = e.width + 4, e.height + 2
        other = self.M[self.owner != who] if who is not None else np.empty((0, 2))
        best = None
        for dist in [d_ for d_ in (10, 14, 20, 27, 36, 48, 62, 80, 100, 130, 170, 220, 280) if d_ <= maxdist]:
            for ax_, ay_ in anchors:
                A = self.ax.transData.transform((ax_, ay_))
                for ang in angs:
                    dx, dy = dist * np.cos(ang), dist * np.sin(ang)
                    ha = "left" if dx > 3 else ("right" if dx < -3 else "center")
                    va = "bottom" if dy > 3 else ("top" if dy < -3 else "center")
                    x0 = A[0] + dx - (0 if ha == "left" else (w if ha == "right" else w / 2))
                    y0 = A[1] + dy - (0 if va == "bottom" else (h if va == "top" else h / 2))
                    bb = matplotlib.transforms.Bbox.from_extents(x0, y0, x0 + w, y0 + h)
                    if not self._free(bb):
                        continue
                    if any(pa.intersects_bbox(bb, filled=True) for pa in avoid):
                        continue
                    if inside is not None and not inside.contains_points(bb.corners()).all():
                        continue
                    sc = dist
                    if len(other):
                        cx = np.clip(other[:, 0], bb.x0, bb.x1); cy = np.clip(other[:, 1], bb.y0, bb.y1)
                        dmin = np.sqrt(((other[:, 0] - cx) ** 2 + (other[:, 1] - cy) ** 2).min())
                        sc += 0.6 * max(0.0, 20 - dmin)
                    if best is None or sc < best[0]:
                        best = (sc, ax_, ay_, dx, dy, ha, va, bb, dist)
            if best is not None and best[0] <= dist:          # nothing farther out can score lower
                break
        if best is None:
            if not self.relaxed:                               # crowded: let it cross a zone outline (never a marker)
                self.relaxed = True
                try:
                    return self.place(text, anchors, angles, leader, who, avoid, inside, maxdist, **kw)
                finally:
                    self.relaxed = False
            raise RuntimeError(f"no free spot for label {text!r}")
        _, ax_, ay_, dx, dy, ha, va, bb, dist = best
        self.boxes.append(bb.expanded(1, 1))
        ap = dict(arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=1, shrinkB=np.sqrt(MS) / 2 + 1)) \
            if leader and dist > 20 else {}
        return self.ax.annotate(text, (ax_, ay_), xytext=(dx * 72 / self.fig.dpi, dy * 72 / self.fig.dpi),
                                textcoords="offset points", ha=ha, va=va, zorder=5, **ap, **kw)


def panel(ax, fig, d, key, n, sc, settings):
    """markers, lines and zones of one panel; returns the label placement, to call after the final layout"""
    global XK
    pts_all, zones, lines, owner = [], {True: [], False: []}, [], []
    fam_pts = defaultdict(list)
    for s in settings:
        fi = family(s); xs, ys = [], []
        for c in (64, 128, 256):
            r = d.get((s, c, n))
            if r is None:
                continue
            x, y = r["ratio"], r[key] * sc
            xs.append(x); ys.append(y); zones[s in LOSSLESS].append((x, y)); pts_all.append((x, y))
            who = LL_NAME.get(s, FAMILIES[fi][0])
            fam_pts[who].append((x, y)); owner.append(who)
            ax.scatter(x, y, s=MS, marker=SIZE_MARK[c], color=SLOTS[fi], alpha=ALPHA, linewidths=0, zorder=3)
        ax.plot(xs, ys, color=SLOTS[fi], lw=1, alpha=0.5, zorder=2); lines.append((xs, ys))
    zl = {}
    P_ = np.asarray(pts_all); Sx = np.ptp(P_[:, 0]) * 1.25; Dy = np.ptp(np.log10(P_[:, 1])) + 0.6
    XK = ASPECT * Sx / Dy                        # width spans ~Sx ratio units, height ~Dy decades
    for lossless, nm in ((True, "Lossless"), (False, "Lossy")):
        ex, ey = ellipse(zones[lossless])
        ax.fill(ex, ey, color=ZONE[nm][0], alpha=0.22, lw=0, zorder=0)     # very light fill, no edge
        zl[nm] = (ex, ey)
    ax.set_xscale("linear"); ax.set_yscale("log")
    allx = np.concatenate([zl[k][0] for k in zl]); ally = np.concatenate([zl[k][1] for k in zl])
    ax.set_xlim(max(0, allx.min() - 0.3), allx.max() + 0.3)
    ax.set_ylim(ally.min() / 1.3, ally.max() * 3.5)                          # room for "Lossless" above
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(mt.LogLocator(base=10, numticks=12))
        a_.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        a_.set_minor_formatter(mt.NullFormatter())
    ax.xaxis.set_major_locator(mt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mt.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    return lambda: labels(ax, fig, pts_all, lines, owner, zl, fam_pts)


def labels(ax, fig, pts_all, lines, owner, zl, fam_pts):
    """label placement, in display pixels: call only after the figure's final layout"""
    fig.canvas.draw()
    rad = np.sqrt(MS) / 2 * fig.dpi / 72 + 2                         # marker radius in pixels, plus a margin
    P = Placer(ax, fig, pts_all, rad, lines, owner=owner)
    paths = {k: Path(ax.transData.transform(np.column_stack(v))) for k, v in zl.items()}
    outside = []
    for nm, other in (("Lossless", "Lossy"), ("Lossy", "Lossless")):  # zone labels inside their own ellipse, near
        ex, ey = zl[nm]                                                  # its top, outside the other one
        cx, cy = ex.mean(), np.exp(np.log(ey).mean())
        anchors = []
        for f in (0.85, 0.75, 0.65, 0.55, 0.45):                         # points on shrunken copies, highest first
            sx, sy = cx + (ex - cx) * f, cy * (ey / cy) ** f
            o = np.argsort(-np.log(sy))[:30:3]
            anchors += [(sx[i], sy[i]) for i in o]
        kw = dict(angles=(270, 0, 180, 315, 225, 90), inside=paths[nm], fontsize=FS, color=ZONE[nm][1])
        try:
            P.place(nm, anchors, avoid=[paths[other]], **kw)
        except RuntimeError:                                             # no room outside the overlap
            try:
                P.place(nm, anchors, **kw)
            except RuntimeError:                                         # too narrow: placed just above it, after
                outside.append((nm, other, ex, ey, kw))                  # the codec labels
    for nm in fam_pts:                                                # codec labels next to their own points
        p = np.array(fam_pts[nm]); g = np.exp(np.log(p).mean(0))
        order = np.argsort(np.hypot(*(np.log(p) - np.log(g)).T))
        P.place(nm, [tuple(p[i]) for i in order], leader=True, who=nm, fontsize=FS, color=INK)
    for nm, other, ex, ey, kw in outside:                             # off the other zone, by its own top
        kw.pop("inside"); kw["angles"] = (90, 60, 120, 30, 150, 0)
        top = [(ex[i], ey[i]) for i in np.argsort(-ey)[:40:2]]
        try:
            P.place(nm, top, avoid=[paths[other]], maxdist=27, **kw)        # stays by its own zone
        except RuntimeError:                                          # no room: anywhere free above its zone
            i = int(np.argmax(ey)); kw.pop("angles", None)
            ax.annotate(nm, (ex[i], ey[i]), xytext=(0, 20), textcoords="offset points", ha="center", va="bottom",
                        **{k: v for k, v in kw.items() if k in ("fontsize", "color", "fontstyle")})


def figure(d, path):
    """encode (a, b) above decode (c, d), each panel half the page wide; the chunk-size legend and the notes below"""
    settings = sorted({s for s, _, _ in d}, key=lambda s: (family(s), s))
    W = 11.2; S = W / 2; L = 1.1; wA = S - L - 0.2; hA = ASPECT * wA
    BOT, GAP, TOP = 2.0, 1.1, 0.5; H = BOT + 2 * hA + GAP + TOP
    fig = plt.figure(figsize=(W, H)); later = []; axs = []
    for y0, key, what in ((BOT + hA + GAP, "codec_enc", "Encode"), (BOT, "codec_dec", "Decode")):
        row = [fig.add_axes([x0 / W, y0 / H, wA / W, hA / H]) for x0 in (L, S + L)]
        for ax, n, lab in ((row[0], 1, f"{what} speed,\n1 core (MB/s)"), (row[1], 128, f"{what} speed,\n128 cores (GB/s)")):
            ax.set_xlabel("Compression ratio"); ax.set_ylabel(lab)
            ax.set_title(f"{what}, {n} core{'s' if n > 1 else ''}", loc="center")
            later.append(panel(ax, fig, d, key, n, 1.0 if n == 1 else 1e-3, settings))
            ax.grid(True, which="major", color=MUTED, alpha=0.25, lw=0.5); ax.set_axisbelow(True)
        axs.append(row)
    h2 = [Line2D([], [], ls="", marker=SIZE_MARK[c], ms=9, color=MUTED, alpha=ALPHA, mew=0, label=f"{c}³ chunks")
          for c in (64, 128, 256)]
    lg = fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(L / W, (BOT - 0.75) / H), ncol=4, title="Chunk size:",
                    alignment="left", borderaxespad=0, columnspacing=1.5)
    lg.get_title().set_fontsize(FS)
    lg._legend_box.align = "left"
    fig.text(L / W, (BOT - 1.4) / H,
             "Codecs are labelled in the plots; the shaded zones mark the lossless and the lossy codecs.\n"
             "Blosc2 (lossless): byte shuffle (2-byte items), then LZ4 (clevel 5) or Zstd (clevel 1 or 3).",
             ha="left", va="top", color=MUTED)
    for f in later:
        f()
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()   # panel letters
    for row, ks in zip(axs, ("ab", "cd")):
        ytop = max(ax.get_tightbbox(r).transformed(inv).y1 for ax in row)
        for ax, k in zip(row, ks):
            fig.text(ax.get_tightbbox(r).transformed(inv).x0 + 0.002, ytop, k, fontsize=14, fontweight="bold",
                     fontfamily="Arial", ha="left", va="top")
    fig.savefig(path, dpi=WPX / W); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "data", "codec_speed.csv"),
                    help="codec speed table (default: data/codec_speed.csv)")
    ap.add_argument("--theme", choices=("light", "dark"), default=THEME, help="page colours (dark: figs/dark)")
    ap.add_argument("--out", default=os.path.join(here, "figs", *(["dark"] if THEME == "dark" else []), "scatter_encdec_doc.png"),
                    help="output PNG (default: figs/scatter_encdec_doc.png, or figs/dark/ with --theme dark)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    figure(load(a.csv), a.out)
    print(a.out)


if __name__ == "__main__":
    main()

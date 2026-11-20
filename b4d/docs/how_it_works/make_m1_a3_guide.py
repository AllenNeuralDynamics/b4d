"""Figures and numbers for README.md (How it works): b4d mode 1 + 'above3' predictor + Rice coder at q = 1, with the
noise-unit table (format v11 = v10 with the table: 8 Rice streams, Camera(exact_vst=True)), followed on a 5 x 5 patch, step by step. The encoder steps are
re-implemented here in plain Python (as b4d_blosc2.c computes them: lut32_get / m1_row / rice_group_plan) and checked
against the real codec before any figure is drawn:
  * every Rice bit stream of the example chunk (real exaSPIM data, example_chunk.npy), rebuilt here, must appear byte for byte inside b4d.compress()'s output
    (this pins the noise units, the predictions and every stored difference r);
  * the decoded pixels shown are b4d.decompress()'s, and each must lie within the q/2 bound of the rebuilt value.

    PYTHONPATH=<b4d build> B4D_DLL_DIR=<blosc2 lib dir> python docs/how_it_works/make_m1_a3_guide.py   # docs/how_it_works/figs/*.png

B4D_GUIDE_THEME=dark draws the same figures for a dark page (figs/dark); B4D_GUIDE_FACTS=<file.json> also writes the
numbers the text quotes.

Style: Arial 12 pt everywhere (panel letters 14 pt bold), figures 1600 px wide; plot axes with outward major
and minor ticks, no top / right spines); one cell size, font and colour scheme for every pixel grid: raw ADU grey, noise
units blue, predictions purple, the stored difference r green (positive) / pink (negative), folded u green, errors
orange / purple. Figures carry panel titles and axis labels only; the explanation is in the markdown.
"""
import math, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.stats import norm, poisson
import b4d

HERE = os.path.dirname(os.path.abspath(__file__))
# theme (figures that follow a light or dark page): B4D_GUIDE_THEME=dark draws the same figures for a dark
# page into figs/dark; the .md shows each with <picture> and prefers-color-scheme
THEME = os.environ.get("B4D_GUIDE_THEME", "light"); assert THEME in ("light", "dark"), THEME
FIG = os.path.join(HERE, "figs", *(["dark"] if THEME == "dark" else [])); os.makedirs(FIG, exist_ok=True)
FACTS = os.environ.get("B4D_GUIDE_FACTS")               # optional: where to write the numbers quoted in the text


def dump_facts():
    if FACTS:
        with open(FACTS, "w") as f_:
            json.dump(facts, f_, indent=1)

# ---- camera (beta3 IMX411, the calibration of the validation runs) and codec settings
CONV, OFFSET, RN = 1.3733347158235059, 34.0927734375, 2.6469157347620023     # ADU per electron, ADU, electrons
Q = 1.0
CAM = b4d.Camera(CONV, OFFSET, RN, exact_vst=True)       # the noise-unit table (variance objective)
CAMA = b4d.Camera(CONV, OFFSET, RN)                      # the old analytic curve, for comparison only
conv, off, rn = (float(np.float32(v)) for v in (CONV, OFFSET, RN))
FIX = 256                                                # fixed point: 1/256 of a noise unit
VST = b4d.exact_vst(CAM)
TAB = VST["table"]                                       # T(x), x = 0..65535 ADU
LUT = np.rint(TAB * FIX).astype(np.int64)                # lut32_get: rnd_d(T(x) * 256)


def T_ans(x):
    """the analytic curve (generalised Anscombe), linear below the offset"""
    x = np.asarray(x, float); e = (x - off) / conv; c = rn * rn + 0.375
    return np.where(e >= 0, 2.0 * np.sqrt(np.maximum(e, 0) + c), 2.0 * np.sqrt(c) + e / np.sqrt(c))


# ---- the steps, as the C encoder does them
def mean3(s, n):
    """rounded mean of n integers summing to s (sign-symmetric, as mean3d_row_i32)"""
    return (s + n // 2) // n if s >= 0 else -((-s + n // 2) // n)


def predict_row(prev, w, y, cur_left=None):
    """above3 predictions for row y from the previous row of decoder values (prev); row 0 uses the left neighbour"""
    p = np.zeros(w, np.int64)
    for x in range(w):
        if y == 0:
            p[x] = 0 if x == 0 else cur_left[x - 1]
        elif w == 1:
            p[x] = prev[0]
        elif x == 0:
            p[x] = mean3(prev[0] + prev[1], 2)
        elif x == w - 1:
            p[x] = mean3(prev[x - 1] + prev[x], 2)
        else:
            p[x] = mean3(prev[x - 1] + prev[x] + prev[x + 1], 3)
    return p


def encode(img, q=Q, open_loop=False):
    """mode 1, above3: per pixel the noise-unit value t, prediction p, residual r (steps of q), rebuilt value f (t, p, f in
    1/256 noise units). open_loop: predict from the true values instead of the decoder's copy (what b4d does NOT do);
    f is then what a decoder, which only has its own copy, would rebuild."""
    h, w = img.shape; qf = int(round(q * 4)) * (FIX // 4)
    t = LUT[img.astype(np.int64)]
    p = np.zeros((h, w), np.int64); r = np.zeros((h, w), np.int64); f = np.zeros((h, w), np.int64)
    for y in range(h):
        if y == 0:
            for x in range(w):
                src = t[0] if open_loop else f[0]
                p[0, x] = 0 if x == 0 else src[x - 1]
                r[0, x] = int(np.rint((t[0, x] - p[0, x]) / qf))
                f[0, x] = (0 if x == 0 else f[0, x - 1]) + r[0, x] * qf if open_loop else p[0, x] + r[0, x] * qf
            continue
        p[y] = predict_row(t[y - 1] if open_loop else f[y - 1], w, y)
        r[y] = np.rint((t[y] - p[y]) / qf).astype(np.int64)
        f[y] = (predict_row(f[y - 1], w, y) if open_loop else p[y]) + r[y] * qf
    return t, p, r, f


def zigzag(r):
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def rice_code(u, k):
    qn = u >> k
    if qn >= 31:
        raise ValueError("escape (not in this example)")
    return "1" * qn + "0" + (format(u & ((1 << k) - 1), f"0{k}b") if k else "")


def rice_plan(u):
    """k for one group of up to 128 values: start from the mean, try k-1, k, k+1, keep the cheapest (rice_group_plan);
    15 = all zero, 14 = sparse"""
    s = int(sum(u)); n = len(u)
    if s == 0:
        return 15, 4, {}
    mean = s // n; k = 0
    while k < 13 and (mean >> (k + 1)):
        k += 1
    cands = sorted({max(k - 1, 0), k, min(k + 1, 13)})
    cost = {kk: 4 + sum(int(v) >> kk for v in u) + n * (kk + 1) for kk in cands}
    best = min(cost, key=lambda kk: (cost[kk], kk))
    nnz = int(np.count_nonzero(u))
    sparse = 4 + 7 + sum(7 + 2 * int(math.floor(math.log2(v))) + 1 for v in u if v) if nnz < 128 else 1 << 30
    if sparse < cost[best]:
        return 14, sparse, cost
    return best, cost[best], cost


def pack(bits):
    bits = bits + "0" * (-len(bits) % 8); return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


# ---- the data: one 64 x 64 x 64 chunk of a real exaSPIM recording (beta3 camera; fetch_example_chunk.py: dataset
# 859684_2026-10-06_16-58-46, channel 488, tile 13, origin z 8120, y 5301, x 10179), coded as b4d codes a chunk: its 64 z
# planes one under the other, one image of 4096 rows of 64 (above3 has no plane boundaries: a plane's first row is
# predicted from the previous plane's last row; only the chunk's first row uses its left neighbour)
CHUNK_NPY = os.path.join(HERE, "example_chunk.npy")
if not os.path.exists(CHUNK_NPY):
    import fetch_example_chunk
    fetch_example_chunk.main()
chunk = np.ascontiguousarray(np.load(CHUNK_NPY))
Z, Y, X = chunk.shape
PZ = 32
R0 = PZ * Y                                              # z plane 32's first row in the chunk's image
img = chunk.reshape(-1, X)
frame = np.ascontiguousarray(chunk[PZ])
tC, pC, rC, fC = encode(img)
tF, pF, rF, fF = (a_[R0:R0 + Y] for a_ in (tC, pC, rC, fC))   # z plane 32's rows of the chunk's coding
uC = zigzag(rC).ravel()
gC = [uC[i:i + 128] for i in range(0, len(uC), 128)]        # the chunk's 2048 groups of 128 values (two rows)
plC = [rice_plan(g) for g in gC]
G0 = R0 * X // 128                                       # z plane 32's first group (its groups are G0 .. G0 + 31)
plF, gF = plC[G0:G0 + Y * X // 128], gC[G0:G0 + Y * X // 128]
_uF = zigzag(rF).ravel()
_kF = [pl[0] for pl in plF]


def find_patch(strict):
    """a 5 x 5 patch with stored differences of both signs, and a followed pixel (centre) with r of +2..+5; strict: no |r|
    above 6 (typical of the chunk, not its sharpest edge, where the codes run to 13 bits)"""
    best = None
    for py in range(1, Y - 5):
        for px in range(1, X - 6):
            rr = rF[py:py + 5, px:px + 5]; c = int(rr[2, 2])
            npos, nneg = int((rr > 0).sum()), int((rr < 0).sum())
            kk_ = _kF[((py + 2) * X + px + 2) // 128]; uu_ = int(_uF[(py + 2) * X + px + 2])
            if strict and (not (2 <= c <= 5) or min(npos, nneg) < 6 or not (1 <= kk_ <= 13) or uu_ < 3 or np.abs(rr).max() > 6):
                continue
            if not strict and (not (1 <= c <= 6) or min(npos, nneg) < 4 or not (0 <= kk_ <= 13)):
                continue
            span = int(frame[py:py + 5, px:px + 5].max()) - int(frame[py:py + 5, px:px + 5].min())
            score = min(npos, nneg) + span / 200.0 + 4 * (kk_ == 2)
            if best is None or score > best[0]:
                best = (score, py, px)
    return best


best = find_patch(True) or find_patch(False)
assert best is not None, "no patch fits the rules"
_, PY, PX = best
HI = (2, 2)
sl = (slice(PY, PY + 5), slice(PX, PX + 5))
patch = np.ascontiguousarray(frame[sl])
t, p, r, f = tF[sl], pF[sl], rF[sl], fF[sl]

# ---- check against the real codec: the whole chunk, its NS Rice streams found byte for byte in b4d's output
NS = 8                                                   # Rice streams (format v10 / v11; v3..v9 had 4)
blob = b4d.compress(chunk, q=Q, mode=1, predictor="above3", camera=CAM, entropy="rice", blocksize=chunk.nbytes)
realC = b4d.decompress(blob).astype(np.int64).reshape(-1, X)
assert all(pl[0] <= 13 for pl in plC), "this chunk has a zero or sparse group; the byte check below assumes plain Rice"
sbits = [""] * NS
for g, (g_u, pl) in enumerate(zip(gC, plC)):
    sbits[g % NS] += format(pl[0], "04b") + "".join(rice_code(int(v), pl[0]) for v in g_u)
sbytes = [pack(b_) for b_ in sbits]
offs = [blob.find(sb) for sb in sbytes]
assert all(o >= 0 for o in offs), ("a stream was not found in the real compressed output", offs)
realF = realC[R0:R0 + Y]
real = realF[sl]
Tdec = TAB[realC]
assert np.abs(Tdec - fC / FIX).max() <= Q / 2 + 0.35, "decoded pixels are not within the bound of the rebuilt values"
gidx = lambda i, j: ((R0 + PY + i) * X + PX + j) // 128   # a patch pixel's group in the chunk
kpx = np.array([[plC[gidx(i, j)][0] for j in range(5)] for i in range(5)])
k, gbits, costs = plC[gidx(*HI)]
codes = [rice_code(int(zigzag(r)[i, j]), int(kpx[i, j])) for i in range(5) for j in range(5)]
# the followed pixel's group: in the chunk (grp_hc) and in z plane 32 (grp_hi, g0 .. g31; the same stream: G0 % NS == 0)
grp_hc = gidx(*HI); grp_hi = grp_hc - G0
assert G0 % NS == 0
grp_bits = format(k, "04b") + "".join(rice_code(int(v), k) for v in gC[grp_hc])

# the whole chunk at q = 1, 2, 3, 4
per_q = {}; r_q = {}
for qq in (1.0, 2.0, 3.0, 4.0):
    tc, pc, rc, fc = (tC, pC, rC, fC) if qq == Q else encode(img, qq)
    r_q[qq] = rc
    blobc = blob if qq == Q else b4d.compress(chunk, q=qq, mode=1, predictor="above3", camera=CAM, entropy="rice", blocksize=chunk.nbytes)
    realc = b4d.decompress(blobc).astype(np.int64).reshape(-1, X)
    uc = zigzag(rc).ravel()
    groups = [uc[i:i + 128] for i in range(0, len(uc), 128)]
    plans = [rice_plan(g) for g in groups]
    gk = [pl[0] for pl in plans]; gb = [pl[1] for pl in plans]
    err = TAB[realc] - TAB[chunk.astype(np.int64)].reshape(-1, X)
    per_q[qq] = dict(blob=len(blobc), ratio=chunk.nbytes / len(blobc), k_hist={int(v): int(c_) for v, c_ in zip(*np.unique(gk, return_counts=True))},
                     streams=[sum(gb[i] for i in range(s_, len(gb), NS)) / 8 for s_ in range(NS)], ngroups=len(groups),
                     max_err_nu=float(np.abs(err).max()), rms_err_adu=float(np.sqrt(np.mean((realc - img) ** 2))),
                     mean_err_adu=float(np.mean(realc - img)), max_rebuilt_err=float(np.abs(fc - tc).max() / FIX),
                     patch=realc.reshape(chunk.shape)[PZ][sl].tolist(), mean_u=float(np.mean(uc)))
    if qq == 1.0:
        chunk_streams_bytes = per_q[qq]["streams"]
        hdr_total = len(blobc) - sum(math.ceil(s_) for s_ in per_q[qq]["streams"])   # includes the 4 x (NS - 4) stream sizes
        open_t, open_p, open_r, open_f = encode(img, qq, open_loop=True)
        per_q[qq]["open_loop_max_err"] = float(np.abs(open_f - open_t).max() / FIX)
        per_q[qq]["open_loop_rms_err"] = float(np.sqrt(np.mean(((open_f - open_t) / FIX) ** 2)))
        per_q[qq]["closed_loop_rms_err"] = float(np.sqrt(np.mean(((fc - tc) / FIX) ** 2)))
# patch bits at q = 2, 3 and 4 with the same group rule
for qq in (2.0, 3.0, 4.0):
    rq = r_q[qq]
    uq = zigzag(rq).ravel(); plq = [rice_plan(uq[a:a + 128]) for a in range(0, len(uq), 128)]
    rqp = rq[R0:R0 + Y][sl]
    bits = 0
    for i in range(5):
        for j in range(5):
            kk = plq[gidx(i, j)][0]
            bits += len(rice_code(int(zigzag(rqp)[i, j]), kk)) if kk <= 13 else 0
    per_q[qq]["patch_bits"] = bits
per_q[1.0]["patch_bits"] = sum(len(c_) for c_ in codes)

# headers of the q = 1 chunk: container (B4DP: magic, version, ndim, shape, npieces, piece size), Blosc2 chunk header +
# block offset + block size, b4d block header (44), the sizes of streams 3..6 (4 x 4 = 16, v10), the curve description (8 + 4 x 48 = 200)
nk = len(VST["g"]); desc = 8 + 4 * nk
container = 7 + 8 * 3 + 4 + 4; blosc = 32 + 4 + 4
facts = dict(
    camera=dict(conversion=CONV, offset=OFFSET, read_noise=RN), patch_at=dict(z=PZ, y=PY, x=PX),
    patch=patch.tolist(), T=(t / FIX).round(2).tolist(), pred=(p / FIX).round(2).tolist(), r=r.tolist(), rebuilt=(f / FIX).round(2).tolist(),
    u=zigzag(r).tolist(), k_patch=kpx.tolist(), k=k, costs={int(a): int(b_) for a, b_ in costs.items()}, group_bits=gbits, codes=codes,
    decoded=real.tolist(), err_nu=(TAB[real] - TAB[patch.astype(np.int64)]).round(3).tolist(),
    hi=dict(raw=int(patch[HI]), T=round(t[HI] / FIX, 2), pred=round(p[HI] / FIX, 2), above=[round(float(fF[PY + HI[0] - 1, PX + HI[1] + d]) / FIX, 2) for d in (-1, 0, 1)], row=int(R0 + PY + HI[0]),
            r=int(r[HI]), u=int(zigzag(r)[HI]), code=rice_code(int(zigzag(r)[HI]), k), rebuilt=round(f[HI] / FIX, 2), decoded=int(real[HI]),
            T_ans=round(float(T_ans(patch[HI])), 2)),
    chunk_stream_bytes=[len(b_) for b_ in sbytes], chunk_stream_offsets=offs, chunk_origin=dict(dataset="859684_2026-10-06_16-58-46", tile=13, z=8120, y=5301, x=10179),
    hi_group=grp_hi, hi_group_chunk=grp_hc, hi_stream=grp_hc % NS, hi_group_hex=pack(grp_bits)[:16].hex(" "), patch_bits=per_q[1.0]["patch_bits"],
    chunk_raw_bytes=chunk.nbytes, per_q=per_q, headers=dict(total=hdr_total, container=container, blosc=blosc, block=44, xs=4 * (NS - 4), curve=desc,
                                                             other=hdr_total - container - blosc - 44 - 4 * (NS - 4) - desc),
    vst=dict(nk=nk, lam_max=float(VST["lam_max"]), max_err=float(VST["max_err"])))
dump_facts()
print(json.dumps({k_: v for k_, v in facts.items() if k_ in ("patch_at", "hi", "k", "costs", "patch_bits", "chunk_stream_bytes",
                                                             "headers", "vst", "hi_group_hex")}, indent=1))
print({q_: {a: v for a, v in d.items() if a != "patch"} for q_, d in per_q.items()})
print(f"checks passed: all {NS} Rice streams of the chunk found byte for byte in the real output; decoded pixels within the bound")

# ================================================================================================================ drawing
STYLE = {"font.family": "Arial", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 12,
         "ytick.labelsize": 12, "legend.fontsize": 12, "figure.titlesize": 12,
         "axes.linewidth": 1.0, "lines.linewidth": 1.0, "patch.linewidth": 1.0, "hatch.linewidth": 1.0, "grid.linewidth": 1.0,
         "lines.markeredgewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 1.0,
         "ytick.minor.width": 1.0, "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 3, "ytick.minor.size": 3,
         "xtick.direction": "out", "ytick.direction": "out", "xtick.minor.visible": True,
         "ytick.minor.visible": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
         "savefig.dpi": 1600 / (14.0 * 0.8), "figure.dpi": 100, "image.interpolation": "nearest", 
         "axes.unicode_minus": True, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
         "legend.handlelength": 1.2, "legend.borderaxespad": 0, "legend.columnspacing": 0.8}
# page colours: PAPER the background, INK text and lines on it, GRIDLINE the edge of an empty cell, EDGE the gap between
# filled cells (the page colour); CELL_INK the text on a light filled cell or box, black in both themes
if THEME == "dark":                                      # GitHub's dark page
    PAPER, INK, MUTED, GRIDLINE = "#0d1117", "#e6edf3", "#8b949e", "#3d444d"
else:
    PAPER, INK, MUTED, GRIDLINE = "#ffffff", "#0b0b0b", "#8c8b87", "#d4d3cf"
EDGE, CELL_INK = PAPER, "#0b0b0b"
STYLE.update({"savefig.facecolor": PAPER, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
              "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
plt.rcParams.update(STYLE)
SCALE = 0.8                                              # geometry in the original inches x 0.8, so the text keeps
_figure = plt.figure                                     # its place; fsize() gives back the unscaled size for inch offsets


def _scaled_figure(*a, figsize=None, **kw):
    if figsize is not None:
        figsize = (figsize[0] * SCALE, figsize[1] * SCALE)
    return _figure(*a, figsize=figsize, **kw)


plt.figure = _scaled_figure


def fsize(fig):
    w, h = fig.get_size_inches(); return w / SCALE, h / SCALE
LIGHT = "#d4d3cf"                                        # single pixels in the PTC
C_ALT = "#eb6834"                                        # second series
HL = "#e08a00"                                           # the followed pixel
C_T, C_P, C_R, C_U = "#2a78d6", "#7b4fc9", "#1a9850", "#2e8b57"
CMAP = dict(raw="Greys", T="Blues", pred="Purples", r="PiYG", u="Greens", err="Reds", code="Greens")
HLW = 1.0                                                # the followed pixel: one black box on the cell's edges
CELL = 0.70                                              # inches per grid cell, every figure
FS = 12                                                  # all text in all figures 12 pt
CFS = 12                                                 # text inside grid cells and other cell-sized boxes
IMG_TICKS = [0, 16, 32, 48, 64]                          # every 64 x 64 pixel image: the same ticks, labelled in pixels


def fmt_num(v, kind):
    if isinstance(v, str):
        return v
    if kind in ("r",):
        v = int(v); return "0" if v == 0 else (f"+{v}" if v > 0 else f"−{-v}")
    if kind in ("tp", "err"):
        s = f"{abs(v):.2f}"; return ("−" if v < 0 else "+") + s if abs(v) >= 0.005 else "0.00"
    if kind in ("T", "pred"):
        return f"{v:.2f}"
    return f"{int(v)}"


def face(kind, v, lim):
    cm = plt.get_cmap(CMAP["r" if kind == "tp" else kind])
    # low values dark, high values light, as in the grey images; not the error scale (the diverging r / tp
    # scale keeps both ends dark: its sign is the colour)
    if kind == "err":                                    # light (no error) to dark red (the largest), so errors stand out
        x = 0.05 + 0.80 * min(1.0, abs(v) / lim)
    elif kind in ("r", "tp"):
        x = 0.5 + 0.42 * max(-1, min(1, v / lim))
    else:
        lo, hi = lim; x = 0.70 - 0.62 * (v - lo) / max(hi - lo, 1e-9)
    c = cm(x)
    return c, CELL_INK                                   # black numbers on every cell


def grid(ax, vals, kind, lim=None, hl=(), outline=(), unknown=(), blank=()):
    """a pixel grid: one cell size (CELL), font (10) and colour map per kind; hl: the followed pixel (one black box, bold),
    outline: cells it is computed from (bold), unknown: '?', blank: not known yet (white)"""
    v = np.asarray(vals, dtype=object); h, w = v.shape
    if kind == "code":                                   # a Rice code: coloured by its length
        num = np.array([[float(len(x)) if x else np.nan for x in row] for row in v])
    else:
        num = np.array([[np.nan if isinstance(x, str) else float(x) for x in row] for row in v])
    if lim is None:
        lim = np.nanmax(np.abs(num)) if kind in ("r", "tp", "err") else (np.nanmin(num), np.nanmax(num))
    for i in range(h):
        for j in range(w):
            if (i, j) in blank or (i, j) in unknown or np.isnan(num[i, j]):
                fc, tc = (PAPER, INK)
            else:
                fc, tc = face(kind, num[i, j], lim)
            ax.add_patch(Rectangle((j, h - 1 - i), 1, 1, facecolor=fc, edgecolor=GRIDLINE if fc == PAPER else EDGE, lw=1))
            if (i, j) in blank:
                continue
            txt = "?" if (i, j) in unknown else fmt_num(v[i, j], kind)
            ax.text(j + 0.5, h - 0.5 - i, txt, ha="center", va="center", fontsize=CFS, color=tc,
                    fontweight="normal")
    for (i, j) in hl:                                    # one black box, on the white lines between the cells
        ax.add_patch(Rectangle((j, h - 1 - i), 1, 1, fill=False, edgecolor=CELL_INK, lw=HLW, zorder=6, clip_on=False))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")


def row_fig(widths, height, gap=0.9, left=0.25, right=0.25, top=0.55, bottom=0.25):
    """one row of panels, sizes in inches; every panel's title at the same height (title())"""
    W = left + sum(widths) + gap * (len(widths) - 1) + right; H = top + height + bottom
    fig = plt.figure(figsize=(W, H)); axs = []; x = left
    for w_ in widths:
        axs.append(fig.add_axes([x / W, bottom / H, w_ / W, height / H])); x += w_ + gap
    fig._geom = (W, H, top, bottom, height)
    return fig, axs


def _register(fig, ax, t):
    """a titled panel: save() re-centres its title after spreading the figure and letters it"""
    if not hasattr(fig, "_titled"):
        fig._titled = []
    fig._titled.append((ax, t))


def title(fig, ax, s):
    W, H, top, bottom, height = fig._geom
    b = ax.get_position()
    t = fig.text((b.x0 + b.x1) / 2, (bottom + height + 0.18) / H, s, ha="center", va="bottom", fontsize=FS, color=INK)
    _register(fig, ax, t)


def arrow(fig, a0, a1, y=0.5):
    b0, b1 = a0.get_position(), a1.get_position()
    yy = b0.y0 + y * (b0.y1 - b0.y0)
    fig.patches.append(FancyArrowPatch((b0.x1 + 0.012, yy), (b1.x0 - 0.012, yy), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color=MUTED, lw=1, shrinkA=0, shrinkB=0))


PAGE_IN = 14.0 * 0.8                                     # 1600 px at savefig.dpi; every figure padded (white, on the right) to this width: shown at
                                                         # the page's width, a grid cell is then CELL / PAGE_IN of it in every figure


def minor_ticks(fig):
    """minor ticks on every numeric linear axis that has none (axes drawn with their own fixed tick list, e.g. category
    labels, keep none); log axes already carry them"""
    for ax in fig.axes:
        if not ax.axison or ax.name == "3d":
            continue
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale != "linear" or isinstance(axis.get_major_locator(), mt.FixedLocator):
                continue
            if isinstance(axis.get_minor_locator(), mt.NullLocator) and len(axis.get_majorticklocs()) > 1:
                axis.set_minor_locator(mt.AutoMinorLocator())


def aspect43(fig):
    """every data plot (axes drawn, no image, not 3-D) in a 4:3 box, centred in the space it was given; cell grids,
    images and the 3-D views keep their geometry"""
    for ax in fig.axes:
        if not ax.axison or ax.name == "3d" or ax.images or getattr(ax, "_free", False) or getattr(fig, "_free", False):
            continue
        y1 = ax.get_position().y1
        ax.set_box_aspect(3 / 4); ax.apply_aspect()
        b = ax.get_position(); ax.set_position(b); ax.set_box_aspect(None)   # the 4:3 box becomes the axes' own position
        for a_, t in getattr(fig, "_titled", []):            # its title follows the box's top
            if a_ is ax:
                t.set_y(t.get_position()[1] - (y1 - b.y1))


def spread(fig):
    """widen the figure to the page (PAGE_IN): every axes keeps its size and its centre moves in proportion, so the gaps
    grow evenly and a row fills the 1600 px canvas. Figure-level texts, arrows and lines move with the panels of their
    own row: x is mapped piecewise linearly through the old and new edges of the axes that share their height."""
    W, H = fig.get_size_inches()
    if abs(W - PAGE_IN) < 1e-6:
        return
    axes = [a for a in fig.axes if a.get_visible()]
    old = {a: a.get_position() for a in axes}
    lo = min(axes, key=lambda a: old[a].x0); hi = max(axes, key=lambda a: old[a].x1)
    cl, wl = (old[lo].x0 + old[lo].x1) / 2 * W, old[lo].width * W
    cr, wr = (old[hi].x0 + old[hi].x1) / 2 * W, old[hi].width * W
    target_hi = PAGE_IN - (W - old[hi].x1 * W)              # same right margin as before
    if cr - cl > 1e-6:                                       # centres: c' = a + b c, left edge and right margin kept
        b_ = ((target_hi - wr / 2) - (old[lo].x0 * W + wl / 2)) / (cr - cl); a_ = old[lo].x0 * W + wl / 2 - b_ * cl
    else:                                                    # one column: centred on the page
        b_, a_ = 1.0, (PAGE_IN - W) / 2
    new = {}
    for a, b in old.items():
        x0, x1 = b.x0 * W, b.x1 * W; c = (x0 + x1) / 2; w = x1 - x0
        new[a] = (a_ + b_ * c - w / 2, a_ + b_ * c + w / 2)

    def mapx(x_in, y_fr):
        """x (inches, old) at height y (figure fraction) -> x (inches, new)"""
        row = [a for a in axes if old[a].y0 - 0.02 <= y_fr <= old[a].y1 + 0.06]
        if not row:
            ys = min(axes, key=lambda a: min(abs(old[a].y0 - y_fr), abs(old[a].y1 - y_fr)))
            row = [a for a in axes if abs(old[a].y0 - old[ys].y0) < 1e-6 or abs(old[a].y1 - old[ys].y1) < 1e-6] or [ys]
        pts = sorted({(old[a].x0 * W, new[a][0]) for a in row} | {(old[a].x1 * W, new[a][1]) for a in row})
        xs = np.array([p_[0] for p_ in pts]); ys_ = np.array([p_[1] for p_ in pts])
        if x_in <= xs[0]:
            return ys_[0] - (xs[0] - x_in)
        if x_in >= xs[-1]:
            return ys_[-1] + (x_in - xs[-1])
        return float(np.interp(x_in, xs, ys_))

    for t in fig.texts:
        x, y = t.get_position(); t.set_x(mapx(x * W, y) / PAGE_IN)
    for pa in fig.patches:
        if isinstance(pa, FancyArrowPatch) and pa._posA_posB is not None:
            (xa, ya), (xb, yb_) = pa._posA_posB
            pa.set_positions((mapx(xa * W, ya) / PAGE_IN, ya), (mapx(xb * W, yb_) / PAGE_IN, yb_))
    for ar in fig.artists:
        if isinstance(ar, Line2D) and ar.get_transform() == fig.transFigure:
            xs, ys = ar.get_xdata(), ar.get_ydata()
            ar.set_xdata([mapx(x_ * W, y_) / PAGE_IN for x_, y_ in zip(xs, ys)])
    fig.set_size_inches(PAGE_IN, H)
    for a in axes:
        b = old[a]; x0, x1 = new[a]
        a.set_position([x0 / PAGE_IN, b.y0, (x1 - x0) / PAGE_IN, b.height])
    for a, t in getattr(fig, "_titled", []):                 # titles centred on their (moved) panel
        b = a.get_position(); t.set_x((b.x0 + b.x1) / 2)


def letters(fig):
    """bold 14 pt Arial a, b, c ... at the top left of each titled panel (level with its title), in reading order;
    none on a figure with a single panel"""
    tl = getattr(fig, "_titled", [])
    if len(tl) < 2:
        return
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    items = []
    for a, t in tl:
        bb = a.get_tightbbox(r).transformed(inv); tb = t.get_window_extent(r).transformed(inv)
        items.append((round(-tb.y0, 2), a.get_position().x0, max(bb.y1, tb.y1), min(bb.x0, tb.x0 - 0.016)))   # last: clear of its title
    order = sorted(items, key=(lambda t: (round(t[1], 2), t[0])) if getattr(fig, "_letter_order", "rows") == "columns" else None)
    for i, (_, _, ytop, x0) in enumerate(order):
        fig.text(max(x0, 0.002), ytop, "abcdefghijklmnop"[i], fontsize=14, fontweight="bold", fontfamily="Arial",
                 ha="left", va="top", color=INK)


def save(fig, name):
    # opaque page background (on a transparent one the text and axes are hard to see where the page colour differs)
    from PIL import Image
    minor_ticks(fig)
    aspect43(fig)
    spread(fig)
    letters(fig)
    pth = os.path.join(FIG, name); fig.savefig(pth); plt.close(fig)
    im = Image.open(pth).convert("RGB"); wpx = int(round(PAGE_IN * plt.rcParams["savefig.dpi"]))
    if im.width > wpx:
        raise ValueError(f"{name} is wider than the page canvas ({im.width} > {wpx} px)")
    if im.width < wpx:
        c = Image.new("RGB", (wpx, im.height), PAPER); c.paste(im, (0, 0)); im = c
    im.save(pth)


G = 5 * CELL                                             # a 5 x 5 grid panel
above3 = [(HI[0] - 1, HI[1] - 1), (HI[0] - 1, HI[1]), (HI[0] - 1, HI[1] + 1)]
Tp, Pp, Fp = (t / FIX).round(2), (p / FIX).round(2), (f / FIX).round(2)
Tlim = (float(min(Tp.min(), Pp.min(), Fp.min())), float(max(Tp.max(), Pp.max(), Fp.max())))
Rlim = max(3, int(np.abs(r).max()))

# ---- fig01: noise grows with brightness; noise units; what the table adds
def xdist(lam):
    """ADU-rounded Poisson(lam) + Gaussian(rn) pixel values: (x, P(x))"""
    nmax = int(lam + 10 * math.sqrt(lam + 1) + 10); n = np.arange(nmax + 1); pn = poisson.pmf(n, lam)
    x = np.arange(max(0, int(off - 8 * conv * rn - conv)), int(off + conv * (nmax + 8 * rn)) + 2)
    hi_ = ((x + 0.5 - off) / conv)[:, None] - n[None, :]; lo_ = ((x - 0.5 - off) / conv)[:, None] - n[None, :]
    px = (norm.cdf(hi_ / rn) - norm.cdf(lo_ / rn)) @ pn
    return x, px / px.sum()


def added_fraction(Tc, lam):
    """variance a step of one noise unit adds, (q^2/12) E[(dADU/dT)^2], over the pixel's own variance"""
    x, px = xdist(lam)
    d = np.diff(Tc); inv = 1 / d ** 2
    s2 = 0.5 * (np.concatenate([[inv[0]], inv]) + np.concatenate([inv, [inv[-1]]]))
    m = (px * x).sum(); var = (px * (x - m) ** 2).sum()
    return (px * s2[x]).sum() / 12 / var


lams = np.concatenate([[0.0], np.geomspace(0.05, 300, 60), np.geomspace(400, 8000, 8)])   # to the brightest flat
TA = T_ans(np.arange(65536))
addA = np.array([added_fraction(TA, l_) for l_ in lams]); addT = np.array([added_fraction(TAB, l_) for l_ in lams])
facts["added_at_dark"] = dict(analytic=float(addA[0]), table=float(addT[0]), target=1 / 12)
dump_facts()

W3_ = 0.3 + 3 * 5 * 0.48 + 2 * 1.15 + 0.3               # section 1 plots: the width of the 1 x 3 grid figures at the 0.48 in cells (kept when cells shrank)
PW = (PAGE_IN / SCALE - 1.0 - 2 * 1.0 - 0.25) / 3; PH = 0.75 * PW   # 4:3 panels filling the page; row_fig adds left 1.0, gaps 1.0, right 0.25
fig, axs = row_fig([PW, PW, PW], PH, gap=1.0, left=1.0, bottom=0.85, top=0.6)
ax = axs[0]
# the photon transfer curve, measured (ptc_beta3.json): each LED flat's photon noise (median
# per-pixel temporal variance with the pixel's dark variance taken out) against its median signal above the offset, on the
# photon-only line; the read noise measured in the darks; the camera model (photon + read) through both
PTC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptc_beta3.json")))
# drawn as a classic photon transfer curve, raw temporal variance (dark not taken out) on log-log axes: single pixels, the
# sensor medians, the dark (read) variance and the camera model; data in ptc_beta3.json["fig1a"] and ptc_beta3_pixels.npz
P1 = PTC["fig1a"]; PXL_ = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptc_beta3_pixels.npz"))
ps_, pv_ = np.array(P1["signal_ADU"]), np.array(P1["variance_ADU2"]); vd_, K_ = P1["dark_variance_ADU2"], P1["K_ADU_per_e"]
pm_, pp_ = PXL_["signal_ADU"], PXL_["variance_ADU2"]; ok_ = pm_ > 0.5
ax.scatter(pm_[ok_], pp_[ok_], s=1, color=LIGHT, lw=0, rasterized=True, label="Single pixels", zorder=1)
xe = np.logspace(0, 4.1, 100)
# the line is b4d's own camera model, not a fit to these points:
# raw variance = conv x signal + (conv x rn)^2. The 3/8 of T = 2 sqrt(e + rn^2 + 3/8) is not in it: that constant only
# makes the transformed noise close to 1; it is not noise in the raw values
pv_model = conv * ps_ + (conv * rn) ** 2
ax.plot(xe, conv * xe + (conv * rn) ** 2, color=C_T, label="b4d camera model", zorder=2)
ax.axhline(vd_, color=MUTED, ls="--", zorder=1)
ax.text(1e4, vd_ * 0.85, "read-noise variance", color=MUTED, ha="right", va="top", fontsize=CFS)
ax.plot(ps_, pv_, ls="none", marker="o", ms=6, mfc=PAPER, mec=INK, zorder=3, label="Whole sensor")
ax.set(xscale="log", yscale="log", xlim=(1, 2e4), ylim=(5, 10 ** 4.5))
for a_ in (ax.xaxis, ax.yaxis):                          # minor ticks at 2..9 x each decade
    a_.set_major_locator(mt.LogLocator(base=10, numticks=12))
    a_.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12)); a_.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("Mean signal above offset (ADU)"); ax.set_ylabel("Temporal variance (ADU²)")
ax.legend(loc="upper left", fontsize=CFS, handlelength=1.2)
facts["ptc"] = dict(dark_variance_ADU2=vd_, read_noise_ADU=float(np.sqrt(vd_)), K_ADU_per_e=K_,
                    signal_ADU=[float(v_) for v_ in ps_], variance_ADU2=[float(v_) for v_ in pv_],
                    largest_departure_percent=float(np.max(np.abs(P1["residual_percent"]))),
                    model_variance_ADU2=[float(v_) for v_ in pv_model],
                    largest_departure_from_model_percent=float(np.max(np.abs(100 * (pv_ - pv_model) / pv_))))
dump_facts()
title(fig, ax, "Photon transfer curve")
ax = axs[1]
xs = np.arange(20, 1500)
x0_ = int(round(off)); TABr = TAB - TAB[x0_]; TAr = TA - TA[x0_]
# measured (ptc_beta3.json round_trip): each recording's median raw value, with its measured
# temporal noise as the bar (+-1 standard deviation in ADU, about +-1 noise unit); the formula dotted so the table shows
RT = PTC["round_trip"]["recordings"]
ax.plot(xs, TABr[xs], color=C_T, label="Fitted transform")
ax.plot(xs, TAr[xs], color=C_ALT, ls=":", lw=2, label="Anscombe")
for nm_, d_ in RT.items():
    lvl = int(round(off + d_["signal_ADU"])); s_ = math.sqrt(d_["raw_var_ADU2"])
    if lvl + s_ < 1500:
        ax.errorbar([lvl], [TABr[lvl]], xerr=[[s_], [s_]], fmt="o", ms=6, mfc=PAPER, mec=INK, ecolor=INK, capsize=3, lw=1,
                    label="Measured ±1σ" if nm_ == "dark_pre" else None)
ax.set_xlabel("Raw value (ADU)"); ax.set_ylabel("T(x) (σ)"); ax.set_xlim(0, 1500); ax.set_ylim(-5, 60)
ax.legend(loc="lower right", fontsize=CFS, handlelength=1.5)
title(fig, ax, "Raw value x to T(x)")
ax = axs[2]
ax.plot(lams, 100 * addT, color=C_T, label="Fitted transform")
ax.plot(lams, 100 * addA, color=C_ALT, ls=":", lw=2, label="Anscombe")
ax.axhline(100 / 12, color=MUTED, ls="--", label="q²/12")
# measured: b4d minus raw variance, same pixels and frames, as a % of the raw variance (ptc_beta3.json round_trip)
se_ = np.array([d_["signal_ADU"] for d_ in RT.values()]) / conv
ax.plot(se_, [d_["added_percent_formula"] for d_ in RT.values()], ls="none", marker="s", ms=4, color=C_ALT, zorder=4,
        label="Measured, Anscombe")
ax.plot(se_, [d_["added_percent_table"] for d_ in RT.values()], ls="none", marker="o", ms=8, mfc="none", mec=C_T, mew=1.5,
        zorder=5, label="Measured, fitted")
ax.set_xscale("symlog", linthresh=1, linscale=0.5); ax.set_xlim(-0.3, 8000); ax.set_ylim(7, 12)
ax.xaxis.set_major_locator(mt.FixedLocator([0, 1, 10, 100, 1000])); ax.xaxis.set_major_formatter(mt.FixedFormatter(["0", "1", "10", "100", "1000"]))
ax.xaxis.set_minor_locator(mt.FixedLocator(list(np.arange(0.1, 1, 0.1)) + [v_ * 10 ** e_ for e_ in range(0, 4) for v_ in range(2, 10)]))
ax.yaxis.set_minor_locator(mt.MultipleLocator(0.5))
ax.set_xlabel("Signal (electrons)"); ax.set_ylabel("Added noise (% of var.)")
ax.legend(loc="upper right", fontsize=CFS, handlelength=1.5)
title(fig, ax, "Rounding noise, q = 1")
save(fig, "fig01_noise_units.png")

# ---- section 1, the camera's per-pixel calibration: histograms of the offset, read noise and gain of every pixel,
# calibrated from the raw frames and from the same frames after b4d (noise-unit table, q = 1; b4d's own rounding variance
# taken back out); sensor_beta3.npz holds the histograms (fixed bin width per quantity) and their summary statistics
def sensor_figs():                                       # a function, so its names (k, e, h, i, ...) stay local
    SM = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_beta3.npz"))
    SMM = json.loads(str(SM["meta"]))
    fig, axs = row_fig([PW, PW, PW], PH, gap=1.0, left=1.0, bottom=0.85, top=0.6)
    for ax, (k, lab, ttl) in zip(axs, [("read_noise", "Read noise (e$^-$)", "Read noise"), ("offset", "Offset (ADU)", "Offset"),
                                       ("gain", "Gain (e$^-$/ADU)", "Gain")]):
        e = SM[f"{k}_bins"]; nd = 3 if k == "gain" else 2
        BW = {"read_noise": "0.7 e$^-$", "offset": "0.3 ADU", "gain": "0.001 e$^-$/ADU"}
        for nm, lab_, kw in (("raw", "Raw", dict(color=LIGHT, fill=True, lw=0)), ("b4d", "b4d", dict(color=C_T, lw=1.5))):
            h = SM[f"{nm}_{k}_hist"]; md = SMM[f"{nm}_{k}"]["median"]
            if nm == "raw":                              # raw: one bar per bin, edges drawn
                ax.bar(e[:-1], h / h.sum(), width=np.diff(e), align="edge", bottom=1e-9, color=LIGHT, edgecolor=MUTED,
                       lw=0.25, label=lab_)
            else:
                ax.stairs(h / h.sum(), e, label=lab_, **kw)
        ax.set_yscale("log"); ax.set_ylim(1e-7, 1e-1 if k == "gain" else 1e0); ax.set_xlabel(f"{lab}, bins of {BW[k]}")   # fixed bin width per panel
        if k == "read_noise":
            ax.set_ylabel("Fraction of pixels per bin")
        ax.yaxis.set_major_locator(mt.LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12)); ax.yaxis.set_minor_formatter(mt.NullFormatter())
        # x ranges and ticks
        if k == "read_noise":
            ax.set_xlim(0, 35); xt_ = [0, 10, 20, 30]   # linear; the data span 2.3-31 e-
        elif k == "offset":
            ax.set_xlim(30, 45); xt_ = [30, 35, 40, 45]
        else:
            ax.set_xlim(0.7, 0.75); xt_ = [0.7, 0.72, 0.74]
        ax.xaxis.set_major_locator(mt.FixedLocator(xt_)); ax.xaxis.set_major_formatter(mt.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(mt.MultipleLocator(0.005 if k == "gain" else 1))   # explicit
        # short labels, top right, clear of the data; the medians are in the text
        hs_, ls_ = ax.get_legend_handles_labels(); o_ = [ls_.index("Raw"), ls_.index("b4d")]
        ax.legend([hs_[i] for i in o_], [ls_[i] for i in o_], loc="upper right", fontsize=CFS, handlelength=1.2)   # Raw first
        title(fig, ax, ttl)
    save(fig, "fig01b_sensor_histograms.png")
    facts["sensor_maps"] = {k_: v_ for k_, v_ in SMM.items() if k_ != "run"}


sensor_figs()

# ---- layout helpers for grids of panels (sizes in inches; every 5 x 5 patch is G x G, every title 0.18 in above its panel)
from matplotlib.lines import Line2D


def panels(nr, nc, w=None, h=None, gapx=1.0, gapy=1.0, left=0.3, right=0.3, top=0.55, bottom=0.3):
    """nr x nc panels of w x h inches (lists per column / row allowed); returns fig, axes[row][col]"""
    ws = w if isinstance(w, (list, tuple)) else [w or G] * nc
    hs = h if isinstance(h, (list, tuple)) else [h or G] * nr
    gx = gapx if isinstance(gapx, (list, tuple)) else [gapx] * (nc - 1)
    gy = gapy if isinstance(gapy, (list, tuple)) else [gapy] * (nr - 1)
    W = left + sum(ws) + sum(gx) + right; H = top + sum(hs) + sum(gy) + bottom
    fig = plt.figure(figsize=(W, H)); axs = []
    y = H - top
    for i in range(nr):
        y -= hs[i]; x = left; row = []
        for j in range(nc):
            row.append(fig.add_axes([x / W, y / H, ws[j] / W, hs[i] / H])); x += ws[j] + (gx[j] if j < nc - 1 else 0)
        axs.append(row)
        if i < nr - 1:
            y -= gy[i]
    return fig, axs


def ttl(fig, ax, s, box=None):
    W, H = fsize(fig); b = box or ax.get_position()
    t = fig.text((b.x0 + b.x1) / 2, b.y1 + 0.18 / H, s, ha="center", va="bottom", fontsize=FS, color=INK)
    _register(fig, ax, t)


def harrow(fig, a0, a1):
    W, H = fsize(fig); b0, b1 = a0.get_position(), a1.get_position(); yy = (b0.y0 + b0.y1) / 2
    fig.patches.append(FancyArrowPatch((b0.x1 + 0.15 / W, yy), (b1.x0 - 0.15 / W, yy), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color=MUTED, lw=1, shrinkA=0, shrinkB=0))


def varrow(fig, a0, a1):
    """from panel a0 to panel a1 straight up or down (a1's title sits in the gap: the arrow stops below / above it)"""
    W, H = fsize(fig); b0, b1 = a0.get_position(), a1.get_position(); xx = (b0.x0 + b0.x1) / 2
    if b1.y1 < b0.y0:   # down
        y0_, y1_ = b0.y0 - 0.1 / H, b1.y1 + 0.5 / H
    else:               # up (the arrow ends below a1, a0's title is below it... none: a0 is underneath)
        y0_, y1_ = b0.y1 + 0.5 / H, b1.y0 - 0.1 / H
    fig.patches.append(FancyArrowPatch((xx, y0_), (xx, y1_), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1, shrinkA=0, shrinkB=0))


def wrap_arrow(fig, a0, a1):
    """from the end of one row (a0, right) to the start of the next (a1, left): down, left, down"""
    W, H = fsize(fig); b0, b1 = a0.get_position(), a1.get_position()
    xa, xb = (b0.x0 + b0.x1) / 2, (b1.x0 + b1.x1) / 2
    ym = (b0.y0 + b1.y1 + 0.45 / H) / 2
    fig.add_artist(Line2D([xa, xa, xb], [b0.y0 - 0.1 / H, ym, ym], transform=fig.transFigure, color=MUTED, lw=1))
    fig.patches.append(FancyArrowPatch((xb, ym), (xb, b1.y1 + 0.5 / H), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1, shrinkA=0, shrinkB=0))


ULIM = (0, max(2 * Rlim, int(zigzag(r).max())))
uu = zigzag(r)
RAWLIM = (float(min(patch.min(), real.min())), float(max(patch.max(), real.max())))

# ---- fig02: the example, three columns of the same height: (1) the array (a grid of shards), one shard (a grid of chunks),
# one chunk; (2) all 64 planes one under the other: the one 4096 x 64 array the encoder runs over;
# (3) N rows x 5 of that array, in ADU and in noise units (T), with the 5 x 5 patch and the followed pixel
NR = 13                                                   # rows in column 3: N x CELL = the columns' height
Y0 = PY + HI[0] - NR // 2 + 1                             # first row shown (the followed pixel near the middle)
Y0 = max(0, min(Y - NR, Y0))
HC = NR * CELL                                            # every column's height (inches)
TOP, BOT = 1.15, 0.75
W2, H2 = 3.35 + HC * X / 508 + 1.3 + 2 * G + 0.3 + 0.4, TOP + HC + BOT   # natural width; save() spreads the gaps to the page
fig = plt.figure(figsize=(W2, H2))
def _ax(x, y, w, h, **kw):                                # inches from the bottom left
    return fig.add_axes([x / W2, y / H2, w / W2, h / H2], **kw)
def _ttl(x, ytop, s, ax):
    b = ax.get_position()
    t = fig.text((b.x0 + b.x1) / 2, (ytop + 0.18) / H2, s, ha="center", va="bottom", fontsize=FS, color=INK)
    _register(fig, ax, t)
yb = BOT                                                  # the columns' bottom; top = yb + HC

# column 1: array -> shard -> chunk (3D)
C1X, C1W, P3 = 0.45, 1.7, 1.7; G3 = (HC - 3 * P3) / 2
def blocks3d(ax, n, base, hi_idx, hi_col):
    filled = np.ones((n, n, n), bool); cols_ = np.empty(filled.shape, dtype=object)
    for a in range(n):
        for b_ in range(n):
            for c_ in range(n):
                cols_[a, b_, c_] = base
    cols_[hi_idx] = hi_col
    ax.voxels(filled, facecolors=cols_, edgecolor="#ffffff", linewidth=0.8)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-58); ax.set_axis_off()
c1 = []
for i_, (n_, base_, hic_, ttl_) in enumerate([(3, plt.get_cmap("RdPu")(0.55), plt.get_cmap("Greens")(0.35), "The array: shards"),
                                             (4, plt.get_cmap("Greens")(0.35), (0.72, 0.72, 0.72, 1.0), "One shard: chunks")]):
    y_ = yb + HC - (i_ + 1) * P3 - i_ * G3
    ax_ = _ax(C1X, y_, C1W, P3, projection="3d"); blocks3d(ax_, n_, base_, (n_ - 1, 0, n_ - 1), hic_); _ttl(C1X, y_ + P3, ttl_, ax_); c1.append(ax_)
y_ = yb
ax3 = _ax(C1X, y_, C1W, P3, projection="3d"); c1.append(ax3)
n8 = 8
filled = np.ones((n8, n8, n8), bool)
cols = np.empty(filled.shape, dtype=object)
small = chunk.reshape(n8, 8, n8, 8, n8, 8).mean(axis=(1, 3, 5))       # z, y, x
lo_, hi_ = small.min(), small.max()
for a in range(n8):
    for b_ in range(n8):
        for c_ in range(n8):
            g_ = 0.92 - 0.6 * (small[c_, b_, a] - lo_) / (hi_ - lo_)       # voxel index (x, y, z)
            cols[a, b_, c_] = (g_, g_, g_, 1.0)
zsl = PZ // 8
for a in range(n8):
    for b_ in range(n8):
        cols[a, b_, zsl] = (0.22, 0.22, 0.22, 1.0)
ax3.voxels(filled, facecolors=cols, edgecolor="#ffffff", linewidth=0.3)
ax3.set_box_aspect((1, 1, 1)); ax3.view_init(elev=22, azim=-58)
for setter in (ax3.set_xticks, ax3.set_yticks, ax3.set_zticks):
    setter(list(range(9)))                              # every 8 pixels
for setter in (ax3.set_xticklabels, ax3.set_yticklabels, ax3.set_zticklabels):
    setter([str(8 * i_) if 8 * i_ in (0, 32, 64) else "" for i_ in range(9)], fontsize=CFS)
ax3.set_xlabel("x (pixels)", labelpad=2, fontsize=CFS); ax3.set_ylabel("y (pixels)", labelpad=2, fontsize=CFS)
ax3.set_zlabel("z (pixels)", labelpad=0, fontsize=CFS)
ax3.grid(False)
for a_ in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    a_.set_minor_locator(mt.NullLocator())
    a_.pane.set_visible(False); a_.line.set_linewidth(1)
    a_._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.3, linewidth={True: 1.0, False: 1.0})
ax3.tick_params(pad=-1)
_ttl(C1X, y_ + P3, "One chunk: 64³ pixels", ax3)
for a0_, a1_ in zip(c1[:-1], c1[1:]):                     # down arrows: array -> shard -> chunk
    b0_, b1_ = a0_.get_position(), a1_.get_position(); xx_ = (b0_.x0 + b0_.x1) / 2
    fig.patches.append(FancyArrowPatch((xx_, b0_.y0 + 0.05 / H2), (xx_, b1_.y1 + 0.5 / H2), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color=MUTED, lw=1, shrinkA=0, shrinkB=0))

# column 2: all 64 planes stacked into one array (4096 rows x 64), the full column height; zigzag cuts skip the planes
# between the three pieces drawn
SEGS = [[0, 1], [PZ - 1, PZ, PZ + 1], [Z - 2, Z - 1]]     # the planes drawn
CUT = 30                                                  # a cut's height, in rows of the drawing
units = sum(Y * len(s_) for s_ in SEGS) + CUT * (len(SEGS) - 1)
C2X = 3.35
WS = HC * X / units                                       # the stack's width (inches): pixels square
VMIN, VMAX = float(np.percentile(chunk, 0.5)), float(np.percentile(chunk, 99.9))
axs_ = _ax(C2X, yb, WS, HC)
y_ = 0; tops = {}; ymaj, ymin_ = [], []; ends = []   # ends: (drawing y, array row) of each piece's bottom edge
for si, seg in enumerate(SEGS):
    for z_ in seg:
        axs_.imshow(chunk[z_], cmap="gray", interpolation="nearest", extent=(0, X, y_ + Y, y_), vmin=VMIN, vmax=VMAX)
        if z_ != seg[0]:
            axs_.plot([0, X], [y_, y_], color="white", lw=0.6)
        axs_.text(3, y_ + 4, f"z {z_}", ha="left", va="top", fontsize=CFS, color="white", fontweight="normal")
        tops[z_] = y_; ymaj.append(y_); ymin_ += [y_ + d_ for d_ in range(16, Y, 16)]
        y_ += Y
    ends.append((y_, Y * (seg[-1] + 1)))                  # label each piece's bottom row too
    if si < len(SEGS) - 1:                                # the cut: a zigzag edge on each side of a gap
        zx = np.linspace(0, X, 9); zy = np.where(np.arange(9) % 2, 5, 0)
        axs_.fill_between(zx, y_ - 1, y_ + zy, color=PAPER, zorder=3)
        axs_.fill_between(zx, y_ + CUT - 5 + zy, y_ + CUT + 1, color=PAPER, zorder=3)
        axs_.plot(zx, y_ + zy, color=MUTED, lw=1, zorder=4, clip_on=False)
        axs_.plot(zx, y_ + CUT - 5 + zy, color=MUTED, lw=1, zorder=4, clip_on=False)
        axs_.text(X / 2, y_ + CUT / 2 + 1, f"z {seg[-1] + 1}–{SEGS[si + 1][0] - 1}", ha="center", va="center", fontsize=CFS,
                  color=MUTED, style="italic", zorder=5)
        y_ += CUT
axs_.set_xlim(0, X); axs_.set_ylim(units, 0); axs_.set_aspect("auto")
axs_.set_yticks(ymaj + [e_[0] for e_ in ends]); axs_.set_yticklabels([str(Y * z_) for z_ in tops] + [str(e_[1]) for e_ in ends], fontsize=CFS)
axs_.set_yticks(ymin_, minor=True)
axs_.set_xticks([0, 32, 64]); axs_.xaxis.set_minor_locator(mt.MultipleLocator(8))
axs_.tick_params(labelsize=CFS)
for s_ in axs_.spines.values():
    s_.set_visible(False)
axs_.set_xlabel("x (pixels)", fontsize=CFS, labelpad=1)
axs_.set_ylabel("Row of the array", fontsize=CFS, labelpad=4)
axs_.add_patch(Rectangle((PX, tops[PZ] + Y0), 5, NR, fill=False, edgecolor=HL, lw=1.5, zorder=5))
_bs = axs_.get_position()
_register(fig, axs_, fig.text((_bs.x0 + _bs.x1) / 2, (yb + HC + 0.18) / H2, f"All {Z} planes stacked:\none array, {Z * Y} rows × {X}",
                              ha="center", va="bottom", fontsize=FS, color=INK))
fig._letter_order = "columns"

# column 3: N x 5 of the array, raw and T (one panel)
C3X = C2X + HC * X / units + 1.3
axr = _ax(C3X, yb, G, HC); axt = _ax(C3X + G + 0.3, yb, G, HC)
rows_ = slice(Y0, Y0 + NR); cols_ = slice(PX, PX + 5)
rawN = frame[rows_, cols_]; TN = (LUT[rawN.astype(np.int64)] / FIX).round(2)
hiN = [(PY + HI[0] - Y0, HI[1])]
grid(axr, rawN, "raw", lim=(float(rawN.min()), float(rawN.max())), hl=hiN)
grid(axt, TN, "T", lim=(float(TN.min()), float(TN.max())), hl=hiN)
for a_ in (axr, axt):                                     # the 5 x 5 patch the next sections follow
    a_.add_patch(Rectangle((0, NR - (PY - Y0) - 5), 5, 5, fill=False, edgecolor=HL, lw=2.5, zorder=7, clip_on=False))
    for j in range(5):
        a_.text(j + 0.5, -0.25, str(PX + j), ha="center", va="top", fontsize=CFS, color=MUTED)
_ttl(C3X, yb + HC, "Raw (ADU)", axr); _ttl(C3X + G + 0.3, yb + HC, "T(x) (σ)", axt)
# dotted lines: the stack's orange box -> the N x 5 grids
def _fig_xy(ax, x, y):
    return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
for yy_, gy_ in ((tops[PZ] + Y0, NR), (tops[PZ] + Y0 + NR, 0)):
    p0 = _fig_xy(axs_, PX + 5, yy_); p1 = _fig_xy(axr, 0, gy_)
    fig.add_artist(Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=fig.transFigure, color=HL, lw=1, ls=":"))
save(fig, "fig02_example.png")

# ---- fig03: the prediction. The decoder working down the patch, each "?" predicted as the average of the three rebuilt
# pixels above it (in bold; the sum written under the panel), then all 25 predictions; all in the prediction's purple.
# One row only; prediction + r x q = rebuilt is told in section 4.
stages = [((1, 1), "Row 2, pixel 2"), (HI, "Row 3, pixel 3")]
fig, A = panels(1, 3, gapx=1.15, bottom=0.55)
FW, FH = fsize(fig)
def _under(ax, s, color=INK):                              # a line of arithmetic under a panel
    b_ = ax.get_position()
    fig.text(b_.x0, b_.y0 - 0.14 / FH, s, ha="left", va="top", fontsize=CFS, color=color)
for ax, (at, nm) in zip(A[0][:2], stages):
    done = [(i, j) for i in range(5) for j in range(5) if (i, j) < at]
    src = [(at[0] - 1, at[1] + d) for d in (-1, 0, 1)]
    vals = [[Fp[i, j] if (i, j) in done else "" for j in range(5)] for i in range(5)]
    grid(ax, vals, "pred", lim=Tlim, hl=[at], outline=src, unknown=[at], blank=[(i, j) for i in range(5) for j in range(5) if (i, j) not in done and (i, j) != at])
    ttl(fig, ax, nm)
    sv = [Fp[s_] for s_ in src]
    _under(ax, f"? = ({sv[0]:.2f} + {sv[1]:.2f} + {sv[2]:.2f}) ÷ 3 = {Pp[at]:.2f}", C_P)
grid(A[0][2], Pp, "pred", lim=Tlim, hl=[HI]); ttl(fig, A[0][2], "All 25 predictions")
harrow(fig, A[0][0], A[0][1]); harrow(fig, A[0][1], A[0][2])
save(fig, "fig03_predict.png")

# ---- fig04: three ways a pixel is predicted, with real numbers from this chunk
ex_y = PY + HI[0]
cases = [("Most pixels", [fF[ex_y - 1, PX + HI[1] + d] / FIX for d in (-1, 0, 1)], pF[ex_y, PX + HI[1]] / FIX, 1),
         ("Left edge", [fF[ex_y - 1, 0] / FIX, fF[ex_y - 1, 1] / FIX], pF[ex_y, 0] / FIX, 0),
         ("First row", [fC[0, 9] / FIX], pC[0, 10] / FIX, 1)]
# (no figure; the numbers are in the facts)
facts["predict_cases"] = [dict(name=nm, src=[round(float(s_), 2) for s_ in src], pred=round(float(pv), 2)) for nm, src, pv, _ in cases]

# ---- fig05: how the difference that is rounded to r is made: T(x) − p (no
# number line, no rebuilt T, no r grid here; r is shown in section 5)
tp = (t - p) / FIX
fig, A = panels(1, 3, gapx=1.15, bottom=0.6)
FW, FH = fsize(fig)
grid(A[0][0], Tp, "T", lim=Tlim, hl=[HI]); ttl(fig, A[0][0], "T(x) (σ)")
grid(A[0][1], Pp, "pred", lim=Tlim, hl=[HI]); ttl(fig, A[0][1], "Prediction p (σ)")
grid(A[0][2], tp.round(2), "tp", lim=Rlim, hl=[HI]); ttl(fig, A[0][2], "T(x) − p (σ)")
for a0, a1_, sym in ((A[0][0], A[0][1], "−"), (A[0][1], A[0][2], "=")):
    b0_, b1_ = a0.get_position(), a1_.get_position()
    fig.text((b0_.x1 + b1_.x0) / 2, (b0_.y0 + b0_.y1) / 2, sym, ha="center", va="center", fontsize=12, color=INK)
b_ = A[0][0].get_position()
fig.text(b_.x0, b_.y0 - 0.14 / FH, f"Our pixel: {Tp[HI]:.2f} − {Pp[HI]:.2f} = {tp[HI]:.2f}, rounded to r = {int(r[HI]):+d}",
         ha="left", va="top", fontsize=CFS, color=INK)
save(fig, "fig05_quantise.png")

# ---- fig06: fold the signs away
fig, A = panels(1, 3, w=[G, G, 4.3], gapx=1.15)
grid(A[0][0], r, "r", lim=Rlim, hl=[HI]); ttl(fig, A[0][0], f"r (rounded to whole steps of q = {Q:g})")
grid(A[0][1], uu, "u", lim=ULIM, hl=[HI]); ttl(fig, A[0][1], "u (folded)")
harrow(fig, A[0][0], A[0][1])
# the folding rule, worked for r = 0, -1, +1, ... +3 (how u = -2r - 1 works): r, the sum, u
ax = A[0][2]; ax.axis("off")
rr = np.array([0, -1, 1, -2, 2, -3, 3]); ur = zigzag(rr)
cw = (G / 7) / 4.3                                         # one cell's width, as a fraction of the panel
HLF = [(int(np.where(rr == r[HI])[0][0]), 0)] if r[HI] in rr else []   # our pixel's r, if listed
grid(ax.inset_axes([0, 0, cw, 1]), [[v_] for v_ in rr], "r", lim=Rlim, hl=HLF)
grid(ax.inset_axes([1 - cw, 0, cw, 1]), [[v_] for v_ in ur], "u", lim=ULIM, hl=HLF)
for i_, (rv, uv) in enumerate(zip(rr, ur)):
    yy = 1 - (i_ + 0.5) / 7
    s_ = f"2 × {rv} = {uv}" if rv >= 0 else f"−2 × (−{-rv}) − 1 = {uv}"
    ax.text(cw + 0.06, yy, "u = " + s_, transform=ax.transAxes, ha="left", va="center", fontsize=CFS,
            color=face("r", 2.0 if rv > 0 else (-2.0 if rv < 0 else 0.0), 2.0)[0] if rv != 0 else INK,
            fontweight="normal")
    ax.annotate("", xy=(1 - cw - 0.02, yy), xytext=(0.68, yy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10))
ttl(fig, ax, "Folding rule: u = 2r, or −2r − 1 if r < 0")
save(fig, "fig06_fold.png")

# ---- fig07: the Rice code: u split into u >> k and its last k bits; the codes for this k; every pixel's code
uh = int(zigzag(r)[HI]); qh = uh >> k; rem = uh & ((1 << k) - 1); code_hi = rice_code(uh, k)
fig, A = panels(1, 3, gapx=1.15)
ax = A[0][0]
ub = format(uh, "b").rjust(k + 1, "0"); nh = len(ub) - k
xs0 = (5 - len(ub)) / 2
for j, bt in enumerate(ub):
    fc_ = "#d9ead3" if j < nh else "#cfe0f5"
    ax.add_patch(Rectangle((xs0 + j, 3.3), 1, 1, facecolor=fc_, edgecolor=EDGE, lw=1))
    ax.text(xs0 + j + 0.5, 3.8, bt, ha="center", va="center", fontsize=CFS, color=CELL_INK)
# the two labels meet at the split between the bits, one ending and one starting there (no overlap with 0.40 in cells)
ax.text(xs0 + nh - 0.12, 3.1, f"{uh} ÷ {1 << k} = {qh}", ha="right", va="top", fontsize=CFS, color=C_R)
ax.text(xs0 + nh + 0.12, 3.1, "Last k bits", ha="left", va="top", fontsize=CFS, color=C_T)
kinds = ["one"] * qh + ["stop"] + ["rem"] * k
bc = {"one": C_R, "stop": "#d4d3cf", "rem": C_T}
xs1 = (5 - len(code_hi)) / 2
YC = 0.95                                                # the code's row
for j, (bt, kd) in enumerate(zip(code_hi, kinds)):
    ax.add_patch(Rectangle((xs1 + j, YC), 1, 1, facecolor=bc[kd], edgecolor=EDGE, lw=1))
    ax.text(xs1 + j + 0.5, YC + 0.5, bt, ha="center", va="center", fontsize=CFS, color="white" if kd != "stop" else CELL_INK)
# what each part of the code is (e.g. the 0 in 1 0 00): the 1s, the stop bit, the last k bits
ax.text(xs1 + qh / 2, YC - 0.1, "1s", ha="center", va="top", fontsize=CFS, color=C_R)
ax.text(xs1 + qh + 0.5, YC - 0.1, "Stop", ha="center", va="top", fontsize=CFS, color=MUTED)
ax.text(xs1 + qh + 1 + k / 2, YC - 0.1, "Last k bits", ha="center", va="top", fontsize=CFS, color=C_T)
for x0_, x1_ in ((xs0 + nh / 2, xs1 + qh / 2), (xs0 + nh + k / 2, xs1 + qh + 1 + k / 2)):
    ax.annotate("", xy=(x1_, YC + 1.1), xytext=(x0_, 2.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=14,
                                                                           shrinkA=0, shrinkB=0))
ax.text(0, 4.6, f"u = {uh} in binary", ha="left", va="center", fontsize=FS, color=INK)
ax.text(0, 0.05, "Rice code", ha="left", va="center", fontsize=FS, color=INK)
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")
ttl(fig, ax, f"Our pixel: u = {uh}, k = {k}")
ax = A[0][1]
show_u = sorted(set([0, 1, 2, 3, uh, uh + 1, uh + 2, 2 * uh]))[:7]
rows = [["u", "Code", "Bits"]] + [[str(v), rice_code(v, k), str(len(rice_code(v, k)))] for v in show_u]
nr = len(rows); cw = [1.1, 2.8, 1.1]; sc_ = 5 / sum(cw)
for i_, row in enumerate(rows):
    x_ = 0
    for j_, cell in enumerate(row):
        is_hi = i_ > 0 and int(row[0]) == uh
        fc_ = "#f0efeb" if i_ == 0 else (face("code", len(row[1]), (2, 9))[0] if is_hi else PAPER)
        ax.add_patch(Rectangle((x_ * sc_, (nr - 1 - i_) * 5 / nr), cw[j_] * sc_, 5 / nr, facecolor=fc_, edgecolor=GRIDLINE, lw=1))
        ax.text((x_ + cw[j_] / 2) * sc_, (nr - 0.5 - i_) * 5 / nr, cell, ha="center", va="center", fontsize=CFS, color=INK if fc_ == PAPER else CELL_INK,
                fontweight="normal")
        x_ += cw[j_]
    if i_ > 0 and int(row[0]) == uh:
        ax.add_patch(Rectangle((0, (nr - 1 - i_) * 5 / nr), 5, 5 / nr, fill=False, edgecolor=CELL_INK, lw=HLW, zorder=6))
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")
ttl(fig, ax, f"Codes with k = {k}")
cg = np.array(codes, dtype=object).reshape(5, 5)
grid(A[0][2], cg, "code", lim=(2, 9), hl=[HI]); ttl(fig, A[0][2], "Rice code of each pixel")
save(fig, "fig07_rice.png")

# ---- fig08 (streams): the example z plane's 32 groups (two rows each): their bits and k beside the rows they code, dealt to
# four streams; and which k the chunk's groups chose at each q
gk_f = [pl[0] for pl in plF]; gb_f = [pl[1] for pl in plF]
kcol = {k_: plt.get_cmap("Greens")(0.2 + 0.11 * k_) for k_ in range(8)}   # k = 0 .. 7, light to dark
fig, A = panels(2, 3, w=[G, 3.2, 3.2], h=[G, 2.6], gapx=[0.6, 1.3], gapy=1.3, left=1.35, bottom=0.85, top=0.6)
ax = A[0][0]
ax.imshow(frame, cmap="gray", interpolation="nearest", extent=(0, 64, 64, 0))
ax.add_patch(Rectangle((0, 2 * grp_hi), 64, 2, fill=False, edgecolor=HL, lw=1.5))
ax.set_xticks(IMG_TICKS); ax.set_xlabel("x (pixels)"); ax.set_yticks([2 * g_ + 1 for g_ in range(0, 32, 4)]); ax.set_yticklabels([f"g{g_}" for g_ in range(0, 32, 4)])
# only the labelled ticks, no minor ticks: 32 groups in 2 in were a crowd of ticks
ax.yaxis.set_minor_locator(mt.NullLocator()); ax.xaxis.set_minor_locator(mt.MultipleLocator(8))
for s_ in ax.spines.values():
    s_.set_visible(True)
ttl(fig, ax, f"z plane {PZ}, 32 groups")
ax = A[0][1]
for gi in range(32):
    ax.barh(2 * gi + 1, gb_f[gi], height=1.7, color=kcol.get(gk_f[gi], "#274e13"), edgecolor=CELL_INK if gi == grp_hi else "none", lw=1)
ax.set_ylim(64, 0); ax.set_yticks([]); ax.yaxis.set_minor_locator(mt.NullLocator()); ax.spines["left"].set_visible(False)
ax.set_xlabel("Bits in the group"); ax.xaxis.set_major_locator(mt.MultipleLocator(200)); ax.xaxis.set_minor_locator(mt.MultipleLocator(50))
from matplotlib.patches import Patch
ttl(fig, ax, "Bits and k of each group")
ax = A[0][2]
um_ = [float(np.mean(gF[gi])) for gi in range(32)]
ax.plot(um_, np.arange(32) * 2 + 1, color=C_U, marker="o", ms=4, mfc=PAPER, mec=C_U)
ax.set_ylim(64, 0); ax.set_yticks([]); ax.yaxis.set_minor_locator(mt.NullLocator()); ax.spines["left"].set_visible(False)
ax.set_xlabel("Mean u in the group"); ax.set_xlim(0, 1.5 * max(um_))   # room on the right for the k legend
ax.legend(handles=[Patch(facecolor=kcol[v], label=f"k = {v}") for v in sorted(set(gk_f))], loc="lower right")
ttl(fig, ax, "Size of the numbers coded")
A[1][0].remove(); A[1][1].remove()
b0_, b1_ = A[1][0].get_position(), A[1][1].get_position()
ax = fig.add_axes([b0_.x0, b0_.y0, b1_.x1 - b0_.x0, b0_.height]); ax._free = True
for s_ in range(NS):
    x_ = 0
    for gi in range(s_, 32, NS):
        ax.barh(NS - 1 - s_, gb_f[gi], left=x_, height=0.6, color=kcol.get(gk_f[gi], "#274e13"), edgecolor=EDGE, lw=1)
        ax.text(x_ + gb_f[gi] / 2, NS - 1 - s_, f"g{gi}", ha="center", va="center", fontsize=FS, color=CELL_INK)
        x_ += gb_f[gi]
    if s_ == grp_hi % NS:
        xg = sum(gb_f[g_] for g_ in range(s_, grp_hi, NS))
        ax.add_patch(Rectangle((xg, NS - 1 - s_ - 0.3), gb_f[grp_hi], 0.6, fill=False, edgecolor=CELL_INK, lw=HLW, zorder=6))
ax.set_yticks(range(NS)); ax.set_yticklabels([f"Stream {s_}" for s_ in range(NS - 1, -1, -1)]); ax.yaxis.set_minor_locator(mt.NullLocator())
ax.set_xlabel("Bits"); ax.xaxis.set_major_locator(mt.MultipleLocator(500)); ax.xaxis.set_minor_locator(mt.MultipleLocator(100))
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ttl(fig, ax, f"The 32 groups dealt to {NS} streams")
ax = A[1][2]
cats = sorted(set().union(*[per_q[q_]["k_hist"].keys() for q_ in per_q]))
lab = {14: "sparse", 15: "zero"}
QS = sorted(per_q); wb = 0.8 / len(QS)
for qi, (q_, colr) in enumerate(zip(QS, ("#2a78d6", "#eb6834", "#1baf7a", "#7b4fc9"))):
    ax.bar(np.arange(len(cats)) + (qi - (len(QS) - 1) / 2) * wb, [per_q[q_]["k_hist"].get(c_, 0) for c_ in cats], wb, color=colr, edgecolor=INK, lw=1, label=f"q = {q_:g}")
ax.set_xticks(range(len(cats))); ax.set_xticklabels([lab.get(c_, str(c_)) for c_ in cats]); ax.xaxis.set_minor_locator(mt.NullLocator())
ax.set_xlabel("Selected k"); ax.set_ylabel("Groups in the chunk"); ax.legend(loc="upper left", bbox_to_anchor=(0.42, 1.0))
ttl(fig, ax, "The chunk's 2,048 groups")
save(fig, "fig08_streams.png")
facts["frame_groups"] = [dict(k=int(gk_f[gi]), bits=int(gb_f[gi]), mean_u=round(um_[gi], 2)) for gi in range(32)]

# ---- fig09: the file: raw against compressed, to scale (and the ratio); what the headers hold; chunks -> shard -> array
H_ = facts["headers"]
fig, A = panels(3, 1, w=[10.5], h=[1.7, 0.8, 1.9], gapy=[1.9, 1.3], left=2.3, bottom=0.3, top=0.6)
fig._free = True                                          # diagrams to scale: no 4:3 boxes
FW, FH = fsize(fig)
ax = A[0][0]
raw_b = chunk.nbytes; comp_b = per_q[1.0]["blob"]
YR = 1.4                                                 # the raw bar's row (b4d's is 0; the ratio arrow between them)
ax.barh(YR, raw_b, height=0.6, color="#d4d3cf", edgecolor=EDGE, lw=1)
ax.text(raw_b / 2, YR, f"{raw_b:,} bytes (16 bits per pixel)", ha="center", va="center", fontsize=FS, color=CELL_INK)
x_ = 0
parts = [("headers", hdr_total, HL)] + [(str(s_), math.ceil(v), plt.get_cmap("Blues")(0.35 + 0.07 * s_)) for s_, v in enumerate(chunk_streams_bytes)]
for nm, v, c_ in parts:
    ax.barh(0, v, left=x_, height=0.6, color=c_, edgecolor=EDGE, lw=1)
    if nm != "headers":
        ax.text(x_ + v / 2, 0, nm, ha="center", va="center", fontsize=FS, color="white")
    x_ += v
ax.text((hdr_total + x_) / 2, 0.36, "Streams", ha="center", va="bottom", fontsize=FS, color=INK)
ax.text(x_ + 6000, 0, f"{comp_b:,} bytes ({8 * comp_b / chunk.size:.1f} bits per pixel)", ha="left", va="center", fontsize=FS, color=INK)
# the compression ratio: an arrow from the raw size back to the compressed size
ax.annotate("", xy=(comp_b, YR / 2), xytext=(raw_b, YR / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=16, shrinkA=0, shrinkB=0))
ax.text((raw_b + comp_b) / 2 + 40000, YR / 2, f"{raw_b / comp_b:.2f} times smaller", ha="center", va="center", fontsize=FS, color=INK,
        bbox=dict(facecolor=PAPER, edgecolor="none", pad=2))
ax.set_ylim(-0.5, YR + 0.5)
ax.set_yticks([0, YR]); ax.set_yticklabels(["b4d (q = 1)", "Raw"]); ax.yaxis.set_minor_locator(mt.NullLocator()); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, raw_b); ax.set_xlabel("Bytes"); ax.xaxis.set_major_locator(mt.MultipleLocator(100000)); ax.xaxis.set_minor_locator(mt.MultipleLocator(20000))
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{int(v):,}" if v > 0 else ""))   # no "0": the dotted lines pass there
ttl(fig, ax, f"One chunk to scale: raw, and compressed (headers, then the {NS} streams)")
axh = A[1][0]
hp = [("Container", H_["container"], "#f6d7a7"), ("Blosc2", H_["blosc"], "#f2c078"), ("Block", H_["block"], "#eda100"), ("Sizes", H_["xs"], "#dd9500"), ("Transform curve", H_["curve"], "#c98a00")]
x_ = 0
for nm, v, c_ in hp:
    axh.barh(0, v, left=x_, height=0.6, color=c_, edgecolor=EDGE, lw=1)
    axh.text(x_ + v / 2, 0, f"{nm}\n{v} B", ha="center", va="center", fontsize=FS, color=CELL_INK)
    x_ += v
axh.set_xlim(0, x_); axh.set_ylim(-0.4, 0.4); axh.set_yticks([0]); axh.set_yticklabels(["Headers"]); axh.yaxis.set_minor_locator(mt.NullLocator()); axh.tick_params(axis="y", length=0)
axh.spines["left"].set_visible(False); axh.set_xlabel("Bytes"); axh.xaxis.set_major_locator(mt.MultipleLocator(50)); axh.xaxis.set_minor_locator(mt.MultipleLocator(10))
# dotted lines from the orange sliver to the headers opened out: straight down past the tick labels at the left edge, then
# out to the bar's right end; the title sits inside them, clear of both
to_fig = lambda a_, xy: fig.transFigure.inverted().transform(a_.transData.transform(xy))
pa0, pa1 = to_fig(ax, (0, -0.3)), to_fig(ax, (hdr_total, -0.3))
pb0, pb1 = to_fig(axh, (0, 0.3)), to_fig(axh, (x_, 0.3))
yband = ax.get_position().y0 - 0.1 / FH
fig.add_artist(Line2D([pa0[0], pb0[0]], [pa0[1], pb0[1]], transform=fig.transFigure, color=MUTED, lw=1, ls=":"))
fig.add_artist(Line2D([pa1[0], pa1[0], pb1[0]], [pa1[1], yband, pb1[1]], transform=fig.transFigure, color=MUTED, lw=1, ls=":"))
bh = axh.get_position()
_register(fig, axh, fig.text((bh.x0 + bh.x1) / 2, bh.y1 + 0.18 / FH, f"The {hdr_total} header bytes, opened out", ha="center",
                             va="bottom", fontsize=FS, color=INK))
# chunks add up to a shard (one file); shards and one zarr.json make the array (a folder); not to scale
axs_ = A[2][0]; axs_.set_xlim(0, 10.5); axs_.set_ylim(0, 1.9); axs_.axis("off")
def box(x0, w, y0, h, fc, txt, tc=CELL_INK):
    axs_.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=EDGE, lw=1))
    if txt:
        axs_.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=CFS, color=tc)
CB = plt.get_cmap("Greens")(0.65)                       # chunks green (streams blue above, shards magenta)
YS, YA, HB = 1.25, 0.15, 0.5                             # the shard's row, the array's row, box height
xs_ = [0, 1.4, 2.8, 4.9]
for x0, nm in zip(xs_, ["Chunk 0", "Chunk 1", "Chunk 2", "Last chunk"]):
    box(x0, 0.1, YS, HB, HL, "")                         # each chunk starts with its own headers
    box(x0 + 0.1, 1.2, YS, HB, CB, nm, "white")
axs_.text(4.45, YS + HB / 2, "…", ha="center", va="center", fontsize=FS, color=INK)
box(6.3, 3.4, YS, HB, "#d4d3cf", "Index: 16 B per chunk + 4 B check")
box(0, 1.3, YA, HB, "#f0efeb", "zarr.json")
for x0, nm in zip(xs_[1:3] + [4.9], ["Shard 0", "Shard 1", "Last shard"]):
    box(x0, 1.3, YA, HB, plt.get_cmap("RdPu")(0.55), nm, "white")
axs_.text(4.45, YA + HB / 2, "…", ha="center", va="center", fontsize=FS, color=INK)
for xa_, xb_ in ((0, 1.4), (9.7, 2.7)):                  # the shard row is one shard file of the array
    axs_.plot([xa_, xb_], [YS, YA + HB], color=MUTED, lw=1, ls=":")
for y_, nm in ((YS, "A shard (one file)"), (YA, "The array (a folder)")):
    axs_.text(-0.2, y_ + HB / 2, nm, ha="right", va="center", fontsize=FS, color=INK)
ttl(fig, axs_, "Chunks packed into a shard; shards and one zarr.json make the array (not to scale)")
save(fig, "fig09_file.png")

# ---- fig10: decoding. read u -> unfold to r; prediction (from the rebuilt rows above) + r x q = rebuilt T; then back to
# ADU: decoded -> compare with the raw -> error (exactly how rebuilt T is made)
errp = TAB[real] - TAB[patch.astype(np.int64)]
fig, A = panels(3, 3, gapx=1.15, gapy=[1.2, 1.5], bottom=0.6, right=1.6)
FW, FH = fsize(fig)
grid(A[0][0], uu, "u", lim=ULIM, hl=[HI]); ttl(fig, A[0][0], "u (read)")
grid(A[0][1], r, "r", lim=Rlim, hl=[HI]); ttl(fig, A[0][1], "r (unfolded)")
harrow(fig, A[0][0], A[0][1])
# the unfolding rule, worked for u = 0 .. 6: u, the sum, r; wider than a grid panel
b_ = A[0][2].get_position(); A[0][2].remove(); WR = 4.3
ax = fig.add_axes([b_.x0, b_.y0, WR / FW, b_.height]); ax.axis("off")
uL = np.arange(7); rL = np.where(uL % 2 == 0, uL // 2, -(uL + 1) // 2)
cw = (G / 7) / WR
HLU = [(int(uu[HI]), 0)] if uu[HI] < 7 else []
grid(ax.inset_axes([0, 0, cw, 1]), [[v_] for v_ in uL], "u", lim=ULIM, hl=HLU)
grid(ax.inset_axes([1 - cw, 0, cw, 1]), [[v_] for v_ in rL], "r", lim=Rlim, hl=HLU)
for i_, (uv, rv) in enumerate(zip(uL, rL)):
    yy = 1 - (i_ + 0.5) / 7
    s_ = f"r = {uv} ÷ 2 = {rv:+d}" if uv % 2 == 0 else f"r = −({uv} + 1) ÷ 2 = −{-rv}"
    if uv == 0: s_ = "r = 0 ÷ 2 = 0"
    ax.text(cw + 0.06, yy, s_, transform=ax.transAxes, ha="left", va="center", fontsize=CFS,
            color=face("r", float(np.sign(rv)) * 2.0, 2.0)[0] if rv != 0 else INK, fontweight="normal")
    ax.annotate("", xy=(1 - cw - 0.02, yy), xytext=(0.68, yy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10))
ttl(fig, ax, "Unfolding rule: u even: r = u ÷ 2; u odd: r = −(u + 1) ÷ 2")
grid(A[1][0], Pp, "pred", lim=Tlim, hl=[HI]); ttl(fig, A[1][0], "Prediction p (σ)")
grid(A[1][1], r, "r", lim=Rlim, hl=[HI]); ttl(fig, A[1][1], f"r × q (q = {Q:g})")
grid(A[1][2], Fp, "T", lim=Tlim, hl=[HI]); ttl(fig, A[1][2], "Reconstructed T̂ (σ)")
for a0, a1_, sym in ((A[1][0], A[1][1], "+"), (A[1][1], A[1][2], "=")):
    b0_, b1_ = a0.get_position(), a1_.get_position()
    fig.text((b0_.x1 + b1_.x0) / 2, (b0_.y0 + b0_.y1) / 2, sym, ha="center", va="center", fontsize=12, color=INK)
b0_, b1_ = A[0][1].get_position(), A[1][1].get_position(); xx_ = b1_.x1 + 0.25 / FW   # r goes down into the sum
fig.patches.append(FancyArrowPatch((xx_, (b0_.y0 + b0_.y1) / 2), (xx_, (b1_.y0 + b1_.y1) / 2), transform=fig.transFigure,
                                   connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1))
b_ = A[1][0].get_position()
fig.text(b_.x0, b_.y0 - 0.14 / FH, f"Our pixel: {Pp[HI]:.2f} {'+' if r[HI] >= 0 else '−'} {abs(int(r[HI]))} × {Q:g} = {Fp[HI]:.2f}",
         ha="left", va="top", fontsize=CFS, color=INK)
grid(A[2][0], real, "raw", lim=RAWLIM, hl=[HI]); ttl(fig, A[2][0], "Decoded (ADU)")
grid(A[2][1], patch, "raw", lim=RAWLIM, hl=[HI]); ttl(fig, A[2][1], "Raw (ADU)")
grid(A[2][2], errp.round(2), "err", lim=0.5, hl=[HI]); ttl(fig, A[2][2], "Error (σ)")
wrap_arrow(fig, A[1][2], A[2][0])
for a0, a1_, sym in ((A[2][0], A[2][1], "−"), (A[2][1], A[2][2], "=")):   # decoded - raw = error
    b0_, b1_ = a0.get_position(), a1_.get_position()
    fig.text((b0_.x1 + b1_.x0) / 2, (b0_.y0 + b0_.y1) / 2, sym, ha="center", va="center", fontsize=12, color=INK)
b_ = A[2][0].get_position(); dA = int(real[HI]) - int(patch[HI])
sg = lambda v_, f_: ("−" if v_ < 0 else "+") + format(abs(v_), f_)
fig.text(b_.x0, b_.y0 - 0.14 / FH, f"Our pixel: {int(real[HI])} − {int(patch[HI])} = {sg(dA, 'd')} ADU, which is "
         f"{sg(errp[HI], '.2f')}σ",
         ha="left", va="top", fontsize=CFS, color=INK)
save(fig, "fig10_decode.png")

# ---- fig11: the q dial: decoded at q = 1, 2, 3 and their errors
QD = (1.0, 2.0, 3.0)
fig, A = panels(2, 3, gapx=1.15, gapy=1.0)
pats = [np.array(per_q[q_]["patch"], np.int64) for q_ in QD]   # the chunk coded at each q, the patch of z plane 32
lim_q = (float(min([patch.min()] + [a_.min() for a_ in pats])), float(max([patch.max()] + [a_.max() for a_ in pats])))
errs = [TAB[a_] - TAB[patch.astype(np.int64)] for a_ in pats]
elim = 1.5                                               # one scale for the three: white 0 .. red 1.5 noise units
for j, (q_, a_, e_) in enumerate(zip(QD, pats, errs)):
    grid(A[0][j], a_, "raw", lim=lim_q, hl=[HI]); ttl(fig, A[0][j], f"Decoded, q = {q_:g} (ADU)")
    grid(A[1][j], e_.round(2), "err", lim=elim, hl=[HI]); ttl(fig, A[1][j], f"Error, q = {q_:g}σ")
facts["q_patch_err_max"] = {f"{q_:g}": round(float(np.abs(e_).max()), 2) for q_, e_ in zip(QD, errs)}
save(fig, "fig11_q.png")

dump_facts()
print("figures in", FIG)

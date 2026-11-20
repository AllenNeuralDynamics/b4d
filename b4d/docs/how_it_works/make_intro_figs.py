"""Introduction figure of README.md (Figure 2): sustained write rate of the b4d writer on CPU and GPU against the
maximum rate of four scientific cameras, the cost of copying frames, and the same with multiscale on. Reads the
benchmark result tables in data/ (described in data/README.md); measures nothing. A table that does not exist leaves its
curves or bars out.

    python make_intro_figs.py [--data DIR] [--theme dark] [--facts FILE.json]

Rates are writer rates with an in-memory sink (compression + shard assembly), q = 1, closed loop, three-pixel-mean
predictor, Rice coding, 64^3 chunks. Capacity of a ladder = the writer rate of the highest offered frame rate that ran
30 s with zero dropped frames (see ladder()). Style: Arial 12 pt, 1600 px wide."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# camera: (label, frame bytes as written, maximum full-frame rate of the real camera (fps)). The benchmark simulates each
# camera's frame and noise model and raises the frame rate until frames drop; the real maximums are much lower:
#   IMX411 6.3 fps (12-bit readout; 2.0 fps at 16 bit), IMX811 12.4 fps (12-bit; 5.3 fps at 16 bit): Sony datasheets as
#   published by FRAMOS / Macnica; GSPRINT4521 1,000 fps (8 bit); Teledyne Kinetix 500 fps (8 bit).
CAMERAS = {
    "imx411": ("Sony IMX411", 10640 * 14192 * 2, 6.3),
    "imx811": ("Sony IMX811", 12800 * 19200 * 2, 12.4),
    "gsprint4521": ("GSPRINT4521", 4096 * 5120, 1000.0),
    "kinetix": ("Teledyne Kinetix", 3200 * 3200, 500.0),
}
COLORS = {"imx411": "#2a78d6", "imx811": "#e34948", "gsprint4521": "#6250d6", "kinetix": "#1baf7a"}
SETS = (("CPU (128 cores)", 0, 0.35), ("1 GPU", 1, 0.65), ("2 GPUs", 2, 1.0))     # bar groups: label, GPUs, shade


def style(dark):
    # the page colours of make_m1_a3_guide.py (dark: GitHub's dark page), so Figure 2 matches Figures 3-13
    paper, ink, muted = ("#0d1117", "#e6edf3", "#8b949e") if dark else ("#ffffff", "#0b0b0b", "#8c8b87")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.facecolor": paper, "figure.facecolor": paper, "axes.facecolor": paper,
        "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink})
    return ink, muted


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ok(r):
    return float(r["dropped"] or 0) == 0


def sweep(data, name):
    """camera -> {units: GB/s}: the confirm run where there is one, else the best zero-drop probe"""
    best = defaultdict(dict)
    for r in rows(os.path.join(data, name)):
        if not ok(r) or not r["writer_gbs"]:
            continue
        k = int(r["units"]); rank = (1 if r["stage"] in ("confirm", "confirm2", "confirm3", "monitor") else 0, float(r["writer_gbs"]))
        if r["stage"] == "monitor":
            continue
        cur = best[r["camera"]].get(k)
        if cur is None or rank > cur:
            best[r["camera"]][k] = rank
    return {c: {k: v[1] for k, v in d.items()} for c, d in best.items()}


def unconfirmed(data, name):
    """(camera, units) whose confirm run dropped frames and never passed: the plotted value is only a probe's capacity"""
    good, bad = set(), set()
    for r in rows(os.path.join(data, name)):
        if r["stage"] in ("confirm", "confirm2", "confirm3"):        # pinned sweeps step the confirm down (sweep.py --pin)
            (good if ok(r) else bad).add((r["camera"], int(r["units"])))
    return bad - good


def ladder(data, name):
    """(camera, ngpu) -> sustained fps: the writer's rate at the highest offer sustained with zero drops, capped at the
    lowest offer above it that dropped frames. Where the writer's speed is the limit this
    keeps its resolution (finer than the x1.10 ladder steps); where drops come from stalls (CPU multiscale: writer 2789 fps,
    sustained 1305 fps) the cap holds the value within one step of what the camera actually streamed."""
    passed, failed = {}, {}
    for r in rows(os.path.join(data, name)):
        k = (r["camera"], int(r["ngpu"] or 0)); off = float(r["fps_offered"])
        if ok(r):
            if k not in passed or off > passed[k][0]:
                passed[k] = (off, float(r["writer_fps"]))
        else:
            failed.setdefault(k, []).append(off)
    out = {}
    for k, (off, wfps) in passed.items():
        above = [f for f in failed.get(k, []) if f > off]
        out[k] = max(off, min([wfps] + above))
    return out


def cpu_panel(ax, cpu, title, facts, key, open_pts=frozenset()):
    for c, (lab, fb, fmax) in CAMERAS.items():
        d = cpu.get(c)
        if not d:
            continue
        k = sorted(d); v = [d[i] for i in k]
        ax.plot(k, v, "-o", ms=2.5, color=COLORS[c], label=lab)
        ko = [i for i in k if (c, i) in open_pts]                  # confirm failed: drawn open (probe capacity only)
        if ko:
            ax.plot(ko, [d[i] for i in ko], "o", ms=3.2, color=COLORS[c], mfc=ax.get_facecolor(), mew=0.9, zorder=3)
        k0 = k[0]; per = d[k0] / k0                                     # linear scaling from the smallest core count measured
        ax.plot([k0, k[-1]], [d[k0], per * k[-1]], ":", color=COLORS[c], lw=0.8)
        facts[f"{key}_{c}"] = dict(k0=k0, gbs_k0=d[k0], gbs_per_core=per, gbs_32=d.get(32), gbs_64=d.get(64), gbs_128=d.get(128),
                                   eff_64=d.get(64, 0) / (64 * per), eff_128=d.get(128, 0) / (128 * per))
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlim(3.4, 160)
    ax.set_xticks([4, 8, 16, 32, 64, 128]); ax.xaxis.set_major_formatter(mt.ScalarFormatter())
    ax.xaxis.set_minor_locator(mt.LogLocator(base=2, subs=(1.25, 1.5, 1.75), numticks=20)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.yaxis.set_major_locator(mt.FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.yaxis.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_ylim(0.8, 90)
    ax.set_xlabel("CPU cores"); ax.set_ylabel("Write rate (GB/s)"); ax.set_title(title)


def gpu_panel(ax, one, two, title, facts, key, muted):
    for c in CAMERAS:
        for d, ls in ((one.get(c), "-"), (two.get(c), "--")):
            if d:
                k = sorted(d); ax.plot(k, [d[i] for i in k], ls, marker="o", ms=2.5, color=COLORS[c])
        if one.get(c):
            facts[f"{key}_{c}"] = {str(i): one[c][i] for i in sorted(one[c])}
        if two.get(c):
            facts[f"{key}_2gpu_{c}"] = {str(i): two[c][i] for i in sorted(two[c])}
    top = max([v for d in list(one.values()) + list(two.values()) for v in d.values()] + [20])
    ax.set_xlim(0, 88); ax.set_ylim(0, 25 if top <= 23 else 5 * np.ceil(top * 1.1 / 5))
    ax.xaxis.set_major_locator(mt.MultipleLocator(20)); ax.xaxis.set_minor_locator(mt.MultipleLocator(4))
    ax.yaxis.set_major_locator(mt.MultipleLocator(5 if top <= 23 else 10)); ax.yaxis.set_minor_locator(mt.AutoMinorLocator())
    ax.set_xlabel("SMs per GPU (of 84)"); ax.set_ylabel("Write rate (GB/s)"); ax.set_title(title)


def bar_rows(ax):
    """horizontal bars: one group of three per camera, the first camera on top"""
    cams = list(CAMERAS); y = np.arange(len(cams)); ax.set_ylim(len(cams) - 0.5, -0.5)
    ax.set_yticks(y); ax.tick_params(axis="y", which="both", left=False)
    return cams, y, 0.28


def rate_panel(ax, lad, title, facts, key, ink):
    """sustained rate relative to the real camera's maximum: CPU (128 cores), 1 GPU, 2 GPUs"""
    cams, y, wb = bar_rows(ax)
    for j, (lab, ng, al) in enumerate(SETS):
        for i, c in enumerate(cams):
            f = lad.get((c, ng))
            if f is None:
                continue
            v = f / CAMERAS[c][2]
            ax.barh(y[i] + (j - 1) * wb, v, wb * 0.9, color=COLORS[c], alpha=al)
            ax.text(v * 1.12, y[i] + (j - 1) * wb, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="left", va="center",
                    fontsize=11, color=ink)
            facts.setdefault(f"{key}_{c}", {})[lab] = dict(fps=f, gbs=f * CAMERAS[c][1] / 1e9, frac=v)
    ax.axvline(1.0, color=ink, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlim(0.3, 300)
    ax.xaxis.set_major_locator(mt.FixedLocator([1, 10, 100]))
    ax.xaxis.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlabel("Rate relative to\nthe real camera"); ax.set_title(title)


def cost_panel(ax, pairs, title, xlabel, facts, key):
    """rate lost (%) = 1 - rate / reference rate; pairs: (label, reference table, table, GPUs, shade)"""
    cams, y, wb = bar_rows(ax)
    for j, (lab, ref, cur, ng, al) in enumerate(pairs):
        for i, c in enumerate(cams):
            a_, b_ = ref.get((c, ng)), cur.get((c, ng))
            if a_ is None or b_ is None:
                continue
            l_ = 100 * (1 - b_ / a_)
            ax.barh(y[i] + (j - 1) * wb, l_, wb * 0.9, color=COLORS[c], alpha=al)
            facts.setdefault(f"{key}_{c}", {})[lab] = dict(reference_fps=a_, fps=b_, loss_pct=l_)
    top = max([p_.get_width() for p_ in ax.patches] + [0])
    ax.set_xlabel(xlabel); ax.set_xlim(0, 30 if top <= 28 else 10 * np.ceil(top * 1.08 / 10))
    ax.xaxis.set_major_locator(mt.MultipleLocator(10)); ax.xaxis.set_minor_locator(mt.MultipleLocator(5))
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "data"), help="directory of the result tables (default: data/)")
    ap.add_argument("--theme", default=os.environ.get("B4D_GUIDE_THEME", "light"))
    ap.add_argument("--out", default=os.path.join(here, "figs"))
    ap.add_argument("--facts", help="also write the plotted numbers to this JSON file")
    a = ap.parse_args()
    dark = a.theme == "dark"
    ink, muted = style(dark)
    out = os.path.join(a.out, "dark") if dark else a.out
    os.makedirs(out, exist_ok=True)
    D = a.data

    # SM and core sweeps (a, b, e, f); each CPU run held to N physical cores
    CPU_OFF, CPU_MS = "sweep_cpu_cores.csv", "sweep_cpu_cores_multiscale.csv"
    cpu = sweep(D, CPU_OFF)
    cpu_ms = sweep(D, CPU_MS)
    sm1 = sweep(D, "sweep_gpu_sms_1gpu.csv")
    kin512 = sweep(D, "sweep_gpu_sms_1gpu_kinetix512.csv")              # Kinetix at 512-frame blocks, as in the ladders
    if kin512.get("kinetix"):
        sm1["kinetix"] = kin512["kinetix"]
    sm2 = sweep(D, "sweep_gpu_sms_2gpu.csv")
    sm1_ms = sweep(D, "sweep_gpu_sms_1gpu_multiscale.csv")
    sm2_ms = sweep(D, "sweep_gpu_sms_2gpu_multiscale.csv")
    # the Kinetix multiscale sweeps are limited by their saturated probe (2 GPUs: 6-7 GB/s against 24.2 in the ladder;
    # 1 GPU: 8.7 GB/s at 84 SMs against 13.0), so Kinetix in f comes from ladders at 32/48/64/84 SMs
    for ng, d_ in ((1, sm1_ms), (2, sm2_ms)):
        d_.pop("kinetix", None)
        for sm in (32, 48, 64, 84):
            f_ = ladder(D, f"ladder_gpu_kinetix_multiscale_{ng}gpu_sm{sm}.csv").get(("kinetix", ng))
            if f_:
                d_.setdefault("kinetix", {})[sm] = f_ * CAMERAS["kinetix"][1] / 1e9
    # ladders at full size (c, d, g, h): 128 cores, 84 SMs per GPU
    zc = {**ladder(D, "ladder_cpu_zero_copy.csv"), **ladder(D, "ladder_gpu_zero_copy.csv")}
    cp = {**ladder(D, "ladder_cpu_copy.csv"),
          **{k: v for k, v in ladder(D, "ladder_gpu_copy.csv").items() if k[1] == 1},
          **{k: v for k, v in ladder(D, "ladder_gpu_copy_split.csv").items() if k[1] == 2}}
    ms = {**ladder(D, "ladder_cpu_multiscale.csv"), **ladder(D, "ladder_gpu_multiscale.csv")}
    # the CPU copy cost against the zero-copy ladder run back to back with the copy ladder (same build)
    zc_cpu_old = ladder(D, "ladder_cpu_zero_copy_paired.csv")

    # layout (inches): two rows; a, b (e, f) share the first 60 % of the width as 4:3 panels, c, d (g, h) the rest at
    # the same height, as horizontal bars with the camera names left of c (g); the keys above; the notes below
    W = 11.2; T = 0.6 * W / 2; R = W - 2 * T; wA = T - 0.85 - 0.25; hA = 0.75 * wA; LB = 1.35
    wC = (R - LB - 0.3 - 0.15) / 2
    BOT, GAP, TOP = 3.15, 1.9, 1.75; H = BOT + 2 * hA + GAP + TOP
    fig = plt.figure(figsize=(W, H))
    xs = [(0.85, wA), (T + 0.85, wA), (2 * T + LB, wC), (2 * T + LB + wC + 0.3, wC)]
    y1, y2 = BOT + hA + GAP, BOT
    r1 = [fig.add_axes([x0 / W, y1 / H, w_ / W, hA / H]) for x0, w_ in xs]
    r2 = [fig.add_axes([x0 / W, y2 / H, w_ / W, hA / H]) for x0, w_ in xs]
    facts = {}

    cpu_panel(r1[0], cpu, "CPU (AMD EPYC 9754),\nmultiscale off", facts, "cpu", unconfirmed(D, CPU_OFF))
    gpu_panel(r1[1], sm1, sm2, "GPU (RTX A6000, PCIe Gen4),\nmultiscale off", facts, "sm", muted)
    rate_panel(r1[2], zc, "Sustained rate,\nzero dropped\nframes", facts, "rate", ink)
    cost_panel(r1[3], [("CPU (128 cores)", zc_cpu_old, cp, 0, 0.35), ("1 GPU", zc, cp, 1, 0.65), ("2 GPUs", zc, cp, 2, 1.0)],
               "Cost of\ncopying frames", "Rate lost to\ncopying frames (%)", facts, "copy")
    cpu_panel(r2[0], cpu_ms, "CPU (AMD EPYC 9754),\nmultiscale on", facts, "cpu_ms", unconfirmed(D, CPU_MS))
    gpu_panel(r2[1], sm1_ms, sm2_ms, "GPU (RTX A6000, PCIe Gen4),\nmultiscale on", facts, "sm_ms", muted)
    rate_panel(r2[2], ms, "Sustained rate,\nzero dropped\nframes", facts, "rate_ms", ink)
    cost_panel(r2[3], [(lab, zc, ms, ng, al) for lab, ng, al in SETS], "Cost of\nmultiscale",
               "Rate lost to\nmultiscale (%)", facts, "ms_cost")

    ykey = (y1 + hA + 1.0) / H                                  # the keys' bottom, above the titles
    fig.legend(handles=[Line2D([], [], color=COLORS[c], marker="o", ms=3, label=CAMERAS[c][0]) for c in CAMERAS]
               + [Line2D([], [], color=muted, ls="-", label="1 GPU"), Line2D([], [], color=muted, ls="--", label="2 GPUs")],
               loc="lower left", bbox_to_anchor=(0.85 / W, ykey), ncol=3, handlelength=1.8, handletextpad=0.5,
               columnspacing=1.2, borderaxespad=0)
    fig.legend(handles=[Patch(color=muted, alpha=al, label=lab) for lab, _, al in SETS]
               + [Line2D([], [], color=ink, lw=0.8, ls="--", label="Real camera")],
               loc="lower left", bbox_to_anchor=((2 * T + 0.6) / W, ykey), ncol=2,
               handlelength=1.2, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
    notes = ("8 bit: GSPRINT4521, Teledyne Kinetix; 16 bit: Sony IMX411, Sony IMX811. 8-bit and 16-bit frames are encoded "
             "at similar pixel rates (0.33–0.39 Gpixel/s per core at 4 cores; 6.1 and 6.4 Gpixel/s at 32 SMs), so the 16-bit "
             "sensors are written at about twice the byte rate. On the GPU the 16-bit rate levels off at 20–21 GB/s from 64 SMs, "
             "near the PCIe Gen4 x16 limit. SMs: streaming multiprocessors of an NVIDIA RTX A6000. a, e: each run held to N physical cores (both SMT threads), saturated 20 s probe, then "
             "a 30 s confirmation with zero dropped frames. c, d, g, h: ladders in x1.10 steps (resolution about 10 %). d: the "
             "2-GPU copy run splits each frame by rows across the GPUs; the 2-GPU zero-copy run uses the default settings. "
             "e–h: multiscale levels halve x, y and z down to about 256 pixels, each level compressed with b4d; on the GPU the "
             "full-resolution level is encoded on the GPU and the coarser levels on the CPU.")
    import textwrap
    fig.text(0.85 / W, (BOT - 0.95) / H, "\n".join(textwrap.wrap(notes, 118)), ha="left", va="top", color=muted)

    cams = list(CAMERAS)
    for ax in (r1[2], r2[2]):                                   # camera names left of c and g only
        ax.set_yticklabels([f"{CAMERAS[c][0]}\n({'8' if CAMERAS[c][1] < 3e7 else '16'} bit)" for c in cams])
    for ax in (r1[3], r2[3]):
        ax.set_yticklabels([])
    for ax in r1 + r2:
        ax.tick_params(which="minor", length=2.5, width=0.6)
    for ax in r1[:2] + r2[:2]:                                  # faint gridlines on the line plots
        ax.grid(True, which="major", color=muted, alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for row, ks in ((r1, "abcd"), (r2, "efgh")):
        ytop = max(ax.get_tightbbox(rr).transformed(inv).y1 for ax in row)
        for ax, k in zip(row, ks):
            fig.text(ax.get_tightbbox(rr).transformed(inv).x0 + 0.002, ytop, k, fontsize=14, fontweight="bold",
                     fontfamily="Arial", ha="left", va="top")
    p = os.path.join(out, "fig00b_write_rates.png")
    fig.savefig(p, dpi=1600 / W); plt.close(fig); print(p)
    if a.facts:
        import json
        with open(a.facts, "w") as f:
            json.dump(facts, f, indent=1)


if __name__ == "__main__":
    main()

"""b4d-tune: find the best B4D writer settings for a camera on the machine it runs on, automatically, by measurement.

    b4d-tune --camera imx811 --q 1 --out tune_imx811_q1
    b4d-tune --shape 2048,2048 --conv 3.94 --offset 100 --rn 0.71 --max-adu 65535 --fps-start 1500 --q 2 --name aries
    b4d-tune --camera kinetix --q 2 --quick            # 30 s probes only, no 60 s confirmations (coarse, ~3x faster)
    b4d-record --source ... --tuned tune_imx811_q1/recommendation.json     # record with the result

The camera is simulated (tools/stream_sim.py, run as a subprocess; found next to a source checkout, or give --sim /
$B4D_STREAM_SIM: a sphere phantom through the camera model at a fixed frame rate, zero-copy
into the native writer, memory sink = CPU/pipeline capacity with no drive in the loop). For one setting the
*capacity* is the highest frame rate with zero dropped frames, found by bracketing (+-10% steps, geometric bisection
to +-2%, then a 60 s confirmation). The tuner then changes one parameter at a time, in the order that mattered in
the writer tuning measurements (docs/WRITER_DESIGN.md), and keeps a change only when it is at least --min-gain better:

    1. block depth (frames per block = shard depth)            2. chunk footprint (y, x) at that depth
    3. blocks in flight (--max-commits)                        4. compression threads
    5. shard footprint (rows per shard file)

Every run is one CSV row (results.csv in --out), its stdout is kept under --out/logs, and the final answer is written
to --out/recommendation.json and printed: the settings, the measured capacity, the recommended operating rate
(capacity / 1.2, the margin that survived the soak tests), the compression ratio and the compressed data rate the
storage must sustain. Expect 1-3 hours per camera and q on a 128-core machine; --quick or a smaller candidate list
(--blocks, --footprints, --commits, --threads, --shard-rows) shortens it. The default candidates come from the
machine (b4d.hardware): thread counts around the physical cores, block depths that fit the RAM budget of the pool. Resume: rows already in results.csv are
reused, so an interrupted tuning continues where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

from . import hardware


def find_stream_sim(explicit=None):
    """Path of tools/stream_sim.py, the measuring instrument: --sim, $B4D_STREAM_SIM, or searched upwards from the
    current directory and from this module (a source checkout has b4d/tools/stream_sim.py)."""
    cands = [explicit, os.environ.get("B4D_STREAM_SIM")]
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        d = os.path.abspath(start)
        while True:
            cands += [os.path.join(d, "tools", "stream_sim.py"), os.path.join(d, "b4d", "tools", "stream_sim.py")]
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit("stream_sim.py not found: run from a b4d source checkout, or pass --sim / set B4D_STREAM_SIM")

#: camera presets: shape rows,cols; conv = DN per e- (1/gain); offset DN; rn e-; max_adu = full well of the container
CAMERAS = {
    "imx411": dict(shape="10640,14192", conv=1 / 7.25, offset=2.05, rn=6.89, max_adu=4095, fps_start=110),
    "imx811": dict(shape="12800,19200", conv=1 / 7.25, offset=2.05, rn=6.89, max_adu=4095, fps_start=70),
    "kinetix": dict(shape="3200,3200", conv=0.851, offset=6.0, rn=2.35, max_adu=255, fps_start=1500),
    "aries": dict(shape="2048,2048", conv=3.94, offset=100.0, rn=0.71, max_adu=65535, fps_start=2000),
}
COLUMNS = [
    "timestamp",
    "stage",
    "q",
    "threads",
    "block",
    "chunks",
    "shards",
    "max_commits",
    "blocks",
    "ring",
    "fps",
    "seconds",
    "sustained",
    "dropped",
    "produced",
    "fps_achieved",
    "camera_limited",
    "ratio",
    "compress_s_per_block",
    "io_s_per_block",
    "lat_mean",
    "groups",
    "wall_s",
    "log",
]


# ------------------------------------------------------------------------------------------------- one run
class Runner:
    def __init__(self, a):
        self.a = a
        self.csv = os.path.join(a.out, "results.csv")
        self.logs = os.path.join(a.out, "logs")
        os.makedirs(self.logs, exist_ok=True)
        self.cache = {}  # (config key, fps, seconds) -> row, from results.csv (resume)
        if os.path.exists(self.csv):
            with open(self.csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("sustained") in ("0", "1"):
                        self.cache[(self.key_of(row), float(row["fps"]), float(row["seconds"]))] = row
        self.last_end = 0.0
        self.last_clean = True
        ny, nx = (int(v) for v in a.shape.split(","))
        self.pixels = ny * nx

    @staticmethod
    def key_of(row):
        return (int(row["threads"]), int(row["block"]), row["chunks"], row["shards"], int(row["max_commits"]))

    def cmd(self, cfg, fps, seconds):
        a = self.a
        return [
            sys.executable,
            a.sim,
            "--zero-copy",
            "--sink",
            "memory",
            "--multiscale",
            "0",
            "--shape",
            a.shape,
            "--conv",
            str(a.conv),
            "--offset",
            str(a.offset),
            "--rn",
            str(a.rn),
            "--spot-peak",
            a.spot_peak,
            "--max-adu",
            str(a.max_adu),
            "--q",
            str(a.q),
            "--threads",
            str(cfg["threads"]),
            "--block",
            str(cfg["block"]),
            "--bank",
            str(cfg["block"]),
            "--blocks",
            str(cfg["max_commits"] + 1),
            "--ring",
            str(cfg["block"]),
            "--chunks",
            cfg["chunks"],
            "--shards",
            cfg["shards"],
            "--max-commits",
            str(cfg["max_commits"]),
            "--fps",
            f"{fps:.5g}",
            "--seconds",
            str(seconds),
            "--warmup-blocks",
            str(a.warmup_blocks),
            "--status",
            "5",
            "--abort-drops",
            "32",
        ]

    def run(self, cfg, fps, seconds, stage):
        """-> (sustained, row); cached rows are reused"""
        key = (self.key_of(cfg), round(fps, 5), float(seconds))
        if key in self.cache:
            row = self.cache[key]
            print(
                f"   [cached] {stage}: {fps:.5g} fps -> {'SUSTAINED' if row['sustained'] == '1' else 'DROPS'}",
                flush=True,
            )
            return row["sustained"] == "1", row
        a = self.a
        if self.last_end:  # let the OS tear down the previous run's buffers; longer after a dropping run
            want = a.pause if self.last_clean else 3 * a.pause
            gap = time.time() - self.last_end
            if gap < want:
                time.sleep(want - gap)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(
            self.logs,
            f"{ts}_t{cfg['threads']}_b{cfg['block']}_c{cfg['chunks'].replace(',', 'x')}_m{cfg['max_commits']}_f{fps:.5g}.log",
        )
        cmd = self.cmd(cfg, fps, seconds)
        print(
            f"\n=== {stage}: {fps:.5g} fps, {seconds:g} s  (threads {cfg['threads']}, block {cfg['block']}, chunks {cfg['chunks']}, shards {cfg['shards']}, "
            f"commits {cfg['max_commits']})",
            flush=True,
        )
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            lf.write(" ".join(cmd) + "\n\n")
            p = subprocess.Popen(
                cmd,
                cwd=a.out,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            out = []
            for line in p.stdout:
                lf.write(line)
                out.append(line)
                if line.startswith(("SUSTAINED", "NOT SUSTAINED")) or "native writer: compress" in line:
                    print("   " + line.rstrip(), flush=True)
            p.wait()
        text = "".join(out)
        m = re.search(r"^RESULT (.*)$", text, re.M)
        if not m:
            raise RuntimeError(f"stream_sim failed (exit {p.returncode}); see {log}")
        r = json.loads(m.group(1))
        mc = re.search(
            r"native writer: compress ([\d.]+) s \+ shard IO ([\d.]+) s per block; worker threads observed in (\d+) processor group",
            text,
        )
        sustained = bool(r["sustained"])
        el = r["written"] * self.pixels * 2 / max(r["raw_mbs"] * 1e6, 1e-9)
        achieved = r["written"] / max(el - r["lat_mean"], 1e-9)
        camera_limited = sustained and achieved < 0.97 * fps  # the Python frame loop, not the writer, was the limit
        row = dict(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            stage=stage,
            q=a.q,
            threads=cfg["threads"],
            block=cfg["block"],
            chunks=cfg["chunks"],
            shards=cfg["shards"],
            max_commits=cfg["max_commits"],
            blocks=cfg["max_commits"] + 1,
            ring=cfg["block"],
            fps=f"{fps:.5g}",
            seconds=seconds,
            sustained=int(sustained),
            dropped=r["dropped"],
            produced=r["produced"],
            fps_achieved=f"{achieved:.5g}",
            camera_limited=int(camera_limited),
            ratio=f"{r['ratio']:.3f}",
            compress_s_per_block=mc.group(1) if mc else "",
            io_s_per_block=mc.group(2) if mc else "",
            lat_mean=f"{r['lat_mean']:.2f}",
            groups=mc.group(3) if mc else "",
            wall_s=f"{time.time() - t0:.0f}",
            log=os.path.basename(log),
        )
        new = not os.path.exists(self.csv)
        with open(self.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        self.cache[key] = {k: str(v) for k, v in row.items()}
        self.last_end, self.last_clean = time.time(), sustained
        if camera_limited:
            print(
                f"   !! simulator-limited at {fps:.5g} fps (achieved {achieved:.4g}): the frame loop, not the writer, is the ceiling",
                flush=True,
            )
        return sustained, self.cache[key]

    def capacity(self, cfg, fps_start, stage):
        """highest sustained fps for cfg: +-10% bracketing, bisection to +-2%, 60 s confirmation (unless --quick)"""
        a = self.a
        f = fps_start
        ok, row = self.run(cfg, f, a.seconds, stage)
        if row["camera_limited"] == "1":
            return None, row
        lo = hi = None
        if ok:
            lo = f
            while hi is None:
                f *= 1.10
                ok, row = self.run(cfg, f, a.seconds, stage)
                if row["camera_limited"] == "1":
                    return None, row
                if ok:
                    lo = f
                else:
                    hi = f
        else:
            hi = f
            while lo is None:
                f /= 1.10
                if f < 0.05 * fps_start:
                    return 0.0, row
                ok, row = self.run(cfg, f, a.seconds, stage)
                if ok:
                    lo = f
                else:
                    hi = f
        while hi / lo > 1.04:
            f = (lo * hi) ** 0.5
            ok, row = self.run(cfg, f, a.seconds, stage)
            if ok:
                lo = f
            else:
                hi = f
        if a.quick:
            return lo, self.cache[(self.key_of(cfg), round(lo, 5), float(a.seconds))]
        f = lo
        for i in range(a.max_confirms):  # back off 2% per failure, then 5%
            ok, row = self.run(cfg, f, a.confirm_seconds, stage + " confirm")
            if ok:
                return f, row
            f /= 1.02 if i < a.max_confirms // 2 else 1.05
        return None, row


# ------------------------------------------------------------------------------------------------- geometry helpers
def shard_for(rows, cols, chunk_y, chunk_x, shard_rows):
    sy = max(chunk_y, (min(shard_rows, rows) // chunk_y) * chunk_y)
    sx = (
        max(chunk_x, (cols // chunk_x) * chunk_x)
        if cols % chunk_x == 0
        else max(chunk_x, (min(3840, cols) // chunk_x) * chunk_x)
    )
    return sy, sx


def default_blocks(fps_start, frame_bytes, max_commits=3):
    """block depths worth trying: >= 0.25 s of frames or >= 2 GB per block, powers of two from 32 to 512, and only
    depths whose pool (max_commits + 1 blocks) fits the RAM budget (b4d.hardware.pool_budget_bytes)"""
    cands = []
    for b in (32, 64, 128, 256, 512):
        if b / fps_start >= 0.25 or b * frame_bytes >= 2e9 or b == 512:
            cands.append(b)
    cands = cands[:3] if cands else [64]
    fit = hardware.max_block_frames(frame_bytes, max_commits + 1)
    if fit is not None:
        cands = [b for b in cands if b <= fit] or [min(cands)]
    return cands


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument(
        "--camera", choices=sorted(CAMERAS), help="preset; or give --shape --conv --offset --rn --max-adu --fps-start"
    )
    p.add_argument("--name", default=None)
    p.add_argument("--shape", default=None, help="rows,cols")
    p.add_argument("--conv", type=float, default=None, help="DN per e- (1/gain)")
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--rn", type=float, default=None, help="read noise e-")
    p.add_argument(
        "--max-adu",
        type=int,
        default=None,
        help="full well in ADU of the container (4095 for 12-bit, 255 for 8-bit, 65535)",
    )
    p.add_argument("--fps-start", type=float, default=None, help="first guess of the capacity")
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--spot-peak", default="64,64", help="phantom brightness range in e- (ratio and speed depend on it)")
    p.add_argument(
        "--blocks", default=None, help="block depths to try, e.g. 32,64 (default from the frame rate and size)"
    )
    p.add_argument(
        "--footprints", default="64x64,32x32,128x128", help="chunk (y x x) footprints to try at the chosen depth"
    )
    p.add_argument(
        "--chunk-depth",
        type=int,
        default=32,
        help="chunk depth (<= block depth; the best in the tuning measurements (docs/WRITER_DESIGN.md) for 64-frame blocks)",
    )
    p.add_argument("--commits", default="2,1,3", help="blocks in flight to try")
    p.add_argument(
        "--threads", default=None, help="thread counts to try (default: the physical cores, then 0.75x, 1.25x and 1.5x of it, within the logical CPUs)"
    )
    p.add_argument("--shard-rows", default=None, help="shard heights to try (default: frame/8, frame/16, frame/4)")
    p.add_argument(
        "--min-gain", type=float, default=0.03, help="keep a change only if the capacity improves by this fraction"
    )
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--confirm-seconds", type=float, default=60)
    p.add_argument("--max-confirms", type=int, default=6)
    p.add_argument("--warmup-blocks", type=int, default=4)
    p.add_argument("--pause", type=float, default=10, help="idle seconds between runs")
    p.add_argument("--quick", action="store_true", help="no 60 s confirmations (coarser, ~3x faster)")
    p.add_argument("--out", default=None, help="results directory (default tune_<name>_q<q>)")
    p.add_argument("--sim", default=None, help="path of tools/stream_sim.py (default: searched, or $B4D_STREAM_SIM)")
    a = p.parse_args(argv)
    if a.camera:
        c = CAMERAS[a.camera]
        for k, v in c.items():
            if getattr(a, k.replace("-", "_"), None) is None:
                setattr(a, k, v)
        a.name = a.name or a.camera
    for k in ("shape", "conv", "offset", "rn", "max_adu", "fps_start"):
        if getattr(a, k) is None:
            raise SystemExit(f"--{k.replace('_', '-')} is required without --camera")
    a.name = a.name or "camera"
    a.out = os.path.abspath(a.out or f"tune_{a.name}_q{a.q:g}")
    os.makedirs(a.out, exist_ok=True)
    rows, cols = (int(v) for v in a.shape.split(","))
    frame_bytes = rows * cols * 2
    a.sim = find_stream_sim(a.sim)
    max_commits = max(int(v) for v in a.commits.split(","))
    blocks = [int(v) for v in a.blocks.split(",")] if a.blocks else default_blocks(a.fps_start, frame_bytes, max_commits)
    threads = [int(v) for v in a.threads.split(",")] if a.threads else hardware.thread_candidates()
    shard_rows = (
        [int(v) for v in a.shard_rows.split(",")]
        if a.shard_rows
        else [max(64, rows // 8), max(64, rows // 16), max(64, rows // 4)]
    )
    foot = [tuple(int(v) for v in f.split("x")) for f in a.footprints.split(",")]
    R = Runner(a)
    print(
        f"tuning {a.name}: {cols}x{rows} ({frame_bytes / 1e6:.0f} MB/frame), conv {a.conv:.4g} offset {a.offset:g} rn {a.rn:g}, q {a.q:g}, peak {a.spot_peak} e-; "
        f"blocks {blocks}, footprints {foot}, commits {a.commits}, threads {threads}, shard rows {shard_rows}; results in {a.out}",
        flush=True,
    )

    def cfg_of(block, fy, fx, commits, thr, srows):
        cz = min(a.chunk_depth, block)
        sy, sx = shard_for(rows, cols, fy, fx, srows)
        return dict(
            threads=thr, block=block, chunks=f"{cz},{fy},{fx}", shards=f"{block},{sy},{sx}", max_commits=commits
        )

    best = dict(
        block=blocks[0],
        fy=foot[0][0],
        fx=foot[0][1],
        commits=int(a.commits.split(",")[0]),
        thr=threads[0],
        srows=shard_rows[0],
    )
    fps_guess = a.fps_start
    results = []

    def evaluate(stage, **override):
        nonlocal fps_guess
        c = dict(best, **override)
        cfg = cfg_of(c["block"], c["fy"], c["fx"], c["commits"], c["thr"], c["srows"])
        cap, row = R.capacity(cfg, fps_guess, stage)
        results.append(
            dict(
                stage=stage,
                cfg=cfg,
                capacity=cap,
                ratio=float(row["ratio"]) if row and row.get("ratio") else None,
                compress_s=row.get("compress_s_per_block") if row else None,
                camera_limited=row.get("camera_limited") == "1" if row else False,
            )
        )
        print(
            f"##### {stage}: {'capacity %.4g fps' % cap if cap else 'no capacity found (simulator-limited or unstable)'}  {cfg}",
            flush=True,
        )
        if cap:
            fps_guess = cap
        return cap

    def adopt(stage, key, candidates, best_cap):
        for cand in candidates:
            if cand == best[key]:
                continue
            cap = evaluate(f"{stage} {key}={cand}", **{key: cand})
            if cap and best_cap and cap > best_cap * (1 + a.min_gain):
                best[key], best_cap = cand, cap
                print(f"##### adopted {key}={cand} ({cap:.4g} fps)", flush=True)
            elif cap and best_cap is None:
                best[key], best_cap = cand, cap
        return best_cap

    best_cap = evaluate("baseline")
    best_cap = adopt("1 block depth", "block", blocks, best_cap)
    foot_c = [(fy, fx) for fy, fx in foot]
    for fy, fx in foot_c:
        if (fy, fx) == (best["fy"], best["fx"]):
            continue
        cap = evaluate(f"2 chunk footprint {fy}x{fx}", fy=fy, fx=fx)
        if cap and best_cap and cap > best_cap * (1 + a.min_gain):
            best["fy"], best["fx"], best_cap = fy, fx, cap
            print(f"##### adopted chunk footprint {fy}x{fx} ({cap:.4g} fps)", flush=True)
    best_cap = adopt("3 blocks in flight", "commits", [int(v) for v in a.commits.split(",")], best_cap)
    best_cap = adopt("4 threads", "thr", threads, best_cap)
    best_cap = adopt("5 shard rows", "srows", shard_rows, best_cap)

    cfg = cfg_of(best["block"], best["fy"], best["fx"], best["commits"], best["thr"], best["srows"])
    ratio = next((r["ratio"] for r in reversed(results) if r["cfg"] == cfg and r["ratio"]), None)
    rec = dict(
        camera=a.name,
        shape=a.shape,
        conv=a.conv,
        offset=a.offset,
        rn=a.rn,
        max_adu=a.max_adu,
        q=a.q,
        spot_peak=a.spot_peak,
        settings=dict(
            threads=cfg["threads"],
            block=cfg["block"],
            bank=cfg["block"],
            blocks=cfg["max_commits"] + 1,
            ring=cfg["block"],
            chunks=cfg["chunks"],
            shards=cfg["shards"],
            max_commits=cfg["max_commits"],
        ),
        capacity_fps=best_cap,
        recommended_fps=best_cap / 1.2 if best_cap else None,
        ratio=ratio,
        raw_gbs_at_capacity=best_cap * frame_bytes / 1e9 if best_cap else None,
        compressed_gbs_at_capacity=(best_cap * frame_bytes / 1e9 / ratio) if (best_cap and ratio) else None,
        machine=hardware.describe(),
        quick=a.quick,
        stages=results,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(a.out, "recommendation.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\n================= recommendation =================")
    print(
        f"{a.name} at q={a.q:g}: capacity {best_cap:.4g} fps -> run at {best_cap / 1.2:.4g} fps (capacity / 1.2)"
        if best_cap
        else f"{a.name}: no sustained capacity found"
    )
    print("settings: " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in rec["settings"].items()))
    if ratio and best_cap:
        print(
            f"ratio {ratio:.2f}x; raw {rec['raw_gbs_at_capacity']:.2f} GB/s, compressed {rec['compressed_gbs_at_capacity']:.2f} GB/s at capacity "
            f"({rec['compressed_gbs_at_capacity'] / 1.2:.2f} GB/s at the recommended rate)"
        )
    print(f"details: {os.path.join(a.out, 'recommendation.json')}, every run in results.csv")
    return 0


def load_recommendation(path):
    """Read a ``recommendation.json`` written by b4d-tune.

    Returns ``(recorder_kwargs, source_kwargs)``: the :class:`b4d.ingest.RecorderConfig` fields (``threads``,
    ``chunks``, ``shards``, ``max_commits``, ``q``) and the block-source pool (``frames_per_block``, ``nblocks``)."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    s = rec["settings"]
    recorder = dict(
        threads=int(s["threads"]),
        chunks=tuple(int(v) for v in s["chunks"].split(",")),
        shards=tuple(int(v) for v in s["shards"].split(",")),
        max_commits=int(s["max_commits"]),
        q=float(rec["q"]),
    )
    source = dict(frames_per_block=int(s["block"]), nblocks=int(s["blocks"]))
    return recorder, source


if __name__ == "__main__":
    sys.exit(main())

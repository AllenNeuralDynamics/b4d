"""egrabber-grab: camera-only ingest test through EGrabberSource (no compression, no b4d).

    egrabber-grab --seconds 10                       # rate, losses, timestamps, pixel summary of the first frame
    egrabber-grab --frames 128 --save first.npy      # also save the first frame
    egrabber-grab --info                             # just print the camera information and exit

For recording into B4D use b4d-record (from the b4d package) with --source egrabber_camera.source:EGrabberSource.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .source import EGrabberSource


def grab_blocks(src, max_wait_s: float):
    """Consume blocks until the source is drained; returns (first frame copy, frames delivered, timestamp arrays)."""
    first = None
    got = 0
    ts_all = []
    t_start = time.perf_counter()
    while True:
        try:
            blk = src.wait_block(2000)
        except StopIteration:
            break
        if blk is None:
            if time.perf_counter() - t_start > max_wait_s:
                print("timeout waiting for frames", flush=True)
                break
            continue
        if first is None:
            first = np.array(blk.array[0])
        ts_all.append(blk.frames["timestamp_ns"].copy())
        got += blk.nframes
        src.release(blk)
        print(
            f"  block {blk.seq}: {blk.nframes} frames, ids {blk.frames['frame_id'][0]}..{blk.frames['frame_id'][-1]}",
            flush=True,
        )
    return first, got, ts_all


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="egrabber-grab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--serial", default=None)
    p.add_argument("--fps", type=float, default=0.0, help="0 = the camera's maximum")
    p.add_argument("--exposure-us", type=float, default=0.0, help="0 = leave unchanged")
    p.add_argument("--block", type=int, default=16, help="frames per block")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--frames", type=int, default=0, help="overrides --seconds (rounded up to whole blocks)")
    p.add_argument("--save", default=None, help="save the first frame to this .npy")
    p.add_argument("--info", action="store_true", help="print camera information and exit")
    p.add_argument("--large-pages", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    src = EGrabberSource(
        a.serial,
        frames_per_block=a.block,
        nblocks=a.blocks,
        fps=a.fps,
        exposure_us=a.exposure_us,
        large_pages=a.large_pages,
        verbose=a.verbose,
    )
    i = src.info
    print(
        f"camera: {i.name} SN {i.serial}; {i.shape[1]}x{i.shape[0]} {np.dtype(i.dtype).name}, "
        f"{i.frame_bytes / 1e6:.1f} MB/frame, "
        f"{i.fps:.4f} fps ({i.fps * i.frame_bytes / 1e6:,.0f} MB/s); camera model {i.camera_model}; {i.extra}"
    )
    if a.info:
        src.close()
        return 0
    nblk = max(1, -(-a.frames // a.block)) if a.frames else max(1, int(round(i.fps * a.seconds / a.block)))
    total = nblk * a.block
    t0 = time.perf_counter()
    src.start(max_frames=total)
    print(
        f"started in {time.perf_counter() - t0:.1f} s (pool {a.blocks} x {a.block} frames = "
        f"{a.blocks * a.block * i.frame_bytes / 1e9:.1f} GB); grabbing {total} frames"
    )
    t_start = time.perf_counter()
    first, got, ts_all = grab_blocks(src, max_wait_s=a.seconds + 30)
    src.stop()
    s = src.stats()
    el = time.perf_counter() - t_start
    ts = np.concatenate(ts_all) if ts_all else np.zeros(0)
    res = dict(frames=got, seconds=el, lost=s.lost, incomplete=s.incomplete, fps_timestamp=s.fps_measured, **s.extra)
    if len(ts) > 1:
        dt = np.diff(ts.astype(np.float64)) / 1e6
        res.update(period_ms_mean=float(dt.mean()), period_ms_min=float(dt.min()), period_ms_max=float(dt.max()))
    if first is not None:
        res.update(
            first_frame_mean=float(first.mean()), first_frame_min=int(first.min()), first_frame_max=int(first.max())
        )
        if a.save:
            np.save(a.save, first)
            print(f"saved first frame to {a.save}")
    print(
        f"delivered {got} frames in {el:.1f} s; LOST {s.lost} (incomplete {s.incomplete}); {s.fps_measured:.4f} fps "
        f"from timestamps"
    )
    print("RESULT " + json.dumps(res, default=str))
    src.close()
    return 0 if s.lost == 0 and s.incomplete == 0 else 3


if __name__ == "__main__":
    sys.exit(main())

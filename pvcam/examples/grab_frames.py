"""Camera only, no compression: take blocks of frames from the PVCAM source and look at them.

    python examples/grab_frames.py --frames 64
    python examples/grab_frames.py --serial <serial> --port 1 --frames 256 --block 64     # Kinetix 8-bit Speed port

This is the vendor side of the interface on its own: the source hands out blocks that are views of the circular
buffer PVCAM writes into; whoever consumes them must call release() when done with the memory.
"""

import argparse

import numpy as np
from pvcam_camera import PvcamSource

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--port", type=int, default=None, help="readout port value (Kinetix: 0 Sensitivity, 1 Speed, ...)")
    p.add_argument("--exposure-us", type=float, default=None)
    a = p.parse_args()

    with PvcamSource(
        a.serial, frames_per_block=a.block, nblocks=a.blocks, readout_port=a.port, exposure_us=a.exposure_us
    ) as src:
        print(src)
        print({k: src.info.extra[k] for k in ("readout_port_name", "bit_depth", "exposure_us", "readout_time_us")})
        src.start(max_frames=a.frames)
        while True:
            try:
                blk = src.wait_block(timeout_ms=2000)
            except StopIteration:
                break  # stopped after max_frames and every block was handed out
            if blk is None:
                continue  # timeout: nothing complete yet
            frames = blk.array  # (nframes, H, W) view of the circular buffer PVCAM wrote: no copy
            print(
                f"{blk}: mean {frames[0].mean():.1f} ADU, "
                f"FrameNr {blk.frames['frame_id'][0]}..{blk.frames['frame_id'][-1]}"
                f", timestamps {blk.frames['timestamp_ns'][0]}..{blk.frames['timestamp_ns'][-1]} ns"
            )
            src.release(blk)  # from here on PVCAM may refill this block
        s = src.stats()
        print(
            f"delivered {s.delivered}, lost {s.lost}, incomplete {s.incomplete}; {s.fps_measured:.2f} fps from "
            f"FRAME_INFO timestamps; lock mode {s.extra['lock_mode']}; error {s.extra['error'] or 'none'}"
        )
        print(f"dtype {np.dtype(src.info.dtype).name}, camera model (conv, offset, rn) = {src.info.camera_model}")

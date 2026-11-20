"""Camera only, no compression: take blocks of frames from the eGrabber source and look at them.

    python examples/grab_frames.py --frames 64

This is the vendor side of the interface on its own: a source produces FrameBlocks in pinned memory; whoever consumes
them must call release() when done with the memory.
"""

import argparse

import numpy as np
from egrabber_camera import EGrabberSource

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--block", type=int, default=16)
    a = p.parse_args()

    with EGrabberSource(a.serial, frames_per_block=a.block, nblocks=3) as src:
        print(src)
        src.start(max_frames=a.frames)
        while True:
            try:
                blk = src.wait_block(timeout_ms=2000)
            except StopIteration:
                break  # stopped after max_frames and every block was handed out
            if blk is None:
                continue  # timeout: nothing complete yet
            frames = blk.array  # (nframes, H, W) uint16 view over the grabber's DMA memory
            print(
                f"{blk}: mean {frames[0].mean():.1f} ADU, timestamps "
                f"{blk.frames['timestamp_ns'][0]}..{blk.frames['timestamp_ns'][-1]} ns"
            )
            src.release(blk)  # from here on the grabber may refill this block
        s = src.stats()
        print(
            f"delivered {s.delivered}, lost {s.lost}, incomplete {s.incomplete}; {s.fps_measured:.3f} fps from camera "
            f"timestamps"
        )
        print(
            f"first frame dtype {np.dtype(src.info.dtype).name}, camera model (conv, offset, rn) = "
            f"{src.info.camera_model}"
        )

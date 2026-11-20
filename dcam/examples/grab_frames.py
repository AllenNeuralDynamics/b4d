"""Camera only, no compression: take blocks of frames from the DCAM source and look at them.

    python examples/grab_frames.py --frames 64
    python examples/grab_frames.py --serial <serial> --frames 400 --block 16 --blocks 4 --exposure-us 10000

This is the vendor side of the interface on its own: DCAM writes the frames straight into the source's blocks
(dcambuf_attach); whoever consumes a block must call release() when done with it. DCAM does not wait for the
consumer, so a block held longer than (blocks - 1) * block / fps seconds is overwritten; stats() counts that.
"""

import argparse

import numpy as np

from dcam_camera import DcamSource

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--pixel-type", default="MONO16", choices=["MONO16", "MONO8"])
    p.add_argument("--exposure-us", type=float, default=0.0, help="0: leave the camera's setting")
    a = p.parse_args()

    with DcamSource(
        a.serial, frames_per_block=a.block, nblocks=a.blocks, pixel_type=a.pixel_type, exposure_us=a.exposure_us
    ) as src:
        print(src)
        src.start(max_frames=a.frames)
        while True:
            try:
                blk = src.wait_block(timeout_ms=2000)
            except StopIteration:
                break  # stopped after max_frames and every block was handed out
            if blk is None:
                continue  # timeout: nothing complete yet
            frames = blk.array  # (nframes, H, W) view of the memory DCAM wrote into
            mean = frames[0].mean()
            intact = src.overwritten(blk) == 0  # check after reading, before trusting the data
            print(
                f"{blk}: mean {mean:.1f} ADU, framestamps {blk.frames['frame_id'][0]}..{blk.frames['frame_id'][-1]}"
                f"{'' if intact else ' (OVERWRITTEN while held)'}"
            )
            src.release(blk)  # from here on DCAM may refill this block without loss
        s = src.stats()
        print(
            f"delivered {s.delivered}, lost {s.lost} (overwritten {s.extra['lost_overwritten']}, missed before DCAM "
            f"{s.extra['missed']}); {s.fps_measured:.3f} fps from camera timestamps; stamps {s.extra['stamps']}"
        )
        print(f"pixel dtype {np.dtype(src.info.dtype).name}, camera model {src.info.camera_model}")

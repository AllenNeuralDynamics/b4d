"""Camera only, no compression: take blocks of frames from the Tucsen source and look at them.

    python examples/grab_frames.py --frames 64 --serial <serial>

This is the vendor side of the interface on its own: the source produces FrameBlocks in host memory the camera
writes into; whoever consumes them must call release() when done with the memory (that re-queues the slots).
"""

import argparse

import numpy as np
from tucam_camera import TucamSource

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--sdk-dir", default=None, help="TUCam SDK V3 install root (else TUCAM_ROOT, else the default)")
    p.add_argument("--no-chunk", action="store_true", help="chunk mode off: contiguous blocks, no loss counting")
    a = p.parse_args()

    with TucamSource(
        a.serial, frames_per_block=a.block, nblocks=3, sdk_dir=a.sdk_dir, chunk_mode=not a.no_chunk
    ) as src:
        print(src)
        print(f"frame stride {src.info.extra['frame_stride']} B, contiguous blocks: {src.info.extra['contiguous']}")
        src.start(max_frames=a.frames)
        while True:
            try:
                blk = src.wait_block(timeout_ms=2000)
            except StopIteration:
                break  # stopped after max_frames and every block was handed out
            if blk is None:
                continue  # timeout: nothing complete yet
            frames = blk.array  # (nframes, H, W) view over the announced block memory
            print(
                f"{blk}: mean {frames[0].mean():.1f} ADU, frame ids "
                f"{blk.frames['frame_id'][0]}..{blk.frames['frame_id'][-1]}, incomplete {blk.frames['incomplete'].sum()}"
            )
            src.release(blk)  # from here on the SDK may refill this block
        s = src.stats()
        print(
            f"delivered {s.delivered}, lost {s.lost} (overflow {s.extra['lost_overflow']}, before host "
            f"{s.extra['lost_before_host']}), incomplete {s.incomplete}; {s.fps_measured:.3f} fps from camera timestamps"
        )
        print(f"dtype {np.dtype(src.info.dtype).name}, pixel format {src.info.extra['pixel_format']}")

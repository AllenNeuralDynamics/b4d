"""The vendor side meets the writer side: TucamSource (this package) -> BlockRecorder (b4d).

    python examples/record_with_b4d.py --seconds 30 --q 1 --out D:\\data\\run1          # CPU writer
    python examples/record_with_b4d.py --seconds 30 --gpu staged                          # CUDA encoder

Codec defaults are b4d's: q 1, mode 1, predictor above3, entropy rice.

The recorder reads the frame geometry and rate from the source's .info, takes the block depth from the source and
compresses each block out of the memory the camera wrote. Two caveats specific to this camera (see README.md):

* With chunk mode on (default, needed for loss counting) the frames in a block are ``frame_stride`` bytes apart, so
  the block is not C-contiguous. b4d's CPU writer calls np.ascontiguousarray on the block, which copies it; the
  CUDA staged path uploads with the frame pitch and does not copy. ``--no-chunk`` gives contiguous blocks (no copy
  on either path) but turns loss counting off.
* The camera model (conversion, offset, read noise) is not known from the SDK; pass --gain/--offset/--rn.

The command-line equivalent is
``b4d-record --source tucam_camera.source:TucamSource --source-arg serial=<serial> ...``.
"""

import argparse
import os
from datetime import datetime

from tucam_camera import TucamSource

from b4d.ingest import BlockRecorder, RecorderConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--sdk-dir", default=None)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--block", type=int, default=64, help="frames per block = shard depth")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--no-chunk", action="store_true", help="contiguous blocks; losses are then not counted")
    p.add_argument("--gain", type=float, default=None, help="e-/ADU (camera model)")
    p.add_argument("--offset", type=float, default=0.0, help="ADU (camera model)")
    p.add_argument("--rn", type=float, default=None, help="read noise e- (camera model)")
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--mode", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--predictor", default="above3", choices=["above3", "above", "left_above", "mean3d"])
    p.add_argument("--entropy", default="rice", choices=["rice", "ans"])
    p.add_argument("--gpu", default="none", choices=["none", "staged"])
    p.add_argument("--sink", default="disk", choices=["disk", "memory"])
    p.add_argument("--out", default="b4d_out")
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()

    model = (1.0 / a.gain, a.offset, a.rn) if a.gain and a.rn is not None else None
    src = TucamSource(
        a.serial,
        frames_per_block=a.block,
        nblocks=a.blocks,
        sdk_dir=a.sdk_dir,
        chunk_mode=not a.no_chunk,
        camera_model=model,
    )
    if not src.info.extra["contiguous"] and a.gpu == "none":
        print(
            f"note: frames are {src.info.extra['frame_stride']} B apart (chunk data); the CPU writer copies each block "
            "to make it contiguous. Use --gpu staged, or --no-chunk (no loss counting), to avoid the copy."
        )
    path = os.path.join(a.out, f"example_{datetime.now():%Y%m%d_%H%M%S}.zarr")
    cfg = RecorderConfig(
        path=path,
        sink=a.sink,
        q=a.q,
        mode=a.mode,
        predictor=a.predictor,
        entropy=a.entropy,
        gpu=a.gpu,
        verify=a.sink == "disk",
    )
    with BlockRecorder(src, cfg) as rec:
        print(f"{src} -> {path}; shards {rec.shards}, camera model conv {rec.conv:.4f} offset {rec.offset} rn {rec.rn}")
        rec.start(seconds=a.seconds)
        res = rec.run(status_every=5.0)
        print(rec.noise_line())
        print(rec.report(res))
        if a.sink == "disk":
            print(rec.verify()["text"])
    s = src.stats()
    print(f"source: delivered {s.delivered}, lost {s.lost} {s.extra['lost_overflow']}/{s.extra['lost_before_host']}")
    if not a.keep:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

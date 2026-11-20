"""The vendor side meets the writer side: PvcamSource (this package) -> BlockRecorder (b4d), zero copy.

    python examples/record_with_b4d.py --seconds 30 --q 1 --out D:\\data\\run1            # CPU writer
    python examples/record_with_b4d.py --seconds 30 --port 1 --block 128 --blocks 3       # Kinetix 8-bit Speed port
    python examples/record_with_b4d.py --seconds 30 --gpu staged --entropy ans

Codec defaults are b4d's: q 1, mode 1, predictor above3, entropy rice.

The recorder reads the camera model, frame geometry and rate from the source's .info, takes the block depth from the
source and compresses each block straight out of the PVCAM circular buffer. The command-line equivalent is
``b4d-record --source pvcam_camera.source:PvcamSource --source-arg serial=... --source-arg readout_port=1 ...``.

Ring size: pl_exp_start_cont takes an uns32 byte count, so block * blocks frames must fit in 4 GiB
(full-frame Kinetix: 419 frames at 8 bit, 209 at 12/16 bit).
"""

import argparse
import os
from datetime import datetime

from pvcam_camera import PvcamSource

from b4d.ingest import BlockRecorder, RecorderConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default=None)
    p.add_argument("--port", type=int, default=None, help="readout port value (Kinetix: 0 Sensitivity, 1 Speed, ...)")
    p.add_argument("--exposure-us", type=float, default=None)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--block", type=int, default=64, help="frames per block = shard depth")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--mode", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--predictor", default="above3", choices=["above3", "above", "left_above", "mean3d"])
    p.add_argument("--entropy", default="rice", choices=["rice", "ans"])
    p.add_argument("--gpu", default="none", choices=["none", "staged"])
    p.add_argument("--sink", default="disk", choices=["disk", "memory"])
    p.add_argument("--out", default="b4d_out")
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()

    src = PvcamSource(
        a.serial, frames_per_block=a.block, nblocks=a.blocks, readout_port=a.port, exposure_us=a.exposure_us
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
    print(f"source: delivered {s.delivered}, lost {s.lost}, incomplete {s.incomplete}, error {s.extra['error'] or '-'}")
    src.close()
    if not a.keep:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

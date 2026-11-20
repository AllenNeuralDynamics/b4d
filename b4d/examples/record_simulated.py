"""Record from the built-in simulated camera: the writer side of the interface without any hardware.

    python examples/record_simulated.py                 # 2048x2048 @ 100 fps, 64-frame blocks, CPU writer, memory sink
    python examples/record_simulated.py --sink disk --out C:\\b4d_out --q 2 --gpu staged

Equivalent command line: b4d-record --source simulated --source-arg shape=2048,2048 --source-arg fps=100 --sink memory
"""

import argparse
import os
import shutil

from b4d.ingest import BlockRecorder, RecorderConfig, SimulatedCameraSource

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="2048,2048")
    p.add_argument("--fps", type=float, default=100.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--peak-e", type=float, default=256.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--gpu", default="none", choices=["none", "staged"])
    p.add_argument("--sink", default="memory", choices=["memory", "disk"])
    p.add_argument("--out", default="b4d_recordings")
    a = p.parse_args()

    ny, nx = (int(v) for v in a.shape.split(","))
    src = SimulatedCameraSource(shape=(ny, nx), fps=a.fps, frames_per_block=a.block, nblocks=a.blocks, peak_e=a.peak_e)
    path = os.path.join(a.out, "simulated.zarr")
    cfg = RecorderConfig(
        path=path, sink=a.sink, q=a.q, threads=a.threads, gpu=a.gpu, direct_io=False, verify=a.sink == "disk"
    )
    with BlockRecorder(src, cfg) as rec:
        print(
            f"{src.info.name}: {nx}x{ny} @ {a.fps:g} fps -> shards {rec.shards}, camera model conv {rec.conv:.4f} "
            f"offset {rec.offset} rn {rec.rn}"
        )
        rec.start(seconds=a.seconds)
        res = rec.run(status_every=2.0)
        print(rec.report(res))
        if a.sink == "disk":
            print(rec.verify()["text"])
    shutil.rmtree(path, ignore_errors=True)

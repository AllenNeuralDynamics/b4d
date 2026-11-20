"""The smallest possible camera "vendor" implementation of b4d's BlockSource contract, and a recording with it.

Copy this file to start a driver package for a new camera: keep the constructor convention
(frames_per_block, nblocks, **vendor kwargs), the .info / .frames_per_block / .nblocks attributes and the six
methods; put the vendor SDK underneath. The source never imports b4d (the dataclasses below are plain mirrors of
b4d.ingest.SourceInfo / SourceStats, which is all b4d needs to see). Run:

    python examples/minimal_block_source.py
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

FRAME_DTYPE = np.dtype(
    [
        ("frame_id", "<u8"),
        ("timestamp_ns", "<u8"),
        ("host_ns", "<u8"),
        ("size_filled", "<u8"),
        ("incomplete", "<u4"),
        ("slot", "<u4"),
    ]
)


@dataclass
class Info:  # == b4d.ingest.SourceInfo, structurally
    name: str
    serial: str
    shape: tuple
    dtype: object
    fps: float
    frame_bytes: int
    camera_model: tuple = None
    extra: dict = field(default_factory=dict)


@dataclass
class Stats:  # == b4d.ingest.SourceStats
    delivered: int = 0
    lost: int = 0
    incomplete: int = 0
    blocks_completed: int = 0
    blocks_released: int = 0
    free_blocks: int = 0
    ready_blocks: int = 0
    fps_measured: float = 0.0
    extra: dict = field(default_factory=dict)


class Block:  # == b4d.ingest.FrameBlock
    def __init__(self, full_array, frames, seq, index, nframes, partial=False):
        self.full_array, self.frames, self.seq, self.index, self.nframes, self.partial = (
            full_array,
            frames,
            seq,
            index,
            nframes,
            partial,
        )
        self.array = full_array[:nframes]


class RampCameraSource:
    """A 'camera' that writes a moving ramp into blocks at fps. The pool of nblocks blocks is owned here, as a
    grabber's DMA pool would be; a frame that is due while no block is free is counted as lost."""

    def __init__(self, frames_per_block=32, nblocks=3, shape=(512, 512), fps=200.0):
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        ny, nx = shape
        self.info = Info(
            "ramp camera", "RAMP-1", (ny, nx), np.uint16, float(fps), ny * nx * 2, camera_model=(1.0, 100.0, 1.0)
        )
        self._pool = [np.zeros((self.frames_per_block, ny, nx), np.uint16) for _ in range(self.nblocks)]
        self._free = queue.Queue()
        [self._free.put(i) for i in range(self.nblocks)]
        self._ready = queue.Queue()
        self._st = Stats(free_blocks=self.nblocks)
        self._stop = threading.Event()
        self._thread = None

    def _produce(self, max_frames):
        t0 = time.perf_counter()
        n = seq = 0
        cur = None
        meta = np.zeros(self.frames_per_block, FRAME_DTYPE)
        while not self._stop.is_set() and (max_frames is None or n < max_frames):
            due = t0 + n / self.info.fps
            if due > time.perf_counter():
                time.sleep(min(due - time.perf_counter(), 0.001))
                continue
            if cur is None:
                try:
                    cur = (self._free.get_nowait(), 0)
                    self._st.free_blocks -= 1
                except queue.Empty:
                    self._st.lost += 1
                    n += 1
                    continue
            bi, k = cur
            self._pool[bi][k, :, :] = (
                np.arange(self.info.shape[1], dtype=np.uint16) + n
            ) % 4096 + 100  # the "exposure"
            meta[k] = (n, int((due - t0) * 1e9), time.perf_counter_ns(), self.info.frame_bytes, 0, k)
            self._st.delivered += 1
            n += 1
            k += 1
            if k == self.frames_per_block:
                self._ready.put(Block(self._pool[bi], meta.copy(), seq, bi, k))
                self._st.blocks_completed += 1
                self._st.ready_blocks += 1
                seq += 1
                cur = None
            else:
                cur = (bi, k)
        if cur and cur[1]:
            self._ready.put(Block(self._pool[cur[0]], meta[: cur[1]].copy(), seq, cur[0], cur[1], partial=True))
        self._ready.put(None)

    # ---- the contract
    def start(self, max_frames=None):
        self._thread = threading.Thread(target=self._produce, args=(max_frames,), daemon=True)
        self._thread.start()

    def wait_block(self, timeout_ms=1000):
        try:
            blk = self._ready.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None
        if blk is None:
            self._ready.put(None)
            raise StopIteration
        self._st.ready_blocks -= 1
        return blk

    def release(self, block):
        self._st.blocks_released += 1
        self._st.free_blocks += 1
        self._free.put(block.index)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)

    def stats(self):
        return self._st

    def close(self):
        self.stop()

    def block_array(self, i):  # optional: lets b4d page-lock the pool for the CUDA encoder
        return self._pool[i]


if __name__ == "__main__":
    import shutil

    from b4d.ingest import BlockRecorder, RecorderConfig, validate_source

    src = RampCameraSource(frames_per_block=32, nblocks=3, shape=(512, 512), fps=200.0)
    validate_source(src)  # precise TypeError if something in the contract is missing
    cfg = RecorderConfig(
        path="ramp_example.zarr", q=1.0, threads=8, sink="disk", direct_io=False, shards=(32, 512, 512), verify=True
    )
    with BlockRecorder(src, cfg) as rec:
        rec.start(seconds=5)
        res = rec.run(status_every=2.0)
        print(rec.report(res))
        print(rec.verify()["text"])
    shutil.rmtree("ramp_example.zarr", ignore_errors=True)

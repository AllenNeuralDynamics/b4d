"""A simulated camera that implements the BlockSource contract: frames paced at a fixed rate into a pool of block
buffers, with the same loss semantics as a frame grabber (a frame that is due while every block is held is lost).

    from b4d.ingest import SimulatedCameraSource, BlockRecorder, RecorderConfig
    src = SimulatedCameraSource(shape=(2048, 2048), fps=100, frames_per_block=64, nblocks=3, peak_e=256)
    with BlockRecorder(src, RecorderConfig(path="sim.zarr", q=1)) as rec:
        rec.start(seconds=10); print(rec.report(rec.run()))

Two memory modes:
  mode="alias" (default) the pool's blocks are views of one frame bank of frames_per_block frames, so no bytes move
                         when a frame "arrives": this is the zero-copy case a DMA camera gives the writer, and it
                         measures the writer's capacity without a memcpy in the loop (every block holds the same frames)
  mode="copy"            each due frame is copied from the bank into the current block (one memcpy per frame, as a
                         software camera would do); blocks then differ frame by frame (bank index = frame_id % bank)
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from .protocol import FRAME_DTYPE, FrameBlock, SourceInfo, SourceStats


def synthetic_frames(
    shape,
    nframes: int,
    conversion: float,
    offset: float,
    read_noise: float,
    bg_e: float = 5.0,
    peak_e: float = 256.0,
    fill: float = 0.2,
    psf_fwhm: float = 2.0,
    seed: int = 11,
) -> np.ndarray:
    """Gaussian-blob phantom observed through a camera model (Poisson + read noise), uint16 ADU, shape (nframes, H, W).
    Small enough to be generated on the fly; for large cached banks use the phantom in b4d/tools/stream_sim.py."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    ny, nx = shape
    vol = np.zeros((nframes, ny, nx), np.float32)
    target = fill * vol.size
    n = 0
    while (vol > 0.1 * peak_e).sum() < target and n < 20000:
        sz = rng.uniform(1.0, 6.0)
        amp = rng.uniform(0.2, 1.0) * peak_e
        cz, cy, cx = rng.uniform(0, nframes), rng.uniform(0, ny), rng.uniform(0, nx)
        r = int(3 * sz) + 1
        z0, z1 = max(0, int(cz - r)), min(nframes, int(cz + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(ny, int(cy + r) + 1)
        x0, x1 = max(0, int(cx - r)), min(nx, int(cx + r) + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        vol[z0:z1, y0:y1, x0:x1] += amp * np.exp(-((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sz * sz))
        n += 1
    vol = gaussian_filter(vol, sigma=psf_fwhm / 2.3548, mode="reflect") + bg_e
    noisy = rng.poisson(vol).astype(np.float32) + rng.normal(0.0, read_noise, vol.shape).astype(np.float32)
    return np.clip(np.rint(offset + noisy * conversion), 0, 65535).astype(np.uint16)


class SimulatedCameraSource:
    """BlockSource backed by a frame bank; see the module docstring."""

    def __init__(
        self,
        shape=(1024, 1024),
        fps: float = 50.0,
        frames_per_block: int = 64,
        nblocks: int = 3,
        bank: Optional[np.ndarray] = None,
        mode: str = "alias",
        conversion: float = 1.0 / 7.25,
        offset: float = 2.05,
        read_noise: float = 6.89,
        peak_e: float = 256.0,
        bg_e: float = 5.0,
        fill: float = 0.2,
        seed: int = 11,
        name: str = "simulated camera",
        serial: str = "SIM0",
        buffers=None,
        dtype=np.uint16,
        lost_when_full: bool = True,
    ):
        if mode not in ("alias", "copy"):
            raise ValueError("mode must be 'alias' or 'copy'")
        self.frames_per_block, self.nblocks, self.mode = int(frames_per_block), int(nblocks), mode
        if bank is None:
            bank = synthetic_frames(
                shape, self.frames_per_block, conversion, offset, read_noise, bg_e, peak_e, fill, seed=seed
            )
        self.bank = np.ascontiguousarray(bank)
        ny, nx = self.bank.shape[1:]
        self.info = SourceInfo(
            name=name,
            serial=serial,
            shape=(ny, nx),
            dtype=np.dtype(dtype),
            fps=float(fps),
            frame_bytes=int(ny * nx * np.dtype(dtype).itemsize),
            camera_model=(float(conversion), float(offset), float(read_noise)),
            extra=dict(mode=mode, bank_frames=int(self.bank.shape[0]), peak_e=peak_e),
        )
        if buffers is not None:
            if len(buffers) != self.nblocks:
                raise ValueError("buffers must have nblocks entries")
            self._blocks = [np.asarray(b).reshape((self.frames_per_block, ny, nx)) for b in buffers]
            self.mode = "copy"
        elif mode == "alias":
            if self.bank.shape[0] < self.frames_per_block:
                raise ValueError("alias mode needs a bank of at least frames_per_block frames")
            self._blocks = [self.bank[: self.frames_per_block]] * self.nblocks
        else:
            self._blocks = [np.zeros((self.frames_per_block, ny, nx), dtype) for _ in range(self.nblocks)]
        self._free: "queue.Queue[int]" = queue.Queue()
        for i in range(self.nblocks):
            self._free.put(i)
        self._ready: "queue.Queue[Optional[FrameBlock]]" = queue.Queue()
        self._lock = threading.Lock()
        self._st = SourceStats(free_blocks=self.nblocks)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._max_frames = None
        self._lost_when_full = bool(lost_when_full)
        self._t0 = None

    # ------------------------------------------------------------ producer
    def _produce(self):
        fpb, fps = self.frames_per_block, self.info.fps
        t0 = time.perf_counter()
        self._t0 = t0
        produced = 0
        cur = None  # (block index, filled)
        frames_meta = np.zeros(fpb, FRAME_DTYPE)
        seq = 0
        while not self._stop.is_set() and (self._max_frames is None or produced < self._max_frames):
            due_t = t0 + produced / fps
            now = time.perf_counter()
            if due_t > now:
                time.sleep(min(due_t - now, 0.002))
                continue
            if cur is None:
                try:
                    bi = self._free.get_nowait()
                    cur = (bi, 0)
                    with self._lock:
                        self._st.free_blocks -= 1
                except queue.Empty:
                    bi = None
                if bi is None:
                    with self._lock:  # a frame is due and no block can take it: lost, as a grabber would lose it
                        self._st.lost += 1
                    produced += 1
                    continue
            bi, filled = cur
            if self.mode == "copy":
                np.copyto(self._blocks[bi][filled], self.bank[produced % self.bank.shape[0]])
            ts = int((due_t - t0) * 1e9)
            frames_meta[filled] = (produced, ts, int(time.perf_counter_ns()), self.info.frame_bytes, 0, filled)
            filled += 1
            produced += 1
            with self._lock:
                self._st.delivered += 1
            if filled == fpb:
                blk = FrameBlock(self._blocks[bi], frames_meta[:fpb].copy(), seq, bi, fpb)
                seq += 1
                with self._lock:
                    self._st.blocks_completed += 1
                    self._st.ready_blocks += 1
                self._ready.put(blk)
                cur = None
            else:
                cur = (bi, filled)
        if cur is not None and cur[1] > 0:  # final partial block
            bi, filled = cur
            blk = FrameBlock(self._blocks[bi], frames_meta[:filled].copy(), seq, bi, filled, partial=True)
            with self._lock:
                self._st.blocks_completed += 1
                self._st.ready_blocks += 1
            self._ready.put(blk)
        with self._lock:
            n = self._st.delivered
            self._st.fps_measured = (n - 1) / max(time.perf_counter() - t0, 1e-9) if n > 1 else 0.0
        self._ready.put(None)  # drained

    # ------------------------------------------------------------ contract
    def start(self, max_frames: Optional[int] = None) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._max_frames = int(max_frames) if max_frames else None
        self._stop.clear()
        self._thread = threading.Thread(target=self._produce, name="sim-camera", daemon=True)
        self._thread.start()

    def wait_block(self, timeout_ms: int = 1000):
        try:
            blk = self._ready.get(timeout=max(timeout_ms, 0) / 1000.0)
        except queue.Empty:
            return None
        if blk is None:
            self._ready.put(None)  # keep the sentinel for later callers
            raise StopIteration
        with self._lock:
            self._st.ready_blocks -= 1
        return blk

    def release(self, block) -> None:
        with self._lock:
            self._st.blocks_released += 1
            self._st.free_blocks += 1
        self._free.put(int(block.index))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def stats(self) -> SourceStats:
        with self._lock:
            s = SourceStats(**{k: v for k, v in self._st.__dict__.items() if k != "extra"}, extra=dict(self._st.extra))
        if self._t0 and s.delivered > 1 and not s.fps_measured:
            s.fps_measured = (s.delivered - 1) / max(time.perf_counter() - self._t0, 1e-9)
        return s

    def block_array(self, index: int):
        return self._blocks[int(index)]

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

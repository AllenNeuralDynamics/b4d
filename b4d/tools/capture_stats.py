"""Mean + variance images for capture scripts, in the form calibrate_camera.py reads. Needs only numpy, so it can sit on the capture PC.

    stats = FrameStats(frame_shape)
    for frame in frames:              # uint16
        stats.add(frame)
    stats.save("dark_off.npz")

If frames arrive too fast, skip some: only the frames passed to add() are counted.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class FrameStats:
    """64-bit integer running sums, the whole-frame mean of each frame, and the per-pixel maximum"""

    def __init__(self, shape, threads=4):
        self.shape = tuple(shape); self.n = 0; self.level = []
        self.s1 = np.zeros(shape, np.uint64); self.s2 = np.zeros(shape, np.uint64); self.peak = np.zeros(shape, np.uint16)
        h = shape[0]; threads = max(1, min(threads, h))
        self.rows = [slice(i * h // threads, (i + 1) * h // threads) for i in range(threads)]
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None

    def _part(self, frame, rows):
        f = frame[rows]; v = f.astype(np.uint32)
        self.s1[rows] += v; self.s2[rows] += v * v                      # 65535 squared still fits 32 bits
        np.maximum(self.peak[rows], f, out=self.peak[rows])
        return float(f.sum(dtype=np.uint64))

    def add(self, frame):
        if frame.dtype != np.uint16 or frame.shape != self.shape:
            raise ValueError(f"expected uint16 frames of {self.shape}, got {frame.dtype} of {frame.shape}")
        parts = self.pool.map(lambda r: self._part(frame, r), self.rows) if self.pool else [self._part(frame, self.rows[0])]
        self.level.append(sum(parts) / frame.size); self.n += 1

    def images(self):
        if self.n < 3:
            raise ValueError(f"needs at least 3 frames, got {self.n}")
        a = self.s1.astype(np.float64); mean = a / self.n
        var = (self.s2.astype(np.float64) - a * a / self.n) / (self.n - 1)
        return dict(mean=mean.astype(np.float32), var=var.astype(np.float32), max=self.peak, level=np.array(self.level), n=self.n)

    def save(self, path):
        np.savez(path, **self.images())
        if self.pool:
            self.pool.shutdown()
            self.pool = None

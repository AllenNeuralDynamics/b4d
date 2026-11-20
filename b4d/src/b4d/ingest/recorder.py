"""BlockRecorder: any BlockSource -> B4D sharded zarr (CPU writer or CUDA encoder, optional multiscale pyramid).

    from b4d.ingest import BlockRecorder, RecorderConfig, SimulatedCameraSource
    src = SimulatedCameraSource(shape=(2048, 2048), fps=100, frames_per_block=64, nblocks=3)
    cfg = RecorderConfig(path=r"D:\\data\\run1.zarr", q=1, gpu="none")           # or gpu="staged" for the CUDA encoder
    with BlockRecorder(src, cfg) as rec:
        rec.start(seconds=60)                 # whole blocks; the source stops by itself after the last one
        for blk in rec:                       # yields each block once its write is queued (the view is still valid)
            ...                               # optional per-block hook (do not keep references to blk.array)
        res = rec.result()                    # dict: sustained, lost, ratio, MB/s, latency, per-block timings ...
    # or simply: res = rec.run(status_every=5); print(rec.report(res))

The recorder never copies frames: the source's block memory is what the native writer compresses (or what the CUDA
encoder uploads from). Block depth (= shard depth) comes from the source's frames_per_block; the pool size from its
nblocks; the camera noise model from the config or, when left None, from source.info.camera_model.

Pipelines
  gpu="none"   block -> native C writer (b4d_zarr.c): compress with `threads` threads + shard assembly + IO
  gpu="staged" host block (page-locked for CUDA once) -> one PCIe copy -> CUDA tile kernel -> blobs -> native shard IO
  gpu="rdma"   (experimental) the source delivers device-resident blocks (CuPy arrays); the kernel reads them in place
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from .. import hardware
from .protocol import FRAME_DTYPE, SourceInfo, validate_source


@dataclass
class RecorderConfig:
    # dataset
    path: str = "recording.zarr"
    sink: str = "disk"  # "disk" | "memory" (compress + assemble shards, write nothing)
    # direct_io: write each shard unbuffered + write-through (Windows NO_BUFFERING|WRITE_THROUGH, Linux O_DIRECT) from a
    # sector-aligned buffer: one DMA per shard, nothing left in the OS page cache. Buffered writes (False) go through
    # page cache + fsync: about half the drive's rate (3.4 vs 8.0 GB/s on this server's NVMe) and they fill the standby
    # list, which later slows pinning the camera pool. Volumes that refuse direct IO fall back to buffered alone.
    # (Not to be confused with zero-copy, which is the camera DMA writing straight into the block the compressor reads.)
    direct_io: bool = True
    sync: bool = True
    threads: Optional[int] = (
        None  # CPU compression threads per block write; None = b4d.hardware.auto_threads() (physical cores)
    )
    max_commits: int = 2  # block writes in flight (the source needs >= max_commits + 1 blocks)
    chunks: tuple = (64, 64, 64)
    shards: Optional[tuple] = (
        None  # (frames_per_block, y, x); None = (frames_per_block, <=1280, <=3840) rounded to chunks
    )
    # codec / camera model
    # level-0 chunk codec: b4d | zstd | lz4 | raw. zstd/lz4 = zarr blosc codec, lossless, compressed on the same C pool
    # as B4D (Blosc1 layout, readable by zarr-python/neuroglancer); raw = zarr bytes.
    codec: str = "b4d"
    clevel: int = 5  # zstd/lz4 compression level
    shuffle: str = "shuffle"  # zstd/lz4 filter: shuffle | bitshuffle | noshuffle
    q: float = 1.0  # defaults: m1_a3 (q=1, mode=1, predictor above3, entropy rice), as b4d.compress
    mode: int = 1
    predictor: str = "above3"
    entropy: str = "rice"
    # seeded subtractive dither (format v8, modes 1 and 2): removes the low-ADU mean bias. CPU writer, or the CUDA encoder
    # with the Rice coder (mode 1 above3/above, mode 2 above)
    dither: bool = False
    conv: Optional[float] = None  # DN per e-; None = from source.info.camera_model
    offset: Optional[float] = None  # ADU
    rn: Optional[float] = None  # e-
    exact_vst: object = "variance"  # fitted noise table ("variance", "std") or False = generalised Anscombe
    # per-pixel list (b4d.PixelCalibration): the .npz from tools/calibrate_camera.py, taken in the same camera mode. b4d codec,
    # mode 1, above3, CPU encoder; level 0 only. calibration_origin: (row, col) of the frame's first pixel on the map.
    calibration: Optional[str] = None
    calibration_origin: tuple = (0, 0)
    # multiscale pyramid (OME-NGFF 0.5 group: <path>/0 .. <path>/N)
    multiscale: int = 0  # downsampled levels below level 0 (0 = plain array)
    ms_factors: tuple = (2, 2, 2)
    ms_codec: str = "b4d"  # codec for levels >= 1: b4d (each level's own noise model) | zstd | lz4 | raw
    # GPU
    gpu: str = "none"  # "none" | "staged" | "rdma" (experimental)
    gpu_batch_tiles: int = 2048
    gpu_slots: int = 1
    gpu_check: bool = True  # self-check the first block against the C encoder
    gpu_host_threads: int = 0  # host threads for the ANS stage of the GPU path (0 = half the logical CPUs)
    # run
    warmup_blocks: int = 1  # blocks run through the pipeline before statistics start (written to z 0, overwritten)
    verify: bool = False  # keep sample windows in RAM and check the error bound after close()
    notes: str = ""

    @classmethod
    def from_env(cls, **overrides) -> "RecorderConfig":
        """The writer's execution settings for the host from the environment (then ``overrides``): the knobs a host
        application (voxel) should not have to know about. ``B4D_ENCODER`` cpu|cuda (-> gpu none|staged),
        ``B4D_THREADS`` (see b4d.hardware), ``B4D_MAX_COMMITS`` (blocks in flight), ``B4D_DIRECT_IO`` 0|1,
        ``B4D_GPU_SLOTS``, ``B4D_GPU_BATCH_TILES``, ``B4D_GPU_HOST_THREADS``, ``B4D_GPU_CHECK`` 0|1."""
        env = os.environ
        kw = {}
        enc = env.get("B4D_ENCODER", "").strip().lower()
        if enc:
            if enc not in ("cpu", "cuda"):
                raise ValueError(f"B4D_ENCODER must be cpu or cuda, got {enc!r}")
            kw["gpu"] = "none" if enc == "cpu" else "staged"
        for name, field, conv in (
            ("B4D_THREADS", "threads", int),
            ("B4D_MAX_COMMITS", "max_commits", int),
            ("B4D_DIRECT_IO", "direct_io", lambda v: v.strip().lower() in ("1", "true", "yes")),
            ("B4D_GPU_SLOTS", "gpu_slots", int),
            ("B4D_GPU_BATCH_TILES", "gpu_batch_tiles", int),
            ("B4D_GPU_HOST_THREADS", "gpu_host_threads", int),
            ("B4D_GPU_CHECK", "gpu_check", lambda v: v.strip().lower() in ("1", "true", "yes")),
        ):
            if env.get(name, "").strip():
                kw[field] = conv(env[name])
        kw.update(overrides)
        return cls(**kw)


def load_source(spec: str, **kwargs):
    """Instantiate a BlockSource from "module.path:ClassName" (or "simulated") with keyword arguments."""
    if spec in ("simulated", "sim"):
        from .simulated import SimulatedCameraSource

        return SimulatedCameraSource(**kwargs)
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError("source spec must be 'module:Class' or 'simulated'")
    return getattr(importlib.import_module(mod), cls)(**kwargs)


class BlockRecorder:
    def __init__(self, source, cfg: RecorderConfig):
        import b4d

        self.info: SourceInfo = validate_source(source)
        self.source = source
        self.cfg = cfg
        c = cfg
        self.threads = hardware.auto_threads(c.threads)
        self.block = int(source.frames_per_block)
        self.nblocks = int(source.nblocks)
        ny, nx = self.info.shape
        if c.shards is None:
            cz, cy, cx = c.chunks
            sy = max(cy, min(ny, 1280) // cy * cy)
            sx = max(cx, min(nx, 3840) // cx * cx)
            self.shards = (self.block, sy, sx)
        else:
            self.shards = tuple(int(v) for v in c.shards)
        if self.shards[0] != self.block or any(s % k for s, k in zip(self.shards, c.chunks)):
            raise ValueError(
                f"shards {self.shards} must have depth {self.block} (= source.frames_per_block) and be multiples of "
                f"chunks {c.chunks}"
            )
        if self.nblocks < c.max_commits + 1:
            raise ValueError(
                f"the source has {self.nblocks} blocks; max_commits={c.max_commits} needs at least {c.max_commits + 1}"
            )
        if c.gpu not in ("none", "staged", "rdma"):
            raise ValueError("gpu must be none, staged or rdma")
        if c.gpu != "none" and c.codec != "b4d":
            raise ValueError("the GPU encoder is for the b4d codec; use gpu='none' with zstd/lz4/raw")
        if cfg.calibration and (cfg.gpu != "none" or cfg.codec != "b4d"):
            raise ValueError("a per-pixel calibration needs the b4d codec and gpu='none'")
        cm = self.info.camera_model or (1.0, 0.0, 0.0)
        self.conv = float(c.conv if c.conv is not None else cm[0])
        self.offset = float(c.offset if c.offset is not None else cm[1])
        self.rn = float(c.rn if c.rn is not None else cm[2])
        self.camera_model = b4d.Camera(self.conv, self.offset, self.rn, c.exact_vst)
        self._lock = threading.Lock()
        self._stats = dict(written_frames=0, bytes=0, lat=[], inflight=0, blocks=0, write_fail=0)
        self._frame_table = []
        self._verify_refs = []
        self._warm_futs = []
        self._done = threading.Event()
        self.arr = None
        self.encoder = None
        self.ms = int(c.multiscale) > 0
        self.n_blocks = 0
        self.total_frames = 0
        self.nwarm = max(0, c.warmup_blocks)
        self._t_start = None
        self._t_end = None
        self._stats_reset = False
        self._closed = False
        self.t_open = 0.0
        self.t_close = 0.0
        self.t_register = 0.0
        self.noise_measured = self.noise_model = float("nan")
        self.noise_level = float("nan")

    # ------------------------------------------------------------ setup
    def _writer(self):
        import b4d
        import b4d.zarr_store as zs

        c = self.cfg
        codec = self._level0_codec()
        os.makedirs(os.path.dirname(os.path.abspath(c.path)) or ".", exist_ok=True)
        shutil.rmtree(c.path, ignore_errors=True)
        ny, nx = self.info.shape
        sink = c.sink
        dtype = str(np.dtype(self.info.dtype))
        self._calibration = b4d.PixelCalibration.load(c.calibration, self.camera_model) if c.calibration else None
        cal = dict(calibration=self._calibration, origin=tuple(c.calibration_origin)) if c.calibration else {}
        if self.ms:
            pc = None
            if c.ms_codec != "b4d":
                import numcodecs

                if c.ms_codec == "raw":
                    pc = "raw"
                elif c.ms_codec == "zstd":
                    pc = numcodecs.Zstd(level=5)
                else:
                    pc = numcodecs.Blosc(cname=c.ms_codec, clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
            self.arr = zs.create_multiscale(
                c.path,
                (self.total_frames, ny, nx),
                dtype=dtype,
                shards=self.shards,
                chunks=tuple(c.chunks),
                codec=codec,
                levels=int(c.multiscale),
                factors=tuple(c.ms_factors),
                threads=self.threads,
                sync=c.sync,
                writers=c.max_commits,
                sink=sink,
                pyramid_codec=pc,
                direct_io=c.direct_io,
                **cal,
            )
        else:
            self.arr = zs.create(
                c.path,
                (self.total_frames, ny, nx),
                dtype=dtype,
                shards=self.shards,
                chunks=tuple(c.chunks),
                codec=codec,
                threads=self.threads,
                sync=c.sync,
                writers=c.max_commits,
                sink=sink,
                direct_io=c.direct_io,
                **cal,
            )
        if c.gpu != "none":
            from b4d.cuda_encoder import CudaTileEncoder

            self.encoder = CudaTileEncoder(
                self.camera_model,
                q=c.q,
                mode=c.mode,
                predictor=c.predictor,
                entropy=c.entropy,
                dither=c.dither,
                backend="cuda",
                batch_tiles=c.gpu_batch_tiles,
                slots=c.gpu_slots,
                check=c.gpu_check,
                device_blocks=(c.gpu == "rdma"),
                host_threads=c.gpu_host_threads,
            )
            self.encoder.prepare((self.block, ny, nx), tuple(c.chunks), dtype)
            if c.gpu == "staged" and hasattr(self.source, "block_array"):
                # the source's blocks are DMA targets already; page-lock them for CUDA too, once: one DMA per upload
                t0 = time.perf_counter()
                blocks = [self.source.block_array(i) for i in range(self.nblocks)]
                n = self.encoder.register(blocks)
                if n < len(blocks):  # pragma: no cover - depends on the memory
                    print(
                        f"[recorder] {len(blocks) - n} of {len(blocks)} blocks were not page-locked for CUDA; their uploads "
                        "will be staged copies",
                        flush=True,
                    )
                self.t_register = time.perf_counter() - t0
            # GPU for level 0 only; the pyramid levels (1/7 of the voxels) stay on the C writer
            (self.arr.levels[0] if self.ms else self.arr).encoder = self.encoder.encode_tiles

    def _level0_codec(self):
        from b4d.zarr3 import B4D

        c = self.cfg
        if c.codec == "b4d":
            return B4D(q=c.q, mode=c.mode, camera=self.camera_model, predictor=c.predictor, entropy=c.entropy, dither=c.dither)
        if c.codec == "raw":
            return "raw"
        if c.codec in ("zstd", "lz4"):
            import numcodecs

            sh = {
                "shuffle": numcodecs.Blosc.SHUFFLE,
                "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
                "noshuffle": numcodecs.Blosc.NOSHUFFLE,
            }[c.shuffle]
            return numcodecs.Blosc(cname=c.codec, clevel=int(c.clevel), shuffle=sh)
        raise ValueError("codec must be b4d, zstd, lz4 or raw")

    @property
    def lossless(self) -> bool:
        return self.cfg.codec != "b4d" or self.cfg.q <= 0

    # ------------------------------------------------------------ run control
    def start(self, seconds: Optional[float] = None, frames: Optional[int] = None):
        """Decide the run length (whole blocks), create the dataset, start the source."""
        fps = self.info.fps
        if frames:
            self.n_blocks = max(1, -(-int(frames) // self.block))
        else:
            self.n_blocks = max(1, int(round(fps * float(seconds or 60.0) / self.block)))
        self.total_frames = self.n_blocks * self.block
        max_frames = (self.n_blocks + self.nwarm) * self.block
        if not self.lossless:  # fit the exact VST now (~0.2 s per camera model, cached): on the first write it held the
            import b4d; b4d._vst_of(self.camera_model)  # writers back while the camera ran, and the block pool overflowed
        t0 = time.perf_counter()
        self.source.start(max_frames)
        self.t_open = time.perf_counter() - t0
        self._writer()
        self._t_start = time.perf_counter()

    def _on_done(self, blk, counted: bool, t_submit: float):
        def cb(fut):
            try:
                nb = fut.result()
            except Exception as e:
                print(f"[recorder] WRITE FAILED for block {blk.seq}: {e}", flush=True)
                nb = 0
                with self._lock:
                    self._stats["write_fail"] += 1
            with self._lock:
                if counted:
                    self._stats["lat"].append(time.perf_counter() - t_submit)
                    self._stats["written_frames"] += blk.nframes
                    self._stats["bytes"] += nb
                    self._stats["blocks"] += 1
                self._stats["inflight"] -= 1
                finished = counted and self._stats["blocks"] == self.n_blocks
            self.source.release(blk)
            if finished:
                self._t_end = time.perf_counter()
                self._done.set()

        return cb

    def _reset_statistics(self):
        for f in self._warm_futs:  # the warm-up writes must be finished before the pyramid restarts
            try:
                f.result()
            except Exception:
                pass
        self._warm_futs.clear()
        if self.ms:
            self.arr.reset_pyramid()
            self.arr.stats.update(
                blocks=0,
                raw_bytes=0,
                written_bytes=0,
                seconds=0.0,
                downsample_seconds=0.0,
                level_wait_seconds=0.0,
                fused_blocks=0,
            )
            for lv in self.arr.levels:
                lv.stats.update(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0, t_compress=0.0, t_io=0.0)
        else:
            self.arr.stats.update(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0, t_compress=0.0, t_io=0.0)
        if self.encoder is not None:
            for k in ("upload_s", "kernel_s", "download_s", "host_s"):
                self.encoder.stats[k] = 0.0
            self.encoder.stats["blocks"] = 0
            self.encoder.stats["host_tiles"] = 0
        self._t_start = time.perf_counter()
        self._stats_reset = True

    def pump(self, timeout_ms: int = 500):
        """Wait for the next completed block and queue its write. Returns the block (None on timeout);
        raises StopIteration once every block has been written."""
        if self._done.is_set():
            raise StopIteration
        blk = self.source.wait_block(timeout_ms)  # StopIteration propagates when the source is stopped and drained
        if blk is None:
            return None
        c = self.cfg
        counted = blk.seq >= self.nwarm
        if blk.seq == 0 and blk.nframes >= 2 and c.gpu != "rdma":
            self._measure_noise(blk)
        if counted and not self._stats_reset:
            self._reset_statistics()
        z0 = (blk.seq - self.nwarm) * self.block if counted else 0
        if counted:
            self._frame_table.append(np.asarray(blk.frames))
            if c.verify and c.sink == "disk" and (blk.seq - self.nwarm) in (0, self.n_blocks // 2, self.n_blocks - 1):
                self._take_verify_refs(blk, z0)
        n = blk.nframes
        with self._lock:
            self._stats["inflight"] += 1
        src = blk.array if n == self.block else blk.full_array[:n]
        fut = self.arr[z0 : z0 + n].write(src)
        if not counted:
            self._warm_futs.append(fut)
        fut.add_done_callback(self._on_done(blk, counted, time.perf_counter()))
        return blk

    def __iter__(self):
        while True:
            try:
                blk = self.pump(500)
            except StopIteration:
                return
            if blk is not None:
                yield blk

    def wait(self, timeout: float = 600.0):
        """block until the last counted block is written"""
        return self._done.wait(timeout)

    def run(self, status_every: float = 5.0, status_cb: Optional[Callable[[str], None]] = print) -> dict:
        """start() must have been called. Pumps blocks until done, printing a status line every status_every s."""
        nxt = time.perf_counter() + status_every
        while not self._done.is_set():
            try:
                self.pump(500)
            except StopIteration:
                break
            if status_cb and time.perf_counter() >= nxt:
                status_cb(self.status_line())
                nxt = time.perf_counter() + status_every
        self.wait()
        self.stop()
        return self.result()

    def stop(self):
        self.source.stop()
        if self.arr is not None and not self._closed:
            t0 = time.perf_counter()
            if self.ms:
                self.arr.close(verbose=True)
            else:
                self.arr.close()
            self.t_close = time.perf_counter() - t0
            self._closed = True

    def close(self):
        try:
            if self.arr is not None and not self._closed:
                self.arr.close()
                self._closed = True
        finally:
            self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ measurements
    def _measure_noise(self, blk):
        ny, nx = self.info.shape
        w = min(1024, ny, nx)
        y0 = (ny - w) // 2
        x0 = (nx - w) // 2
        f0 = np.asarray(blk.array[0, y0 : y0 + w, x0 : x0 + w], dtype=np.float32)
        f1 = np.asarray(blk.array[1, y0 : y0 + w, x0 : x0 + w], dtype=np.float32)
        self.noise_measured = float((f1 - f0).std() / np.sqrt(2))
        self.noise_level = float(f0.mean())
        self.noise_model = float(self.camera_model.sigma(np.array([self.noise_level], np.float32))[0])

    def noise_line(self) -> str:
        if not np.isfinite(self.noise_measured):
            return "frame noise: not measured"
        ok = abs(self.noise_measured - self.noise_model) < 0.15 * self.noise_model
        return (
            f"frame noise (central window, two consecutive frames): level {self.noise_level:.2f} ADU, temporal sigma "
            f"{self.noise_measured:.3f} ADU; "
            f"camera model {self.noise_model:.3f} ADU ({'consistent' if ok else 'MISMATCH: check conv/rn'})"
        )

    def _take_verify_refs(self, blk, z0: int):
        rng = np.random.default_rng(7 + z0)
        ny, nx = self.info.shape
        win = min(512, ny, nx)
        for fz in sorted(set([0, blk.nframes - 1])):
            for _ in range(2):
                y0 = int(rng.integers(0, ny - win + 1))
                x0 = int(rng.integers(0, nx - win + 1))
                ref = blk.array[fz, y0 : y0 + win, x0 : x0 + win]
                ref = ref.get() if hasattr(ref, "get") else np.array(ref)
                self._verify_refs.append((z0 + fz, y0, x0, ref))

    def verify(self) -> dict:
        """Read the sample windows back and compare with the source frames: max |error| in model sigma and in ADU.
        Stored values are integers, so the achievable bound is q/2 sigma + 0.5 ADU per pixel. With the exact VST (the
        default) the step is q in the units of the fitted curve T, which near the offset is not exactly one model sigma,
        so the bound is checked there: |T(x) - T(x')| - T'/2 <= q/2 (T'/2: the half-ADU rounding of the integer output
        in curve units, as the + 0.5 ADU above), with the same 5 % margin."""
        if not self._verify_refs:
            return dict(ok=None, text="verify: no reference windows (verify off or memory sink)")
        import zarr

        import b4d.zarr3  # noqa: F401

        c = self.cfg
        t0 = time.perf_counter()
        z = zarr.open_array(os.path.join(c.path, "0") if self.ms else c.path, mode="r")
        worst = worst_adu = worst_adj = worst_t = 0.0
        exact = 0
        import b4d as _b4d_mod
        table = None
        if not self.lossless and _b4d_mod._vst_of(self.camera_model) is not None:
            table = _b4d_mod.exact_vst(self.camera_model)["table"]
            slope = np.abs(np.gradient(table))
        tot = 0.0
        npx = 0
        for zz, y0, x0, ref in self._verify_refs:
            back = z[zz, y0 : y0 + ref.shape[0], x0 : x0 + ref.shape[1]]
            if not self.lossless:
                e = np.abs(ref.astype(np.float32) - back.astype(np.float32))
                sig = self.camera_model.sigma(ref).astype(np.float32)
                if self._calibration is not None:  # a listed pixel's bound follows its own noise, not the model's
                    oy, ox = c.calibration_origin
                    unlisted = self._calibration.tier[oy + y0 : oy + y0 + ref.shape[0], ox + x0 : ox + x0 + ref.shape[1]] == 0
                    e, sig = e[unlisted], sig[unlisted]
                if table is not None:
                    et = np.maximum(np.abs(table[ref] - table[back]) - 0.5 * np.maximum(slope[ref], slope[back]), 0)
                    worst_t = max(worst_t, float((et[unlisted] if self._calibration is not None else et).max()))
                worst = max(worst, float((e / sig).max()))
                worst_adu = max(worst_adu, float(e.max()))
                worst_adj = max(worst_adj, float((np.maximum(e - 0.5, 0) / sig).max()))
                tot += float((e / sig).sum())
                npx += e.size
            else:
                exact += int(np.array_equal(ref, back))
        if self.lossless:
            ok = exact == len(self._verify_refs)
        elif table is not None:
            ok = worst_t <= 0.5 * c.q * 1.05 + 1e-6
        else:
            ok = worst_adj <= 0.5 * c.q * 1.05 + 1e-6
        nfr = len(set(r[0] for r in self._verify_refs))
        if not self.lossless:
            detail = (
                f"|error| mean {tot / max(npx, 1):.3f} sigma, max {worst:.2f} sigma = {worst_adu:.0f} ADU; bound q/2 = "
                f"{c.q / 2:g} sigma "
                f"(+0.5 ADU rounding: {worst_adj:.2f} sigma)"
                + (f"; exact VST: max {worst_t:.2f} curve units after the half-ADU rounding, bound q/2 = {c.q / 2:g}" if table is not None else "")
                + f" -> {'OK' if ok else 'FAIL'}"
            )
        else:
            detail = f"lossless {'OK' if ok else 'FAIL'}"
        text = (
            f"verify: {len(self._verify_refs)} windows from {nfr} frames read back in {time.perf_counter() - t0:.1f} "
            f"s; {detail}"
        )
        return dict(ok=ok, max_sigma=worst, max_adu=worst_adu, max_sigma_adj=worst_adj, max_curve=worst_t if table is not None else None, text=text)

    # ------------------------------------------------------------ reporting
    def frames(self) -> np.ndarray:
        return np.concatenate(self._frame_table) if self._frame_table else np.zeros(0, FRAME_DTYPE)

    def status_line(self) -> str:
        if self._t_start is None:
            return "  (not started)"
        s = self.source.stats()
        with self._lock:
            lat = list(self._stats["lat"])
            wf = self._stats["written_frames"]
            by = self._stats["bytes"]
            infl = self._stats["inflight"]
        el = time.perf_counter() - self._t_start
        return (
            f"  t={el:6.1f}s delivered {max(0, s.delivered - self.nwarm * self.block):5d}  written {wf:5d}  LOST "
            f"{s.lost:4d} (incomplete {s.incomplete})  "
            f"blocks in flight {infl}  free {s.free_blocks}  ready {s.ready_blocks}  "
            f"block latency {np.mean(lat) if lat else 0:4.1f}s (max {max(lat) if lat else 0:4.1f})  "
            f"{by / 1e6 / max(el, 1e-9):,.0f} MB/s to {self.cfg.sink}"
        )

    def result(self) -> dict:
        c = self.cfg
        s = self.source.stats()
        st = dict(self.arr.stats) if self.arr is not None else {}
        ms_extra = {}
        if self.arr is not None and self.ms:  # roll the level arrays' native stats up
            lv0 = self.arr.levels[0].stats
            st.update(
                t_compress=sum(lv.stats.get("t_compress", 0.0) for lv in self.arr.levels),
                t_io=sum(lv.stats.get("t_io", 0.0) for lv in self.arr.levels),
                groups_mask=lv0.get("groups_mask", 0),
                direct_used=sum(lv.stats.get("direct_used", 0) for lv in self.arr.levels),
                direct_fallback=sum(lv.stats.get("direct_fallback", 0) for lv in self.arr.levels),
            )
            lvb = [int(lv.stats.get("written_bytes", 0)) for lv in self.arr.levels]
            nbk = max(st.get("blocks", 0), 1)
            ms_extra = dict(
                ms_levels=len(self.arr.levels) - 1,
                ms_level_bytes=lvb,
                ms_pyramid_pct=100.0 * sum(lvb[1:]) / max(lvb[0], 1),
                ms_downsample_s_per_block=st.get("downsample_seconds", 0.0) / nbk,
                ms_level_wait_s_per_block=st.get("level_wait_seconds", 0.0) / nbk,
                ms_fused_blocks=st.get("fused_blocks", 0),
            )
        nb = max(st.get("blocks", 0), 1)
        frame_bytes = int(self.info.frame_bytes)
        fps = self.info.fps
        ft = self.frames()
        el = (self._t_end or time.perf_counter()) - (self._t_start or time.perf_counter())
        ts = {}
        if len(ft) > 1:
            el = (
                (ft["timestamp_ns"][-1] - ft["timestamp_ns"][0]) / 1e9 * len(ft) / (len(ft) - 1)
            )  # source clock, first exposure to end of last
            dt = np.diff(ft["timestamp_ns"].astype(np.float64)) / 1e6
            ts = dict(
                period_ms_mean=float(dt.mean()),
                period_ms_min=float(dt.min()),
                period_ms_max=float(dt.max()),
                frame_id_first=int(ft["frame_id"][0]),
                frame_id_last=int(ft["frame_id"][-1]),
            )
        raw = self._stats["written_frames"] * frame_bytes
        stored = sum(ms_extra["ms_level_bytes"]) if ms_extra else self._stats["bytes"]
        lost = int(s.lost)
        incomplete = int(s.incomplete)
        sustained = (
            lost == 0
            and incomplete == 0
            and self._stats["written_frames"] == self.total_frames
            and self._stats["write_fail"] == 0
        )
        lat = self._stats["lat"]
        r = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            camera=self.info.name,
            serial=self.info.serial,
            gpu=c.gpu,
            q=c.q,
            fps_camera=fps,
            frames=self.total_frames,
            seconds=el,
            block=self.block,
            blocks=self.nblocks,
            max_commits=c.max_commits,
            threads=self.threads,
            chunks=",".join(map(str, c.chunks)),
            shards=",".join(map(str, self.shards)),
            sink=c.sink,
            direct_io=c.direct_io,
            predictor=c.predictor,
            entropy=c.entropy,
            mode=c.mode,
            conv=self.conv,
            offset=self.offset,
            rn=self.rn,
            written_frames=self._stats["written_frames"],
            delivered=int(s.delivered) - self.nwarm * self.block,
            lost_ts=lost,
            lost_fid=int(s.extra.get("lost_by_frame_id", lost)),
            incomplete=incomplete,
            rejected=int(s.extra.get("rejected", 0)),
            sustained=sustained,
            raw_mbs=raw / 1e6 / max(el, 1e-9),
            disk_mbs=stored / 1e6 / max(el, 1e-9),
            ratio=raw / max(stored, 1),
            bytes_written=stored,
            compress_s_per_block=st.get("t_compress", 0) / nb,
            io_s_per_block=st.get("t_io", 0) / nb,
            write_s_per_block=st.get("seconds", 0) / nb,
            lat_mean=float(np.mean(lat)) if lat else 0.0,
            lat_max=float(max(lat)) if lat else 0.0,
            fill_s=self.block / fps,
            groups=bin(st.get("groups_mask", 0)).count("1"),
            direct_used=st.get("direct_used", 0),
            direct_fallback=st.get("direct_fallback", 0),
            t_open=self.t_open,
            t_close=self.t_close,
            pin_s=float(s.extra.get("t_announce_s", 0.0)),
            prefault_s=float(s.extra.get("t_prefault_s", 0.0)),
            large_pages=bool(s.extra.get("large_pages_used", False)),
            noise_adu_measured=self.noise_measured,
            noise_adu_model=self.noise_model,
            notes=c.notes,
            path=c.path,
            multiscale=int(c.multiscale),
            ms_codec=c.ms_codec,
            **ms_extra,
            **ts,
        )
        if self.encoder is not None:
            g = self.encoder.stats
            gb = max(g.get("blocks", 0), 1)
            r.update(
                gpu_upload_s=g.get("upload_s", 0) / gb,
                gpu_kernel_s=g.get("kernel_s", 0) / gb,
                gpu_download_s=g.get("download_s", 0) / gb,
                gpu_host_s=g.get("host_s", 0) / gb,
                gpu_host_tiles=g.get("host_tiles", 0),
                gpu_upload_mode=g.get("upload_mode", ""),
                gpu_check_bad=g.get("check_bad", None),
                gpu_register_s=self.t_register,
            )
        return r

    def report(self, res: Optional[dict] = None) -> str:
        r = res or self.result()
        c = self.cfg
        lines = [
            f"acquisition {r['seconds']:.1f} s: offered {r['fps_camera'] * self.info.frame_bytes / 1e6:,.0f} MB/s, "
            f"written {r['raw_mbs']:,.0f} MB/s raw "
            f"({r['disk_mbs']:,.0f} MB/s {'to disk' if c.sink == 'disk' else 'compressed, discarded'}), ratio "
            f"{r['ratio']:.2f}x",
            f"frames: delivered {r['delivered']}, written {r['written_frames']}, LOST {r['lost_ts']} (incomplete "
            f"{r['incomplete']})",
        ]
        if "period_ms_mean" in r:
            # A source without timestamps (e.g. TucamSource with chunk_mode=False reports timestamp_ns = 0 for every
            # frame) gives period_ms_mean = 0; don't divide by it.
            p = r["period_ms_mean"]
            fps_ts = f"{1e3 / p:.4f} fps" if p > 0 else "no fps (timestamps do not advance)"
            lines.append(
                f"source timestamps: period mean {r['period_ms_mean']:.3f} ms, min {r['period_ms_min']:.3f}, max "
                f"{r['period_ms_max']:.3f} ms -> "
                f"{fps_ts}; frame ids {r['frame_id_first']}..{r['frame_id_last']}; "
                f"acquisition span {r['seconds']:.1f} s"
            )
        if c.gpu != "none":
            lines.append(
                f"gpu encode ({c.gpu}): per block upload {r['gpu_upload_s']:.2f} s [{r['gpu_upload_mode']}], kernel "
                f"{r['gpu_kernel_s']:.2f} s, "
                f"download/host {r['gpu_download_s']:.2f} s; tiles finished on the host {r['gpu_host_tiles']}; "
                f"first-block self-check differing chunks: {r['gpu_check_bad']}"
            )
            lines.append(
                f"native writer: shard assembly + IO {r['io_s_per_block']:.2f} s per block; "
                f"{r['write_s_per_block']:.2f} s per block inside write"
            )
        else:
            lines.append(
                f"native writer: compress {r['compress_s_per_block']:.2f} s + shard IO {r['io_s_per_block']:.2f} s per "
                f"block; "
                f"{r['write_s_per_block']:.2f} s per block inside write; worker threads observed in {r['groups']} "
                f"processor group(s)"
            )
        if r.get("multiscale"):
            lines.append(
                f"multiscale: {r['ms_levels']} level(s) below level 0 ({c.ms_codec} codec on the pyramid), factors "
                f"{tuple(c.ms_factors)}; "
                f"level-1 planes fused for {r['ms_fused_blocks']} blocks; ordered pyramid step "
                f"{r['ms_downsample_s_per_block']:.2f} s per block; "
                f"levels 1+ add {r['ms_pyramid_pct']:.0f}% to the stored bytes; per-level bytes "
                f"{[round(b / 1e9, 2) for b in r['ms_level_bytes']]} GB"
            )
        lines.append(
            f"block latency mean {r['lat_mean']:.2f} s, max {r['lat_max']:.2f} s (fill time {r['fill_s']:.1f} s; must "
            f"stay below {(self.nblocks - 1) * r['fill_s']:.1f} s); "
            f"source start {r['t_open']:.1f} s, writer close {r['t_close']:.1f} s"
        )
        if c.direct_io and c.sink == "disk":
            lines.append(f"direct IO: {r['direct_used']} shards unbuffered, {r['direct_fallback']} fell back")
        lines.append(
            "SUSTAINED: no frames lost"
            if r["sustained"]
            else "NOT SUSTAINED: frames lost, incomplete, or writes failed"
        )
        return "\n".join(lines)

    def save_sidecar(self, res: dict):
        """frames.npy (per-frame table) and acquisition.json next to the dataset"""
        ft = self.frames()
        if len(ft):
            np.save(os.path.join(self.cfg.path, "frames.npy"), ft)
        with open(os.path.join(self.cfg.path, "acquisition.json"), "w") as f:
            json.dump(dict(res, source_info=self.info.asdict(), config=asdict(self.cfg)), f, indent=1, default=str)


CSV_FIELDS = [
    "timestamp",
    "camera",
    "serial",
    "gpu",
    "multiscale",
    "ms_pyramid_pct",
    "predictor",
    "entropy",
    "mode",
    "q",
    "fps_camera",
    "frames",
    "seconds",
    "block",
    "blocks",
    "max_commits",
    "threads",
    "chunks",
    "shards",
    "sink",
    "direct_io",
    "conv",
    "offset",
    "rn",
    "written_frames",
    "lost_ts",
    "lost_fid",
    "incomplete",
    "sustained",
    "raw_mbs",
    "disk_mbs",
    "ratio",
    "compress_s_per_block",
    "io_s_per_block",
    "write_s_per_block",
    "gpu_upload_s",
    "gpu_kernel_s",
    "gpu_download_s",
    "lat_mean",
    "lat_max",
    "fill_s",
    "bytes_written",
    "noise_adu_measured",
    "noise_adu_model",
    "verify",
    "path",
    "notes",
]


def append_csv(csv_path: str, res: dict):
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: res.get(k, "") for k in CSV_FIELDS})

"""EGrabberSource: the Euresys eGrabber camera as a zero-copy *block source* for the B4D writer.

This is the touch point between the vendor driver and b4d. It implements b4d's BlockSource contract
(``b4d/src/b4d/ingest/protocol.py``) **structurally**: nothing here imports b4d, so this package builds, installs and
runs without b4d, and b4d consumes it without knowing anything about eGrabber. The same file is the template for
another vendor: keep the constructor convention and the seven members below, swap the engine underneath.

    from egrabber_camera import EGrabberSource
    src = EGrabberSource(serial="<serial>", frames_per_block=64, nblocks=3)
    print(src.info)                   # SourceInfo-like: shape, dtype, fps, frame_bytes, camera_model ...
    src.start(max_frames=640)         # pins the pool, announces every slot to the grabber as a DMA target, starts
    while True:
        try:
            blk = src.wait_block(1000)
        except StopIteration:
            break
        if blk is None:
            continue
        ...                           # blk.array is a (nframes, H, W) numpy view over the DMA memory: no copy
        src.release(blk)
    src.close()

With b4d installed: ``b4d.ingest.BlockRecorder(src, RecorderConfig(...))`` compresses the blocks in place
(see examples/record_with_b4d.py), or on the command line
``b4d-record --source egrabber_camera.source:EGrabberSource --source-arg serial=<serial>``.

Buffers: by default the engine allocates and pins its own block pool. A higher layer that wants the frames to land
in its own memory passes ``buffers=[...]`` (``nblocks`` host arrays, or CuPy device arrays with
``buffers_kind="device"`` for GPUDirect RDMA; Euresys supports device announces on Linux only).

Every constructor option after ``serial`` is keyword-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .ingest import Ingest

#: (conversion DN/e-, offset DN, read noise e-) per camera model prefix, measured from dark frames
CAMERA_MODELS = {
    "VPS-245MX2": (1.0 / 7.25, 2.05, 6.89),  # Vieworks VPS-245MX2 (Sony IMX811): gain 7.25 e-/ADU
}


@dataclass
class SourceInfo:
    """Mirror of b4d.ingest.SourceInfo (kept local so this package does not import b4d)."""

    name: str
    serial: str
    shape: Tuple[int, int]
    dtype: Any
    fps: float
    frame_bytes: int
    camera_model: Optional[Tuple[float, float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict:
        d = dict(self.__dict__)
        d["dtype"] = str(np.dtype(self.dtype))
        return d


@dataclass
class SourceStats:
    """Mirror of b4d.ingest.SourceStats."""

    delivered: int = 0
    lost: int = 0
    incomplete: int = 0
    blocks_completed: int = 0
    blocks_released: int = 0
    free_blocks: int = 0
    ready_blocks: int = 0
    fps_measured: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class EGrabberSource:
    """BlockSource over egrabber_ingest.dll (see the module docstring)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        frames_per_block: int = 64,
        nblocks: int = 3,
        fps: float = 0.0,
        exposure_us: float = 0.0,
        pixel_format: Optional[str] = "Mono12",
        large_pages: bool = False,
        prefault_threads: int = 16,
        verbose: bool = False,
        dll: Optional[str] = None,
        buffers: Optional[Sequence] = None,
        buffers_kind: str = "host",
        camera_model: Optional[Tuple[float, float, float]] = None,
    ):
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        if buffers is not None and len(buffers) != self.nblocks:
            raise ValueError(f"buffers has {len(buffers)} entries, nblocks is {nblocks}")
        if buffers_kind not in ("host", "device"):
            raise ValueError("buffers_kind must be 'host' or 'device'")
        self._kw = dict(
            serial=serial,
            fps=float(fps),
            exposure_us=float(exposure_us),
            pixel_format=pixel_format,
            large_pages=large_pages,
            prefault_threads=prefault_threads,
            verbose=verbose,
            dll=dll,
        )
        self._buffers, self._ext_kind = (list(buffers) if buffers is not None else None), (
            1 if buffers_kind == "host" else 2
        )
        self._cam: Optional[Ingest] = None
        self._st = SourceStats(free_blocks=self.nblocks)
        # probe the camera with a 1-frame pool: the engine copies its configuration (incl. max_frames) at open, so the
        # real pool is opened in start(); the probe gives .info right after construction, as the contract asks
        probe = Ingest(frames_per_block=1, nblocks=1, **self._kw)
        try:
            self._camera_info = probe.info
            self._dtype = probe.dtype
        finally:
            probe.close()
        ci = self._camera_info
        model = camera_model
        if model is None:
            for prefix, m in CAMERA_MODELS.items():
                if ci.model.upper().startswith(prefix.upper()):
                    model = m
                    break
        self.info = SourceInfo(
            name=f"{ci.vendor} {ci.model}",
            serial=ci.serial,
            shape=(ci.height, ci.width),
            dtype=self._dtype,
            fps=float(ci.fps),
            frame_bytes=int(ci.payload_bytes),
            camera_model=model,
            extra=dict(
                pixel_format=ci.pixel_format,
                unpacking=ci.unpacking,
                exposure_us=ci.exposure_us,
                fps_max=ci.fps_max,
                interface=ci.interface_id,
                link=ci.link_config,
                firmware=ci.version,
                link_throughput_limit=ci.link_throughput_limit,
            ),
        )

    # ------------------------------------------------------------ contract
    def start(self, max_frames: Optional[int] = None) -> None:
        if self._cam is not None:
            raise RuntimeError("already started")
        self._cam = Ingest(
            frames_per_block=self.frames_per_block,
            nblocks=self.nblocks,
            max_frames=int(max_frames or 0),
            ext_blocks=self._buffers,
            ext_kind=self._ext_kind if self._buffers is not None else 0,
            **self._kw,
        )
        self._cam.start(max_frames)

    def wait_block(self, timeout_ms: int = 1000):
        if self._cam is None:
            raise RuntimeError("start() first")
        # Block has array/full_array/frames/seq/index/nframes/partial; StopIteration when drained
        return self._cam.wait_block(timeout_ms)

    def release(self, block) -> None:
        self._cam.release(block)

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()

    def stats(self) -> SourceStats:
        if self._cam is None:
            return SourceStats(free_blocks=self.nblocks)
        s = self._cam.stats()
        return SourceStats(
            delivered=s.delivered,
            lost=s.lost,
            incomplete=s.incomplete,
            blocks_completed=s.blocks_completed,
            blocks_released=s.blocks_released,
            free_blocks=s.free_blocks,
            ready_blocks=s.ready_blocks,
            fps_measured=s.fps_timestamp,
            extra=dict(
                lost_by_timestamp=s.lost_by_timestamp,
                lost_by_frame_id=s.lost_by_frame_id,
                rejected=s.rejected,
                broken=s.broken,
                period_ns_mean=s.period_ns_mean,
                stream_fps=s.stream_fps,
                stream_data_rate_mbs=s.stream_data_rate_mbs,
                seconds_running=s.seconds_running,
                t_announce_s=s.t_announce_s,
                t_prefault_s=s.t_prefault_s,
                large_pages_used=bool(s.large_pages_used),
            ),
        )

    def block_array(self, index: int):
        """full (frames_per_block, H, W) array of pool block index (lets b4d page-lock the DMA memory for CUDA too)"""
        if self._cam is None:
            raise RuntimeError("start() first")
        return self._cam.block_array(index)

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    # ------------------------------------------------------------ vendor extras (not part of the contract)
    @property
    def camera(self) -> Optional[Ingest]:
        """the underlying Ingest engine while started (GenICam feature access: remote_get/remote_set/stream_get)"""
        return self._cam

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        i = self.info
        return (
            f"EGrabberSource({i.name} SN {i.serial}, {i.shape[1]}x{i.shape[0]} {np.dtype(i.dtype).name} @ {i.fps:.4f} "
            f"fps, {self.frames_per_block}x{self.nblocks} blocks)"
        )

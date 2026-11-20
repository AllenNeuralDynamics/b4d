"""DcamSource: a Hamamatsu DCAM-API camera as a zero-copy *block source* for the B4D writer.

It implements b4d's BlockSource contract (``b4d/src/b4d/ingest/protocol.py``) **structurally**: nothing here
imports b4d, so the package installs and runs without it.

    from dcam_camera import DcamSource
    src = DcamSource(serial="<serial>", frames_per_block=64, nblocks=4, pixel_type="MONO16")
    print(src.info)                   # shape, dtype, fps, frame_bytes ...
    src.start(max_frames=640)         # attaches every frame of the pool to DCAM, starts the capture
    while True:
        try:
            blk = src.wait_block(1000)
        except StopIteration:
            break
        if blk is None:
            continue
        ...                           # blk.array: (nframes, H, W) numpy view of the memory DCAM wrote into
        src.release(blk)
    src.close()

Zero copy
---------
The source allocates ``nblocks`` contiguous, page-aligned host blocks of ``frames_per_block`` frames (or takes the
caller's ``buffers=[...]``) and hands DCAM one pointer per frame with ``dcambuf_attach`` (``ATTACHKIND_FRAME``,
``nblocks * frames_per_block`` pointers, block after block). DCAM writes capture frame k into attached frame
``k % (nblocks * frames_per_block)``, i.e. into slot ``k % frames_per_block`` of pool block
``(k // frames_per_block) % nblocks``. ``wait_block`` returns a view of that block: no per-frame copy.

DCAM has no buffer queue and no back-pressure
----------------------------------------------
DCAM never waits for the consumer: it keeps writing round the attached frames and overwrites old ones. Acquisition
block g (frames ``g*F .. g*F+F-1``, ``F = frames_per_block``) lives in pool block ``g % nblocks``, which DCAM starts
refilling with frame ``(g + nblocks) * F``. The source compares DCAM's captured-frame count
(``dcamcap_transferinfo().nFrameCount``) with that number:

* before delivery: a block whose reuse has begun (count >= ``(g + nblocks) * F``; the frame at the count is taken
  as in flight) is not delivered; its frames count in ``SourceStats.lost`` (``extra["lost_overwritten"]``);
* while the consumer holds it: ``release`` (and ``overwritten(blk)`` at any time) reports how many of its frames
  DCAM has overwritten or is writing; those frames move from ``delivered`` to ``lost``, and the block's ``seq`` is
  recorded in ``extra["overrun_seqs"]`` with ``extra["overrun"] = True``. Data the consumer already read from such
  a block is not trustworthy: check ``overwritten(blk) == 0`` after processing a block and before committing it.

``nblocks`` must be >= 2 (with 1 block DCAM refills the block as soon as it is complete). The time the consumer
may hold a block without loss is ``(nblocks - 1) * frames_per_block / fps``.

Frames lost before DCAM (the camera produced them, DCAM never received them) show up as gaps in the camera's
frame counter (``framestamp``): ``missed = max(framestamp - stamp0 - k)``, with
``stamp0`` taken from the first frame read (the counter does not always start at 0: earlier-driver measurements on an
ORCA-Flash4.0 V3 saw it start at 14 on about one start in three). ``lost = lost_overwritten + missed``.

Per-frame stamps:

* ``stamps="attach"`` (default): next to the frames the source attaches two metadata arrays of N elements
  (``ATTACHKIND_PRIMARY_TIMESTAMP`` and ``ATTACHKIND_PRIMARY_FRAMESTAMP``); DCAM writes frame k's camera timestamp
  and framestamp into element ``k % N`` as it writes the frame. No per-frame call. If DCAM refuses the metadata
  attach, the source records the error in ``extra["stamps_error"]`` and uses ``"lockframe"``.
* ``stamps="lockframe"``: reads framestamp and timestamp of every frame with ``dcambuf_lockframe``. On the
  ORCA-Flash4.0 V3 (DCAM-API 26.11) lockframe *and* copyframe are refused with attached
  frames (``NOTREADY``), while the metadata attach works.
* ``stamps="none"``, or when lockframe fails: ``frame_id`` = DCAM frame index, ``timestamp_ns`` = host time (frames
  first seen in the same poll share it), missed frames not detectable; the error is in ``extra["stamps_error"]``.

``stats().extra["stamps"]`` is the method in use.

Pixel types: MONO16 (default) and MONO8 are zero copy. MONO12 is packed (2 px in 3 bytes) and must be unpacked:
``mono12="refuse"`` (default) raises; ``mono12="copy"`` attaches a packed staging ring and unpacks each block into
the pool at ``wait_block``, reported as ``info.extra["copy"] = True`` / ``stats().extra["copy"] = True``.

GPU: ``dcamapi.py`` declares one attach kind, ``ATTACHKIND_FRAME``, which takes host pointers; no device-memory
attach kind is exposed. GPUDirect RDMA is therefore not available through the DCAM-API as bound here and is not
implemented: ``buffers_kind="device"`` raises NotImplementedError. b4d's staged GPU path can still page-lock the host
blocks (``block_array``) and copy them to the device.

Every constructor option after ``serial`` is keyword-only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import dcamapi as A
from .dcam import Dcam, DCAMError, open_by_serial, parse_serial

GPU_STATUS = (
    "GPUDirect RDMA is not available through the DCAM-API as bound here and is not implemented: the image attach "
    "kind, ATTACHKIND_FRAME, takes host pointers (no device-memory attach kind)"
)

PAGE = 4096

#: DCAM_TIMESTAMP as DCAM writes it into an ATTACHKIND_PRIMARY_TIMESTAMP array
TIMESTAMP_DTYPE = np.dtype([("sec", "<u4"), ("microsec", "<i4")])

#: per-frame metadata table carried by every block (same fields and order as b4d.ingest.FRAME_DTYPE)
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

#: (conversion DN/e-, offset DN, read noise e-) per DCAM MODEL string prefix; empty: no measured values yet
CAMERA_MODELS: Dict[str, Tuple[float, float, float]] = {}

PIXEL_DTYPES = {"MONO8": np.uint8, "MONO16": np.uint16, "MONO12": np.uint16}
WAIT_SLICE_MS = 100  # longest single dcamwait; wait_block rechecks the frame count between waits
_WAIT_TOLERATED = (A.DCAMERR.TIMEOUT, A.DCAMERR.ABORT)
_WAIT_COUNTED = (A.DCAMERR.LOSTFRAME, A.DCAMERR.MISSINGFRAME_TROUBLE)


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


class DcamBlock:
    """A delivered block (the attributes of b4d.ingest.FrameBlock, plus ``gen``).

    seq : 0-based count of delivered blocks; gen : acquisition block number (frames gen*F .. gen*F+F-1 of the
    capture). They differ by the number of blocks lost whole before delivery.
    """

    __slots__ = ("array", "full_array", "frames", "seq", "index", "nframes", "partial", "gen")

    def __init__(self, full_array, frames, seq: int, index: int, nframes: int, partial: bool, gen: int):
        self.full_array = full_array
        self.nframes = int(nframes)
        self.array = full_array[: self.nframes]
        self.frames = frames
        self.seq = int(seq)
        self.index = int(index)
        self.partial = bool(partial)
        self.gen = int(gen)

    @property
    def timestamps_ns(self):
        return self.frames["timestamp_ns"]

    def __repr__(self):
        p = ", partial" if self.partial else ""
        return f"DcamBlock(seq={self.seq}, gen={self.gen}, pool={self.index}, n={self.nframes}{p})"


def aligned_empty(nbytes: int, align: int = PAGE) -> np.ndarray:
    """uint8 array of ``nbytes`` whose first byte is ``align``-aligned (the view keeps the allocation alive)."""
    raw = np.empty(int(nbytes) + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off : off + int(nbytes)]


def unpack_mono12(src: np.ndarray, dst: np.ndarray) -> None:
    """Unpack MONO12 rows (``src``: uint8 (h, rowbytes)) into ``dst`` (uint16 (h, w), w even).

    Layout (verified with the earlier driver on an ORCA-Flash4.0 V3): 2 px in 3 bytes, p0 = b0<<4 | b1&0xF, p1 = b2<<4 | b1>>4.
    """
    w = dst.shape[1]
    b = src[:, : w * 3 // 2].reshape(dst.shape[0], w // 2, 3).astype(np.uint16)
    dst[:, 0::2] = (b[..., 0] << 4) | (b[..., 1] & 0xF)
    dst[:, 1::2] = (b[..., 2] << 4) | (b[..., 1] >> 4)


class DcamSource:
    """BlockSource over the DCAM-API with attached (zero-copy) block memory (see the module docstring)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        frames_per_block: int = 64,
        nblocks: int = 3,
        pixel_type: str = "MONO16",
        roi: Optional[Tuple[int, int, int, int]] = None,
        binning: Optional[int] = None,
        exposure_us: float = 0.0,
        fps: float = 0.0,
        trigger_source: Optional[str] = "INTERNAL",
        trigger_polarity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        mono12: str = "refuse",
        stamps: str = "attach",
        prefault: bool = True,
        buffers: Optional[Sequence] = None,
        buffers_kind: str = "host",
        camera_model: Optional[Tuple[float, float, float]] = None,
        dll: Optional[str] = None,
    ):
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        if self.frames_per_block < 1:
            raise ValueError("frames_per_block must be >= 1")
        if self.nblocks < 2:
            raise ValueError("nblocks must be >= 2: DCAM overwrites without back-pressure")
        if buffers_kind == "device":
            raise NotImplementedError(GPU_STATUS)
        if buffers_kind != "host":
            raise ValueError("buffers_kind must be 'host' (or 'device', which is not implemented)")
        if buffers is not None and len(buffers) != self.nblocks:
            raise ValueError(f"buffers has {len(buffers)} entries, nblocks is {nblocks}")
        if mono12 not in ("refuse", "copy"):
            raise ValueError("mono12 must be 'refuse' or 'copy'")
        if stamps not in ("attach", "lockframe", "none"):
            raise ValueError("stamps must be 'attach', 'lockframe' or 'none'")
        pixel_type = pixel_type.upper()
        if pixel_type not in PIXEL_DTYPES:
            raise ValueError(f"pixel_type must be one of {list(PIXEL_DTYPES)}")
        if pixel_type == "MONO12" and mono12 == "refuse":
            raise ValueError(
                "MONO12 is packed (2 px in 3 bytes) and cannot be zero copy; use pixel_type='MONO16' or 'MONO8', "
                "or mono12='copy' for the unpacking copy path"
            )
        self._copy = pixel_type == "MONO12"
        self._stamps_mode = stamps
        self._prefault = bool(prefault)
        self._ext = list(buffers) if buffers is not None else None
        self._lock = threading.RLock()
        self._cam: Optional[Dcam] = open_by_serial(serial, dll)
        try:
            self._configure(pixel_type, roi, binning, exposure_us, fps, trigger_source, trigger_polarity, properties)
            self._geometry = self._read_geometry()
            self.info = self._make_info(serial, camera_model)
            if self._ext is not None:
                self._check_buffers(self._ext)
        except Exception:
            self._cam.close()
            self._cam = None
            raise
        self._blocks: Optional[list] = None  # nblocks (F, H, W) arrays: the pool (zero copy: DCAM writes here)
        self._staging: Optional[np.ndarray] = None  # MONO12 copy path: N packed frames DCAM writes into
        self._pointers: list = []
        self._attached = False
        self._meta: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (timestamps, framestamps) attached to DCAM
        self._running = False  # between start() and drain
        self._reset_counters(None)

    # ------------------------------------------------------------ setup
    def _configure(self, pixel_type, roi, binning, exposure_us, fps, trigger_source, trigger_polarity, properties):
        cam = self._cam
        for k, v in (properties or {}).items():
            cam.set(k, v)
        if cam.get("IMAGE PIXEL TYPE") != pixel_type:
            cam.set("IMAGE PIXEL TYPE", pixel_type)
        if binning is not None:
            b = int(binning)
            opts = cam.entries("BINNING")
            raw = f"{b}x{b}" if f"{b}x{b}" in opts else b
            cam.set("BINNING", raw)
        if roi is not None:
            self._set_roi(*[int(v) for v in roi])
        if exposure_us:
            cam.set("EXPOSURE TIME", float(exposure_us) * 1e-6)
        if fps:
            p = cam.prop("INTERNAL FRAME RATE")
            if not p.writable:
                raise ValueError("INTERNAL FRAME RATE is not writable in this mode (in AREA mode the exposure sets it)")
            cam.set("INTERNAL FRAME RATE", float(fps))
        if trigger_source is not None and cam.get("TRIGGER SOURCE") != trigger_source:
            cam.set("TRIGGER SOURCE", trigger_source)
        if trigger_polarity is not None:
            cam.set("TRIGGER POLARITY", trigger_polarity)

    def _set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """SUBARRAY values are sensor pixels and apply only with SUBARRAY MODE ON; DCAM cross-checks them while the
        mode is ON, so they are written with the mode OFF and the mode is switched on last."""
        cam = self._cam
        cam.set("SUBARRAY MODE", "OFF")
        for name, v in (("HPOS", 0), ("VPOS", 0), ("HSIZE", w), ("VSIZE", h), ("HPOS", x), ("VPOS", y)):
            cam.set(f"SUBARRAY {name}", v)
        full = (cam.get("IMAGE DETECTOR PIXEL NUM HORZ"), cam.get("IMAGE DETECTOR PIXEL NUM VERT"))
        if (w, h) != tuple(int(v) for v in full):
            cam.set("SUBARRAY MODE", "ON")

    def _read_geometry(self) -> dict:
        cam = self._cam
        if cam.has("NUMBER OF VIEW") and cam.get("NUMBER OF VIEW") > 1:
            raise RuntimeError(f"SENSOR MODE {cam.get('SENSOR MODE')} delivers several views per frame; not supported")
        g = dict(
            height=int(cam.get("IMAGE HEIGHT")),
            width=int(cam.get("IMAGE WIDTH")),
            rowbytes=int(cam.get("BUFFER ROWBYTES")),
            framebytes=int(cam.get("BUFFER FRAMEBYTES")),
            top_offset=int(cam.get("BUFFER TOP OFFSET BYTES")) if cam.has("BUFFER TOP OFFSET BYTES") else 0,
            pixel_type=str(cam.get("BUFFER PIXEL TYPE" if cam.has("BUFFER PIXEL TYPE") else "IMAGE PIXEL TYPE")),
        )
        if g["pixel_type"] not in PIXEL_DTYPES:
            raise RuntimeError(f"buffer pixel type {g['pixel_type']} is not supported")
        if (g["pixel_type"] == "MONO12") != self._copy:
            raise RuntimeError(f"buffer pixel type {g['pixel_type']} does not match the requested pixel type")
        itemsize = np.dtype(PIXEL_DTYPES[g["pixel_type"]]).itemsize
        if self._copy:
            if g["rowbytes"] < g["width"] * 3 // 2 or g["framebytes"] < g["rowbytes"] * g["height"] or g["top_offset"]:
                raise RuntimeError(f"unexpected MONO12 buffer layout {g}")
        contiguous = (g["width"] * itemsize, g["rowbytes"] * g["height"], 0)
        if not self._copy and (g["rowbytes"], g["framebytes"], g["top_offset"]) != contiguous:
            # a padded layout would make the frames non-contiguous in the block; not supported in zero-copy mode
            raise RuntimeError(f"buffer layout is not contiguous (rowbytes/framebytes/top offset): {g}")
        g["dtype"] = np.dtype(PIXEL_DTYPES[g["pixel_type"]])
        g["frame_bytes"] = g["height"] * g["width"] * g["dtype"].itemsize  # as delivered (after unpacking)
        return g

    def _make_info(self, serial, camera_model) -> SourceInfo:
        cam, g = self._cam, self._geometry
        s = {k.name.lower(): cam.string_info(k) or "" for k in A.IDSTR}
        model = camera_model
        if model is None:
            for prefix, m in CAMERA_MODELS.items():
                if s["model"].upper().startswith(prefix.upper()):
                    model = m
                    break
        exposure_s = float(cam.get("EXPOSURE TIME"))
        fps = float(cam.get("INTERNAL FRAME RATE")) if cam.has("INTERNAL FRAME RATE") else 1.0 / max(exposure_s, 1e-6)
        extra = dict(
            vendor=s["vendor"],
            model=s["model"],
            series=s["camera_seriesname"],
            bus=s["bus"],
            versions={k: s[k] for k in ("cameraversion", "driverversion", "moduleversion", "dcamapiversion")},
            pixel_type=g["pixel_type"],
            exposure_us=exposure_s * 1e6,
            trigger_source=cam.get("TRIGGER SOURCE") if cam.has("TRIGGER SOURCE") else None,
            buffer_rowbytes=g["rowbytes"],
            buffer_framebytes=g["framebytes"],
            zero_copy=not self._copy,
            copy=self._copy,
            stamps=self._stamps_mode,
        )
        for name, key in (
            ("BINNING", "binning"),
            ("SENSOR MODE", "sensor_mode"),
            ("READOUT SPEED", "readout_speed"),
            ("CONVERSION FACTOR COEFF", "dcam_conversion_factor_coeff"),  # as DCAM reports it; not interpreted here
            ("CONVERSION FACTOR OFFSET", "dcam_conversion_factor_offset"),
        ):
            if cam.has(name):
                extra[key] = cam.get(name)
        if cam.has("SUBARRAY MODE") and cam.get("SUBARRAY MODE") == "ON":
            extra["roi"] = tuple(int(cam.get(f"SUBARRAY {n}")) for n in ("HPOS", "VPOS", "HSIZE", "VSIZE"))
        return SourceInfo(
            name=f"{s['vendor']} {s['camera_seriesname'] or s['model']}".strip(),
            serial=str(serial) if serial is not None else parse_serial(cam.string_info("CAMERAID")),
            shape=(g["height"], g["width"]),
            dtype=g["dtype"],
            fps=fps,
            frame_bytes=g["frame_bytes"],
            camera_model=model,
            extra=extra,
        )

    def _check_buffers(self, buffers) -> None:
        g, F = self._geometry, self.frames_per_block
        for i, b in enumerate(buffers):
            if not hasattr(b, "__array_interface__"):
                raise TypeError(f"buffers[{i}] is not a host (numpy-compatible) array; device buffers: {GPU_STATUS}")
            a = np.asarray(b)
            if a.dtype != g["dtype"] or a.ndim != 3 or a.shape[0] < F or a.shape[1:] != (g["height"], g["width"]):
                raise ValueError(
                    f"buffers[{i}] must be {g['dtype']} (>= {F}, {g['height']}, {g['width']}), got {a.dtype} {a.shape}"
                )
            if not a[:F].flags.c_contiguous or not a.flags.writeable:
                raise ValueError(f"buffers[{i}] must be C-contiguous and writable")

    def _ensure_pool(self) -> None:
        if self._blocks is not None:
            return
        g, F, nb = self._geometry, self.frames_per_block, self.nblocks
        shape = (F, g["height"], g["width"])
        if self._ext is not None:
            self._blocks = [np.asarray(b)[:F] for b in self._ext]
        else:
            nbytes = F * g["frame_bytes"]
            self._blocks = [aligned_empty(nbytes).view(g["dtype"]).reshape(shape) for _ in range(nb)]
            if self._prefault:
                for b in self._blocks:
                    b.fill(0)  # commit the pages before DCAM is given the pointers
        if self._copy:
            n = nb * F
            self._staging = aligned_empty(n * g["framebytes"])
            if self._prefault:
                self._staging.fill(0)
            base = self._staging.ctypes.data
            self._pointers = [base + k * g["framebytes"] for k in range(n)]
        else:
            fb = g["frame_bytes"]
            self._pointers = [b.ctypes.data + s * fb for b in self._blocks for s in range(F)]

    def _reset_counters(self, max_frames) -> None:
        N = self.nblocks * self.frames_per_block
        self._max_frames = int(max_frames) if max_frames else 0
        self._count = 0  # DCAM nFrameCount at the last poll
        self._final: Optional[int] = None  # nFrameCount after the capture stopped
        self._next_gen = 0
        self._seq = 0
        self._held: Dict[int, int] = {}  # pool block -> gen
        self._delivered = 0
        self._lost_overwritten = 0  # frames overwritten by DCAM before or during use
        self._lost_blocks = 0  # blocks overwritten before delivery
        self._overrun_seqs: list = []  # seq of delivered blocks that DCAM overwrote while held
        self._missed = 0
        self._stamp0: Optional[int] = None
        self._blocks_completed = 0
        self._blocks_released = 0
        self._host_ns = np.zeros(N, np.uint64)  # host time each attached frame was first seen captured
        self._host_k = np.full(N, -1, np.int64)  # capture index that time belongs to
        self._ts_first: Optional[Tuple[int, int]] = None  # (k, timestamp_ns) of the first delivered frame
        self._ts_last: Optional[Tuple[int, int]] = None
        self._stamps_src = self._stamps_mode  # "attach", "lockframe" or "none": the method in use
        self._stamps_error: Optional[str] = None
        self._stamps_unwritten = 0  # stamps="attach": frames whose metadata element DCAM had not written
        self._lockframe_foreign_ptr = 0  # lockframe returned an address other than the attached frame's
        self._dcam_events: Dict[str, int] = {}
        self._t_start = 0.0
        self._t_stop: Optional[float] = None

    # ------------------------------------------------------------ contract
    def start(self, max_frames: Optional[int] = None) -> None:
        with self._lock:
            if self._cam is None:
                raise RuntimeError("closed")
            if self._running and self._final is None:
                raise RuntimeError("already started (stop() first)")
            if self._held:
                raise RuntimeError("release every block before starting again")
            g = self._read_geometry()
            if (g["height"], g["width"], g["pixel_type"]) != (
                self._geometry["height"],
                self._geometry["width"],
                self._geometry["pixel_type"],
            ):
                raise RuntimeError("the frame layout changed since construction; create a new DcamSource")
            self._ensure_pool()
            self._detach()
            self._cam.attach(self._pointers)
            self._attached = True
            self._reset_counters(max_frames)
            if self._stamps_src == "attach":
                self._attach_meta()
            self._cam.start()
            self._t_start = time.perf_counter()
            self._running = True

    def wait_block(self, timeout_ms: int = 1000):
        if self._cam is None:
            raise RuntimeError("closed")
        deadline = time.perf_counter() + max(0, timeout_ms) / 1000.0
        while True:
            with self._lock:
                if not self._running and self._final is None:
                    raise RuntimeError("start() first")
                blk = self._try_deliver()
                if blk is not None:
                    return blk
                if self._drained():
                    self._detach()
                    raise StopIteration
                stopped = self._final is not None
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            if stopped:  # nothing more will arrive; the next block waits for a release
                time.sleep(min(left, 0.005))
                continue
            try:
                ms = max(1, int(min(left * 1000, WAIT_SLICE_MS)))
                self._cam.wait(A.WAITEVENT.FRAMEREADY | A.WAITEVENT.STOPPED, ms)
            except DCAMError as e:
                if e.code in _WAIT_COUNTED:
                    self._dcam_events[e.name] = self._dcam_events.get(e.name, 0) + 1
                elif e.code not in _WAIT_TOLERATED:
                    raise

    def release(self, block) -> None:
        with self._lock:
            b = int(block.index)
            if self._held.get(b) != getattr(block, "gen", None):
                raise ValueError(f"{block!r} is not held")
            n = self._overwritten(block)
            if n:
                self._delivered -= n
                self._lost_overwritten += n
                self._overrun_seqs.append(int(block.seq))
            del self._held[b]
            self._blocks_released += 1

    def stop(self) -> None:
        with self._lock:
            if self._running and self._final is None:
                self._halt()

    def stats(self) -> SourceStats:
        with self._lock:
            if self._final is None and self._running:
                self._poll()
            count = self._final if self._final is not None else self._count
            limit = self._limit(count)
            F = self.frames_per_block
            ready = max(0, min(self.nblocks, limit // F - self._next_gen))
            fps = 0.0
            if self._ts_first and self._ts_last and self._ts_last[0] > self._ts_first[0]:
                dt = (self._ts_last[1] - self._ts_first[1]) / 1e9
                fps = (self._ts_last[0] - self._ts_first[0]) / dt if dt > 0 else 0.0
            t_end = self._t_stop if self._t_stop is not None else time.perf_counter()
            return SourceStats(
                delivered=self._delivered,
                lost=self._lost_overwritten + self._missed,
                incomplete=0,
                blocks_completed=self._blocks_completed,
                blocks_released=self._blocks_released,
                free_blocks=self.nblocks - len(self._held),
                ready_blocks=ready,
                fps_measured=fps,
                extra=dict(
                    captured=count,
                    lost_overwritten=self._lost_overwritten,
                    lost_blocks_before_delivery=self._lost_blocks,
                    overrun=bool(self._overrun_seqs),
                    overrun_seqs=list(self._overrun_seqs[-64:]),
                    overrun_blocks=len(self._overrun_seqs),
                    missed=self._missed,
                    stamp0=self._stamp0,
                    stamps=self._stamps_src,
                    stamps_error=self._stamps_error,
                    stamps_unwritten=self._stamps_unwritten,
                    lockframe_foreign_pointer=self._lockframe_foreign_ptr,
                    dcam_wait_events=dict(self._dcam_events),
                    copy=self._copy,
                    zero_copy=not self._copy,
                    seconds_running=(t_end - self._t_start) if self._t_start else 0.0,
                ),
            )

    def block_array(self, index: int):
        """Full (frames_per_block, H, W) array of pool block ``index`` (zero-copy mode: the memory DCAM writes)."""
        self._ensure_pool()
        return self._blocks[int(index)]

    def close(self) -> None:
        with self._lock:
            if self._cam is None:
                return
            try:
                if self._running and self._final is None:
                    self._halt()
                self._detach()
            finally:
                self._running = False
                self._cam.close()
                self._cam = None

    # ------------------------------------------------------------ vendor extras
    def overwritten(self, block) -> int:
        """Frames of a held block DCAM has overwritten or is writing now (0: the block is intact)."""
        with self._lock:
            return self._overwritten(block)

    def fire_trigger(self) -> None:
        """One software trigger (``trigger_source="SOFTWARE"``)."""
        self._cam.fire_trigger()

    @property
    def camera(self) -> Optional[Dcam]:
        """The DCAM adapter (property access by name: ``camera.get("SENSOR TEMPERATURE")``). Settings that change the
        frame layout must not be changed after construction."""
        return self._cam

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        i = self.info
        return (
            f"DcamSource({i.name} SN {i.serial}, {i.shape[1]}x{i.shape[0]} {np.dtype(i.dtype).name} @ {i.fps:.3f} fps, "
            f"{self.frames_per_block}x{self.nblocks} blocks{', copy' if self._copy else ''})"
        )

    # ------------------------------------------------------------ internals
    def _limit(self, count: int) -> int:
        return min(count, self._max_frames) if self._max_frames else count

    def _poll(self) -> int:
        """Refresh DCAM's captured-frame count, stamp newly captured frames with host time, honour max_frames."""
        if self._final is not None:
            return self._final
        n = int(self._cam.transferinfo()[1])
        if n > self._count:
            N = self._host_k.size
            ks = np.arange(max(self._count, n - N), n, dtype=np.int64)
            self._host_ns[ks % N] = time.monotonic_ns()
            self._host_k[ks % N] = ks
        self._count = n
        if self._max_frames and n >= self._max_frames:
            self._halt()
            return self._final
        return n

    def _halt(self) -> None:
        """Stop the capture (DCAM has no frame-count capture) and fix the final frame count."""
        self._cam.stop()
        self._final = int(self._cam.transferinfo()[1])
        self._count = self._final
        self._t_stop = time.perf_counter()

    def _attach_meta(self) -> None:
        """stamps="attach": attach the timestamp and framestamp arrays; on refusal fall back to lockframe."""
        N = self.nblocks * self.frames_per_block
        ts = np.zeros(N, TIMESTAMP_DTYPE)
        fs = np.full(N, -1, np.int32)  # -1: not written by DCAM yet
        done = []
        try:
            for kind, arr in ((A.ATTACHKIND_PRIMARY_TIMESTAMP, ts), (A.ATTACHKIND_PRIMARY_FRAMESTAMP, fs)):
                self._cam.attach_primary(kind, arr.ctypes.data, N)
                done.append(kind)
        except DCAMError as e:
            for kind in done:
                try:
                    self._cam.release(kind)
                except DCAMError:
                    pass
            self._stamps_src = "lockframe"
            self._stamps_error = f"metadata attach refused: {e}"
            return
        self._meta = (ts, fs)

    def _detach(self) -> None:
        if self._meta is not None and self._cam is not None:
            for kind in (A.ATTACHKIND_PRIMARY_TIMESTAMP, A.ATTACHKIND_PRIMARY_FRAMESTAMP):
                try:
                    self._cam.release(kind)
                except DCAMError:
                    pass
            self._meta = None
        if self._attached and self._cam is not None:
            self._cam.release()
            self._attached = False
        if self._final is not None:
            self._running = False

    def _drained(self) -> bool:
        return self._final is not None and self._next_gen * self.frames_per_block >= self._limit(self._final)

    def _in_flight(self, count: int) -> int:
        """The count of frames that may be in DCAM's hands: +1 for the frame being written while capturing."""
        return count + (0 if self._final is not None else 1)

    def _overwritten(self, block) -> int:
        if self._copy:
            return 0  # the pool is a copy: DCAM writes only the staging ring
        count = self._poll() if self._final is None else self._final
        reuse = (int(block.gen) + self.nblocks) * self.frames_per_block
        return int(min(block.nframes, max(0, self._in_flight(count) - reuse)))

    def _try_deliver(self) -> Optional[DcamBlock]:
        F, nb = self.frames_per_block, self.nblocks
        while True:
            count = self._poll()
            limit = self._limit(count)
            stopped = self._final is not None
            g = self._next_gen
            start = g * F
            if stopped and start >= limit:
                return None
            reuse = (g + nb) * F
            n = min(F, limit - start)
            if self._in_flight(count) > reuse:  # DCAM refills (or refilled) this block's memory: lost whole
                lost = min(F, limit - start)
                self._lost_overwritten += lost
                self._lost_blocks += 1
                self._next_gen += 1
                continue
            if n < F and not (stopped and n > 0):
                return None
            b = g % nb
            if b in self._held:
                return None  # the consumer holds this pool block (an older acquisition block, now overrun)
            frames = self._frame_table(g, n)
            if self._copy:
                self._unpack(g, n)
            count = self._poll()  # stamps/unpack done: the block must still be intact
            if self._in_flight(count) > reuse:
                continue  # counted as lost by the next pass
            return self._hand_out(g, b, n, frames, partial=n < F)

    def _hand_out(self, g, b, n, frames, partial) -> DcamBlock:
        self._held[b] = g
        self._next_gen = g + 1
        blk = DcamBlock(self._blocks[b], frames, self._seq, b, n, partial, g)
        self._seq += 1
        self._delivered += n
        self._blocks_completed += 1
        k0, k1 = g * self.frames_per_block, g * self.frames_per_block + n - 1
        if self._ts_first is None:
            self._ts_first = (k0, int(frames["timestamp_ns"][0]))
        self._ts_last = (k1, int(frames["timestamp_ns"][n - 1]))
        return blk

    def _frame_table(self, g: int, n: int) -> np.ndarray:
        F, N = self.frames_per_block, self.nblocks * self.frames_per_block
        frames = np.zeros(n, FRAME_DTYPE)
        ks = np.arange(g * F, g * F + n, dtype=np.int64)
        slots = ks % N
        now = time.monotonic_ns()
        host = np.where(self._host_k[slots] == ks, self._host_ns[slots], now)
        frames["host_ns"] = host
        frames["slot"] = np.arange(n)
        frames["size_filled"] = self._geometry["framebytes"]
        frames["incomplete"] = 0
        frames["frame_id"] = ks
        frames["timestamp_ns"] = host
        if self._stamps_src == "attach":
            ts, fs = self._meta
            stamp = fs[slots].astype(np.int64)
            ok = stamp >= 0
            self._stamps_unwritten += int((~ok).sum())
            stamp &= 0xFFFFFFFF
            t_ns = ts["sec"][slots].astype(np.int64) * 1_000_000_000 + ts["microsec"][slots].astype(np.int64) * 1000
            frames["frame_id"] = np.where(ok, stamp, ks)
            frames["timestamp_ns"] = np.where(ok, t_ns, host.astype(np.int64))
            if ok.any():
                if self._stamp0 is None:
                    i = int(np.argmax(ok))
                    self._stamp0 = int(stamp[i] - ks[i])
                self._missed = max(self._missed, int((stamp[ok] - self._stamp0 - ks[ok]).max()))
        elif self._stamps_src == "lockframe":
            try:
                for i, k in enumerate(ks):
                    f = self._cam.lockframe(int(k % N))
                    if f.buf and int(f.buf) != self._pointers[int(k % N)]:
                        self._lockframe_foreign_ptr += 1
                    stamp = int(f.framestamp) & 0xFFFFFFFF
                    frames["frame_id"][i] = stamp
                    frames["timestamp_ns"][i] = int(f.timestamp.sec) * 1_000_000_000 + int(f.timestamp.microsec) * 1000
                    if self._stamp0 is None:
                        self._stamp0 = stamp - int(k)
                    self._missed = max(self._missed, stamp - self._stamp0 - int(k))
            except DCAMError as e:  # lockframe refused with attached buffers: host time and DCAM index instead
                self._stamps_src = "none"
                self._stamps_error = (self._stamps_error + "; " if self._stamps_error else "") + str(e)
                frames["frame_id"] = ks
                frames["timestamp_ns"] = host
        return frames

    def _unpack(self, g: int, n: int) -> None:
        """MONO12 copy path: unpack frames g*F .. g*F+n-1 from the staging ring into pool block g % nblocks."""
        geo, F, N = self._geometry, self.frames_per_block, self.nblocks * self.frames_per_block
        h, rb, fb = geo["height"], geo["rowbytes"], geo["framebytes"]
        dst = self._blocks[g % self.nblocks]
        for i in range(n):
            k = (g * F + i) % N
            src = self._staging[k * fb : k * fb + rb * h].reshape(h, rb)
            unpack_mono12(src, dst[i])

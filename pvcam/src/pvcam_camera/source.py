"""PvcamSource: a Teledyne Photometrics camera (target: Kinetix) as a zero-copy *block source* for the B4D writer.

It implements b4d's BlockSource contract (``b4d/src/b4d/ingest/protocol.py``) **structurally**: nothing here imports
b4d. Same class shape as ``egrabber_camera.EGrabberSource``:

    from pvcam_camera import PvcamSource
    src = PvcamSource(serial="<serial>", frames_per_block=64, nblocks=3)
    print(src.info)                   # SourceInfo-like: shape, dtype, fps, frame_bytes, camera_model ...
    src.start(max_frames=640)         # allocates the circular buffer, pl_exp_start_cont into it, starts
    while True:
        try:
            blk = src.wait_block(1000)
        except StopIteration:
            break
        if blk is None:
            continue
        ...                           # blk.array: (nframes, H, W) numpy view of the memory PVCAM wrote: no copy
        src.release(blk)
    src.close()

Zero copy
---------
PVCAM continuous acquisition writes into one caller-allocated circular buffer (``pl_exp_setup_cont`` +
``pl_exp_start_cont(hcam, buffer, size)``). This class allocates ONE contiguous, page-aligned (4096 B) buffer of
``nblocks * frames_per_block`` frame slots; pool block i is slots ``[i*fpb, (i+1)*fpb)``. The ring has exactly
``nblocks * fpb`` slots, so a block never straddles the wrap. ``wait_block`` returns a numpy view of block i; no
frame is copied. Every retrieved frame's address is checked against the slot the block layout expects
(``(address - base) / frame_bytes == frames_retrieved mod ring_frames``); a mismatch stops the acquisition with an
error instead of handing out misaligned blocks.

Buffer mode and back-pressure: the ring is set up with ``CIRC_NO_OVERWRITE``. Frames are taken with
``pl_exp_get_oldest_frame_ex`` (address + FRAME_INFO) and are *not* unlocked when retrieved: a block's frames are
unlocked (``pl_exp_unlock_oldest_frame``, once per frame, oldest first) only when the consumer releases the block
(and every earlier block). pvcam.h describes the two halves differently: ``pl_exp_unlock_oldest_frame`` says a
locked frame "cannot be overwritten", while the ``CIRC_NO_OVERWRITE`` text says the oldest frame "is also replaced
with new frame ... (e.g. via DMA write from camera), but PVCAM recognizes it and returns a buffer overrun error".
The source therefore checks every frame: if a new frame lands in a slot of a block the consumer still holds, the
memory under that block was overwritten; the source counts it (``incomplete``, ``extra['overwritten_held_frames']``)
and stops with ``extra['error']`` set. If locks are honoured, the driver drops frames instead, and the drops show up
as ``FrameNr`` gaps (``lost``). If PVCAM hands out the same frame again until it is unlocked (it then does not allow
several locked frames), the source switches to unlocking on retrieval (``extra['lock_mode'] == 'unlock_on_retrieve'``);
the overwrite check above is then the only protection of held blocks.

Embedded frame metadata (``PARAM_METADATA_ENABLED``) is switched off, so a frame slot is exactly H*W*bytes; the
frame size ``pl_exp_setup_cont`` reports is checked against that.

Buffers: ``buffers=[array]`` (one C-contiguous host array of nblocks*fpb frames) or ``buffers=[b0, ..., b(n-1)]``
(nblocks arrays that are consecutive, adjacent pieces of one allocation) lets the frames land in caller memory.
Separate allocations per block cannot be used: ``pl_exp_start_cont`` takes one buffer pointer and size.

GPU: the PVCAM API (SDK 3.10.1.1 headers) has no GPU, CUDA, RDMA or device-memory path: ``pl_exp_start_cont`` takes
a host pointer. GPUDirect RDMA is not available through the PVCAM API and is not implemented;
``buffers_kind="device"`` raises NotImplementedError. For a GPU encoder, b4d page-locks the host blocks
(``block_array``) and uploads them.

Losses: ``lost`` = sum of gaps in ``FRAME_INFO.FrameNr`` between consecutive retrieved frames (frames the driver could
not place, e.g. ring full of held blocks); ``incomplete`` = frames detected as overwritten while held.
Per frame: ``frame_id`` = FrameNr, ``timestamp_ns`` = FRAME_INFO.TimeStamp (EOF) * ``timestamp_unit_ns`` (pvcam.h:
"precision of 0.1ms", so 100,000 ns per count by default), ``host_ns`` = ``time.perf_counter_ns()`` at retrieval.
``block.frame_info`` keeps FrameNr, TimeStamp, TimeStampBOF and ReadoutTime as delivered. A final partial block
after ``stop()`` or ``max_frames`` is delivered with ``partial=True``.

Every constructor option after ``serial`` is keyword-only.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import pvcam as pv

#: per-frame metadata table of every block: b4d's FRAME_DTYPE, field for field
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

#: FRAME_INFO as PVCAM delivered it, one row per frame (block.frame_info)
FRAME_INFO_DTYPE = np.dtype(
    [("frame_nr", "<i4"), ("timestamp", "<i8"), ("timestamp_bof", "<i8"), ("readout_time", "<i4")]
)

#: (conversion DN/e-, offset DN, read noise e-) keyed by (product name substring, readout port value, host bit depth).
#: Kinetix: the model of the b4d benchmarks (b4d/src/b4d/tune.py CAMERAS["kinetix"], max_adu 255), i.e. the 8-bit
#: "Speed" readout port (port value 1). Those files do not record the gain index; other ports have no model here.
CAMERA_MODELS = {
    ("kinetix", 1, 8): (0.851, 6.0, 2.35),
}

GPU_STATUS = (
    "GPUDirect RDMA is not available through the PVCAM API (PVCAM SDK 3.10.1.1 headers: pl_exp_start_cont takes a "
    "host pointer; no GPU, CUDA, RDMA or device-memory entry point) and is not implemented"
)

PAGE = 4096
_FREE, _FILLING, _READY, _HELD, _RELEASED = "free", "filling", "ready", "held", "released"


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


class Block:
    """A completed block (FrameBlock-like). Valid from wait_block() until release()."""

    __slots__ = ("array", "full_array", "frames", "frame_info", "seq", "index", "nframes", "partial")

    def __init__(self, full_array, frames, frame_info, seq, index, nframes, partial=False):
        self.full_array = full_array
        self.nframes = int(nframes)
        self.array = full_array[: self.nframes]
        self.frames = frames
        self.frame_info = frame_info
        self.seq, self.index, self.partial = int(seq), int(index), bool(partial)

    @property
    def timestamps_ns(self):
        return self.frames["timestamp_ns"]

    def __repr__(self):
        return f"Block(seq={self.seq}, pool={self.index}, n={self.nframes}{', partial' if self.partial else ''})"


def _address(a) -> int:
    return int(a.__array_interface__["data"][0])


class PvcamSource:
    """BlockSource over the PVCAM runtime (see the module docstring)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        frames_per_block: int = 64,
        nblocks: int = 3,
        camera_name: Optional[str] = None,
        exposure_us: Optional[float] = None,
        readout_port: Optional[int] = None,
        speed_index: Optional[int] = None,
        gain_index: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        binning: Tuple[int, int] = (1, 1),
        exp_mode: Optional[int] = None,
        fps: Optional[float] = None,
        buffer_mode: str = "no_overwrite",
        prefault: bool = True,
        poll_interval_s: float = 0.0005,
        timestamp_unit_ns: int = 100_000,
        max_get_failures: Optional[int] = None,
        buffers: Optional[Sequence] = None,
        buffers_kind: str = "host",
        camera_model: Optional[Tuple[float, float, float]] = None,
        runtime=None,
        library: Optional[str] = None,
    ):
        if buffers_kind == "device":
            raise NotImplementedError(GPU_STATUS)
        if buffers_kind != "host":
            raise ValueError("buffers_kind must be 'host' or 'device'")
        if buffer_mode not in ("no_overwrite", "overwrite"):
            raise ValueError("buffer_mode must be 'no_overwrite' or 'overwrite'")
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        if self.frames_per_block < 1 or self.nblocks < 1:
            raise ValueError("frames_per_block and nblocks must be >= 1")
        self.ring_frames = self.frames_per_block * self.nblocks
        self._poll = float(poll_interval_s)
        self._ts_unit = int(timestamp_unit_ns)
        self._max_get_failures = max_get_failures
        self._prefault = bool(prefault)
        self._buffer_mode = pv.CIRC_NO_OVERWRITE if buffer_mode == "no_overwrite" else pv.CIRC_OVERWRITE
        self._rt = runtime if runtime is not None else pv.Pvcam.shared(library)
        self._h: Optional[int] = None
        self._rt.init()
        self._inited = True
        try:
            self._h, cam_name, cam_serial = self._open(serial, camera_name)
            self._configure(
                cam_name,
                cam_serial,
                exposure_us,
                readout_port,
                speed_index,
                gain_index,
                roi,
                binning,
                exp_mode,
                fps,
                camera_model,
                buffer_mode,
            )
            self._buffers = list(buffers) if buffers is not None else None
            if self._buffers is not None:
                self._check_buffers(self._buffers)
        except BaseException:
            self._close_camera()
            raise
        self._ring = None  # (ring_frames, H, W) view; allocated in start() or block_array()
        self._base = 0
        self._lock = threading.Lock()
        self._ready: "queue.Queue[Optional[Block]]" = queue.Queue()
        self._state = [_FREE] * self.nblocks
        self._locked = [0] * self.nblocks
        self._order: deque = deque()  # blocks in fill order (oldest locked frames first)
        self._st = SourceStats(free_blocks=self.nblocks)
        self._x = dict(
            lock_mode="hold_until_release",
            get_frame_errors=0,
            last_get_error="",
            overwritten_held_frames=0,
            frame_nr_backwards=0,
            unlocked=0,
            retrieved=0,
            first_frame_nr=None,
            last_frame_nr=None,
            status=None,
            bytes_arrived=0,
            buffer_cnt=0,
            error="",
            stop_error="",
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False
        self._started = False
        self._max_frames = None

    # ------------------------------------------------------------ camera setup
    def _serial_of(self, h: int) -> str:
        rt = self._rt
        return str(rt.get_param(h, pv.PARAM_HEAD_SER_NUM_ALPHA)) if rt.available(h, pv.PARAM_HEAD_SER_NUM_ALPHA) else ""

    def _open(self, serial, camera_name):
        rt = self._rt
        if camera_name:
            h = rt.open(camera_name)
            s = self._serial_of(h)
            if serial and s != serial:
                rt.close(h)
                raise LookupError(f"camera {camera_name!r} has serial {s!r}, not {serial!r}")
            return h, camera_name, s
        names = rt.camera_names()
        if not names:
            raise LookupError("PVCAM reports no camera")
        if not serial:
            h = rt.open(names[0])
            return h, names[0], self._serial_of(h)
        seen = []
        for name in names:
            try:
                h = rt.open(name)
            except pv.PvcamError as e:
                seen.append(f"{name}: {e.message}")
                continue
            s = self._serial_of(h)
            if s == serial:
                return h, name, s
            seen.append(f"{name}: {s}")
            rt.close(h)
        raise LookupError(f"no PVCAM camera with serial {serial!r} ({'; '.join(seen)})")

    def _get(self, pid, attr=pv.ATTR_CURRENT, default=None):
        rt, h = self._rt, self._h
        return rt.get_param(h, pid, attr) if rt.available(h, pid) else default

    def _set(self, pid, value, what):
        if value is None:
            return
        if not self._rt.available(self._h, pid):
            raise ValueError(f"{what} is not available on this camera")
        self._rt.set_param(self._h, pid, value)

    def _enum_name(self, pid, value):
        if value is None or not self._rt.available(self._h, pid):
            return None
        for v, d in self._rt.enum_values(self._h, pid):
            if v == value:
                return d
        return None

    def _configure(
        self, cam_name, cam_serial, exposure_us, port, speed, gain, roi, binning, exp_mode, fps, model, buffer_mode
    ):
        rt, h = self._rt, self._h
        if rt.available(h, pv.PARAM_CIRC_BUFFER) is False:
            raise RuntimeError("camera does not support circular-buffer acquisition (PARAM_CIRC_BUFFER unavailable)")
        # readout port, then speed (resets gain to 1x), then gain: the order pvcam.h prescribes
        self._set(pv.PARAM_READOUT_PORT, port, "readout_port")
        self._set(pv.PARAM_SPDTAB_INDEX, speed, "speed_index")
        self._set(pv.PARAM_GAIN_INDEX, gain, "gain_index")
        # embedded metadata and centroids change the frame layout: off, so a slot is exactly H*W*bytes
        layout_params = (
            (pv.PARAM_METADATA_ENABLED, "embedded frame metadata"),
            (pv.PARAM_CENTROIDS_ENABLED, "centroids"),
        )
        for pid, what in layout_params:
            if rt.available(h, pid) and rt.get_param(h, pid):
                if not rt.writable(h, pid):
                    raise ValueError(f"{what} is on and read-only: frame slots would not be H*W*bytes")
                rt.set_param(h, pid, False)
                if rt.get_param(h, pid):
                    raise ValueError(f"{what} could not be switched off")
        # exposure resolution: microseconds when offered
        exp_res = pv.EXP_RES_ONE_MILLISEC
        if rt.available(h, pv.PARAM_EXP_RES):
            if rt.writable(h, pv.PARAM_EXP_RES_INDEX):
                for idx, (v, _) in enumerate(rt.enum_values(h, pv.PARAM_EXP_RES)):
                    if v == pv.EXP_RES_ONE_MICROSEC:
                        rt.set_param(h, pv.PARAM_EXP_RES_INDEX, idx)
                        break
            exp_res = int(rt.get_param(h, pv.PARAM_EXP_RES))
        unit_us = {pv.EXP_RES_ONE_MICROSEC: 1.0, pv.EXP_RES_ONE_MILLISEC: 1e3, pv.EXP_RES_ONE_SEC: 1e6}[exp_res]
        if exposure_us is None:
            lo = self._get(pv.PARAM_EXPOSURE_TIME, pv.ATTR_MIN)
            exposure = max(int(lo), 1) if lo is not None else max(int(round(1000.0 / unit_us)), 1)
        else:
            exposure = max(int(round(float(exposure_us) / unit_us)), 1)
        if exp_mode is None:
            exp_mode = pv.TIMED_MODE
            if rt.available(h, pv.PARAM_EXPOSURE_MODE):
                modes = [v for v, _ in rt.enum_values(h, pv.PARAM_EXPOSURE_MODE)]
                if pv.EXT_TRIG_INTERNAL in modes:
                    exp_mode = pv.EXT_TRIG_INTERNAL | int(self._get(pv.PARAM_EXPOSE_OUT_MODE, default=0) or 0)
                elif modes and pv.TIMED_MODE not in modes:
                    exp_mode = modes[0]
        ser, par = int(rt.get_param(h, pv.PARAM_SER_SIZE)), int(rt.get_param(h, pv.PARAM_PAR_SIZE))
        x, y, w, hh = roi if roi is not None else (0, 0, ser, par)
        sbin, pbin = (int(binning[0]), int(binning[1]))
        if x < 0 or y < 0 or w < 1 or hh < 1 or x + w > ser or y + hh > par:
            raise ValueError(f"roi {roi} outside the {ser} x {par} sensor")
        region = pv.Region(x, x + w - 1, sbin, y, y + hh - 1, pbin)
        width, height = w // sbin, hh // pbin
        frame_bytes = int(rt.setup_cont(h, [region], exp_mode, exposure, self._buffer_mode))
        # after setup: host-side format and depth (pvcam.h: rely on the _HOST parameters for the delivered data)
        bits = self._get(pv.PARAM_BIT_DEPTH_HOST, default=self._get(pv.PARAM_BIT_DEPTH, default=16))
        fmt = self._get(pv.PARAM_IMAGE_FORMAT_HOST, default=self._get(pv.PARAM_IMAGE_FORMAT))
        if fmt is None:
            dtype = np.dtype(np.uint8 if int(bits) <= 8 else np.uint16)
        elif fmt == pv.PL_IMAGE_FORMAT_MONO8:
            dtype = np.dtype(np.uint8)
        elif fmt == pv.PL_IMAGE_FORMAT_MONO16:
            dtype = np.dtype(np.uint16)
        else:
            raise ValueError(f"host image format {fmt} is not MONO8/MONO16; b4d takes uint8/uint16 frames")
        expect = width * height * dtype.itemsize
        if frame_bytes != expect:
            raise ValueError(
                f"pl_exp_setup_cont reports {frame_bytes} bytes per frame, {width} x {height} x {dtype.itemsize} = "
                f"{expect}: the frame carries extra data (metadata, compression or several ROIs) and slots would not "
                "be plain frames"
            )
        ring_bytes = self.ring_frames * frame_bytes
        if ring_bytes > pv.MAX_CONT_BUFFER_BYTES:
            raise ValueError(
                f"{self.ring_frames} frames x {frame_bytes} B = {ring_bytes} B: pl_exp_start_cont takes an uns32 size, "
                f"so the circular buffer holds at most {pv.MAX_CONT_BUFFER_BYTES // frame_bytes} frames of this size; "
                "reduce frames_per_block * nblocks"
            )
        readout_us = self._get(pv.PARAM_READOUT_TIME)
        exposure_us_actual = exposure * unit_us
        if fps:
            fps_nominal, fps_from = float(fps), "argument"
        else:
            period = max(float(readout_us or 0), exposure_us_actual)
            fps_nominal, fps_from = 1e6 / period, "max(readout time, exposure)"
        port_v = self._get(pv.PARAM_READOUT_PORT)
        product = self._get(pv.PARAM_PRODUCT_NAME, default="") or ""
        chip = self._get(pv.PARAM_CHIP_NAME, default="") or ""
        vendor = self._get(pv.PARAM_VENDOR_NAME, default="Teledyne Photometrics") or "Teledyne Photometrics"
        if model is None:
            for (sub, mport, mbits), m in CAMERA_MODELS.items():
                if sub in f"{product} {chip}".lower() and port_v == mport and int(bits) == mbits:
                    model = m
                    break
        try:
            version = rt.version()
        except Exception:  # pragma: no cover - informational only
            version = ""
        self._dtype, self._frame_bytes, self._shape = dtype, frame_bytes, (height, width)
        self.info = SourceInfo(
            name=f"{vendor} {product or chip or cam_name}".strip(),
            serial=cam_serial,
            shape=(height, width),
            dtype=dtype,
            fps=float(fps_nominal),
            frame_bytes=frame_bytes,
            camera_model=model,
            extra=dict(
                pvcam_version=version,
                camera_name=cam_name,
                product=product,
                chip=chip,
                firmware=self._get(pv.PARAM_CAM_FW_VERSION),
                readout_port=port_v,
                readout_port_name=self._enum_name(pv.PARAM_READOUT_PORT, port_v),
                speed_index=self._get(pv.PARAM_SPDTAB_INDEX),
                speed_name=self._get(pv.PARAM_SPDTAB_NAME),
                gain_index=self._get(pv.PARAM_GAIN_INDEX),
                gain_name=self._get(pv.PARAM_GAIN_NAME),
                pix_time_ns=self._get(pv.PARAM_PIX_TIME),
                bit_depth=int(bits),
                image_format=fmt,
                region=tuple(region),
                exp_mode=int(exp_mode),
                exposure=exposure,
                exposure_us=exposure_us_actual,
                readout_time_us=readout_us,
                fps_from=fps_from,
                buffer_mode=buffer_mode,
                metadata_enabled=False,
                frame_buffer_size_recommended=self._get(pv.PARAM_FRAME_BUFFER_SIZE, pv.ATTR_DEFAULT),
                ring_frames=self.ring_frames,
                ring_bytes=ring_bytes,
                timestamp_unit_ns=self._ts_unit,
            ),
        )

    def _check_buffers(self, bufs):
        fb, n = self._frame_bytes, self.ring_frames
        arrs = [np.asarray(b) for b in bufs]
        for a in arrs:
            if not a.flags.c_contiguous:
                raise ValueError("caller buffers must be C-contiguous host arrays")
        if len(arrs) == 1:
            if arrs[0].nbytes != n * fb:
                raise ValueError(f"buffers=[array] must hold nblocks*frames_per_block = {n} frames ({n * fb} B)")
        elif len(arrs) == self.nblocks:
            a0 = _address(arrs[0])
            for i, a in enumerate(arrs):
                if a.nbytes != self.frames_per_block * fb or _address(a) != a0 + i * self.frames_per_block * fb:
                    raise ValueError(
                        "PVCAM takes one circular buffer (pl_exp_start_cont(hcam, buffer, size)): nblocks caller "
                        "buffers must be adjacent pieces of one contiguous allocation; pass buffers=[one_array] instead"
                    )
        else:
            raise ValueError(f"buffers must have 1 or nblocks={self.nblocks} entries")

    def _allocate(self):
        if self._ring is not None:
            return
        fb, n = self._frame_bytes, self.ring_frames
        h, w = self._shape
        if self._buffers is None:
            raw = np.empty(n * fb + PAGE, np.uint8)
            off = (-_address(raw)) % PAGE
            buf = raw[off : off + n * fb]
            if self._prefault:
                buf[::PAGE] = 0  # touch every page once, so the first DMA lap does not page-fault
            ring = buf.view(self._dtype).reshape(n, h, w)
        elif len(self._buffers) == 1:
            ring = np.asarray(self._buffers[0]).reshape(-1).view(self._dtype).reshape(n, h, w)
        else:
            a0 = _address(np.asarray(self._buffers[0]))
            mem = (ctypes.c_uint8 * (n * fb)).from_address(a0)  # the caller's allocation; self._buffers keeps it alive
            ring = np.frombuffer(mem, np.uint8).view(self._dtype).reshape(n, h, w)
        self._ring, self._base = ring, _address(ring)
        self.info.extra["ring_page_aligned"] = self._base % PAGE == 0

    # ------------------------------------------------------------ contract
    def start(self, max_frames: Optional[int] = None) -> None:
        if self._started:
            raise RuntimeError("already started")
        self._allocate()
        self._max_frames = int(max_frames) if max_frames else None
        self._rt.start_cont(self._h, self._base, self.ring_frames * self._frame_bytes)
        self._started = self._running = True
        self._thread = threading.Thread(target=self._run, name="pvcam-acquisition", daemon=True)
        self._thread.start()

    def wait_block(self, timeout_ms: int = 1000):
        if not self._started:
            raise RuntimeError("start() first")
        try:
            blk = self._ready.get(timeout=max(int(timeout_ms), 0) / 1000.0)
        except queue.Empty:
            return None
        if blk is None:
            self._ready.put(None)  # keep the sentinel for later callers
            raise StopIteration
        with self._lock:
            self._state[blk.index] = _HELD
        return blk

    def release(self, block) -> None:
        i = int(block.index)
        with self._lock:
            if self._state[i] not in (_HELD, _READY):
                raise ValueError(f"block {i} is not held (state {self._state[i]})")
            self._state[i] = _RELEASED
            self._st.blocks_released += 1
            if not self._running:  # acquisition over: nothing to unlock
                self._state[i] = _FREE
                if i in self._order:
                    self._order.remove(i)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def stats(self) -> SourceStats:
        # no PVCAM parameter reads here: reading non-live parameters during acquisition can stop a Kinetix
        with self._lock:
            s = SourceStats(**{k: v for k, v in self._st.__dict__.items() if k != "extra"}, extra=dict(self._x))
            s.free_blocks = sum(1 for st in self._state if st == _FREE)
            s.ready_blocks = self._ready.qsize() - (1 if not self._running and self._started else 0)
            s.ready_blocks = max(s.ready_blocks, 0)
            span = self._x.get("_ts_span_ns", 0)
        s.extra.pop("_ts_span_ns", None)
        s.extra.pop("_ts_first", None)
        if s.delivered > 1 and span > 0:  # camera frames per second over the frames' own timestamps
            s.fps_measured = (s.extra["last_frame_nr"] - s.extra["first_frame_nr"]) / (span / 1e9)
        return s

    def block_array(self, index: int):
        """full (frames_per_block, H, W) array of pool block index: a view of the circular buffer PVCAM writes"""
        self._allocate()
        f = self.frames_per_block
        return self._ring[int(index) * f : (int(index) + 1) * f]

    def close(self) -> None:
        self.stop()
        self._close_camera()

    # ------------------------------------------------------------ acquisition thread (owns every PVCAM call)
    def _run(self):
        rt, h = self._rt, self._h
        fpb = self.frames_per_block
        self._cur, self._filled, self._seq = None, 0, 0
        self._meta = np.zeros((self.nblocks, fpb), FRAME_DTYPE)
        self._finfo = np.zeros((self.nblocks, fpb), FRAME_INFO_DTYPE)
        self._last = None  # (address, FrameNr, block) of the previous frame
        self._done = False
        consecutive = 0
        try:
            while True:
                self._unlock_released()
                stopping = self._stop_evt.is_set()
                n, consecutive = self._drain(consecutive)
                if self._done or stopping:
                    break
                if n == 0:
                    time.sleep(self._poll)
        except Exception as e:  # PvcamError, overwrite or order errors
            with self._lock:
                self._x["error"] = str(e)
        finally:
            try:
                rt.stop_cont(h, pv.CCS_HALT)
            except Exception as e:
                with self._lock:
                    self._x["stop_error"] = str(e)
            self._finish()

    def _unlock_released(self):
        while True:
            with self._lock:
                if not self._order or self._state[self._order[0]] != _RELEASED:
                    return
                b = self._order.popleft()
                n, self._locked[b] = self._locked[b], 0
            for _ in range(n):
                self._rt.unlock_oldest_frame(self._h)
            with self._lock:
                self._x["unlocked"] += n
                self._state[b] = _FREE

    def _drain(self, consecutive):
        """take every frame PVCAM has ready; returns (frames taken, consecutive get failures)"""
        rt, h = self._rt, self._h
        n = 0
        for _ in range(self.ring_frames + 1):
            status, arrived, laps = rt.check_cont_status(h)
            with self._lock:
                self._x.update(status=status, bytes_arrived=arrived, buffer_cnt=laps)
            if status == pv.READOUT_FAILED:
                raise RuntimeError("pl_exp_check_cont_status reports READOUT_FAILED")
            if status != pv.FRAME_AVAILABLE:
                break
            try:
                addr, rec = rt.get_oldest_frame(h)
            except pv.PvcamError as e:
                consecutive += 1
                with self._lock:
                    self._x["get_frame_errors"] += 1
                    self._x["last_get_error"] = str(e)
                if self._max_get_failures and consecutive >= self._max_get_failures:
                    raise
                break
            consecutive = 0
            n += 1
            if not self._take(addr, rec):
                break
        return n, consecutive

    def _take(self, addr: int, rec) -> bool:
        """book one retrieved frame into its block; False when max_frames is reached"""
        rt, h, fpb, fb = self._rt, self._h, self.frames_per_block, self._frame_bytes
        if self._last is not None and self._last[:2] == (addr, rec.frame_nr):
            if self._x["lock_mode"] != "hold_until_release":
                raise RuntimeError(f"PVCAM returned frame {rec.frame_nr} twice after it was unlocked")
            # PVCAM hands out the next frame only after an unlock: one locked frame at a time
            rt.unlock_oldest_frame(h)
            with self._lock:
                self._x["lock_mode"] = "unlock_on_retrieve"
                self._x["unlocked"] += 1
                self._locked[self._last[2]] -= 1
            return True
        off = addr - self._base
        if off < 0 or off >= self.ring_frames * fb or off % fb:
            raise RuntimeError(f"PVCAM returned frame {rec.frame_nr} at 0x{addr:x}, not a slot of the circular buffer")
        slot, expected = off // fb, self._x["retrieved"] % self.ring_frames
        if slot != expected:
            raise RuntimeError(
                f"frame {rec.frame_nr} is in slot {slot}, the block layout expects slot {expected}: PVCAM did not fill "
                "the circular buffer in order (overwritten unretrieved frames?)"
            )
        b, i = divmod(slot, fpb)
        now = time.perf_counter_ns()
        with self._lock:
            if i == 0:
                if self._state[b] != _FREE:
                    self._st.incomplete += 1
                    self._x["overwritten_held_frames"] += 1
                    raise RuntimeError(
                        f"frame {rec.frame_nr} was written into block {b} ({self._state[b]}), whose memory the "
                        "consumer has not released: held frames were overwritten"
                    )
                self._state[b], self._cur, self._filled = _FILLING, b, 0
                self._order.append(b)
                self._meta[b] = 0
                self._finfo[b] = 0
            last_nr = self._x["last_frame_nr"]
            gap = rec.frame_nr - (last_nr if last_nr is not None else 0) - 1
            if gap > 0:
                self._st.lost += gap
            elif gap < 0:
                self._x["frame_nr_backwards"] += 1
            if self._x["first_frame_nr"] is None:
                self._x["first_frame_nr"] = rec.frame_nr
                self._x["_ts_first"] = rec.timestamp
            self._x["last_frame_nr"] = rec.frame_nr
            self._x["_ts_span_ns"] = (rec.timestamp - self._x["_ts_first"]) * self._ts_unit
            self._meta[b][i] = (max(rec.frame_nr, 0), max(rec.timestamp, 0) * self._ts_unit, now, fb, 0, i)
            self._finfo[b][i] = (rec.frame_nr, rec.timestamp, rec.timestamp_bof, rec.readout_time)
            self._x["retrieved"] += 1
            self._st.delivered += 1
            self._filled = i + 1
            hold = self._x["lock_mode"] == "hold_until_release"
            if hold:
                self._locked[b] += 1
        self._last = (addr, rec.frame_nr, b)
        if not hold:
            rt.unlock_oldest_frame(h)
            with self._lock:
                self._x["unlocked"] += 1
        if i == fpb - 1:
            self._complete(b, fpb, partial=False)
        if self._max_frames and self._st.delivered >= self._max_frames:
            self._done = True
            return False
        return True

    def _complete(self, b, n, partial):
        blk = Block(
            self.block_array(b), self._meta[b][:n].copy(), self._finfo[b][:n].copy(), self._seq, b, n, partial=partial
        )
        with self._lock:
            self._state[b] = _READY
            self._st.blocks_completed += 1
            self._cur, self._filled = None, 0
        self._seq += 1
        self._ready.put(blk)

    def _finish(self):
        if self._cur is not None and self._filled > 0:
            self._complete(self._cur, self._filled, partial=True)
        with self._lock:
            self._running = False
            for i, st in enumerate(self._state):
                if st in (_RELEASED, _FILLING):
                    self._state[i] = _FREE
                    if i in self._order:
                        self._order.remove(i)
        self._ready.put(None)

    def _close_camera(self):
        try:
            if self._h is not None:
                self._rt.close(self._h)
        finally:
            self._h = None
            if self._inited:
                self._inited = False
                self._rt.uninit()

    # ------------------------------------------------------------ extras (not part of the contract)
    @property
    def ring(self):
        """the whole circular buffer, (nblocks * frames_per_block, H, W); None before start()/block_array()"""
        return self._ring

    @property
    def runtime(self):
        return self._rt

    @property
    def hcam(self) -> Optional[int]:
        return self._h

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        i = self.info
        return (
            f"PvcamSource({i.name} SN {i.serial}, {i.shape[1]}x{i.shape[0]} {np.dtype(i.dtype).name} @ {i.fps:.2f} "
            f"fps, {self.frames_per_block}x{self.nblocks} blocks)"
        )

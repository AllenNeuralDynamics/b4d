"""TucamSource: a Tucsen camera (TUCam SDK V3) as a zero-copy *block source* for the B4D writer.

This is the touch point between the vendor SDK and b4d. It implements b4d's BlockSource contract
(``b4d/src/b4d/ingest/protocol.py``) **structurally**: nothing here imports b4d.

    from tucam_camera import TucamSource
    src = TucamSource(serial="<serial>", frames_per_block=64, nblocks=3)
    print(src.info)                   # shape, dtype, fps, frame_bytes, extra (pixel format, frame stride, ...)
    src.start(max_frames=640)         # allocates the pool, announces every frame slot to the SDK, starts
    while True:
        try:
            blk = src.wait_block(1000)
        except StopIteration:
            break
        if blk is None:
            continue
        ...                           # blk.array: (nframes, H, W) numpy view over the memory the camera wrote
        src.release(blk)              # re-queues the block's slots to the SDK
    src.close()

Zero copy
---------
The package allocates ``nblocks`` page-aligned host blocks of ``frames_per_block`` frame slots each. Every slot is
announced to the SDK as one acquisition buffer (``TucBufferAnnounce``) with the MANUAL queue policy, so the SDK
writes each frame straight into a slot of a block and hands the slot back to the host through the buffer callback;
the callback only records the frame's metadata. A block goes to ``wait_block`` when its last slot is filled. Its
slots are given back to the SDK (``TucBufferQueue``) only in ``release(block)``: while the consumer holds blocks,
the SDK has fewer buffers, and a frame the camera produces while no buffer is queued is lost (and counted, below).

Frame stride. The slot size is the SDK's required announce size (``TucBuffersGetRequiredSize``), rounded up to
``stride_align`` bytes. With chunk mode on (the default, needed for loss counting) that size exceeds
``H * W * bytes_per_pixel``: an earlier internal driver measured 128 extra bytes per frame. ``block.array`` is then a
strided ``(n, H, W)`` view with ``strides = (stride, W * bpp, bpp)``: each frame's pixels are contiguous, the frames
are not contiguous with each other (``array.flags.c_contiguous`` is False). ``info.extra["frame_stride"]`` and
``info.extra["contiguous"]`` report it. With ``chunk_mode=False`` the stride equals the frame size and blocks are
C-contiguous, but the frame counter is not delivered and losses are not counted. The pixel data is assumed to
start at the slot's first byte (``image_offset=0``) with the chunk data after it; every frame's ``imageOffset``
is checked against that, and a frame that disagrees is flagged incomplete and counted in ``extra["layout_mismatch"]``.

Pixel formats. Mono8 (uint8), Mono12 and Mono16 (uint16 container) are delivered as is: zero copy. Mono12Packed
(2 px in 3 bytes; the Aries 6504 delivers it in Speed mode, its fastest: 297 fps at 2048 x 2048) needs unpacking,
which is a copy. ``packed="refuse"`` (default) raises a ValueError at construction. ``packed="copy"``
announces a packed *staging* slot per pool slot instead; the buffer callback unpacks each
frame from its staging slot into the matching uint16 pool slot (``unpack_threads`` numpy threads, row bands; 1.7 ms
per 2048 x 2048 frame with 8 threads) before the frame is counted. The pool frames are then contiguous
(``block.array`` is C-contiguous) and hold the 12-bit values (0..4095); ``info.extra["copy"]`` and
``stats().extra["copy"]`` are True, and ``stats().extra["unpack_ms_mean"]`` / ``["unpack_ms_max"]`` report the cost.
Staging slots go back to the SDK in ``release`` with their pool block, so back-pressure and loss counting are the
same as in zero-copy mode. Layout, checked against frames from the Aries 6504: GigE-style
"Mono12Packed", p0 = b0 << 4 | b1 & 0xF, p1 = b2 << 4 | b1 >> 4 (the SDK reports it with PFNC code 0x010C0047, which
PFNC calls Mono12p, but the bit order is the GigE one; ``TucBufferCopyToMemory`` copies the packed bytes unchanged).

Camera settings. ``nodes={"SensorOperationMode": "Speed", "Speed": "HighGain", ...}``
writes GenICam nodes in the given order at construction, before the frame layout is read: sensor and gain mode, ROI
(``Width``, ``Height``, ``OffsetX``, ``OffsetY``) and anything else the camera exposes. The camera keeps them after
``close``.

Losses
------
With chunk mode on, the SDK puts the camera's frame counter in each buffer's ``frameId`` and the camera clock (us)
in ``timestamp``. A counter jump of k means k-1 frames were lost: counted as ``lost_overflow`` when the queued
buffer count had fallen to ``starve_margin`` (default 2, the earlier driver's threshold) or below after the
previous frame (back-pressure: the consumer held the blocks), else as ``lost_before_host`` (lost between the sensor
and the host while buffers were queued). ``stats().lost`` is their sum. Frames lost at the very end of a run, after
the last delivered frame, leave no counter jump and are not counted. Damaged frames (SDK status not OK, incomplete,
no image, unexpected geometry or format) stay in their slot, flagged ``incomplete`` in the frame table, and are
counted in ``stats().incomplete``.

GPU
---
GPUDirect RDMA is not available through the TUCam SDK as bound here and is not implemented: the binding's announce
call takes a host address, and nothing in the SDK binding or the earlier driver refers to device memory.
``buffers_kind="device"`` raises NotImplementedError. A consumer can still page-lock the host blocks for CUDA
(``block_array(i)``, e.g. ``cudaHostRegister``) and upload them.

Every constructor option after ``serial`` is keyword-only.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tucam as T

PAGE = 4096
#: camera timestamp ticks -> ns (the SDK reports the camera clock in us with chunk mode on, per the earlier driver)
TIMESTAMP_NS_PER_TICK = 1000

GPU_UNAVAILABLE = (
    "GPUDirect RDMA is not available through the TUCam SDK as bound here and is not implemented: the SDK's buffer "
    "announce takes host memory only. Use host buffers (buffers_kind='host') and page-lock them for the GPU."
)

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


class FrameBlock:
    """A completed block (same attributes as b4d.ingest.FrameBlock). Valid until ``release``."""

    __slots__ = ("array", "full_array", "frames", "seq", "index", "nframes", "partial")

    def __init__(self, array, frames, seq: int, index: int, nframes: int, partial: bool = False):
        self.full_array = array
        self.nframes = int(nframes)
        self.array = array[: self.nframes]
        self.frames = frames
        self.seq = int(seq)
        self.index = int(index)
        self.partial = bool(partial)

    @property
    def timestamps_ns(self):
        return self.frames["timestamp_ns"]

    def __repr__(self):
        return f"FrameBlock(seq={self.seq}, pool={self.index}, n={self.nframes}{', partial' if self.partial else ''})"


_FREE, _READY, _HELD = 0, 1, 2


def unpack_mono12packed(src: np.ndarray, dst: np.ndarray) -> None:
    """Unpack Mono12Packed bytes (``src``: uint8, ``dst.size * 3 // 2`` bytes) into ``dst`` (uint16, C-contiguous rows).

    GigE layout, 2 px in 3 bytes: p0 = b0 << 4 | b1 & 0xF, p1 = b2 << 4 | b1 >> 4 (checked on the Aries 6504).
    """
    p = src.reshape(-1, 3)
    o = dst.reshape(-1, 2)
    a, b = o[:, 0], o[:, 1]
    b1 = p[:, 1]
    np.left_shift(p[:, 0], 4, out=a, dtype=np.uint16)
    a |= b1 & 0xF
    np.left_shift(p[:, 2], 4, out=b, dtype=np.uint16)
    b |= b1 >> 4


class _Unpacker:
    """Unpack one frame in row bands on a small thread pool (numpy releases the GIL in these ufuncs)."""

    def __init__(self, hw: Tuple[int, int], threads: int):
        h, w = hw
        self._w = w
        n = max(1, min(int(threads), h))
        self._bands = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(h), n)]
        self._pool = ThreadPoolExecutor(n, thread_name_prefix="tucam-unpack") if n > 1 else None

    def __call__(self, src: np.ndarray, dst: np.ndarray) -> None:
        rb = self._w * 3 // 2
        if self._pool is None:
            unpack_mono12packed(src, dst)
            return
        futs = [self._pool.submit(unpack_mono12packed, src[r0 * rb : r1 * rb], dst[r0:r1]) for r0, r1 in self._bands]
        for f in futs:
            f.result()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


class TucamSource:
    """BlockSource over the TUCam SDK V3 (see the module docstring)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        frames_per_block: int = 64,
        nblocks: int = 3,
        fps: float = 0.0,
        exposure_us: float = 0.0,
        chunk_mode: bool = True,
        free_run: bool = True,
        stride_align: int = 64,
        image_offset: int = 0,
        starve_margin: int = 2,
        sdk_dir: Optional[str] = None,
        buffers: Optional[Sequence] = None,
        buffers_kind: str = "host",
        camera_model: Optional[Tuple[float, float, float]] = None,
        nodes: Optional[Dict[str, Any]] = None,
        packed: str = "refuse",
        unpack_threads: int = 8,
    ):
        """serial: camera serial number (None = the first camera found). fps: AcquisitionFrameRate (0 = its maximum;
        Tucsen's software leaves it near 100 Hz). exposure_us: ExposureTime in whole us (0 = leave as is).
        chunk_mode: ChunkModeActive (frame counter and timestamp per frame; needed for loss counting). free_run: reset
        TriggerMode to FreeRunning on open. stride_align: frame stride granularity in bytes. image_offset: expected
        byte offset of the pixels inside a slot. starve_margin: queued-buffer count at or below which a counter gap
        counts as back-pressure overflow. buffers: nblocks caller-owned C-contiguous host arrays of at least
        frames_per_block * frame_stride bytes each. buffers_kind: "host" ("device" raises NotImplementedError).
        nodes: GenICam node values written first, in order (sensor / gain mode, ROI, ...). packed: "refuse" or "copy"
        for packed pixel formats (Mono12Packed). unpack_threads: threads unpacking one frame with packed="copy"."""
        if buffers_kind not in ("host", "device"):
            raise ValueError("buffers_kind must be 'host' or 'device'")
        if buffers_kind == "device":
            raise NotImplementedError(GPU_UNAVAILABLE)
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        if self.frames_per_block < 1 or self.nblocks < 1:
            raise ValueError("frames_per_block and nblocks must be >= 1")
        if buffers is not None and len(buffers) != self.nblocks:
            raise ValueError(f"buffers has {len(buffers)} entries, nblocks is {nblocks}")
        if packed not in ("refuse", "copy"):
            raise ValueError("packed must be 'refuse' or 'copy'")
        self._packed_mode = packed
        self._unpack_threads = max(1, int(unpack_threads))
        self._nodes = dict(nodes or {})
        self._copy = False  # packed="copy" and the camera delivers a packed format
        self._unpacker: Optional[_Unpacker] = None
        self._buffers = list(buffers) if buffers is not None else None
        self._chunk = bool(chunk_mode)
        self._align = max(1, int(stride_align))
        self._offset = int(image_offset)
        self._margin = int(starve_margin)

        devs = T.devices(sdk_dir)
        if not devs:
            raise RuntimeError("no Tucsen camera found by the TUCam SDK")
        if serial is None:
            dev = devs[0]
        else:
            dev = next((d for d in devs if d["serialNumber"] == serial), None)
            if dev is None:
                raise RuntimeError(f"serial number {serial} not found; available: {[d['serialNumber'] for d in devs]}")
        self.cam = T.TUCam(dev["serialNumber"])
        try:
            self._configure(fps, exposure_us, free_run)
            self.info = self._read_info(dev, camera_model)
            if self._buffers is not None:
                self._check_buffers(self._buffers)
        except BaseException:
            self.cam.close()
            raise

        self._lock = threading.Condition(threading.Lock())
        self._mems: List[np.ndarray] = []  # raw uint8 memory of each block
        self._views: List[np.ndarray] = []  # (frames_per_block, H, W) strided views
        self._staging: List[np.ndarray] = []  # packed="copy": raw uint8 staging memory of each block
        self._frames: List[np.ndarray] = []
        self._handles: List[List[int]] = []
        self._slot_of: Dict[int, Tuple[int, int]] = {}
        self._cb = None
        self._registered = False
        self._started = False
        self._sdk_running = False
        self._closed = False
        self._reset_counters()

    # ------------------------------------------------------------ setup
    def _configure(self, fps: float, exposure_us: float, free_run: bool) -> None:
        cam = self.cam
        for name, value in self._nodes.items():  # first: modes change what the other nodes allow
            try:
                cam.set(name, value)
            except T.TUCamError as e:
                raise ValueError(f"nodes: cannot set {name}={value!r}: {e}") from e
        if free_run and cam.try_get("TriggerMode") not in (None, "FreeRunning"):
            cam.set("TriggerMode", "FreeRunning")  # TriggerMode persists in the camera between sessions
        if cam.try_get("ChunkModeActive") is not None and bool(cam.get("ChunkModeActive")) != self._chunk:
            cam.set("ChunkModeActive", self._chunk)
        if exposure_us > 0:
            cam.set("ExposureTime", max(1, round(exposure_us)))
        try:  # read-only in trigger modes
            cam.set("AcquisitionFrameRate", float(fps) if fps > 0 else cam.limits("AcquisitionFrameRate")[1])
        except (T.TUCamError, TypeError):
            pass

    def _read_info(self, dev: Dict[str, str], camera_model) -> SourceInfo:
        cam = self.cam
        pf = str(cam.get("PixelFormat"))
        if pf not in T.PIXEL_FORMATS:
            raise ValueError(f"pixel format {pf} is not supported (supported: Mono8, Mono12, Mono16)")
        pfnc, dt = T.PIXEL_FORMATS[pf]
        if dt is None and self._packed_mode != "copy":
            raise ValueError(
                f"pixel format {pf} is packed and needs unpacking, which is a per-frame copy; this source is zero copy "
                "by default and refuses it. Pass packed='copy' to unpack it, or select a sensor operation / gain mode "
                "that delivers Mono12 or Mono16 (on the Aries 6504, Speed mode delivers Mono12Packed and Dynamic HDR "
                "delivers Mono16), e.g. nodes={'SensorOperationMode': 'Dynamic', 'Dynamic': 'HDR'}."
            )
        self._copy = dt is None
        self._pfnc, self._dtype = pfnc, np.dtype(dt or "uint16")
        h, w = int(cam.get("Height")), int(cam.get("Width"))
        if self._copy and w % 2:
            raise ValueError(f"{pf} with an odd width ({w}) is not supported")
        self._hw = (h, w)
        self._frame_bytes = h * w * self._dtype.itemsize  # as delivered (unpacked)
        self._packed_bytes = h * w * 3 // 2 if self._copy else 0
        self._required = cam.required_size()
        align = math.lcm(self._align, self._dtype.itemsize)  # keep every frame aligned to the pixel size
        if self._copy:
            # the SDK writes packed staging slots; the pool holds contiguous unpacked frames
            need = max(self._required, self._offset + self._packed_bytes)
            self._staging_stride = -(-need // self._align) * self._align
            self._stride = self._frame_bytes
            self._pool_offset = 0
        else:
            need = max(self._required, self._offset + self._frame_bytes)
            self._staging_stride = 0
            self._stride = -(-need // align) * align
            self._pool_offset = self._offset
        fps = float(cam.try_get("AcquisitionFrameRate", 0.0) or 0.0)
        try:
            fps_max = float(cam.limits("AcquisitionFrameRate")[1])
        except (T.TUCamError, TypeError):
            fps_max = fps
        mode = cam.try_get("SensorOperationMode")
        extra = dict(
            pixel_format=pf,
            pfnc=pfnc,
            copy=self._copy,
            zero_copy=not self._copy,
            delivered_format="Mono12 in uint16 (unpacked)" if self._copy else pf,
            staging_stride=self._staging_stride,
            chunk_mode=self._chunk,
            loss_counting=self._chunk,
            required_size=self._required,
            frame_stride=self._stride,
            image_offset=self._offset,
            contiguous=self._stride == self._frame_bytes and self._pool_offset == 0,
            exposure_us=cam.try_get("ExposureTime"),
            fps_max=fps_max,
            sensor_operation_mode=mode,
            nodes=dict(self._nodes),
            gain_mode=cam.try_get(mode) if mode else None,
            trigger_mode=cam.try_get("TriggerMode"),
            system_gain_k_e_per_adu=cam.try_get("OverallSystemGainK"),
            average_black_level_adu=cam.try_get("AverageBlackLevel"),
            firmware=cam.try_get("DeviceFirmwareVersion"),
            transport=dev.get("transportLayerType", ""),
            sdk_root=str(cam.sdk.root),
            buffers="caller" if self._buffers is not None else "package",
        )
        return SourceInfo(
            name=f"{dev.get('vendorName', 'Tucsen')} {dev.get('modelName', '')}".strip(),
            serial=dev["serialNumber"],
            shape=(h, w),
            dtype=self._dtype,
            fps=fps if fps > 0 else max(fps_max, 1.0),
            frame_bytes=self._frame_bytes,
            camera_model=camera_model,
            extra=extra,
        )

    def _block_bytes(self) -> int:
        return self.frames_per_block * self._stride

    def _check_buffers(self, buffers) -> None:
        need = self._block_bytes()
        for i, b in enumerate(buffers):
            if hasattr(b, "__cuda_array_interface__"):
                raise TypeError(f"buffers[{i}] is device memory; {GPU_UNAVAILABLE}")
            a = np.asarray(b)
            if not a.flags.c_contiguous or not a.flags.writeable:
                raise ValueError(f"buffers[{i}] must be a writable C-contiguous host array")
            if a.nbytes < need:
                raise ValueError(
                    f"buffers[{i}] has {a.nbytes} bytes; a block needs frames_per_block * frame_stride = "
                    f"{self.frames_per_block} * {self._stride} = {need} bytes (the stride includes the SDK's chunk "
                    "data; see info.extra['frame_stride'])"
                )
            if a.ctypes.data % self._dtype.itemsize:
                raise ValueError(f"buffers[{i}] is not aligned to the pixel size")

    def _alloc_pool(self) -> None:
        h, w = self._hw
        bpp = self._dtype.itemsize
        nbytes = self._block_bytes()
        for k in range(self.nblocks):
            if self._buffers is not None:
                mem = np.asarray(self._buffers[k]).reshape(-1).view(np.uint8)[:nbytes]
            else:
                raw = np.empty(nbytes + PAGE, np.uint8)
                off = (-raw.ctypes.data) % PAGE
                mem = raw[off : off + nbytes]
                mem[::PAGE] = 0  # fault every page in now, not during acquisition
            view = np.ndarray(
                (self.frames_per_block, h, w), self._dtype, buffer=mem, offset=self._pool_offset,
                strides=(self._stride, w * bpp, bpp),
            )
            if self._copy:  # packed staging slots the SDK writes; unpacked into the view by the callback
                sn = self.frames_per_block * self._staging_stride
                raw = np.empty(sn + PAGE, np.uint8)
                off = (-raw.ctypes.data) % PAGE
                st = raw[off : off + sn]
                st[::PAGE] = 0
                self._staging.append(st)
            self._mems.append(mem)
            self._views.append(view)
            self._frames.append(np.zeros(self.frames_per_block, FRAME_DTYPE))

    def _reset_counters(self) -> None:
        self._state = [_FREE] * self.nblocks
        self._filled = [0] * self.nblocks
        self._ready: collections.deque = collections.deque()
        self._seq = 0
        self._queued = 0
        self._starved = False
        self._starve_events = 0
        self._last_id = -1
        self._first = None  # (frame_id, timestamp ticks, host_ns) of the first delivered frame
        self._last = None
        self._delivered = 0
        self._incomplete = 0
        self._overflow = 0
        self._missed = 0
        self._layout_mismatch = 0
        self._out_of_order = 0
        self._unknown = 0
        self._ignored = 0
        self._queue_errors = 0
        self._unpack_s = 0.0
        self._unpack_max = 0.0
        self._unpacked = 0
        self._completed = 0
        self._released = 0
        self._done = False
        self._stopped = False
        self._max_frames = 0
        self._t_start = 0.0

    # ------------------------------------------------------------ contract
    def start(self, max_frames: Optional[int] = None) -> None:
        if self._closed:
            raise RuntimeError("closed")
        if self._started:
            raise RuntimeError("already started (one acquisition per source)")
        cam = self.cam
        pf = str(cam.get("PixelFormat"))
        hw = (int(cam.get("Height")), int(cam.get("Width")))
        if pf != self.info.extra["pixel_format"] or hw != self._hw or cam.required_size() != self._required:
            raise RuntimeError("camera settings changed the frame layout since construction; make a new TucamSource")
        self._alloc_pool()
        cam.set_manual_queue(True)
        if self._copy:
            self._unpacker = _Unpacker(self._hw, self._unpack_threads)
        for k in range(self.nblocks):
            base = (self._staging[k] if self._copy else self._mems[k]).ctypes.data
            stride = self._staging_stride if self._copy else self._stride
            hs = []
            for s in range(self.frames_per_block):
                hnd = cam.announce(base + s * stride, stride)
                hs.append(hnd)
                self._slot_of[int(hnd)] = (k, s)
            self._handles.append(hs)
        self._cb = cam.make_buffer_callback(self._on_buffer)  # keep a reference: the SDK holds a raw pointer
        cam.register_buffer_callback(self._cb)
        self._registered = True
        self._reset_counters()
        self._max_frames = int(max_frames or 0)
        # the SDK puts every announced buffer in its input pool at acquisition start (per the earlier driver)
        self._queued = self.nblocks * self.frames_per_block
        self._started = True
        self._t_start = time.perf_counter()
        self._sdk_running = True
        try:
            cam.start(self._max_frames)
        except BaseException:
            self._sdk_running = False
            raise

    # SDK grab thread: one call per delivered buffer; records metadata only
    def _on_buffer(self, _device, buffer, _ctx) -> None:
        host_ns = time.monotonic_ns()
        status, info = self.cam.buffer_info(buffer)
        if self._copy and status == self.cam.status_ok and not info.isIncomplete and bool(info.imagePresent):
            # unpack outside the lock: a staging slot is queued only while its pool block is FREE. After a partial
            # flush the SDK may still fill slots of a block that is READY or HELD now: those are not unpacked (the
            # callback below counts them as ignored)
            loc = self._slot_of.get(int(buffer) if not hasattr(buffer, "value") else int(buffer.value))
            layout_ok = (int(info.imageHeight), int(info.imageWidth), int(info.pixelFormat),
                         int(info.imageOffset)) == (*self._hw, self._pfnc, self._offset)
            if loc is not None and layout_ok and self._state[loc[0]] == _FREE and not (self._done or self._stopped):
                k, s = loc
                a = self._staging_stride * s + self._offset
                t0 = time.perf_counter()
                self._unpacker(self._staging[k][a : a + self._packed_bytes], self._views[k][s])
                dt = time.perf_counter() - t0
                self._unpack_s += dt
                self._unpack_max = max(self._unpack_max, dt)
                self._unpacked += 1
        with self._lock:
            loc = self._slot_of.get(int(buffer) if not hasattr(buffer, "value") else int(buffer.value))
            if loc is None:
                self._unknown += 1
                return
            k, s = loc
            self._queued -= 1
            if self._done or self._stopped or self._state[k] != _FREE:
                self._ignored += 1
                return
            ok = status == self.cam.status_ok and not info.isIncomplete and bool(info.imagePresent)
            if ok and (
                int(info.imageHeight) != self._hw[0]
                or int(info.imageWidth) != self._hw[1]
                or int(info.pixelFormat) != self._pfnc
                or int(info.imageOffset) != self._offset
            ):
                self._layout_mismatch += 1
                ok = False
            fid = int(info.frameId) if status == self.cam.status_ok else 0
            ts = int(info.timestamp) if status == self.cam.status_ok else 0
            if self._chunk and status == self.cam.status_ok:
                if self._last_id >= 0 and fid > self._last_id + 1:
                    gap = fid - self._last_id - 1
                    if self._starved:
                        self._overflow += gap
                    else:
                        self._missed += gap
                self._last_id = max(self._last_id, fid)
            self._starved = self._queued <= self._margin
            self._starve_events += int(self._queued == 0)
            row = self._frames[k][s]
            row["frame_id"] = fid
            row["timestamp_ns"] = ts * TIMESTAMP_NS_PER_TICK
            row["host_ns"] = host_ns
            row["size_filled"] = self._frame_bytes if ok else 0  # derived from the checked geometry, not an SDK field
            row["incomplete"] = 0 if ok else 1
            row["slot"] = s
            if s != self._filled[k]:
                self._out_of_order += 1
            self._filled[k] = max(self._filled[k], s + 1)
            self._delivered += 1
            self._incomplete += int(not ok)
            if self._first is None:
                self._first = (fid, ts, host_ns)
            self._last = (fid, ts, host_ns)
            if self._filled[k] == self.frames_per_block:
                self._complete(k)
            if self._max_frames and self._delivered + self._overflow + self._missed >= self._max_frames:
                self._flush_partial()
                self._done = True
            self._lock.notify_all()

    def _complete(self, k: int, partial: bool = False) -> None:
        self._state[k] = _READY
        self._ready.append((k, self._seq, self._filled[k], partial))
        self._seq += 1
        self._completed += 1

    def _flush_partial(self) -> None:
        for k in range(self.nblocks):
            if self._state[k] == _FREE and self._filled[k] > 0:
                self._complete(k, partial=True)

    def wait_block(self, timeout_ms: int = 1000):
        if not self._started:
            raise RuntimeError("start() first")
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        with self._lock:
            while not self._ready:
                if self._done or self._stopped:
                    raise StopIteration
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._lock.wait(left)
            k, seq, n, partial = self._ready.popleft()
            self._state[k] = _HELD
            frames = self._frames[k][:n].copy()  # metadata only; the consumer may keep it after release
        return FrameBlock(self._views[k], frames, seq, k, n, partial)

    def release(self, block) -> None:
        k = int(block.index)
        with self._lock:
            if self._state[k] != _HELD:
                raise ValueError(f"block {k} is not held by the consumer")
            self._state[k] = _FREE
            self._filled[k] = 0
            self._frames[k][:] = 0
            self._released += 1
            requeue = self._sdk_running and not self._done and not self._stopped
            if requeue:
                self._queued += self.frames_per_block
        if requeue:  # outside the lock: the SDK may call back into _on_buffer from its own thread
            ok = self.cam.status_ok
            errors = sum(int(self.cam.queue(hnd) != ok) for hnd in self._handles[k])
            if errors:
                with self._lock:
                    self._queue_errors += errors
                    self._queued -= errors
                raise T.TUCamError(f"TucBufferQueue (block {k}, {errors} slots)", -1)

    def stop(self) -> None:
        if not self._started:
            return
        if self._sdk_running:
            self._sdk_running = False
            try:
                self.cam.stop()
            except T.TUCamError:
                pass  # already stopped by itself after max_frames
        with self._lock:
            if not self._done:
                self._flush_partial()
            self._stopped = True
            self._lock.notify_all()

    def stats(self) -> SourceStats:
        with self._lock:
            fps = 0.0
            if self._first is not None and self._last is not None:
                (i0, t0, h0), (i1, t1, h1) = self._first, self._last
                if self._chunk and i1 > i0 and t1 > t0:
                    fps = (i1 - i0) * 1e9 / ((t1 - t0) * TIMESTAMP_NS_PER_TICK)
                elif self._delivered > 1 and h1 > h0:
                    fps = (self._delivered - 1) * 1e9 / (h1 - h0)
            return SourceStats(
                delivered=self._delivered,
                lost=self._overflow + self._missed,
                incomplete=self._incomplete,
                blocks_completed=self._completed,
                blocks_released=self._released,
                free_blocks=sum(1 for k in range(self.nblocks) if self._state[k] == _FREE and self._filled[k] == 0),
                ready_blocks=len(self._ready),
                fps_measured=fps,
                extra=dict(
                    lost_overflow=self._overflow,
                    lost_before_host=self._missed,
                    loss_counting=self._chunk,
                    pool_empty_events=self._starve_events,
                    queued_buffers=self._queued,
                    layout_mismatch=self._layout_mismatch,
                    out_of_order=self._out_of_order,
                    unknown_buffers=self._unknown,
                    ignored=self._ignored,
                    queue_errors=self._queue_errors,
                    copy=self._copy,
                    unpacked=self._unpacked,
                    unpack_ms_mean=(self._unpack_s / self._unpacked * 1e3) if self._unpacked else 0.0,
                    unpack_ms_max=self._unpack_max * 1e3,
                    frame_stride=self._stride,
                    seconds_running=(time.perf_counter() - self._t_start) if self._started else 0.0,
                ),
            )

    def block_array(self, index: int):
        """full (frames_per_block, H, W) view of pool block index (strided when the frame stride exceeds the frame)"""
        if not self._views:
            raise RuntimeError("start() first")
        return self._views[index]

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        cam = self.cam
        try:
            if self._registered:
                cam.unregister_buffer_callback()
                self._registered = False
            for hs in self._handles:
                for hnd in hs:
                    cam.revoke(hnd)
            if self._handles:
                cam.set_manual_queue(False)
        finally:
            self._handles.clear()
            self._slot_of.clear()
            self._views.clear()
            self._mems.clear()
            self._staging.clear()
            if self._unpacker is not None:
                self._unpacker.close()
                self._unpacker = None
            self._closed = True
            cam.close()

    # ------------------------------------------------------------ vendor extras (not part of the contract)
    @property
    def camera(self) -> T.TUCam:
        """the open camera (GenICam node access: get/set/limits/entries/execute)"""
        return self.cam

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        i = self.info
        return (
            f"TucamSource({i.name} SN {i.serial}, {i.shape[1]}x{i.shape[0]} {np.dtype(i.dtype).name} "
            f"{i.extra['pixel_format']} @ {i.fps:.3f} fps, stride {self._stride} B, "
            f"{self.frames_per_block}x{self.nblocks} blocks)"
        )

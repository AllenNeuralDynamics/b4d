"""ctypes binding for egrabber_ingest.dll: zero-copy camera ingest into a pool of pinned block buffers.

    from egrabber_camera import Ingest
    with Ingest("<serial>", frames_per_block=64, nblocks=3) as cam:
        print(cam.info)                      # CameraInfo(model=..., width=19200, height=12800, fps=12.438, ...)
        cam.start(max_frames=64 * 10)
        while True:
            blk = cam.wait_block(timeout_ms=2000)   # None on timeout; raises StopIteration when stopped and drained
            if blk is None: continue
            frames = blk.array                       # numpy view (nframes, H, W) uint16 over the DMA memory - no copy
            ...
            cam.release(blk)                         # slots back to the grabber
        cam.stop()

Block memory can also be supplied by the caller (`ext_blocks=[...]`, `ext_kind=1` host / `2` NVIDIA device memory):
host memory for the GPU-staged path (the same pages are DMA targets for the grabber and the GPU), device memory for
GPUDirect RDMA (Euresys supports NvidiaRdmaMemory on Linux only; on Windows the announce is refused).

The numpy view is only valid until release(); the consumer must have finished reading it.
"""

from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
DLL_NAME = "egrabber_ingest.dll"
#: environment variable naming the DLL explicitly (looked at after the package and the checkout's bin/)
DLL_ENV = "EGRABBER_INGEST_DLL"

EG_OK, EG_TIMEOUT, EG_ERR, EG_ERR_ARG = 0, 1, -1, -2
EG_ERR_STATE, EG_ERR_NOCAM, EG_ERR_ALLOC, EG_ERR_ORDER = -3, -4, -5, -6


class _Config(C.Structure):
    _fields_ = [
        ("serial", C.c_char_p),
        ("pixel_format", C.c_char_p),
        ("fps", C.c_double),
        ("exposure_us", C.c_double),
        ("frames_per_block", C.c_int),
        ("nblocks", C.c_int),
        ("large_pages", C.c_int),
        ("prefault_threads", C.c_int),
        ("max_frames", C.c_uint64),
        ("verbose", C.c_int),
        ("ext_blocks", C.POINTER(C.c_void_p)),
        ("ext_kind", C.c_int),
        ("attach_only", C.c_int),
    ]


class _CameraInfo(C.Structure):
    _fields_ = [
        ("vendor", C.c_char * 64),
        ("model", C.c_char * 64),
        ("serial", C.c_char * 64),
        ("version", C.c_char * 64),
        ("pixel_format", C.c_char * 32),
        ("unpacking", C.c_char * 32),
        ("interface_id", C.c_char * 128),
        ("link_config", C.c_char * 32),
        ("width", C.c_int64),
        ("height", C.c_int64),
        ("bytes_per_pixel", C.c_int),
        ("payload_bytes", C.c_uint64),
        ("fps", C.c_double),
        ("fps_max", C.c_double),
        ("exposure_us", C.c_double),
        ("link_throughput_limit", C.c_double),
    ]


class _FrameInfo(C.Structure):
    _fields_ = [
        ("frame_id", C.c_uint64),
        ("timestamp_ns", C.c_uint64),
        ("host_ns", C.c_uint64),
        ("size_filled", C.c_uint64),
        ("incomplete", C.c_uint32),
        ("slot", C.c_uint32),
    ]


class _BlockInfo(C.Structure):
    _fields_ = [
        ("index", C.c_int),
        ("nframes", C.c_int),
        ("partial", C.c_int),
        ("base", C.c_void_p),
        ("slot_bytes", C.c_uint64),
        ("block_bytes", C.c_uint64),
        ("frames", C.POINTER(_FrameInfo)),
        ("first_frame_id", C.c_uint64),
        ("last_frame_id", C.c_uint64),
        ("seq", C.c_uint64),
    ]


class _Stats(C.Structure):
    _fields_ = [
        ("delivered", C.c_uint64),
        ("incomplete", C.c_uint64),
        ("lost_by_timestamp", C.c_uint64),
        ("lost_by_frame_id", C.c_uint64),
        ("rejected", C.c_uint64),
        ("broken", C.c_uint64),
        ("blocks_completed", C.c_uint64),
        ("blocks_released", C.c_uint64),
        ("period_ns_mean", C.c_double),
        ("fps_timestamp", C.c_double),
        ("stream_fps", C.c_double),
        ("stream_data_rate_mbs", C.c_double),
        ("seconds_running", C.c_double),
        ("t_announce_s", C.c_double),
        ("t_prefault_s", C.c_double),
        ("large_pages_used", C.c_int),
        ("ext_kind", C.c_int),
        ("ready_blocks", C.c_int),
        ("free_blocks", C.c_int),
        ("slots_queued", C.c_uint64),
        ("slots_ready", C.c_uint64),
        ("triggered", C.c_int),
        ("order_mismatch", C.c_uint64),
    ]


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
class CameraInfo:
    vendor: str
    model: str
    serial: str
    version: str
    pixel_format: str
    unpacking: str
    interface_id: str
    link_config: str
    width: int
    height: int
    bytes_per_pixel: int
    payload_bytes: int
    fps: float
    fps_max: float
    exposure_us: float
    link_throughput_limit: float

    @classmethod
    def _from(cls, c: _CameraInfo):
        d = {}
        for name, _ in _CameraInfo._fields_:
            v = getattr(c, name)
            d[name] = v.decode(errors="replace") if isinstance(v, bytes) else v
        return cls(**d)

    @property
    def shape(self):
        return (int(self.height), int(self.width))

    def asdict(self):
        return dict(self.__dict__)


@dataclass
class Stats:
    delivered: int
    incomplete: int
    lost_by_timestamp: int
    lost_by_frame_id: int
    rejected: int
    broken: int
    blocks_completed: int
    blocks_released: int
    period_ns_mean: float
    fps_timestamp: float
    stream_fps: float
    stream_data_rate_mbs: float
    seconds_running: float
    t_announce_s: float
    t_prefault_s: float
    large_pages_used: int
    ext_kind: int
    ready_blocks: int
    free_blocks: int
    slots_queued: int
    slots_ready: int
    triggered: int
    order_mismatch: int

    @property
    def lost(self) -> int:
        """frames the camera produced that never reached a slot (the larger of the independent counts). With an external
        trigger the frame spacing is not the frame period, so timestamp gaps are not counted: frame-id gaps are."""
        return max(self.lost_by_frame_id if self.triggered else self.lost_by_timestamp, self.rejected)


class Block:
    """A completed block: `array` is a (nframes, H, W) uint16 view over the DMA memory; `frames` the per-frame table.
    For device-resident blocks (ext_kind 2) `array` is the caller's CuPy array sliced to nframes."""

    def __init__(self, bi: _BlockInfo, shape, dtype, ext_arrays=None):
        self.index = int(bi.index)
        self.nframes = int(bi.nframes)
        self.partial = bool(bi.partial)
        self.seq = int(bi.seq)
        self.first_frame_id = int(bi.first_frame_id)
        self.last_frame_id = int(bi.last_frame_id)
        self.base = int(bi.base)
        self.slot_bytes = int(bi.slot_bytes)
        self.block_bytes = int(bi.block_bytes)
        n_slots = self.block_bytes // self.slot_bytes
        if ext_arrays is not None:
            self.full_array = ext_arrays[self.index]
        else:
            buf = (C.c_uint8 * self.block_bytes).from_address(self.base)
            self.full_array = np.frombuffer(buf, dtype=dtype).reshape((n_slots,) + tuple(shape))
        self.array = self.full_array[: self.nframes]
        if self.nframes:
            raw = C.string_at(bi.frames, self.nframes * C.sizeof(_FrameInfo))
            self.frames = np.frombuffer(raw, dtype=FRAME_DTYPE).copy()
        else:
            self.frames = np.zeros(0, FRAME_DTYPE)

    @property
    def timestamps_ns(self):
        return self.frames["timestamp_ns"]

    def __repr__(self):
        return (
            f"Block(seq={self.seq}, pool={self.index}, frames={self.first_frame_id}..{self.last_frame_id} "
            f"n={self.nframes}{', partial' if self.partial else ''})"
        )


class IngestError(RuntimeError):
    pass


class OrderError(IngestError):
    """a frame landed in a slot other than the predicted one (eg_next_slot)"""


def dll_candidates() -> list:
    """Where the engine DLL is looked for, in order: inside the installed package (the wheel ships it there; build.py
    copies it there for editable installs), the source checkout's ``egrabber/bin``, then ``$EGRABBER_INGEST_DLL``."""
    cands = [os.path.join(_HERE, DLL_NAME), os.path.join(_HERE, "..", "..", "bin", DLL_NAME)]
    env = os.environ.get(DLL_ENV, "").strip()
    if env:
        cands.append(env)
    return cands


def find_dll(dll: Optional[str] = None) -> str:
    cands = [dll] if dll else dll_candidates()
    for p in cands:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    raise IngestError(
        f"{DLL_NAME} not found (looked in {cands}); build it with python egrabber/build.py or set {DLL_ENV}"
    )


class Ingest:
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
        max_frames: int = 0,
        verbose: bool = False,
        dll: Optional[str] = None,
        ext_blocks: Optional[Sequence] = None,
        ext_kind: int = 0,
        attach_only: bool = False,
    ):
        """ext_blocks: nblocks caller-owned buffers (numpy arrays, CuPy arrays or anything with __array_interface__ /
        __cuda_array_interface__ / a .ptr) of >= frames_per_block * payload bytes; ext_kind 1 = host memory, 2 = device
        memory.

        Device-only use (a host application that owns the camera settings): ``attach_only=True`` leaves the
        camera exactly as configured and ``nblocks=0`` opens the device without a pool; allocate it later with
        :meth:`alloc_pool` (after the host has set ROI, binning and pixel format) and release it with :meth:`free_pool`.
        """
        self._lib = self._load(dll)
        self._ext_arrays = None
        self._ext_ptrs = None
        if ext_blocks is not None:
            if ext_kind not in (1, 2):
                raise ValueError("ext_kind must be 1 (host) or 2 (device) when ext_blocks are given")
            if len(ext_blocks) != nblocks:
                raise ValueError(f"ext_blocks has {len(ext_blocks)} entries, nblocks is {nblocks}")
            self._ext_arrays = list(ext_blocks)
            self._ext_ptrs = (C.c_void_p * nblocks)(*[_ptr_of(b) for b in ext_blocks])
        self._cfg = _Config(
            serial=(serial or "").encode(),
            pixel_format=(pixel_format or "").encode(),
            fps=float(fps),
            exposure_us=float(exposure_us),
            frames_per_block=int(frames_per_block),
            nblocks=int(nblocks),
            large_pages=1 if large_pages else 0,
            prefault_threads=int(prefault_threads),
            max_frames=int(max_frames),
            verbose=1 if verbose else 0,
            attach_only=1 if attach_only else 0,
            ext_blocks=self._ext_ptrs if self._ext_ptrs is not None else None,
            ext_kind=int(ext_kind) if ext_blocks is not None else 0,
        )
        ci = _CameraInfo()
        self._h = self._lib.eg_open(C.byref(self._cfg), C.byref(ci))
        if not self._h:
            raise IngestError(self._lib.eg_last_error(None).decode(errors="replace"))
        self.info = CameraInfo._from(ci)
        self.frames_per_block, self.nblocks = int(frames_per_block), int(nblocks)
        self.dtype = np.dtype("uint16") if self.info.bytes_per_pixel == 2 else np.dtype("uint8")
        self._pool = int(nblocks) > 0
        if self._ext_arrays is not None:  # present caller buffers as (frames, H, W) arrays of the right dtype
            self._ext_arrays = [
                _as_block_view(b, self.frames_per_block, self.info.shape, self.dtype) for b in self._ext_arrays
            ]

    # ---- loading
    @staticmethod
    def _load(dll):
        lib = C.CDLL(find_dll(dll))
        sigs = {
            # name: (restype, argtypes); eg_start (start with config.max_frames) is not bound: start() uses eg_start_n
            "eg_open": (C.c_void_p, [C.POINTER(_Config), C.POINTER(_CameraInfo)]),
            "eg_wait_block": (C.c_int, [C.c_void_p, C.c_int, C.POINTER(_BlockInfo)]),
            "eg_release_block": (C.c_int, [C.c_void_p, C.c_int]),
            "eg_block_ptr": (C.c_int, [C.c_void_p, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]),
            "eg_start_n": (C.c_int, [C.c_void_p, C.c_uint64]),
            "eg_refresh_info": (C.c_int, [C.c_void_p, C.POINTER(_CameraInfo)]),
            "eg_pool_alloc": (
                C.c_int,
                [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.c_int],
            ),
            "eg_pool_free": (C.c_int, [C.c_void_p]),
            "eg_next_slot": (C.c_int, [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            "eg_wait_slot": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(_FrameInfo)]),
            "eg_stop": (C.c_int, [C.c_void_p]),
            "eg_is_running": (C.c_int, [C.c_void_p]),
            "eg_get_stats": (C.c_int, [C.c_void_p, C.POINTER(_Stats)]),
            "eg_last_error": (C.c_char_p, [C.c_void_p]),
            "eg_close": (None, [C.c_void_p]),
            "eg_remote_get": (C.c_int, [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]),
            "eg_remote_set": (C.c_int, [C.c_void_p, C.c_char_p, C.c_char_p]),
            "eg_stream_get": (C.c_int, [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]),
        }
        for name, (restype, argtypes) in sigs.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        return lib

    def _err(self):
        return self._lib.eg_last_error(self._h).decode(errors="replace")

    def _check(self, r, what):
        if r < 0:
            raise IngestError(f"{what}: code {r}: {self._err()}")
        return r

    # ---- control
    def start(self, max_frames: Optional[int] = None):
        """Start the camera. ``max_frames``: the engine stops by itself after this many delivered frames (None = the
        constructor's value, 0 = until stop()). May be called again after stop() or after the run ended by itself, once
        every block has been released; the pool is reset and reused."""
        n = int(self._cfg.max_frames if max_frames is None else max_frames)
        self._check(self._lib.eg_start_n(self._h, n), "eg_start")

    # ---- pool and geometry (device-only use)
    def refresh_info(self) -> CameraInfo:
        """re-read size, payload, frame rate and exposure after the host changed ROI / binning / pixel format"""
        ci = _CameraInfo()
        self._check(self._lib.eg_refresh_info(self._h, C.byref(ci)), "eg_refresh_info")
        self.info = CameraInfo._from(ci)
        self.dtype = np.dtype("uint16") if self.info.bytes_per_pixel == 2 else np.dtype("uint8")
        return self.info

    @property
    def has_pool(self) -> bool:
        return self._pool

    def alloc_pool(
        self,
        frames_per_block: int,
        nblocks: int,
        *,
        large_pages: bool = False,
        prefault_threads: int = 16,
        buffers: Optional[Sequence] = None,
        buffers_kind: str = "host",
    ) -> None:
        """allocate, pre-fault, announce and queue a block pool (the payload is re-read first, so call it after the host
        has finished configuring ROI, binning and pixel format)"""
        if self._pool:
            raise IngestError("a pool is already allocated; free_pool() first")
        ptrs, ext_kind, arrays = None, 0, None
        if buffers is not None:
            if len(buffers) != nblocks:
                raise ValueError(f"buffers has {len(buffers)} entries, nblocks is {nblocks}")
            ext_kind = 1 if buffers_kind == "host" else 2
            arrays = list(buffers)
            ptrs = (C.c_void_p * nblocks)(*[_ptr_of(b) for b in buffers])
        self.refresh_info()
        r = self._lib.eg_pool_alloc(
            self._h, int(frames_per_block), int(nblocks), 1 if large_pages else 0, int(prefault_threads), ptrs, ext_kind
        )
        self._check(r, "eg_pool_alloc")
        self.frames_per_block, self.nblocks, self._pool = int(frames_per_block), int(nblocks), True
        self._ext_ptrs = ptrs
        self._ext_arrays = (
            [_as_block_view(b, self.frames_per_block, self.info.shape, self.dtype) for b in arrays]
            if arrays is not None
            else None
        )

    def free_pool(self) -> None:
        """stop, revoke the announced buffers and free the pool; the device stays open"""
        if not self._pool:
            return
        self._check(self._lib.eg_pool_free(self._h), "eg_pool_free")
        self._pool, self._ext_arrays, self._ext_ptrs = False, None, None
        self.__dict__.pop("_views", None)
        self.nblocks = 0

    def wait_block(self, timeout_ms: int = 1000) -> Optional[Block]:
        bi = _BlockInfo()
        r = self._lib.eg_wait_block(self._h, int(timeout_ms), C.byref(bi))
        if r == EG_TIMEOUT:
            return None
        if r == EG_ERR_STATE:
            raise StopIteration
        self._check(r, "eg_wait_block")
        return Block(bi, self.info.shape, self.dtype, self._ext_arrays)

    def release(self, blk: Block):
        self._check(self._lib.eg_release_block(self._h, int(blk.index)), "eg_release_block")

    # ---- frame-at-a-time, zero copy: the destination of the next frame is known before it arrives
    def next_slot(self):
        """(block index, slot) that the grabber will fill next (the head of its input FIFO, mirrored by the engine)"""
        b, s = C.c_int(), C.c_int()
        r = self._lib.eg_next_slot(self._h, C.byref(b), C.byref(s))
        if r == EG_ERR_ORDER:
            raise OrderError(
                "slot order unknown after a partial block was released; it is reset by the next start() with no "
                "block held"
            )
        self._check(r, "eg_next_slot")
        return int(b.value), int(s.value)

    def slot_view(self, block: int, slot: int):
        """the (H, W) array of one slot of a pool block (a view over the DMA memory; cached per block)"""
        return self._block_view(block)[slot]

    def wait_slot(self, block: int, slot: int, timeout_ms: int = 1000):
        """block until that slot holds a frame; returns its frame record (FRAME_DTYPE) or None on timeout.
        Raises OrderError if a frame landed in another slot first, StopIteration when stopped and drained."""
        fi = _FrameInfo()
        r = self._lib.eg_wait_slot(self._h, int(block), int(slot), int(timeout_ms), C.byref(fi))
        if r == EG_TIMEOUT:
            return None
        if r == EG_ERR_ORDER:
            raise OrderError("a frame landed in a slot other than the predicted one")
        if r == EG_ERR_STATE:
            raise StopIteration
        self._check(r, "eg_wait_slot")
        return np.frombuffer(C.string_at(C.addressof(fi), C.sizeof(_FrameInfo)), dtype=FRAME_DTYPE).copy()[0]

    def _block_view(self, index: int):
        cache = self.__dict__.setdefault("_views", {})
        v = cache.get(index)
        if v is None:
            v = cache[index] = self.block_array(index)
        return v

    def block_array(self, index: int):
        """the full (frames_per_block, H, W) array of a pool block (numpy view, or the caller's device array)"""
        if self._ext_arrays is not None:
            return self._ext_arrays[index]
        base = C.c_void_p()
        nbytes = C.c_uint64()
        self._check(self._lib.eg_block_ptr(self._h, int(index), C.byref(base), C.byref(nbytes)), "eg_block_ptr")
        buf = (C.c_uint8 * nbytes.value).from_address(base.value)
        return np.frombuffer(buf, dtype=self.dtype).reshape((self.frames_per_block,) + self.info.shape)

    def stop(self):
        if self._h:
            self._check(self._lib.eg_stop(self._h), "eg_stop")

    @property
    def running(self) -> bool:
        return bool(self._lib.eg_is_running(self._h))

    def stats(self) -> Stats:
        s = _Stats()
        self._check(self._lib.eg_get_stats(self._h, C.byref(s)), "eg_get_stats")
        return Stats(**{n: getattr(s, n) for n, _ in _Stats._fields_})

    def remote_get(self, feature: str) -> str:
        buf = C.create_string_buffer(8192)
        self._check(self._lib.eg_remote_get(self._h, feature.encode(), buf, 8192), f"get {feature}")
        return buf.value.decode(errors="replace")

    def remote_set(self, feature: str, value):
        self._check(self._lib.eg_remote_set(self._h, feature.encode(), str(value).encode()), f"set {feature}")

    def stream_get(self, feature: str) -> str:
        buf = C.create_string_buffer(8192)
        self._check(self._lib.eg_stream_get(self._h, feature.encode(), buf, 8192), f"stream get {feature}")
        return buf.value.decode(errors="replace")

    def close(self):
        if getattr(self, "_h", None):
            self._lib.eg_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _ptr_of(b) -> int:
    if hasattr(b, "__cuda_array_interface__"):
        return int(b.__cuda_array_interface__["data"][0])
    if hasattr(b, "__array_interface__"):
        return int(b.__array_interface__["data"][0])
    if hasattr(b, "ptr"):
        return int(b.ptr)
    if hasattr(b, "ctypes"):
        return int(b.ctypes.data)
    raise TypeError(f"cannot take the address of {type(b)}")


def _as_block_view(b, frames, shape, dtype):
    """present a caller buffer as a (frames, H, W) array of dtype (numpy for host memory, CuPy for device memory)"""
    n = frames * int(np.prod(shape))
    full = (frames,) + tuple(shape)
    if hasattr(b, "__cuda_array_interface__"):
        import cupy as cp

        a = cp.asarray(b)
        if a.dtype == dtype and a.shape == full:
            return a
        return a.view(cp.uint8).ravel()[: n * dtype.itemsize].view(dtype).reshape(full)
    if hasattr(b, "__array_interface__"):
        a = np.asarray(b)
        if a.dtype == dtype and a.shape == full:
            return a
        return a.view(np.uint8).ravel()[: n * dtype.itemsize].view(dtype).reshape(full)
    # CuPy PinnedMemoryPointer or similar: wrap through a ctypes buffer
    buf = (C.c_uint8 * (n * dtype.itemsize)).from_address(_ptr_of(b))
    return np.frombuffer(buf, dtype=dtype).reshape(full)

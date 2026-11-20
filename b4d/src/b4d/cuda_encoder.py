"""Native CUDA B4D tile encoder (b4d_rice_kernel.h) for the zarr writer: one thread block per chunk, output
byte-identical to the C codec, compacted on the device. Needs CuPy (pip install cupy-cuda12x). The same
source can run through a CPU emulation (backend="emu", compiled with gcc at first use) for tests.

    from b4d.cuda_encoder import CudaTileEncoder
    enc = CudaTileEncoder(cam, device=0, batch_tiles=256)     # defaults as b4d.compress: q=1, mode=1, above3, rice
    arr.encoder = enc.encode_tiles

Progress and device-memory messages go to the "b4d.cuda_encoder" logger.

Scope: uint16; mode 2 (and 0) with the 'above' or 'mean3d' predictor, or mode 1 with 'above3' (closed loop, residuals
from b4d_resid_m1_tiles); Rice coder on the device or ANS on the host; tiles of <= 262144 elements (e.g. 64^3 or
32x64x64). Tiles the kernel flags (all-zero, repeated byte, incompressible, a mode-1 residual outside int16) are
finished on the host. The per-pixel calibration (format v7) runs on the C encoder only. dither=True (format v8, mode 2
'above' or mode 1 'above3', Rice, tile_copy): each tile's seed is reduced on the device, so the bytes are the C
encoder's.

Device memory: two pipeline slots, each holding one block (block bytes), scratch (batch_tiles x 2 x tile bytes)
and an output buffer (batch_tiles x tile bytes): ~2 x (9.7 + 1 + 0.5) GB for 9.7 GB blocks of 32x64x64 tiles at
the default batch of 2048. Keep the total well below the GPU's memory: on Windows (WDDM) oversubscription is
paged to system RAM instead of failing, which slows the whole pipeline by 10x or more.
"""
from __future__ import annotations

import ctypes
import logging
import os
import queue
import struct
import threading
import time

import numpy as np

from . import hardware
from . import Camera, _MAGIC, _VERSION, _DTYPES, _meta, PREDICTORS, compress as _cpu_compress, _b4d, _anscombe

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b4d_rice_kernel.h")
_log = logging.getLogger(__name__)

# caller buffers page-locked for DMA, shared by every encoder in the process (several GPUs may upload from the same
# ring buffer): (ptr, nbytes) -> [registered ok, array kept alive, encoders using it]. Registered portable, so any
# device's context sees them as pinned; unregistered when the last encoder using one closes.
_PIN_LOCK = threading.Lock()
_PINNED = {}


def _frame_pitch(a):
    """bytes between frames of a host block whose frames are whole contiguous rows (a C-contiguous block, or a band of
    rows of one: block[:, y0:y1, :]), else None"""
    ts = a.dtype.itemsize
    if a.ndim != 3 or a.strides[2] != ts or a.strides[1] != a.shape[2] * ts or a.strides[0] < a.shape[1] * a.strides[1]:
        return None
    return a.strides[0]


def _unregister(ptr):
    try:
        import cupy as cp
        cp.cuda.runtime.hostUnregister(ptr)
    except Exception:
        pass


# Device-to-host copies into one page-locked registration of >= ~2 GB can fail at random with cudaErrorMemoryAllocation
# on Windows (WDDM, driver 553.50 observed, both GPUs; uploads from 5 GB registrations are fine), and a failed copy
# poisons the CUDA context. The encoder's output buffers are therefore registered in pieces of _PIECE bytes (pieces=True)
# and every download into them is split at the piece boundaries (_d2h).
_PIECE = 1 << 30


def _unregister_all(ptrs):
    for p in ptrs:
        _unregister(p)


def _host_pinned(nbytes, dtype=np.uint8, pieces=False):
    """a page-locked host buffer: ordinary memory, touched, then cudaHostRegister'ed (portable); unregistered when the last
    view of it is gone. Under WDDM cudaHostAlloc (CuPy alloc_pinned_memory) slows from seconds to minutes per 10 GB once
    ~20 GB is allocated (measured: 4.0, 3.9, 127, 260 s for successive 10.5 GB pieces) and CuPy's pinned pool
    keeps freed blocks, while registering existing memory stays at ~4 s per 31.5 GB up to 189 GB: the encoder's large output
    buffers (a third of a block each) stalled the 2-GPU IMX811 runs. Falls back to alloc_pinned_memory.
    pieces=True: the buffer starts on a page and is registered in _PIECE-byte pieces (copies into it go through _d2h)."""
    import weakref
    import cupy as cp
    n = max(int(nbytes), 1); itemsize = np.dtype(dtype).itemsize
    if pieces and n > _PIECE:
        raw = np.zeros(n + 4096, np.uint8)                           # zeros: every page committed and touched before pinning
        off = (-raw.ctypes.data) % 4096
        buf = raw[off:off + n - n % itemsize].view(dtype)
        base = buf.ctypes.data; ptrs = []
        try:
            for p in range(0, buf.nbytes, _PIECE):
                cp.cuda.runtime.hostRegister(base + p, min(_PIECE, buf.nbytes - p), 1); ptrs.append(base + p)
        except Exception:
            _unregister_all(ptrs)
            return np.frombuffer(cp.cuda.alloc_pinned_memory(buf.nbytes), dtype, buf.size)
        weakref.finalize(buf, _unregister_all, ptrs)
        return buf
    buf = np.zeros(-(-n // itemsize), dtype)                         # zeros: every page committed and touched before pinning
    try:
        cp.cuda.runtime.hostRegister(buf.ctypes.data, buf.nbytes, 1)
    except Exception:
        return np.frombuffer(cp.cuda.alloc_pinned_memory(buf.nbytes), dtype, buf.size)
    weakref.finalize(buf, _unregister, buf.ctypes.data)
    return buf


def _d2h(cp, host_np, off, dev_ptr, n, stream_ptr):
    """async device-to-host copy of n bytes into host_np[off:off + n], split at the _PIECE boundaries of host_np (a
    buffer from _host_pinned(..., pieces=True) starts on a piece; for a single registration the split is harmless)"""
    base = host_np.ctypes.data
    while n > 0:
        m = min(n, _PIECE - off % _PIECE)
        cp.cuda.runtime.memcpyAsync(base + off, dev_ptr, m, cp.cuda.runtime.memcpyDeviceToHost, stream_ptr)
        off += m; dev_ptr += m; n -= m


def _out_need(ntiles, nbytes, itemsize):
    """first size of a block's page-locked output buffer: a third of the raw bytes for 16-bit tiles, half for 8-bit
    (Teledyne Kinetix 8-bit blocks compress to 0.33-0.34 of raw, just over a third, so every output buffer
    was regrown once - a new 2.5 GB registration per block for the first blocks, the GPU startup transient). Blocks
    that still outgrow it raise the encoder's per-geometry hint (_out_hint), so each pool buffer is replaced at most once."""
    return max(ntiles * nbytes // (2 if itemsize == 1 else 3), 64 << 20)


def _span(a):
    """(address, bytes) of the memory a host block (C-contiguous, or a band of frame rows) touches"""
    if a.flags["C_CONTIGUOUS"]:
        return a.ctypes.data, a.nbytes
    return a.ctypes.data, (a.shape[0] - 1) * a.strides[0] + a.shape[1] * a.strides[1]


_NS = 8   # Rice streams per chunk, as B4DK_NS in b4d_rice_kernel.h and RICE_NS in b4d_blosc2.c (format v10 / v11)


class _Params(ctypes.Structure):
    _fields_ = [("bz", ctypes.c_int), ("by", ctypes.c_int), ("bx", ctypes.c_int), ("ty", ctypes.c_int), ("tx", ctypes.c_int),
                ("cz", ctypes.c_int), ("cy", ctypes.c_int), ("rows", ctypes.c_int), ("cols", ctypes.c_int), ("nelem", ctypes.c_int),
                ("ngroups", ctypes.c_int), ("T", ctypes.c_int), ("mode", ctypes.c_int), ("qi", ctypes.c_int), ("pred", ctypes.c_int),
                ("conv", ctypes.c_float), ("offset", ctypes.c_float), ("rn", ctypes.c_float),
                ("out_cap", ctypes.c_ulonglong), ("scratch_words", ctypes.c_int), ("ntiles", ctypes.c_int), ("ts", ctypes.c_int),
                ("vst_bytes", ctypes.c_int), ("vst", ctypes.c_void_p)]


def _lut32(conv, offset, rn):
    """mode 1 stabilised samples in fixed point: rnd(T(x) * 256), as lut32_get in b4d_blosc2.c (double, round half even)"""
    conv, offset, rn = float(np.float32(conv)), float(np.float32(offset)), float(np.float32(rn))
    t = _anscombe(np.arange(65536, dtype=np.float64), conv, offset, rn * rn + 0.375)
    return np.rint(t * 256).astype(np.int32)


def _lutd(meta, conv, offset, rn):
    """mode 2 dither table: rnd(T(x) / q * 256) - base, clipped at 0, as lutd_get in b4d_blosc2.c; and whether it is valid"""
    from .chunk_format import _vst_fwd
    q = (meta >> 2) * 0.25
    t = _vst_fwd(np.arange(65536, dtype=np.float64), np.float32(conv), np.float32(offset), np.float32(rn))
    base = np.rint(_vst_fwd(np.array([0.0]), np.float32(conv), np.float32(offset), np.float32(rn))[0] / q * 256)
    v = np.maximum(np.rint(t / q * 256) - base, 0).astype(np.int32)
    return v, bool(((int(v[-1]) + 256) >> 8) <= 65535)


def _lut16(meta, conv, offset, rn):
    from .chunk_format import _build_lut
    return _build_lut(meta, np.float32(conv), np.float32(offset), np.float32(rn)).astype(np.uint16)


def _vst_tables(meta, cap):
    """The exact VST's tables, as lut16_get / lut32_get / lutd_get build them from T in b4d_blosc2.c (format v9): mode 2
    rnd(T/q) - rnd(T(0)/q) clipped to 16 bits, mode 1 rnd(T * 256), dithered mode 2 rnd(T/q * 256) - base clipped at 0;
    and the description that follows the 44-byte header (u8 nk, 3 zero bytes, f32 lam_max, f32 g[nk])."""
    T = np.frombuffer(_b4d.vst_table(cap), np.float64)
    q = (meta >> 2) * 0.25
    lut32 = np.rint(T * 256).astype(np.int32)
    l16 = lutd = None; ok = True
    if q > 0:
        smin = np.rint(T[0] / q)
        l16 = np.clip(np.rint(T / q) - smin, 0, 65535).astype(np.uint16)
        base = np.rint(T[0] / q * 256)
        lutd = np.maximum(np.rint(T / q * 256) - base, 0).astype(np.int32)
        ok = ((int(lutd[-1]) + 256) >> 8) <= 65535
    lam, g = _b4d.vst_knots(cap)
    desc = struct.pack("<B3xf", len(g), lam) + struct.pack(f"<{len(g)}f", *g)
    return l16, lut32, (lutd, ok), np.frombuffer(desc, np.uint8).copy()


class _DeviceView:
    """numpy-like slicing of a CuPy block for the few host-side tile reads (fallback tiles, self-check)"""
    def __init__(self, a): self.a = a; self.shape = tuple(a.shape)
    def __getitem__(self, sl): return self.a[sl].get()


class _Slices:
    """lazily issued upload slices of one block: plan[k] = (last tile index, (device offset, host offset), width, height, event)"""
    def __init__(self, plan, issue, ahead):
        self.plan, self._issue, self.ahead, self.issued = plan, issue, max(0, int(ahead)), 0

    def issue_to(self, k):
        """issue slices up to k + ahead"""
        while self.issued < len(self.plan) and self.issued <= k + self.ahead:
            self._issue(self.issued); self.issued += 1

    def event_for(self, last_tile):
        """the event after which tiles [0, last_tile] are on the device (issuing what that needs, plus the look-ahead)"""
        k = next(i for i, p in enumerate(self.plan) if p[0] >= last_tile)
        self.issue_to(k)
        return self.plan[k][4]

    def finish(self):
        self.issue_to(len(self.plan))
        return self.plan[-1][4]


class CudaTileEncoder:
    def __init__(self, camera: Camera, *, q: float = 1.0, mode: int = 1, predictor: str = "above3", entropy: str = "rice",
                 dither: bool = False, device: int = 0, backend: str = "cuda", batch_tiles: int = 2048, slots: int = 2,
                 check: bool = True, tile_copy: bool = True, device_blocks: bool = False, host_threads: int = 0,
                 sm_limit: int = 0):
        """q, mode, predictor, entropy, dither: as b4d.compress (defaults m1_a3). predictor: 'above3' (format v4, mode 1
        only: closed-loop residuals from b4d_resid_m1_tiles), 'above' (modes 1 and 2) or 'mean3d' (format v4), computed by
        the kernel. entropy: 'rice' (coded on the device) or 'ans' (the kernel exports the residuals,
        b4d_encode_residual_block2 codes them on host_threads CPU threads; the bytes are what the C encoder produces for
        the same settings). device: CUDA device index. backend: 'cuda' or 'emu' (the kernel source compiled for the CPU,
        for tests). sm_limit > 0 runs the kernels in persistent mode on SMs 0..sm_limit-1 only (a compute axis for scaling
        measurements; same bytes). Blocks may be uint16 or uint8 (Rice; uint8 needs tile_copy)."""
        if backend not in ("cuda", "emu"):
            raise ValueError(f"backend must be 'cuda' or 'emu', not {backend!r}")
        if predictor not in ("above", "mean3d", "above3"):
            raise ValueError("the CUDA encoder supports predictor 'above', 'mean3d' or 'above3' ('b3d' runs on the C encoder only)")
        if (predictor == "above3" and int(mode) != 1) or (int(mode) == 1 and predictor not in ("above3", "above")):
            raise ValueError("the CUDA encoder runs mode 1 with predictor 'above3' or 'above', and 'above3' in mode 1 only")
        if entropy not in ("rice", "ans"):
            raise ValueError("the CUDA encoder supports entropy 'rice' (device) or 'ans' (device residuals, host coder)")
        self.dither = bool(dither)
        if self.dither and (entropy != "rice" or backend == "emu" or not tile_copy or
                            not ((int(mode) == 2 and predictor == "above") or (int(mode) == 1 and predictor in ("above3", "above")))):
            raise ValueError("the CUDA encoder dithers mode 2 'above' or mode 1 'above3'/'above' with the Rice coder on the device (tile_copy)")
        self.predictor, self.entropy = predictor, entropy
        self.pred_id = PREDICTORS[predictor]; self.ent_id = 2 if entropy == "ans" else 1
        self.kpred = 3 if predictor == "above3" else self.pred_id   # the kernel's id: 1 above, 2 mean3d, 3 above3 (header only)
        self.m1 = int(mode) == 1                                    # closed loop: b4d_resid_m1_tiles ('above3' or 'above')
        self.host_threads = int(host_threads) or max(8, hardware.logical_cpus() // 2)
        self.sm_limit = int(sm_limit)
        self.device_blocks = bool(device_blocks)                  # blocks arrive as CuPy arrays (no upload, no per-slot block buffer)
        # tile_copy: re-arrange each batch of tiles into a contiguous device buffer before the kernel (0.5 GB per slot at
        # 2048 x 32x64x64; ~0.03 s per block). In-place addressing saves that memory but triples the kernel's GPU time
        # because the 'above' row of every element is a block row (28 KB) away instead of adjacent.
        self.tile_copy = bool(tile_copy)
        self.camera, self.q, self.mode = camera, float(q), int(mode)
        self.meta = _meta(self.mode, self.q)
        self.backend = backend
        self.device = int(device)
        self.batch_tiles = int(batch_tiles)
        self.conv, self.offset, self.rn = float(np.float32(camera.conversion)), float(np.float32(camera.offset)), float(np.float32(camera.read_noise))
        # exact VST (format v9): the tables come from the camera's exact curve and every chunk carries its description
        from . import _vst_of
        self.vst_cap = _vst_of(camera) if (self.meta & 3) else None
        if self.vst_cap is not None and entropy == "ans":
            raise ValueError("the exact VST runs with the Rice coder on the device (entropy='rice'), or on the C encoder")
        self._vtabs = _vst_tables(self.meta, self.vst_cap) if self.vst_cap is not None else None
        self.vst_np = self._vtabs[3] if self._vtabs else np.zeros(1, np.uint8)
        if self._vtabs:
            self.lut_np = self._vtabs[0] if (self.meta & 3) == 2 else np.zeros(65536, np.uint16)
        else:
            self.lut_np = _lut16(self.meta, self.conv, self.offset, self.rn) if (self.meta & 3) == 2 else np.zeros(65536, np.uint16)
        self.stats = dict(blocks=0, upload_s=0.0, kernel_s=0.0, download_s=0.0, host_s=0.0, host_tiles=0)
        # overlap_upload: a registered (page-locked) block is uploaded in slices on a second stream, and each batch of tiles
        # starts as soon as the slices holding its rows have arrived, so the PCIe copy overlaps the kernels of the same
        # block (WDDM runs a copy and a kernel concurrently on separate streams). False: whole-block upload.
        self.overlap_upload = True
        self.upload_slices = 64                                     # IMX411 block, 84 SMs: 16/1 16.5, 32/2 16.6, 64/4 17.8, 128/6 17.6 GB/s (resident 19.9)
        self.upload_ahead = 4                                       # slices queued beyond the one the next batch needs
        # overlap_download: two sets of batch output buffers; batch k's compressed bytes go to the host on a third stream
        # while batch k+1's kernels run (the copy was serial: 0.125 of 1.01 s per IMX411 block). False: one set, serial copy.
        self.overlap_download = True
        # out_pool: the block's result is a view of a page-locked buffer from a small pool instead of a copy of the slot's
        # buffer (that copy, into freshly allocated memory, was ~0.15-0.2 s per IMX411 block with the slot held). A pool
        # buffer is reused only once nothing references the previous result (weakref); past out_pool_max buffers the
        # result is copied as before. False: always copied.
        self.out_pool = True
        self.out_pool_max = 6
        self._hpool = []
        self._out_hint = {}                                           # (tiles, tile bytes) -> output bytes a block needed x 1.15
        self._pool_lock = threading.Lock()
        self._compiled = {}; self._registered = {}; self._registered_keys = {}; self.check = check
        self._compile_lock = threading.Lock()                      # two slot threads reach the first block together
        self._kern_lock = threading.Lock()                         # persistent mode: kernels of one slot at a time
        self._slots = queue.Queue()
        if backend == "emu":
            self._lib = self._build_emu()
        else:
            import cupy as cp
            _prev_dev = cp.cuda.runtime.getDevice()                  # CuPy's current device is per thread: every entry point
            cp.cuda.Device(self.device).use()                        # below runs under `with Device(self.device)`
            self.cp = cp
            self.lut_dev = cp.asarray(self.lut_np)
            self.lut32_dev = cp.asarray(self._vtabs[1] if self._vtabs else _lut32(self.conv, self.offset, self.rn)) if self.m1 else None
            self.lutd_dev = None
            self.vst_dev = cp.asarray(self.vst_np)
            self.vst_len = len(self.vst_np) if self._vtabs else 0
            if self.dither and not self.m1:
                lutd, ok = self._vtabs[2] if self._vtabs else _lutd(self.meta, self.conv, self.offset, self.rn)
                if not ok:
                    raise ValueError("dither: the stabilised range overflows 16 bits at this q (the C encoder stores such tiles raw)")
                self.lutd_dev = cp.asarray(lutd)
            props = cp.cuda.runtime.getDeviceProperties(self.device)
            self.n_sm = props["multiProcessorCount"]
            if self.sm_limit > self.n_sm:
                raise ValueError(f"sm_limit {self.sm_limit} > the device's {self.n_sm} SMs")
            self._works = {}                                     # persistent-mode work counters, one per stream: two slots run kernels concurrently
            self.nslots = max(1, int(slots))
            for _ in range(self.nslots):                         # pipeline slots: the upload of one block overlaps the kernel of another
                self._slots.put({"stream": cp.cuda.Stream(non_blocking=True)})
            free, total = cp.cuda.runtime.memGetInfo()
            self.stats["dev_mem_free_at_start_gb"] = free / 1e9; self.stats["dev_mem_total_gb"] = total / 1e9
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            _log.info("device %d %s: %.1f of %.1f GB free before allocation", self.device, name, free / 1e9, total / 1e9)
            if free < 0.5 * total:
                _log.warning("device %d: less than half of the device memory is free; other processes (stale python runs?) may be "
                             "holding it. Check nvidia-smi. Oversubscription on Windows/WDDM pages GPU memory to system RAM and "
                             "slows everything 10-30x.", self.device)
            cp.cuda.runtime.setDevice(_prev_dev)

    # ---- compilation
    def _kernel_for(self, nelem):
        """CUDA kernel compiled for a tile size class (shared memory is sized by B4DK_MAXE)"""
        maxe = 1 << max(10, int(np.ceil(np.log2(max(nelem, 1)))))
        if maxe > 262144:
            raise ValueError(f"tile of {nelem} elements exceeds the GPU encoder limit (262144); use smaller chunks or the CPU writer")
        with self._compile_lock:
            return self._kernel_for_locked(maxe)

    def _kernel_for_locked(self, maxe):
        if maxe not in self._compiled:
            cp = self.cp
            src = open(_SRC).read()
            dyn = maxe > 131072                                   # 64^3 class: ~55 KB of shared memory, above the 48 KB static limit
            opts = (f"-DB4DK_MAXE={maxe}", "--std=c++11") + (("-DB4DK_DYNSMEM",) if dyn else ())
            mod = cp.RawModule(code=src, options=opts, name_expressions=["b4d_rice_tiles", "b4d_gather_tiles", "b4d_smem_bytes", "b4d_resid_tiles",
                                                                         "b4d_resid_m1_tiles", "b4d_gather_tiles8", "b4d_seed8_tiles", "b4d_resid_d_tiles"])
            kern, gather, resid = mod.get_function("b4d_rice_tiles"), mod.get_function("b4d_gather_tiles"), mod.get_function("b4d_resid_tiles")
            self._resid_m1 = mod.get_function("b4d_resid_m1_tiles"); self._gather8 = mod.get_function("b4d_gather_tiles8")
            self._seed8 = mod.get_function("b4d_seed8_tiles"); self._resid_d = mod.get_function("b4d_resid_d_tiles")
            smem = 0
            if dyn:
                out = cp.zeros(1, cp.uint32); mod.get_function("b4d_smem_bytes")((1,), (32,), (out,)); smem = int(out.get()[0])
                kern.max_dynamic_shared_size_bytes = smem           # opt in above 48 KB (sm_7x+: up to 96-227 KB)
            self._compiled[maxe] = (kern, gather, smem, resid)
        return self._compiled[maxe]

    def _build_emu(self):
        import subprocess, tempfile
        out = os.path.join(tempfile.gettempdir(), "b4d_rice_emu.so" if os.name != "nt" else "b4d_rice_emu.dll")
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(_SRC):
            src = os.path.join(tempfile.gettempdir(), "b4d_rice_emu.c")
            with open(src, "w") as f:
                f.write("#define B4D_EMU\n#define B4DK_MAXE 262144\n#include \"%s\"\n" % _SRC.replace("\\", "/"))
            subprocess.check_call(["gcc", "-O2", "-std=gnu11", "-w", "-fPIC", "-shared", src, "-o", out])
        lib = ctypes.CDLL(out)
        lib.b4d_rice_tiles_emu.restype = None
        return lib

    # ---- the writer hook: (block, origin, order, chunks, shape) -> blobs in writer order
    def prepare(self, block_shape, chunks, dtype="uint16"):
        """allocate every slot's device/host buffers for this block and tile geometry now (first-touch and
        cudaMalloc cost ~1-2 s per 10 GB; doing it during the acquisition drops frames)"""
        if self.backend == "emu":
            return
        with self.cp.cuda.Device(self.device):
            self._prepare(block_shape, chunks, dtype)

    def _prepare(self, block_shape, chunks, dtype):
        cp = self.cp
        cz, cy, cx = chunks; nelem = cz * cy * cx; dt = np.dtype(dtype); nbytes = nelem * dt.itemsize
        tz, ty, tx = (-(-s // c) for s, c in zip(block_shape, chunks)); ntiles = tz * ty * tx
        sw = nelem // _NS + 64; bt = min(self.batch_tiles, ntiles)
        taken = [self._slots.get() for _ in range(self.nslots)]
        try:
            for slot in taken:
                with slot["stream"]:
                    if not self.device_blocks and (slot.get("dev") is None or slot["dev"].shape != tuple(block_shape) or slot["dev"].dtype != dt):
                        slot["dev"] = cp.zeros(tuple(block_shape), dt)               # zeros: touches the pages
                    if slot.get("key") != (bt, sw, nbytes):
                        out_cap = bt * (nbytes + 1024)
                        slot.update(key=(bt, sw, nbytes), scratch=cp.zeros(bt * _NS * sw, cp.uint32), out=cp.zeros(out_cap, cp.uint8),
                                    used=cp.zeros(1, cp.uint64), offs=cp.zeros(bt, cp.uint64), lens=cp.zeros(bt, cp.uint32), st=cp.zeros(bt, cp.int32), out_cap=out_cap)
                    need = max(_out_need(ntiles, nbytes, dt.itemsize), self._out_hint.get((ntiles, nbytes), 0))
                    if slot.get("hout") is None or slot["hout_np"].nbytes < need:
                        slot["hout"] = slot["hout_np"] = _host_pinned(need, pieces=True)
                    if self.tile_copy and (slot.get("tiles") is None or slot["tiles"].shape != (bt, cz * cy, cx)):
                        slot["tiles"] = cp.zeros((bt, cz * cy, cx), cp.uint16)
                    slot["stream"].synchronize()
            if self.out_pool:                                         # results in flight: one per slot plus two being written
                es = [self._pool_take(need) for _ in range(min(self.out_pool_max, self.nslots + 2))]
                for e in es:
                    if e is not None:
                        self._dma_touch(e["np"], h2d=False); self._pool_release(e)
            free, total = cp.cuda.runtime.memGetInfo()
            _log.info("device %d: %d slot(s) prepared for blocks of %.1f GB; %.1f GB device memory free", self.device, self.nslots,
                      np.prod(block_shape) * dt.itemsize / 1e9, free / 1e9)
        finally:
            for slot in taken:
                self._slots.put(slot)

    def encode_tiles(self, block, origin, order, chunks, shape):
        t0 = time.perf_counter()
        nz, ny, nx = block.shape; cz, cy, cx = chunks; z0, y0, x0 = origin; az, ay, ax = shape
        ez, ey, ex = min(nz, az - z0), min(ny, ay - y0), min(nx, ax - x0)
        tz, ty, tx = -(-ez // cz), -(-ey // cy), -(-ex // cx)
        ntiles = tz * ty * tx; nelem = cz * cy * cx
        dt = np.dtype(block.dtype)
        if dt not in (np.dtype("uint8"), np.dtype("uint16")):
            raise TypeError("the CUDA encoder takes uint8 or uint16 blocks")
        ts = dt.itemsize
        if ts == 1 and (self.ent_id == 2 or not self.tile_copy or self.backend == "emu"):
            raise ValueError("uint8 blocks need the Rice coder on the CUDA device with tile_copy")
        P = _Params(ez, ey, ex, ty, tx, cz, cy, cz * cy, cx, nelem, -(-nelem // 128), -(-(-(-nelem // 128)) // _NS), self.meta & 3, self.meta >> 2,
                    self.kpred, self.conv, self.offset, self.rn, 0, nelem // _NS + 64, 0, ts)
        if self._vtabs:                                            # emulation: the description from host memory
            P.vst_bytes = len(self.vst_np); P.vst = self.vst_np.ctypes.data
        hdr3 = struct.pack("<4sBBB", _MAGIC, _VERSION, _DTYPES[dt], 3) + struct.pack("<3Q", cz, cy, cx)
        nbytes = nelem * ts
        blobs_grid = [None] * ntiles
        if self.backend == "emu":
            src = np.ascontiguousarray(block[:ez, :ey, :ex])
            t1 = time.perf_counter(); self.stats["upload_s"] += t1 - t0
            self._run_emu(src, ntiles, P, blobs_grid, hdr3, nbytes, cz, cy, cx, tz, ty, tx)
            out = []
            for (bz, by, bx) in order:
                i, j, k = bz // cz, by // cy, bx // cx
                out.append(blobs_grid[(i * ty + j) * tx + k] if (i < tz and j < ty and k < tx) else None)
            self.stats["blocks"] += 1
            return out
        with self.cp.cuda.Device(self.device):                    # writer threads: CuPy's current device is per thread
            hbuf, g_off, g_len = self._run_cuda(block, ez, ey, ex, tz, ty, tx, cz, cy, cx, P, blobs_grid, hdr3, nbytes, t0)
        th = time.perf_counter()
        o = np.asarray(order, dtype=np.int64)                       # writer order -> tile grid index, vectorised
        gi, gj, gk = o[:, 0] // cz, o[:, 1] // cy, o[:, 2] // cx
        inside = (gi < tz) & (gj < ty) & (gk < tx)
        idx = np.where(inside, (gi * ty + gj) * tx + gk, 0)
        w_off = np.where(inside, g_off[idx], -1); w_len = np.where(inside, g_len[idx], -1)
        self.stats["host_s"] += time.perf_counter() - th
        self.stats["blocks"] += 1
        return (hbuf, np.ascontiguousarray(w_off), np.ascontiguousarray(w_len))

    def _finish_host(self, tile_np, st, hdr3, nbytes):
        """status 2: all zero -> zero-run chunk; 1/3: C encoder"""
        from .chunk_format import _zero_chunk
        if st == 2:
            chunk = _zero_chunk(nbytes, tile_np.dtype.itemsize, nbytes, self.meta)
            return hdr3 + struct.pack("<II", 1, len(chunk)) + chunk
        return _cpu_compress(tile_np, q=self.q, mode=self.mode, camera=self.camera, predictor=self.predictor, entropy=self.entropy, dither=self.dither)

    def _tile_np(self, src, i, tz, ty, tx, cz, cy, cx):
        if hasattr(src, "__cuda_array_interface__"):
            src = _DeviceView(src)
        """host copy of tile i (zero-padded at the block edge)"""
        a, r = divmod(i, ty * tx); b, k = divmod(r, tx)
        t = np.zeros((cz, cy, cx), src.dtype); sub = src[a * cz:(a + 1) * cz, b * cy:(b + 1) * cy, k * cx:(k + 1) * cx]
        t[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub; return t

    def _run_emu(self, src, ntiles, P, blobs_grid, hdr3, nbytes, cz, cy, cx, tz, ty, tx):
        lib = self._lib
        for b0 in range(0, ntiles, self.batch_tiles):
            b1 = min(b0 + self.batch_tiles, ntiles); n = b1 - b0
            P.ntiles = n; P.out_cap = n * (nbytes + 1024)
            scratch = np.zeros(n * _NS * P.scratch_words, np.uint32)
            out = np.zeros(P.out_cap, np.uint8); used = np.zeros(1, np.uint64); offs = np.zeros(n, np.uint64); lens = np.zeros(n, np.uint32); st = np.zeros(n, np.int32)
            t0 = time.perf_counter()
            lib.b4d_rice_tiles_emu(src.ctypes.data_as(ctypes.c_void_p), self.lut_np.ctypes.data_as(ctypes.c_void_p), ctypes.byref(P), ctypes.c_int(b0),
                                   scratch.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), used.ctypes.data_as(ctypes.c_void_p),
                                   offs.ctypes.data_as(ctypes.c_void_p), lens.ctypes.data_as(ctypes.c_void_p), st.ctypes.data_as(ctypes.c_void_p))
            self.stats["kernel_s"] += time.perf_counter() - t0
            self._collect(out, offs, lens, st, lambda i, b0=b0: self._tile_np(src, b0 + i, tz, ty, tx, cz, cy, cx), b0, blobs_grid, hdr3, nbytes)

    def _collect(self, out_host, offs, lens, st, tiles_for_fallback, b0, blobs_grid, hdr3, nbytes):
        t0 = time.perf_counter()
        for i in range(len(lens)):
            if st[i] == 0:
                o = int(offs[i]); blobs_grid[b0 + i] = out_host[o:o + int(lens[i])].tobytes()
            else:
                self.stats["host_tiles"] += 1
                blobs_grid[b0 + i] = self._finish_host(tiles_for_fallback(i) if callable(tiles_for_fallback) else tiles_for_fallback[i], int(st[i]), hdr3, nbytes)
        self.stats["host_s"] += time.perf_counter() - t0

    def _register_pinned(self, arr):
        """page-lock the caller's block buffer once (by address) so uploads are true DMA at PCIe speed"""
        cp = self.cp
        ptr, nb = _span(arr); key = (ptr, nb)
        ok = self._registered.get(key)
        if ok is not None:
            return ok
        with _PIN_LOCK:
            e = _PINNED.get(key)
            if e is None and not arr.flags["C_CONTIGUOUS"]:       # a band of a block registered whole (MultiGpuEncoder split="rows")
                key = next((k for k, v in _PINNED.items() if v[0] and k[0] <= ptr and ptr + nb <= k[0] + k[1]), key)
                e = _PINNED.get(key)
            if e is None:
                try:
                    cp.cuda.runtime.hostRegister(ptr, nb, 1); ok = True   # cudaHostRegisterPortable: pinned for every device
                except Exception:
                    ok = False
                # keep the array: freed memory that stays registered makes later copies from a reused address fail
                # (cudaErrorInvalidValue); the last close() unregisters
                e = _PINNED[key] = [ok, arr, 0]
            e[2] += 1
        self._registered[(ptr, nb)] = e[0]; self._registered_keys[(ptr, nb)] = key
        return e[0]

    def _dma_touch(self, host_np, h2d):
        """one DMA pass over a newly page-locked host buffer (device-to-host into it, or host-to-device from it). The
        first DMA through newly registered memory runs at ~8 GB/s instead of ~27 (measured under WDDM): without this pass
        the first blocks of a recording took 0.1-1.7 s longer and frames were dropped. Chunks never cross a _PIECE."""
        cp = self.cp
        flat = host_np.reshape(-1).view(np.uint8); nb = flat.nbytes; step = min(256 << 20, nb)
        if nb == 0:
            return
        with cp.cuda.Device(self.device):
            tmp = cp.empty(step, cp.uint8); st = cp.cuda.Stream(non_blocking=True); base = flat.ctypes.data
            kind = cp.cuda.runtime.memcpyHostToDevice if h2d else cp.cuda.runtime.memcpyDeviceToHost
            for off in range(0, nb, step):
                m = min(step, nb - off, _PIECE - off % _PIECE)
                if h2d:
                    cp.cuda.runtime.memcpyAsync(tmp.data.ptr, base + off, m, kind, st.ptr)
                else:
                    cp.cuda.runtime.memcpyAsync(base + off, tmp.data.ptr, m, kind, st.ptr)
            st.synchronize(); del tmp

    def register(self, arrays):
        """page-lock host block buffers now (each distinct buffer once; ~4 s per 31.5 GB) instead of at their first
        encode_tiles, where the registration delays that block, and run one upload over each (see _dma_touch).
        Returns the number of buffers newly page-locked."""
        if self.backend == "emu":
            return 0
        seen, n = set(), 0
        for a in arrays:
            if not isinstance(a, np.ndarray) or _frame_pitch(a) is None or _span(a) in seen:
                continue
            seen.add(_span(a)); known = _span(a) in self._registered
            with self.cp.cuda.Device(self.device):
                new = self._register_pinned(a) and not known
            if new and a.flags["C_CONTIGUOUS"]:
                self._dma_touch(a, h2d=True)
            n += int(new)
        return n

    def close(self):
        """release everything this encoder holds (also on garbage collection): the page-locked caller buffers (once no
        other encoder uses them), the slots' device and pinned buffers, the output pool and the cached kernels, then
        return the freed blocks of CuPy's pools to the driver. Results already returned stay valid (a pooled result
        keeps its own buffer alive). The encoder is unusable afterwards."""
        reg, self._registered = getattr(self, "_registered", {}), {}
        rkeys, self._registered_keys = getattr(self, "_registered_keys", {}), {}
        cp = getattr(self, "cp", None)
        with _PIN_LOCK:
            for key in (rkeys.get(k, k) for k in reg):
                e = _PINNED.get(key)
                if e is None:
                    continue
                e[2] -= 1
                if e[2] <= 0:
                    del _PINNED[key]
                    if e[0] and cp is not None:
                        try:
                            cp.cuda.runtime.hostUnregister(key[0])
                        except Exception:
                            pass
        pool = getattr(self, "_copy_pool", None)
        if pool is not None:
            pool.shutdown(wait=False); self._copy_pool = None
        if cp is None or getattr(self, "_closed", False):
            return
        self._closed = True
        with cp.cuda.Device(self.device):
            try:
                cp.cuda.Device(self.device).synchronize()
            except Exception:
                pass
            while True:                                               # idle slots (a slot in use by another thread is not touched)
                try:
                    self._slots.get_nowait().clear()
                except queue.Empty:
                    break
            with self._pool_lock:
                self._hpool = []
            self._compiled = {}; self._works = {}
            self.lut_dev = self.lut32_dev = self.lutd_dev = None
            for k in [k for k in vars(self) if k.startswith("_gather8") or k in ("_seed8", "_resid_m1", "_resid_d")]:
                setattr(self, k, None)
            import gc
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _staged_upload(self, src, dev, slot):
        """fallback when the block cannot be page-locked: multi-threaded copy into a pinned staging buffer, then DMA"""
        cp = self.cp
        from concurrent.futures import ThreadPoolExecutor
        if slot.get("stage") is None or slot["stage_np"].shape != src.shape or slot["stage_np"].dtype != src.dtype:
            slot["stage"] = _host_pinned(src.nbytes)
            slot["stage_np"] = np.frombuffer(slot["stage"], dtype=src.dtype, count=src.size).reshape(src.shape)
            slot["pool"] = ThreadPoolExecutor(16)
        n = src.shape[0]; step = -(-n // 16)
        list(slot["pool"].map(lambda i: np.copyto(slot["stage_np"][i:i + step], src[i:i + step]), range(0, n, step)))
        dev.set(slot["stage_np"], stream=slot["stream"])

    def _sliced_upload(self, src, dev, slot, tz, ty, tx, cz, cy):
        """plan the H2D copy of a registered block as ~upload_slices slices on the slot's copy stream, in tile order: each
        slice is one z slab of tiles (cz frames) times a band of tile rows (whole rows of the frame), copied as a 2-D copy
        (one row per frame). Slices are issued lazily, upload_ahead beyond the one a batch needs: queuing the whole block
        at once made each batch's small device-to-host copies wait behind the rest of the upload (GPU time 0.67 -> 1.31 s
        per IMX411 block). Returns a _Slices; the first slices are issued here."""
        cp = self.cp
        if slot.get("cstream") is None:
            slot["cstream"] = cp.cuda.Stream(non_blocking=True)
        cs = slot["cstream"]
        ez, ey, ex = src.shape; ts = src.dtype.itemsize
        frame = ey * ex * ts; row = ex * ts; spitch = _frame_pitch(src)   # device frame / host frame (a band: the block's frame)
        per_slab = max(1, -(-self.upload_slices // tz))           # bands of tile rows per z slab
        band = max(1, -(-ty // per_slab))
        evs = slot.setdefault("cevents", [])
        plan = []
        for i in range(tz):
            z0, z1 = i * cz, min((i + 1) * cz, ez)
            for j0 in range(0, ty, band):
                j1 = min(j0 + band, ty)
                y0, y1 = j0 * cy, min(j1 * cy, ey)
                off = z0 * frame + y0 * row; soff = z0 * spitch + y0 * row
                if len(plan) == len(evs):
                    evs.append(cp.cuda.Event(disable_timing=True))
                plan.append(((i * ty + j1 - 1) * tx + tx - 1, (off, soff), (y1 - y0) * row, z1 - z0, evs[len(plan)]))

        def issue(k):
            _, (off, soff), width, height, ev = plan[k]
            cp.cuda.runtime.memcpy2DAsync(dev.data.ptr + off, frame, src.ctypes.data + soff, spitch, width, height,
                                          cp.cuda.runtime.memcpyHostToDevice, cs.ptr)
            ev.record(cs)
        s = _Slices(plan, issue, self.upload_ahead)
        s.issue_to(0)
        return s

    def _grid(self, n_blocks, per_sm):
        """(grid, sm_limit, work) for a launch of n_blocks thread blocks: as is (no work counter: a null pointer the
        kernels do not read), or persistent on sm_limit SMs (the grid fills every SM; blocks on SMs >= sm_limit exit at
        once and the rest take work items from the counter)"""
        if self.sm_limit <= 0:
            return n_blocks, np.int32(0), np.uint64(0)
        st = self.cp.cuda.get_current_stream().ptr             # launches run inside their slot's `with stream:`
        w = self._works.get(st)
        if w is None:
            w = self._works[st] = self.cp.zeros(2, self.cp.uint64)   # [next work item, blocks started]
        w.fill(0)                                              # on that stream: ordered before this launch, after the previous one
        return min(n_blocks, self.n_sm * per_sm), np.int32(self.sm_limit), w

    def _gather(self, gather, dev, tiles, ez, ey, ex, ty, tx, b0, cz, cy, cx, n, stream):
        """contiguous zero-padded uint16 tiles [b0, b0 + n) of the block (uint8 blocks are widened)"""
        total = n * cz * cy * cx
        g, lim, work = self._grid((total + 255) // 256, 6)
        k = self._gather8 if dev.dtype == self.cp.uint8 else gather
        k((g,), (256,), (dev, tiles, np.int32(ez), np.int32(ey), np.int32(ex), np.int32(ty), np.int32(tx), np.int32(b0),
                         np.int32(cz), np.int32(cy), np.int32(cx), np.int32(n), lim, work), stream=stream)

    def _seeds(self, tiles, nelem, ts, n, seeds, stream):
        """v8 dither seeds of contiguous tiles [0, n) (one thread block per tile)"""
        g, lim, work = self._grid(n, 8)
        self._seed8((g,), (256,), (tiles, np.int32(nelem), np.int32(ts), np.int32(n), seeds, lim, work), stream=stream)

    def _residuals(self, resid_kernel, src_dev, contig, ez, ey, ex, ty, tx, b0, cz, cy, cx, mode, n, res, stream, bad=None, seeds=None):
        """int16 residuals of tiles [b0, b0 + n) into res: open loop (b4d_resid_tiles, one thread per element), mode 2 with
        dither (b4d_resid_d_tiles, contiguous tiles) or mode-1 above3 (b4d_resid_m1_tiles, one block per tile; bad[t] != 0
        where a residual left int16; dithered with seeds)"""
        null = np.uint64(0)
        if self.m1:
            bad.fill(0)
            nth = min(1024, -(-cx // 32) * 32)
            g, lim, work = self._grid(n, 16)
            self._resid_m1((g,), (nth,), (src_dev, self.lut32_dev, np.int32(ez), np.int32(ey), np.int32(ex), np.int32(ty), np.int32(tx),
                                          np.int32(b0), np.int32(contig), np.int32(cz), np.int32(cy), np.int32(cx),
                                          np.int32((self.meta >> 2) * 64), np.int32(n), res, bad, lim, work,
                                          seeds if seeds is not None else null, np.int32(self.predictor == "above")), stream=stream, shared_mem=8 * cx)
        elif seeds is not None:
            total_el = n * cz * cy * cx
            self._resid_d(((total_el + 255) // 256,), (256,), (src_dev, self.lutd_dev, seeds, np.int32(cz * cy * cx), np.int32(cx), np.int32(n), res),
                          stream=stream)
        else:
            total_el = n * cz * cy * cx
            resid_kernel(((total_el + 255) // 256,), (256,), (src_dev, self.lut_dev, np.int32(ez), np.int32(ey), np.int32(ex), np.int32(ty), np.int32(tx),
                                                              np.int32(b0), np.int32(contig), np.int32(cz), np.int32(cy), np.int32(cx), np.int32(mode),
                                                              np.int32(self.pred_id), np.int32(n), res), stream=stream)

    def _pool_take(self, need):
        """a page-locked output buffer from the pool that no earlier result still references, or None (pool full)"""
        import weakref  # noqa: F401
        with self._pool_lock:
            free = [e for e in self._hpool if not e["busy"] and (e["ref"] is None or e["ref"]() is None)]
            for e in sorted(free, key=lambda e: e["np"].nbytes):
                if e["np"].nbytes >= need:
                    e["busy"] = True; e["ref"] = None; return e
            if free:                                                  # a free one that is too small: replace it
                e = free[0]; self._hpool.remove(e)
            if len(self._hpool) >= self.out_pool_max:
                return None
            mem = _host_pinned(need, pieces=True)
            e = dict(mem=mem, np=np.frombuffer(mem, np.uint8, need), ref=None, busy=True)
            self._hpool.append(e)
            return e

    def _pool_give(self, e, pos):
        """the result: a view of the first pos bytes of pool buffer e; e is reusable once the view (and all views of it) are gone"""
        import weakref
        res = np.frombuffer(e["mem"], np.uint8, pos)
        with self._pool_lock:
            e["ref"] = weakref.ref(res); e["busy"] = False
        self.stats["out_pool_buffers"] = len(self._hpool)
        return res

    def _pool_release(self, e):
        with self._pool_lock:
            e["ref"] = None; e["busy"] = False

    def _copy_out(self, hout_np, pos):
        """the compressed bytes out of the slot's pinned buffer (reused by the next block), on 32 threads: a
        single-threaded copy of a block's output (GBs) took ~0.45 s of a 2 s block"""
        out = np.empty(pos, np.uint8)
        if pos < (64 << 20):
            out[:] = hout_np[:pos]; return out
        if getattr(self, "_copy_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._copy_pool = ThreadPoolExecutor(32)
        step = -(-pos // 32)
        list(self._copy_pool.map(lambda i: np.copyto(out[i:i + step], hout_np[i:min(i + step, pos)]), range(0, pos, step)))
        return out

    def _slot(self):
        """one of two pipeline slots (stream + device/host buffers); blocks until one is free"""
        return self._slots.get()

    def _run_cuda(self, block, ez, ey, ex, tz, ty, tx, cz, cy, cx, P, blobs_grid, hdr3, nbytes, t_start):
        cp = self.cp
        nelem = cz * cy * cx; ntiles = tz * ty * tx
        kernel, gather, smem, resid_kernel = self._kernel_for(nelem)
        on_device = hasattr(block, "__cuda_array_interface__")      # block already in device memory (e.g. GPUDirect RDMA): no upload
        if on_device:
            src = block if (ez, ey, ex) == tuple(block.shape) else cp.ascontiguousarray(block[:ez, :ey, :ex])
        else:
            src = block if (ez, ey, ex) == block.shape else block[:ez, :ey, :ex]
            if _frame_pitch(src) is None:                           # a band of whole frame rows (frame split) is uploaded as is
                src = np.ascontiguousarray(src)
        slot = self._slot()
        extra = []
        ready = None                                                # sliced upload: [(last tile index, event)] in tile order
        pe = None                                                   # output pool buffer (Rice path)
        try:
            stream = slot["stream"]
            with stream:
                if on_device:
                    dev = src; self.stats["upload_mode"] = "none (block resident in device memory)"
                else:
                    if slot.get("dev") is None or slot["dev"].shape != src.shape or slot["dev"].dtype != src.dtype:   # persistent per slot
                        slot["dev"] = None; cp.get_default_memory_pool().free_all_blocks()
                        slot["dev"] = cp.empty(src.shape, src.dtype)
                    dev = slot["dev"]
                    if self.overlap_upload and self._register_pinned(src):
                        ready = self._sliced_upload(src, dev, slot, tz, ty, tx, cz, cy)
                        self.stats["upload_mode"] = "pinned (registered block), sliced and overlapped with the kernels"
                    elif self._register_pinned(src):
                        # explicit async H2D from the registered buffer (then the stream sync below): dev.set() held the
                        # interpreter for the whole copy, so the other slot's thread could not launch its kernels meanwhile
                        fb = src.shape[1] * src.shape[2] * src.dtype.itemsize
                        cp.cuda.runtime.memcpy2DAsync(dev.data.ptr, fb, src.ctypes.data, _frame_pitch(src), fb, src.shape[0],
                                                      cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                        self.stats["upload_mode"] = "pinned (registered block)"
                    else:
                        self._staged_upload(src, dev, slot); self.stats["upload_mode"] = "staged (16-thread copy into pinned memory)"
                if ready is None:
                    stream.synchronize()
                elif self.ent_id == 2:                                # ANS path: whole block first (not pipelined by slice)
                    ready.finish().synchronize(); ready = None
                t1 = time.perf_counter(); self.stats["upload_s"] += t1 - t_start    # with sliced upload: issue time only
                bt = min(self.batch_tiles, ntiles)
                if slot.get("key") != (bt, P.scratch_words, nbytes):
                    out_cap = bt * (nbytes + 1024)
                    slot.update(key=(bt, P.scratch_words, nbytes), scratch=cp.empty(bt * _NS * P.scratch_words, cp.uint32), out=cp.empty(out_cap, cp.uint8),
                                used=cp.zeros(1, cp.uint64), offs=cp.zeros(bt, cp.uint64), lens=cp.zeros(bt, cp.uint32), st=cp.zeros(bt, cp.int32), out_cap=out_cap)
                scratch, out, used, offs, lens, st, out_cap = slot["scratch"], slot["out"], slot["used"], slot["offs"], slot["lens"], slot["st"], slot["out_cap"]
                if self.tile_copy and (slot.get("tiles") is None or slot["tiles"].shape != (bt, cz * cy, cx)):
                    slot["tiles"] = cp.empty((bt, cz * cy, cx), cp.uint16)
                slot.pop("padded", None)                           # (older builds kept a padded block copy here)
                need = max(_out_need(ntiles, nbytes, nbytes // nelem), self._out_hint.get((ntiles, nbytes), 0))
                pe = self._pool_take(need) if (self.out_pool and self.ent_id != 2) else None
                if pe is not None:
                    hout_np = pe["np"]
                else:
                    if slot.get("hout") is None or slot["hout_np"].nbytes < need:
                        slot["hout"] = slot["hout_np"] = _host_pinned(need, pieces=True)
                    hout_np = slot["hout_np"]
                all_off = np.zeros(ntiles, np.int64); all_len = np.zeros(ntiles, np.int64); all_st = np.zeros(ntiles, np.int64)
                pos = 0
                if self.ent_id == 2:
                    # ---- ANS: a two-stage pipeline per batch. GPU: gather + residual kernel + D2H of batch k+1 run while the
                    # host (writer thread pool, C) entropy-codes and frames batch k. Double-buffered tiles/residual buffers.
                    nb = (ntiles + bt - 1) // bt
                    # double buffers, (re)sized when a block brings a larger batch or another tile shape
                    if slot.get("tiles2") is None or slot["tiles2"].shape != (bt, cz * cy, cx): slot["tiles2"] = cp.empty((bt, cz * cy, cx), cp.uint16)
                    for key in ("res", "res2"):
                        if slot.get(key) is None or slot[key].size < bt * nelem: slot[key] = cp.empty(bt * nelem, cp.int16)
                    if slot.get("hres") is None or slot["hres_np"].size < bt * nelem:
                        slot["hres"] = _host_pinned(bt * nelem * 2); slot["hres_np"] = np.frombuffer(slot["hres"], dtype=np.int16)
                        slot["hres2"] = _host_pinned(bt * nelem * 2); slot["hres2_np"] = np.frombuffer(slot["hres2"], dtype=np.int16)
                    if slot.get("ev") is None: slot["ev"] = [cp.cuda.Event(disable_timing=True), cp.cuda.Event(disable_timing=True)]
                    if slot.get("bad2") is None or slot["bad2"][0].size < bt: slot["bad2"] = [cp.zeros(bt, cp.int32), cp.zeros(bt, cp.int32)]
                    tiles_b = [slot["tiles"], slot["tiles2"]] if self.tile_copy else [None, None]
                    res_b = [slot["res"], slot["res2"]]; hres_b = [slot["hres_np"], slot["hres2_np"]]; ev = slot["ev"]
                    gpu_t = [0.0]
                    def launch(k):
                        b0 = k * bt; n = min(bt, ntiles - b0); buf = k % 2; tk = time.perf_counter()
                        if self.tile_copy:
                            self._gather(gather, dev, tiles_b[buf], ez, ey, ex, ty, tx, b0, cz, cy, cx, n, stream)
                            sd, contig = tiles_b[buf], 1
                        else:
                            sd, contig = dev, 0
                        total_el = n * nelem
                        self._residuals(resid_kernel, sd, contig, ez, ey, ex, ty, tx, b0, cz, cy, cx, P.mode, n, res_b[buf], stream, slot["bad2"][buf])
                        cp.cuda.runtime.memcpyAsync(hres_b[buf].ctypes.data, res_b[buf].data.ptr, int(total_el * 2), cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
                        ev[buf].record(stream); gpu_t[0] += time.perf_counter() - tk
                    launch(0)
                    for k in range(nb):
                        b0 = k * bt; n = min(bt, ntiles - b0); b1 = b0 + n; buf = k % 2
                        tw = time.perf_counter(); ev[buf].synchronize(); self.stats["kernel_s"] += time.perf_counter() - tw
                        if k + 1 < nb:
                            launch(k + 1)                                                   # GPU: next batch, overlapped with the host coder
                        th = time.perf_counter()
                        blobs, boff, blen, bst = _b4d.encode_residual_tiles(hres_b[buf][:n * nelem], n, nelem, P.mode, P.qi, self.conv, self.offset, self.rn,
                                                                            cx, cy, self.pred_id, 2, self.meta, self.host_threads)
                        boff = np.frombuffer(boff, np.int64); blen = np.frombuffer(blen, np.int64); bst = np.frombuffer(bst, np.int8)
                        if self.m1:                                                 # a residual left int16: the C encoder stores the tile raw
                            bst = bst | (slot["bad2"][buf][:n].get(stream=stream) != 0).astype(np.int8)
                        fall = [(t, self._finish_host(self._tile_np(src, b0 + t, tz, ty, tx, cz, cy, cx), 1, hdr3, nbytes)) for t in np.flatnonzero(bst).tolist()]
                        self.stats["host_tiles"] += len(fall)
                        need_b = len(blobs) + sum(len(f[1]) for f in fall)
                        if pos + need_b > hout_np.nbytes:
                            newbuf = newnp = _host_pinned(int((pos + need_b) * 1.5))
                            newnp[:pos] = hout_np[:pos]; slot["hout"], slot["hout_np"] = newbuf, newnp; hout_np = newnp
                        hout_np[pos:pos + len(blobs)] = np.frombuffer(blobs, np.uint8)
                        all_off[b0:b1] = boff + pos; all_len[b0:b1] = blen; all_st[b0:b1] = 0; pos += len(blobs)
                        for t, bb in fall:
                            L = len(bb); hout_np[pos:pos + L] = np.frombuffer(bb, np.uint8); all_off[b0 + t] = pos; all_len[b0 + t] = L; pos += L
                        self.stats["download_s"] += time.perf_counter() - th
                    self.stats["kernel_s"] += gpu_t[0]
                # Rice path. Batch k's kernels write into output set k % nbuf; with overlap_download (nbuf = 2) its bytes go to the
                # host on the slot's download stream while batch k+1's kernels run. The per-tile meta and the byte count come back
                # with batch k's kernels (one small copy), so the host waits once per batch.
                nbuf = 2 if (self.overlap_download and self.ent_id != 2) else 1
                if self.ent_id != 2:
                    pk = (slot["key"], nbuf, self.m1)
                    if slot.get("pkey") != pk:
                        bufs = []
                        for i in range(nbuf):
                            b = dict(out=out, used=used, offs=offs, lens=lens, st=st) if i == 0 else dict(
                                out=cp.empty(out_cap, cp.uint8), used=cp.zeros(1, cp.uint64), offs=cp.zeros(bt, cp.uint64), lens=cp.zeros(bt, cp.uint32), st=cp.zeros(bt, cp.int32))
                            b["bad"] = cp.zeros(bt, cp.int32) if self.m1 else None
                            b["meta"] = cp.zeros((bt + 1, 4), cp.int64)   # per tile: offset, length, status, bad; row bt: [bytes used, ...]
                            b["hmeta"] = cp.cuda.alloc_pinned_memory((bt + 1) * 32)
                            b["hmeta_np"] = np.frombuffer(b["hmeta"], np.int64, (bt + 1) * 4).reshape(bt + 1, 4)
                            b["kev"] = cp.cuda.Event(disable_timing=True); b["dev"] = None
                            bufs.append(b)
                        slot["pbufs"], slot["pkey"] = bufs, pk
                    bufs = slot["pbufs"]
                    if nbuf > 1 and slot.get("dstream") is None:
                        slot["dstream"] = cp.cuda.Stream(non_blocking=True)
                    ds = slot["dstream"] if nbuf > 1 else stream
                D2H = cp.cuda.runtime.memcpyDeviceToHost
                for k, b0 in enumerate(range(0, ntiles, bt) if self.ent_id != 2 else ()):
                    b1 = min(b0 + bt, ntiles); n = b1 - b0; B = bufs[k % nbuf]
                    tk = time.perf_counter()
                    # persistent mode (sm_limit > 0): one slot's kernels at a time. Two concurrent persistent launches leave the second one
                    # only SMs >= sm_limit, where its blocks may not work (b4dk_admit), so it ran on one SM; serialising keeps 'N SMs'
                    # exact while the other slot's upload / download still overlaps.
                    if ready is not None:                           # the slices holding this batch's rows: wait on the host (outside the
                        tw = time.perf_counter()                    # kernel lock) and order the compute stream after them
                        ev = ready.event_for(b1 - 1)
                        ev.synchronize(); stream.wait_event(ev)
                        self.stats["upload_wait_s"] = self.stats.get("upload_wait_s", 0.0) + time.perf_counter() - tw
                        tk = time.perf_counter()
                    klock = self._kern_lock if self.sm_limit > 0 else None
                    if klock is not None:
                        klock.acquire()
                    try:
                        if B["dev"] is not None:                        # this output set's previous batch is on the host
                            stream.wait_event(B["dev"])
                        B["used"].fill(0)
                        if self.tile_copy:                              # contiguous tiles for this batch (gather kernel, zero at the edges)
                            self._gather(gather, dev, slot["tiles"], ez, ey, ex, ty, tx, b0, cz, cy, cx, n, stream)
                            src_dev, contig = slot["tiles"], 1
                        else:
                            src_dev, contig = dev, 0
                        # residuals computed once, before the coder (always: recomputing them in the coder's two passes was slower, 14.3 vs
                        # 15.9 GB/s for mode 2 above on an A6000, block in device memory)
                        use_res = True
                        seeds = None
                        if self.dither:
                            if slot.get("seeds") is None or slot["seeds"].size < bt:
                                slot["seeds"] = cp.zeros(bt, cp.uint32)
                            seeds = slot["seeds"]; self._seeds(src_dev, nelem, P.ts, n, seeds, stream)
                        if use_res:
                            if slot.get("res") is None or slot["res"].size < bt * nelem:
                                slot["res"] = cp.empty(bt * nelem, cp.int16)
                            self._residuals(resid_kernel, src_dev, contig, ez, ey, ex, ty, tx, b0, cz, cy, cx, P.mode, n, slot["res"], stream, B["bad"], seeds)
                        g, lim, work = self._grid(n, 2)
                        kernel((g,), (256,), (src_dev, self.lut_dev, np.int32(ez), np.int32(ey), np.int32(ex), np.int32(ty), np.int32(tx), np.int32(b0), np.int32(contig),
                                              np.int32(cz), np.int32(cy), np.int32(cx), np.int32(P.mode), np.int32(P.qi), np.int32(self.kpred),
                                              np.float32(P.conv), np.float32(P.offset), np.float32(P.rn), np.uint64(out_cap), np.int32(P.scratch_words), np.int32(n),
                                              scratch, B["out"], B["used"], B["offs"], B["lens"], B["st"], slot["res"] if use_res else self.lut_dev, np.int32(1 if use_res else 0),
                                              np.int32(P.ts), lim, work, seeds if seeds is not None else np.uint64(0), self.vst_dev, np.int32(self.vst_len)),
                               stream=stream, shared_mem=smem)
                        M = B["meta"]
                        M[:n, 0] = B["offs"][:n]; M[:n, 1] = B["lens"][:n]; M[:n, 2] = B["st"][:n]
                        if self.m1:
                            M[:n, 3] = B["bad"][:n]
                        M[bt, 0] = B["used"][0]
                        cp.cuda.runtime.memcpyAsync(B["hmeta_np"].ctypes.data, M.data.ptr, int((bt + 1) * 32), D2H, stream.ptr)
                        B["kev"].record(stream); B["kev"].synchronize()
                    finally:
                        if klock is not None:
                            klock.release()
                    tk2 = time.perf_counter(); self.stats["kernel_s"] += tk2 - tk
                    hm = B["hmeta_np"]; nused = int(hm[bt, 0]); meta = hm[:n, :3].copy()
                    if self.m1:                                                     # a residual left int16: the C encoder stores the tile raw
                        meta[:, 2] = np.where(hm[:n, 3] != 0, 1, meta[:, 2])
                    if pos + nused > hout_np.nbytes:
                        ds.synchronize()                                            # earlier batches' copies into the old buffer
                        newbuf = newnp = _host_pinned(int((pos + nused) * 1.5), pieces=True)
                        newnp[:pos] = hout_np[:pos]; hout_np = newnp
                        if pe is not None:
                            pe["mem"], pe["np"] = newbuf, newnp
                        else:
                            slot["hout"], slot["hout_np"] = newbuf, newnp
                    # explicit async D2H into the page-locked buffer (CuPy's get(out=) does not treat it as pinned)
                    if nbuf > 1:
                        ds.wait_event(B["kev"])
                    _d2h(cp, hout_np, pos, B["out"].data.ptr, int(nused), ds.ptr)
                    if nbuf > 1:
                        if B["dev"] is None:
                            B["dev"] = cp.cuda.Event(disable_timing=True)
                        B["dev"].record(ds)
                    else:
                        stream.synchronize()
                    all_off[b0:b1] = np.where(meta[:, 2] == 0, meta[:, 0] + pos, -1); all_len[b0:b1] = np.where(meta[:, 2] == 0, meta[:, 1], -1); all_st[b0:b1] = meta[:, 2]
                    pos += nused
                    self.stats["download_s"] += time.perf_counter() - tk2
                if nbuf > 1:
                    td = time.perf_counter(); ds.synchronize(); self.stats["download_s"] += time.perf_counter() - td
                host_tiles = np.flatnonzero(all_st != 0)
                for i in host_tiles.tolist():
                    extra.append((i, self._finish_host(self._tile_np(src, i, tz, ty, tx, cz, cy, cx), int(all_st[i]), hdr3, nbytes)))
                self.stats["host_tiles"] += len(host_tiles)
                if self.check and self.stats["blocks"] == 0:
                    idx = np.linspace(0, ntiles - 1, min(64, ntiles)).astype(int); bad = 0; ex_d = dict(extra)
                    for i in idx.tolist():
                        ref = _cpu_compress(self._tile_np(src, i, tz, ty, tx, cz, cy, cx), q=self.q, mode=self.mode, camera=self.camera, predictor=self.predictor,
                                            entropy=self.entropy, dither=self.dither)
                        got = hout_np[all_off[i]: all_off[i] + all_len[i]].tobytes() if all_st[i] == 0 else ex_d[i]
                        bad += (got != ref)
                    (_log.info if bad == 0 else _log.warning)("device %d self-check on %d chunks of the first block: %s", self.device, len(idx),
                                                              "all identical to the C encoder" if bad == 0 else f"{bad} DIFFER from the C encoder")
                    self.stats["check_bad"] = bad
                if pos > need:                                     # outgrew the first size: later blocks start bigger
                    with self._pool_lock:
                        self._out_hint[(ntiles, nbytes)] = max(self._out_hint.get((ntiles, nbytes), 0), int(pos * 1.15))
                if pe is not None:
                    hbuf = self._pool_give(pe, pos); pe = None
                else:
                    hbuf = self._copy_out(hout_np, pos)
        finally:
            if pe is not None:
                self._pool_release(pe)
            self._slots.put(slot)
        if extra:
            parts = [hbuf]; p = pos
            for i, blob in extra:
                all_off[i] = p; all_len[i] = len(blob); parts.append(np.frombuffer(blob, np.uint8)); p += len(blob)
            hbuf = np.concatenate(parts)
        return hbuf, all_off, all_len


class _MultiStats:
    """stats of a MultiGpuEncoder: numbers summed over the devices (per-block times divided by the summed block count are
    per-device means), other values from the first device; update() resets every device"""
    def __init__(self, encs): self._encs = encs

    def _merged(self):
        out = {}
        for e in self._encs:
            for k, v in e.stats.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("dev_mem"):
                    out[k] = out.get(k, 0) + v
                else:
                    out.setdefault(k, v)
        out["blocks_per_device"] = [e.stats.get("blocks", 0) for e in self._encs]
        return out

    def update(self, *a, **kw):
        for e in self._encs:
            e.stats.update(*a, **kw)

    def __getitem__(self, k): return self._merged()[k]
    def __contains__(self, k): return k in self._merged()
    def get(self, k, d=None): return self._merged().get(k, d)
    def items(self): return self._merged().items()
    def keys(self): return self._merged().keys()


class MultiGpuEncoder:
    """CudaTileEncoder on several GPUs (devices: CUDA device indices; every other keyword goes to CudaTileEncoder).

    split="blocks": each block (one encode_tiles call) is encoded whole on one device, the one with the fewest blocks in flight
    (ties round robin). The writer needs slots x devices blocks in flight (max_commits) to keep every device busy.
    split="rows": every block is shared out as bands of whole tile rows, one band per device (frames are cut along y, so
    each device uploads its band of every frame at the block's frame pitch; the block is page-locked once for all
    devices). A device then holds only its band, so a block that leaves room for one slot per GPU leaves room for two.
    The chunk blobs are the same bytes in both modes (and those of one GPU); with split="rows" the result is a list of
    per-chunk memoryviews of the devices' output buffers.

        enc = MultiGpuEncoder(cam, devices=(0, 1), split="rows")
        arr.encoder = enc.encode_tiles
    """
    def __init__(self, camera, *, devices=(0, 1), split: str = "blocks", **kw):
        if "device" in kw:
            raise TypeError("MultiGpuEncoder takes devices=(...), not device=")
        if split not in ("blocks", "rows"):
            raise ValueError(f"split must be 'blocks' or 'rows': {split!r}")
        self.split = split
        self.devices = [int(d) for d in devices]
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError(f"devices must be distinct and non-empty: {devices}")
        self.encs = [CudaTileEncoder(camera, device=d, **kw) for d in self.devices]
        self.backend = self.encs[0].backend
        self.nslots = sum(getattr(e, "nslots", 1) for e in self.encs)
        self._lock = threading.Lock(); self._inflight = [0] * len(self.encs); self._rr = 0
        self._pool = None
        self.stats = _MultiStats(self.encs)

    def _bands(self, ey, cy):
        """[y0, y1) of each device's band: whole tile rows, as even as they divide"""
        ty = -(-ey // cy); n = len(self.encs)
        b = [min(ey, cy * ((ty * k) // n)) for k in range(n + 1)]
        return list(zip(b[:-1], b[1:]))

    def register(self, arrays):
        """page-lock host block buffers now (registrations are portable: one serves every device)"""
        n = self.encs[0].register(arrays)
        for e in self.encs[1:]:
            e.register(arrays)
        return n

    def prepare(self, block_shape, chunks, dtype="uint16"):
        from concurrent.futures import ThreadPoolExecutor
        if self.split == "rows":
            shapes = [(block_shape[0], y1 - y0, block_shape[2]) for y0, y1 in self._bands(block_shape[1], chunks[1])]
        else:
            shapes = [tuple(block_shape)] * len(self.encs)
        with ThreadPoolExecutor(len(self.encs)) as ex:
            list(ex.map(lambda a: a[0].prepare(a[1], chunks, dtype) if a[1][1] > 0 else None, zip(self.encs, shapes)))

    def encode_tiles(self, block, origin, order, chunks, shape):
        if self.split == "rows":
            return self._encode_rows(block, origin, order, chunks, shape)
        n = len(self.encs)
        with self._lock:
            i = min(((self._rr + j) % n for j in range(n)), key=lambda j: self._inflight[j])
            self._inflight[i] += 1; self._rr = (i + 1) % n
        try:
            return self.encs[i].encode_tiles(block, origin, order, chunks, shape)
        finally:
            with self._lock:
                self._inflight[i] -= 1

    def _encode_rows(self, block, origin, order, chunks, shape):
        from concurrent.futures import ThreadPoolExecutor
        if hasattr(block, "__cuda_array_interface__"):
            raise TypeError("split='rows' takes host blocks")
        block = np.asarray(block)
        if _frame_pitch(block) is None:
            block = np.ascontiguousarray(block)
        cy = chunks[1]; ey = min(block.shape[1], shape[1] - origin[1])
        o = np.asarray(order, np.int64).reshape(-1, 3)
        if self.backend == "cuda":                                    # page-lock the block once; the bands use that registration
            self.encs[0]._register_pinned(block)
        jobs = []
        for k, (y0, y1) in enumerate(self._bands(ey, cy)):
            if y1 <= y0:
                continue
            sel = np.flatnonzero((o[:, 1] >= y0) & (o[:, 1] < y1))
            sub_order = [(int(a), int(b) - y0, int(c)) for a, b, c in o[sel].tolist()]
            jobs.append((k, sel, block[:, y0:y1, :], (origin[0], origin[1] + y0, origin[2]), sub_order))
        with self._lock:
            if self._pool is None:                                    # one worker per device slot (concurrent writer calls)
                self._pool = ThreadPoolExecutor(max(2, self.nslots + len(self.encs)), thread_name_prefix="b4d-gpu-band")
        futs = [self._pool.submit(self.encs[k].encode_tiles, sub, org, so, chunks, shape) for k, _, sub, org, so in jobs[1:]]
        res = [self.encs[jobs[0][0]].encode_tiles(jobs[0][2], jobs[0][3], jobs[0][4], chunks, shape)] if jobs else []
        res += [f.result() for f in futs]
        out = [None] * len(o)
        for (_, sel, _, _, _), r in zip(jobs, res):
            idx = sel.tolist()
            if isinstance(r, tuple):                                 # (buffer, offsets, lengths) in the band's order
                mv = memoryview(r[0]).cast("B"); offs = r[1].tolist(); lens = r[2].tolist()
                for j, i in enumerate(idx):
                    L = lens[j]
                    if L >= 0:
                        out[i] = mv[offs[j]: offs[j] + L]
            else:
                for j, i in enumerate(idx):
                    out[i] = r[j]
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True); self._pool = None
        for e in self.encs:
            e.close()

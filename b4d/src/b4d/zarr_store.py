"""A small spec-based API over the native zarr v3 sharded writer (b4d_zarr.c).

    import b4d.zarr_store as zs
    arr = zs.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": "/data/stack.zarr"},
                   "metadata": {"shape": [4096, 10640, 14192], "data_type": "uint16",
                                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [32, 1408, 3584]}},  # shard
                                "codecs": [{"name": "sharding_indexed", "configuration": {
                                    "chunk_shape": [32, 64, 64],                              # inner chunk
                                    "codecs": [{"name": "b4d", "configuration": {"q": 1, "mode": 1, "conversion": 2,
                                                "offset": 100, "read_noise": 3, "predictor": "above3", "entropy": "rice"}}]}}]}},
                  create=True, threads=64)
    fut = arr[z0:z0 + 32].write(block)          # block: contiguous (32, Y, X) uint16; returns a Future
    fut.result()                                # bytes written; the block buffer is free once this returns
    # writers=2 lets one block's file IO overlap the next block's compression (each write uses `threads` threads);
    # max_threads=96 caps the total (threads per write = max_threads // writers); write(block, threads=...) overrides per call

    arr = zs.open(path)                         # open existing (reads zarr.json)
    arr[z0:z0 + 32, 1408:2816].write(band)     # any y/x window that starts on shard boundaries
    arr.read(z0:z1)  /  arr[z0:z1].read()       # via zarr-python (needs b4d.zarr3 registered for b4d stores)

Convenience constructor:  zs.create(path, shape, shards=..., chunks=..., dtype="uint16", codec=B4D(...) | numcodecs codec | "raw")

Codecs: "b4d", "bytes"/raw and the zarr "blosc" codec (lz4, zstd, blosclz, zlib with shuffle/bitshuffle) are encoded
natively on the C thread pool; any other numcodecs codec (plain zstd, gzip, ...) is encoded
in Python threads and handed to the native shard assembler. Writes are executed on a background thread so write() returns at once; each write
compresses with `threads` native threads, assembles the shard files with a crc32c index, flushes them
(sync=True) and renames them into place. Blocks must start on shard boundaries and cover whole shards
(except at the array edge).
"""
from __future__ import annotations

import io, json, os, threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

from . import Camera, _b4d, ENTROPY_CODERS, PREDICTORS, _meta

_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
_DTYPES = {"uint8": np.dtype("uint8"), "uint16": np.dtype("uint16")}
_SINKS = ("disk", "memory")           # memory: compress + assemble the shards, write nothing (benchmarking)
_FUSED_DOWNSAMPLE = True              # multiscale: level-1 planes come out of the level-0 compression pass (C gather)
_ASYNC_CASCADE = True                 # multiscale: levels >= 2 are derived from a full level-1 buffer on a pyramid thread,
                                      # off the ordered step (it held the next block back: 0.02-0.06 s per 16-bit block)
_FUSED_DTYPES = (np.uint16, np.uint8)  # uint8 too (with uint16 only, 8-bit cameras paid a separate pass)


def _prefaulted_zeros(shape, dtype, threads=8):
    """np.zeros that is really backed by memory: np.zeros maps the pages lazily (calloc), so their first touch happens in
    the writer's hot path (Windows: ~0.5 s per GB, serialised in the kernel). Writing the zeros in parallel slabs at
    allocation moves that cost out of the acquisition (without it the first 4 multiscale blocks ran 3-7x slow)."""
    a = np.empty(shape, dtype)
    if a.size == 0:
        return a
    flat = a.reshape(-1); n = flat.size; k = max(1, min(int(threads), n // (1 << 20) or 1))
    step = -(-n // k)
    if k == 1:
        flat[:] = 0
    else:
        with ThreadPoolExecutor(k) as ex:                  # numpy releases the GIL for the fill
            list(ex.map(lambda i: flat.__setitem__(slice(i, min(i + step, n)), 0), range(0, n, step)))
    return a


def _check_sink(sink):
    if sink not in _SINKS:
        raise ValueError(f"sink must be one of {_SINKS}, not {sink!r}")
    return sink


def b4d_camera(cfg: dict):
    """the Camera in a B4D codec configuration; float32 as stored"""
    return Camera(float(cfg["conversion"]), float(cfg["offset"]), float(cfg["read_noise"]), cfg.get("exact_vst", False) or False)


def _b4d_codec_cfg(codec) -> dict:
    """B4D codec object (b4d.zarr3.B4D) or dict -> configuration dict"""
    if isinstance(codec, dict):
        return dict(codec)
    cfg = dict(q=codec.q, mode=codec.mode, conversion=codec.conversion, offset=codec.offset, read_noise=codec.read_noise,
               predictor=codec.predictor, entropy=codec.entropy)
    if getattr(codec, "dither", False):
        cfg["dither"] = codec.dither if codec.dither == "listed" else True
    cfg["exact_vst"] = getattr(codec, "exact_vst", False)   # always stored; absent (older metadata) means False
    return cfg


_NO_BLOCK = np.zeros(0, np.uint8)     # stand-in buffer for the C call when the chunks are pre-encoded from a device-resident block


class ZarrArray:
    def __init__(self, path: str, metadata: dict, threads: int, sync: bool, writers: int = 1, sink: str = "disk", direct_io: bool = False):
        self.path = path.rstrip("/\\")
        self.sink = _check_sink(sink)         # "disk" or "memory" (compress + assemble shards, write nothing: benchmarking)
        # unbuffered write-through shard files (NO_BUFFERING|WRITE_THROUGH / O_DIRECT): one DMA per shard straight from the
        # assembly buffer, no page-cache copy or residue; ~2x the buffered rate on NVMe. Falls back per shard where refused.
        self.direct_io = bool(direct_io)
        self.encoder = None                   # optional external chunk encoder (see b4d.cuda_encoder.CudaTileEncoder)
        self._pixmap = None; self._map_origin = (0, 0)   # per-pixel calibration (set_calibration)
        self.metadata = metadata
        full_shape = tuple(int(v) for v in metadata["shape"])
        full_shards = tuple(int(v) for v in metadata["chunk_grid"]["configuration"]["chunk_shape"])
        sh = metadata["codecs"][0]
        if sh["name"] != "sharding_indexed":
            raise ValueError("only sharding_indexed arrays are supported")
        cfg = sh["configuration"]
        full_chunks = tuple(int(v) for v in cfg["chunk_shape"])
        # leading axes (e.g. a channel axis in (c, z, y, x) OME arrays): chunk and shard extent must be 1 there;
        # the writer addresses them through the chunk-key prefix and works on the trailing (z, y, x) block
        self.ndim = len(full_shape)
        if self.ndim < 3:
            raise ValueError("arrays must have at least 3 dimensions (z, y, x)")
        self.lead = self.ndim - 3
        if any(c != 1 or s != 1 for c, s in zip(full_chunks[:self.lead], full_shards[:self.lead])):
            raise ValueError("leading (non-spatial) axes must have chunk and shard extent 1")
        self.full_shape = full_shape
        self.shape = full_shape[self.lead:]
        self.shards = full_shards[self.lead:]
        self.chunks = full_chunks[self.lead:]
        self.dtype = _DTYPES[metadata["data_type"]]
        self.inner = cfg["codecs"]
        self.key_sep = metadata.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
        self.threads = int(threads); self.sync = bool(sync)
        self._exec = ThreadPoolExecutor(max(1, int(writers)), thread_name_prefix="zarr-writer")   # blocks in flight
        self._lock = threading.Lock()
        self.stats = dict(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0)
        # resolve codec
        names = [c["name"] for c in self.inner]
        self._numcodec = None
        if "b4d" in names:
            c = next(c for c in self.inner if c["name"] == "b4d")["configuration"]
            self._kind = 0
            self._params = (float(c["conversion"]), float(c["offset"]), float(c["read_noise"]),
                            PREDICTORS[c["predictor"]] if isinstance(c["predictor"], str) else int(c["predictor"]),
                            ENTROPY_CODERS[c["entropy"]] if isinstance(c["entropy"], str) else int(c["entropy"]),
                            _meta(int(c["mode"]), float(c["q"])), 2 if c.get("dither") == "listed" else 1 if c.get("dither") else 0)
            self._blosc = ("", 0, 0)
            from . import _vst_of
            # exact VST (format v9) or None; lossless never uses it (the fit costs ~0.2 s per camera model)
            self._vst = _vst_of(b4d_camera(c)) if (self._params[5] & 3) else None
        else:
            self._params = (1.0, 0.0, 0.0, 0, 0, 0, 0)
            comp = [c for c in self.inner if c["name"] != "bytes"]
            self._blosc = ("", 0, 0)
            if not comp:
                self._kind = 1
            elif len(comp) == 1 and comp[0]["name"] == "blosc":        # zarr blosc codec: native, on the C pool (lz4/zstd/...)
                cfg = comp[0].get("configuration", {})
                sh = cfg.get("shuffle", "shuffle")
                self._blosc = (str(cfg.get("cname", "lz4")), int(cfg.get("clevel", 5)), _SHUFFLE.get(sh, sh if isinstance(sh, int) else 1))
                self._kind = 2
            else:                                                     # other numcodecs codecs: Python-thread encoding
                self._kind = 3
                import numcodecs
                def mk(c):
                    cfg = dict(c.get("configuration", {}))
                    if c["name"] == "blosc":
                        cfg["shuffle"] = _SHUFFLE.get(cfg.get("shuffle", "shuffle"), cfg.get("shuffle", 1)); cfg.pop("typesize", None)
                    return numcodecs.get_codec({"id": c["name"], **cfg})
                self._numcodec = [mk(c) for c in comp]

    def set_calibration(self, calibration, origin=(0, 0)):
        """Use a b4d.PixelCalibration for the chunks written from now on (B4D codec, mode 1 with above3 or mode 2 with above; not with an external
        encoder). origin: (row, col) of the array's first pixel on the map. The chunks hold what they need: readers change nothing."""
        if self._kind != 0:
            raise ValueError("a per-pixel calibration needs the B4D codec")
        c = next(c for c in self.inner if c["name"] == "b4d")["configuration"]
        cam = b4d_camera(c)
        self._map_origin = calibration._check(c["mode"], c["q"], c["predictor"], cam, self.shape[1:], origin)
        self._calibration = calibration; self._pixmap = calibration._capsule

    # ---- slice indexing: arr[z0:z1] / arr[z0:z1, y0:y1] / arr[z0:z1, y0:y1, x0:x1];
    # for arrays with leading axes (e.g. (c, z, y, x)): arr[c, z0:z1] with an integer per leading axis
    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        lead_idx = []
        for ax in range(self.lead):
            s = idx[ax] if ax < len(idx) else 0
            if isinstance(s, slice):
                start, stop, step = s.indices(self.full_shape[ax])
                if stop - start != 1:
                    raise IndexError("leading axes must be indexed by a single integer")
                s = start
            lead_idx.append(int(s))
        idx = idx[self.lead:]
        sl = []
        for ax in range(3):
            s = idx[ax] if ax < len(idx) else slice(None)
            if isinstance(s, int):
                s = slice(s, s + 1)
            start, stop, step = s.indices(self.shape[ax])
            if step != 1:
                raise IndexError("step must be 1")
            sl.append((start, stop))
        return _View(self, sl, tuple(lead_idx))

    @property
    def domain(self):
        return tuple((0, n) for n in self.shape)

    def spec(self) -> dict:
        return {"driver": "zarr3", "kvstore": {"driver": "file", "path": self.path}, "metadata": self.metadata}

    def read(self, *idx):
        return self[idx if idx else (slice(None),)].read()

    def flush(self):
        """wait for all queued writes"""
        self._exec.submit(lambda: None).result()

    def close(self):
        self.flush(); self._exec.shutdown(wait=True)
        _b4d.zarr_free_cache()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ---- the write
    def _write(self, block: np.ndarray, origin, threads, ds_out=None, factors=None, lead_idx=()) -> int:
        """ds_out: optional contiguous buffer (the block's dtype, uint8 or uint16) of shape ceil(block / factors) that receives the box mean of
        the block (factors 1 or 2 per axis), computed from the bands already gathered for compression (no extra pass)."""
        import time
        t0 = time.perf_counter()
        on_device = hasattr(block, "__cuda_array_interface__")          # device-resident block: only an external encoder can read it
        if on_device and self.encoder is None:
            raise TypeError("a device-resident block needs an external (GPU) encoder")
        if not on_device:
            block = np.ascontiguousarray(block)
        nz, ny, nx = block.shape
        z0, y0, x0 = origin
        ts_ = np.dtype(block.dtype).itemsize
        blobs = None; kind = self._kind
        if self.encoder is not None:                               # external encoder (e.g. GPU): blobs in writer order
            blobs = self.encoder(block, origin, self._chunk_order(block.shape), self.chunks, self.shape); kind = 3
        elif kind == 3:                                            # generic numcodecs: encode chunks in Python threads
            blobs = self._encode_blobs(block, origin)
        prefix = self.key_sep.join(str(int(i)) for i in lead_idx) if lead_idx else ""
        extra = (ds_out if ds_out is not None else None, tuple(int(f) for f in factors) if factors else (1, 1, 1), 1 if self.direct_io else 0, prefix)
        if self._pixmap is not None:
            if kind != 0:
                raise ValueError("a per-pixel calibration does not work with an external encoder")
            extra = extra + (self._pixmap, *self._map_origin)
        else:
            extra = extra + (None, 0, 0)
        extra = extra + (getattr(self, "_vst", None) if kind == 0 else None,)
        r = _b4d.zarr_write_block(block if not on_device else _NO_BLOCK, ts_, nz, ny, nx, int(z0), int(y0), int(x0), tuple(int(v) for v in self.shape),
                                  tuple(self.shards), tuple(self.chunks), kind, self._params, self._blosc[0], self._blosc[1], self._blosc[2],
                                  blobs, self.path if self.sink == "disk" else "", self.key_sep, 1 if self.sync else 0, int(threads or self.threads), *extra)
        written = r["bytes"]
        with self._lock:
            self.stats["blocks"] += 1; self.stats["raw_bytes"] += block.nbytes; self.stats["written_bytes"] += written
            self.stats["seconds"] += time.perf_counter() - t0
            self.stats["t_compress"] = self.stats.get("t_compress", 0.0) + r["t_compress"]
            self.stats["t_io"] = self.stats.get("t_io", 0.0) + r["t_io"]
            self.stats["groups_mask"] = self.stats.get("groups_mask", 0) | r["groups_mask"]
            self.stats["direct_used"] = self.stats.get("direct_used", 0) + r["direct_used"]
            self.stats["direct_fallback"] = self.stats.get("direct_fallback", 0) + r["direct_fallback"]
            self.stats["blosc1_fallback"] = self.stats.get("blosc1_fallback", 0) + r["blosc1_fallback"]
        return written

    def _chunk_order(self, block_shape):
        """chunk origins (in block coordinates) in the writer's order: C order over shards, then chunks in a shard"""
        nz, ny, nx = block_shape
        sz, sy, sx = self.shards; cz, cy, cx = self.chunks
        nsz, nsy, nsx = -(-nz // sz), -(-ny // sy), -(-nx // sx)
        ncz, ncy, ncx = sz // cz, sy // cy, sx // cx
        order = []
        for szi in range(nsz):
            for syi in range(nsy):
                for sxi in range(nsx):
                    for kz in range(ncz):
                        for ky in range(ncy):
                            for kx in range(ncx):
                                order.append((szi * sz + kz * cz, syi * sy + ky * cy, sxi * sx + kx * cx))
        return order

    def _encode_blobs(self, block, origin):
        """generic numcodecs path: one blob per inner chunk, C order over shards then chunks; None = absent"""
        nz, ny, nx = block.shape; z0, y0, x0 = origin
        cz, cy, cx = self.chunks
        order = self._chunk_order(block.shape)
        def enc(o):
            bz, by, bx = o
            dz = min(cz, nz - bz, self.shape[0] - (z0 + bz)); dy = min(cy, ny - by, self.shape[1] - (y0 + by)); dx = min(cx, nx - bx, self.shape[2] - (x0 + bx))
            if dz <= 0 or dy <= 0 or dx <= 0:
                return None
            tile = np.zeros((cz, cy, cx), block.dtype); tile[:dz, :dy, :dx] = block[bz:bz + dz, by:by + dy, bx:bx + dx]
            buf = tile.tobytes()
            for c in self._numcodec:
                buf = c.encode(buf)
            return bytes(buf)
        with ThreadPoolExecutor(max(1, self.threads)) as pool:
            return list(pool.map(enc, order))


class _View:
    def __init__(self, arr: ZarrArray, sl, lead_idx=()):
        self.arr, self.sl, self.lead_idx = arr, sl, tuple(lead_idx)

    @property
    def shape(self):
        return tuple(b - a for a, b in self.sl)

    def write(self, block: np.ndarray, *, threads: Optional[int] = None) -> Future:
        """Queue the write of `block` into this window. Returns a Future -> bytes written. The block must
        start on shard boundaries in every axis and cover whole shards except at the array edge. threads overrides the
        array's native threads for this write."""
        if not hasattr(block, "__cuda_array_interface__"):
            block = np.asarray(block)
        if block.dtype != self.arr.dtype:
            raise TypeError(f"block dtype {block.dtype} != array dtype {self.arr.dtype}")
        if block.ndim == 3 + len(self.lead_idx) and all(s == 1 for s in block.shape[:len(self.lead_idx)]):
            block = block.reshape(block.shape[len(self.lead_idx):])          # accept (1, z, y, x) for a channel write
        if block.shape != self.shape:
            raise ValueError(f"block shape {block.shape} != window shape {self.shape}")
        origin = tuple(a for a, _ in self.sl)
        for ax, (o, s, n, e) in enumerate(zip(origin, self.arr.shards, block.shape, self.arr.shape)):
            if o % s:
                raise ValueError(f"window start {o} on axis {ax} is not a multiple of the shard size {s}")
            if n % s and o + n < e:
                raise ValueError(f"window extent {n} on axis {ax} must cover whole shards ({s}) unless it ends at the array edge")
        return self.arr._exec.submit(self.arr._write, block, origin, threads, None, None, self.lead_idx)

    def read(self):
        import zarr
        import b4d.zarr3  # noqa: F401  (registers the b4d codec with zarr-python)
        z = zarr.open_array(self.arr.path, mode="r")
        return z[tuple(self.lead_idx) + tuple(slice(a, b) for a, b in self.sl)]


# ------------------------------------------------------------------ multiscale (OME-NGFF 0.5 group of levels)
def _ceil_div(a, b): return -(-a // b)


def auto_levels(shape, factors, min_size=256):
    """number of downsampled levels: halve (per `factors`) until the smallest downsampled axis would drop below min_size"""
    n = 0; cur = list(shape)
    while True:
        nxt = [_ceil_div(c, f) if f > 1 else c for c, f in zip(cur, factors)]
        if min(v for v, f in zip(nxt, factors) if f > 1) < min_size:
            return n
        n += 1; cur = nxt


class MultiscaleArray:
    """Level 0 plus N box-averaged levels, written together by the native writer.

        ms = zs.create_multiscale(path, shape, shards=..., chunks=..., codec=..., levels=N,   # N required ("auto" allowed)
                                  factors=(2, 2, 2),          # 2x2x2 box mean per level (default); (1, 2, 2) for xy only
                                  threads=..., writers=2, scale=(dz, dy, dx))
        ms[z0:z0 + 32].write(block)   # whole frames only (full y and x); lower levels are derived in C

    Layout: <path>/zarr.json (OME-NGFF 0.5 'multiscales' group), <path>/0, <path>/1, ... (zarr v3 sharded arrays).
    Chunk and shard DEPTH are the same at every level: with z downsampling the downsampled planes of level k are
    accumulated across 2**k blocks until a full shard depth is available, then written (so level k lags level 0
    by up to 2**k blocks; close() flushes the remainder). Memory for the accumulators is < 1/3 of a block.

    Arrays with leading axes (OME (c, z, y, x), chunk and shard extent 1 on c): ms[c, z0:z1].write(block) writes
    channel c at every level. One MultiscaleArray instance writes one channel (its pyramid accumulators belong to
    that channel); open another instance over the same arrays for another channel."""

    def __init__(self, path, levels_arrays, factors, threads):
        self.path = path; self.levels = levels_arrays; self.factors = tuple(factors); self.threads = threads
        self.block_depth = self.levels[0].shards[0]                      # multiscale blocks are whole shards deep
        self.shape = self.levels[0].shape; self.dtype = self.levels[0].dtype
        self.shards = self.levels[0].shards; self.chunks = self.levels[0].chunks
        self._exec = ThreadPoolExecutor(self.levels[0]._exec._max_workers, thread_name_prefix="ms-writer")
        self._acc_lock = threading.Lock()
        self._pyr_exec = ThreadPoolExecutor(1, thread_name_prefix="ms-pyramid")   # levels >= 2, in order
        self._order = threading.Condition(); self._next_submit = 0; self._next_accum = 0   # accumulate in submission order
        # accumulators for levels >= 1, double-buffered: a full buffer is handed to the level's writer thread and
        # accumulation continues in the other one, so level IO never blocks the ordered section
        self._acc = {}
        for k in range(1, len(self.levels)):
            a = self.levels[k]
            bufs = [_prefaulted_zeros((a.shards[0], a.shape[1], a.shape[2]), a.dtype, threads) for _ in range(2)]
            self._acc[k] = dict(bufs=bufs, cur=0, filled=0, z0=0, futs=[None, None], freed=[0, 0], casc=[None, None])
        self._level_futs = []
        self._scratch_buf = {}
        self._l1_pool = []
        self.lead_idx = ()                     # leading (channel) index of the arrays' writes, fixed by the first write
        # optional hook on_written(level, z0, z1): the shard files of level `level` covering planes [z0, z1) are durable
        # (level 0 after each block, levels >= 1 after each accumulator flush). A host that stages to object storage
        # uploads exactly those files; called on writer threads, must not block for long.
        self.on_written = None
        self.stats = dict(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0, downsample_seconds=0.0)

    @property
    def nlevels(self): return len(self.levels)
    def level(self, k): return self.levels[k]
    def __getitem__(self, idx):
        v = self.levels[0][idx]
        if v.lead_idx != self.lead_idx:
            if self._next_submit:
                raise ValueError(f"this MultiscaleArray writes channel {self.lead_idx}; open another instance for {v.lead_idx}")
            self.lead_idx = v.lead_idx
        return _MSView(self, v)
    def flush(self): self._exec.submit(lambda: None).result(); self._drain_cascade(); [a.flush() for a in self.levels]
    def _drain_cascade(self):
        """wait until every level >= 2 derivation handed to the pyramid thread has run (and submitted its level writes)"""
        self._pyr_exec.submit(lambda: None).result()
    def close(self, verbose=False):
        import time
        t0 = time.perf_counter()
        self._exec.submit(lambda: None).result()                     # all block writes submitted so far are done
        t1 = time.perf_counter()
        self._exec.submit(self._flush_levels).result()               # partial pyramid accumulators -> padded shards
        t2 = time.perf_counter()
        self._exec.shutdown(wait=True); self._pyr_exec.shutdown(wait=True); [a.close() for a in self.levels]
        t3 = time.perf_counter()
        self._acc.clear(); self._scratch_buf.clear(); self._l1_pool.clear()   # release the pyramid buffers
        if verbose:
            print(f"close: pending blocks {t1 - t0:.1f} s, level flush {t2 - t1:.1f} s, level writers {t3 - t2:.1f} s", flush=True)
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    def _downsample_into(self, cur, dst):
        """box-average cur into dst (dst shape = ceil(cur / factors)); dst must be contiguous and pre-faulted"""
        fz, fy, fx = self.factors
        nz, ny, nx = cur.shape
        _b4d.downsample(cur, dst, cur.dtype.itemsize, nz, ny, nx, fz, fy, fx, int(self.threads))
        return dst

    def reset_pyramid(self):
        """discard everything accumulated for levels >= 1 and start the pyramid again at z = 0 (after warm-up
        writes that will be overwritten): waits for pending level writes, clears the accumulators and offsets"""
        self._exec.submit(lambda: None).result()                     # every block write submitted so far has finished
        with self._order:
            while self._next_accum != self._next_submit:
                self._order.wait()
        self._drain_cascade()
        for k, acc in self._acc.items():
            for i in range(2):
                f = acc["futs"][i]
                if f is not None:
                    f.result(); acc["futs"][i] = None
            acc["filled"] = 0; acc["z0"] = 0; acc["cur"] = 0; acc["freed"] = [0, 0]; acc["casc"] = [None, None]
            # the submission-side mirror of the level-1 fill (fused path) must restart with the fill itself; otherwise
            # the next blocks' fused planes land at a stale offset and the first level-1 shards hold warm-up data
            acc["reserved"] = 0; acc["rcur"] = 0; acc["gen_r"] = 0; acc["gen_done"] = 0
        for f in self._level_futs:
            f.result()
        self._level_futs.clear()
        for lv in self.levels[1:]:
            lv.stats.update(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0)
        self.stats["downsample_seconds"] = 0.0; self.stats["level_wait_seconds"] = 0.0; self.stats["fused_blocks"] = 0

    def _kind_ok_for_fused(self):
        return all(f in (1, 2) for f in self.factors) and all((c % 2 == 0) or f == 1 for c, f in zip(self.levels[0].chunks, self.factors))

    def _take_scratch(self, shape):
        """pool of pre-faulted level-1 scratch buffers (one per block in flight), under self._order"""
        for i, (s, busy) in enumerate(self._l1_pool):
            if not busy and s.shape == shape:
                self._l1_pool[i][1] = True; return s
        s = _prefaulted_zeros(shape, self.dtype, self.threads); self._l1_pool.append([s, True]); return s

    def _free_scratch(self, s):
        for ent in self._l1_pool:
            if ent[0] is s: ent[1] = False; return

    def _scratch(self, k, shape):
        """persistent, pre-faulted scratch for level k's downsampled planes (allocated once; Windows first-touch
        page faults would otherwise cost ~0.5 s per GB on every block)"""
        s = self._scratch_buf.get(k)
        if s is None or s.shape != shape:
            s = _prefaulted_zeros(shape, self.dtype, self.threads); self._scratch_buf[k] = s
        return s

    def _submit_level(self, k, acc, n_planes, cascade=True):
        """write the current buffer of level k (n_planes real planes, padded to the shard depth unless at the array
        edge) on the level's writer thread; swap buffers; wait for the old write of the buffer we are about to reuse.
        Level 1 with cascade: also hand the derivation of levels >= 2 from this buffer to the pyramid thread and return
        True (the caller then does not cascade itself); the buffer is reused only after both have finished."""
        a = self.levels[k]; buf = acc["bufs"][acc["cur"]]; z0 = acc["z0"]
        n_shard = min(buf.shape[0], a.shape[0] - z0)
        if n_shard <= 0:
            # more planes than the level array can hold (e.g. writes beyond the declared z extent): drop them, but
            # keep the accumulator consistent so the caller's loop always progresses
            self.stats["overshoot_planes"] = self.stats.get("overshoot_planes", 0) + int(acc["filled"])
            acc["z0"] = z0 + buf.shape[0]; acc["filled"] = 0; acc["cur"] ^= 1
            acc["gen_done"] = acc.get("gen_done", 0) + 1
            return False
        if n_planes < n_shard:
            buf[n_planes:n_shard] = 0                                   # stale planes from an earlier fill
        fut = a._exec.submit(a._write, buf[:n_shard], (z0, 0, 0), None, None, None, self.lead_idx)
        acc["futs"][acc["cur"]] = fut; self._level_futs.append(fut)
        if self.on_written is not None:
            fut.add_done_callback(lambda f, k=k, z0=z0, z1=z0 + n_shard: f.exception() is None and self._notify_written(k, z0, z1))
        casc = cascade and k == 1 and _ASYNC_CASCADE and len(self.levels) > 2
        if casc:                                                        # before gen_done: a reservation that sees this generation sees its cascade
            src = buf if n_planes == buf.shape[0] else buf[:n_planes]
            acc["casc"][acc["cur"]] = self._pyr_exec.submit(self._cascade_task, src)
        acc["gen_done"] = acc.get("gen_done", 0) + 1                   # this buffer generation is submitted (fused reservations may reuse it after it completes)
        cur_buf = acc["cur"]
        def _freed(f, acc=acc, b=cur_buf):
            with self._order:
                acc["freed"][b] += 1; self._order.notify_all()
        fut.add_done_callback(_freed)
        acc["z0"] = z0 + buf.shape[0]; acc["filled"] = 0; acc["cur"] ^= 1
        old = acc["futs"][acc["cur"]]; oldc = acc["casc"][acc["cur"]] if "casc" in acc else None
        if old is not None or oldc is not None:
            import time
            t = time.perf_counter()
            if old is not None:
                old.result()                                            # the buffer we reuse must be free: written ...
            if oldc is not None:
                oldc.result()                                           # ... and read by the levels >= 2 derivation
            self.stats["level_wait_seconds"] = self.stats.get("level_wait_seconds", 0.0) + (time.perf_counter() - t)
        return casc

    def _cascade_task(self, src):
        """pyramid thread: levels >= 2 from one full level-1 buffer (in buffer order: one thread)"""
        import time
        t = time.perf_counter()
        self._append_from(2, src)
        self.stats["cascade_seconds"] = self.stats.get("cascade_seconds", 0.0) + (time.perf_counter() - t)

    def _append_from(self, k, src, final=False):
        """downsample `src` (level k-1 planes) into level k's accumulator; when it fills (or at the end), submit it
        for writing and cascade into level k+1. Downsamples straight into the accumulator when the planes fit,
        otherwise through a persistent scratch buffer; no per-block allocations."""
        if k >= len(self.levels):
            return
        acc = self._acc[k]
        fz = self.factors[0]
        nplanes = _ceil_div(src.shape[0], fz)
        buf = acc["bufs"][acc["cur"]]
        if nplanes <= buf.shape[0] - acc["filled"]:
            dst = buf[acc["filled"]:acc["filled"] + nplanes]
            self._downsample_into(src, dst); acc["filled"] += nplanes
            if acc["filled"] == buf.shape[0]:
                if not self._submit_level(k, acc, buf.shape[0]):
                    self._append_from(k + 1, buf)
        else:
            planes = self._downsample_into(src, self._scratch(k, (nplanes,) + buf.shape[1:]))
            take = 0
            while take < nplanes:
                buf = acc["bufs"][acc["cur"]]
                m = min(buf.shape[0] - acc["filled"], nplanes - take)
                if m <= 0:                                              # cannot happen after _submit_level resets; never spin
                    raise RuntimeError(f"level {k} accumulator made no progress (filled={acc['filled']}, depth={buf.shape[0]})")
                buf[acc["filled"]:acc["filled"] + m] = planes[take:take + m]; acc["filled"] += m; take += m
                if acc["filled"] == buf.shape[0]:
                    if not self._submit_level(k, acc, buf.shape[0]):
                        self._append_from(k + 1, buf)
        if final:
            if k == 1:
                self._drain_cascade()                                   # the end of the pyramid runs here, in order
            if acc["filled"]:
                buf = acc["bufs"][acc["cur"]]; n = acc["filled"]
                self._submit_level(k, acc, n, cascade=False)
                self._append_from(k + 1, buf[:n], final=True)
            else:                                                       # nothing left here, but higher levels may hold planes
                self._append_from(k + 1, np.empty((0,) + self.levels[k].shape[1:], self.dtype), final=True)

    def _flush_levels(self):
        """end of the acquisition: write the partially filled accumulators, cascading to the top level"""
        with self._order:
            while self._next_accum != self._next_submit:
                self._order.wait()
        if len(self.levels) > 1:
            self._append_from(1, np.empty((0,) + self.shape[1:], self.dtype), final=True)
        total = 0
        for f in self._level_futs:
            total += f.result()
        self._level_futs.clear()
        self.stats["written_bytes"] += total
        return total

    def _fused_slot(self, block, seq):
        """destination of this block's level-1 planes inside the level-1 accumulator, if the fused path applies:
        (buffer index, offset, planes, generation to wait for) or None"""
        if not (_FUSED_DOWNSAMPLE and len(self.levels) > 1 and block.dtype in _FUSED_DTYPES and self._kind_ok_for_fused()):
            return None
        if getattr(self.levels[0], "encoder", None) is not None:        # external (GPU) encoder: no C gather
            return None
        nz = block.shape[0]; fz = self.factors[0]
        if nz != self.block_depth or nz % fz:                           # only full, evenly divisible blocks
            return None
        acc = self._acc[1]; D = acc["bufs"][0].shape[0]; p = nz // fz
        if D % p:
            return None
        start = seq * p
        b = (start // D) % 2; off = start % D; gen = (start // D) // 2   # gen-th reuse of buffer b
        return b, off, p, gen

    def _write_all(self, block, origin, seq, ds_view=None):
        """level 0 compression, then (in block order) the pyramid step. With a reserved level-1 destination the
        compression workers write the level-1 planes straight into the accumulator (fused), and the ordered step
        only commits them; otherwise the ordered step downsamples the block itself."""
        import time
        t0 = time.perf_counter(); total = 0

        def pyramid_step():
            with self._order:                                                # pyramid planes must be appended in z order
                while self._next_accum != seq:
                    self._order.wait()
            try:
                if len(self.levels) > 1:
                    t1 = time.perf_counter()
                    if ds_view is not None:
                        self._commit_planes(1, ds_view.shape[0])
                    else:
                        self._append_from(1, block)
                    self.stats["downsample_seconds"] += time.perf_counter() - t1
            finally:
                with self._order:
                    self._next_accum = seq + 1
                    self._order.notify_all()

        if ds_view is not None:                                              # fused: level-1 planes come out of the compression pass
            self.stats["fused_blocks"] = self.stats.get("fused_blocks", 0) + 1
            total += self.levels[0]._write(block, tuple(origin), None, ds_out=ds_view, factors=self.factors, lead_idx=self.lead_idx)
            pyramid_step()
        else:
            total += self.levels[0]._write(block, tuple(origin), None, lead_idx=self.lead_idx)
            pyramid_step()
        self.stats["blocks"] += 1; self.stats["raw_bytes"] += block.nbytes; self.stats["written_bytes"] += total
        self.stats["seconds"] += time.perf_counter() - t0
        self._notify_written(0, origin[0], origin[0] + block.shape[0])
        return total

    def _notify_written(self, level, z0, z1):
        cb = self.on_written
        if cb is not None:
            try:
                cb(level, int(z0), int(z1))
            except Exception as e:   # pragma: no cover - a host's hook must not take the writer down
                self.stats["hook_errors"] = self.stats.get("hook_errors", 0) + 1
                self.stats["hook_error"] = repr(e)

    def _reserve_l1(self, block_shape):
        """called in submission order (under self._order): mirror the level-1 accumulator's fill state ahead of the
        completions and hand out a contiguous destination view when this block's planes fit in one buffer whose
        previous contents (two buffer generations back) have already been submitted for writing. Returns
        (view, future_to_wait_for) or (None, None); the caller waits on the future outside the lock before it
        starts the block, so the workers never write into a buffer still being written out."""
        if not (_FUSED_DOWNSAMPLE and self._kind_ok_for_fused() and self.dtype in _FUSED_DTYPES):
            return None, None
        if getattr(self.levels[0], "encoder", None) is not None:        # external (GPU) encoder: no C gather, no fused planes
            return None, None
        acc = self._acc[1]; depth = acc["bufs"][0].shape[0]
        nplanes = _ceil_div(block_shape[0], self.factors[0])
        acc.setdefault("reserved", 0); acc.setdefault("rcur", 0); acc.setdefault("gen_r", 0); acc.setdefault("gen_done", 0)
        view = wait = None
        if acc["reserved"] + nplanes <= depth:
            gen = acc["gen_r"]                                   # generation of the buffer being reserved into
            if gen < 2 or acc["gen_done"] >= gen - 1:            # generation gen-2 (same buffer) already submitted
                wait = _WaitAll(acc["futs"][acc["rcur"]], acc["casc"][acc["rcur"]]) if gen >= 2 else None
                view = acc["bufs"][acc["rcur"]][acc["reserved"]:acc["reserved"] + nplanes]
        acc["reserved"] += nplanes                               # mirror the completion side's accounting, fused or not
        while acc["reserved"] >= depth:
            acc["reserved"] -= depth; acc["rcur"] ^= 1; acc["gen_r"] += 1
        return view, wait

    def _commit_planes(self, k, nplanes):
        """the planes are already in level k's current buffer (written by the compression workers): advance the fill
        count; when the buffer is full, write it and cascade (as _append_from does after downsampling)"""
        acc = self._acc[k]; buf = acc["bufs"][acc["cur"]]
        acc["filled"] += nplanes
        if acc["filled"] > buf.shape[0]:
            raise RuntimeError(f"level {k} accumulator overfilled ({acc['filled']} > {buf.shape[0]})")
        if acc["filled"] == buf.shape[0]:
            if not self._submit_level(k, acc, buf.shape[0]):
                self._append_from(k + 1, buf)

class _WaitAll:
    """result() waits for every given future (None entries are skipped)"""
    def __init__(self, *futs): self.futs = [f for f in futs if f is not None]
    def result(self):
        for f in self.futs:
            f.result()


class _MSView:
    def __init__(self, ms, view0): self.ms, self.v = ms, view0
    @property
    def shape(self): return self.v.shape
    def write(self, block) -> Future:
        block = np.asarray(block)
        if block.dtype != self.ms.dtype: raise TypeError(f"block dtype {block.dtype} != {self.ms.dtype}")
        if block.shape != self.v.shape: raise ValueError(f"block shape {block.shape} != window {self.v.shape}")
        origin = tuple(a for a, _ in self.v.sl)
        if origin[1] != 0 or origin[2] != 0 or block.shape[1:] != self.ms.shape[1:]:
            raise ValueError("multiscale writes must cover whole frames (full y and x)")
        s0 = self.ms.shards[0]
        if origin[0] % s0 or (block.shape[0] % s0 and origin[0] + block.shape[0] < self.ms.shape[0]):
            raise ValueError(f"z window must start on a shard boundary ({s0}) and cover whole shards except at the end")
        with self.ms._order:
            seq = self.ms._next_submit; self.ms._next_submit += 1
            block = np.ascontiguousarray(block)
            ds_view, wait = self.ms._reserve_l1(block.shape) if len(self.ms.levels) > 1 else (None, None)   # under the order lock
        if wait is not None:
            wait.result()                                        # the reserved buffer's previous write must be finished
        return self.ms._exec.submit(self.ms._write_all, block, origin, seq, ds_view)
    def read(self): return self.v.read()


def create_multiscale(path, shape, *, shards, chunks, dtype="uint16", codec=None, levels=None, factors=(2, 2, 2),
                      scale=None, axes=("z", "y", "x"), units=None, threads=None, sync=True, writers=1, max_threads=None,
                      overwrite=True, min_size=256, sink="disk", pyramid_codec=None, name=None,
                      shard_policy="constant_bytes", level_shards=None, direct_io=False, calibration=None, origin=(0, 0)) -> MultiscaleArray:
    """Create an OME-NGFF 0.5 multiscale group written by the native writer. levels (required): number of
    downsampled levels below level 0; "auto" halves until the smallest axis would drop below min_size.
    factors: per-axis box size per level, default (2, 2, 2) = isotropic 2x2x2 mean; use (1, 2, 2) for xy only.
    scale: physical voxel size of level 0 (per axis) for the metadata. pyramid_codec: codec for levels >= 1
    (default: the same as level 0); e.g. numcodecs.Zstd(3) makes the pyramid readable by viewers that do not
    know the b4d codec (neuroglancer, viv) while level 0 stays B4D. Output: OME-NGFF 0.5 (zarr v3).
    Chunks are the same at every level (constant read cost per request). Shards per level follow shard_policy:
    "constant_bytes" (default) keeps voxels per shard roughly constant by doubling the shard's xy footprint each
    level (alternating y, x; clamped to the level), so files stay a similar size up the pyramid; "same" keeps the
    level-0 xy shard shape (files shrink 4x per level). level_shards: explicit list of (sz, sy, sx) per level.
    calibration, origin: as for create; level 0 only (averaging already pulls a noisy pixel towards its neighbours).
    threads / max_threads / writers / sink: as for create."""
    import shutil
    _check_sink(sink)
    if levels is None:
        raise ValueError("levels is required: an integer number of downsampled levels, or 'auto'")
    shape = tuple(int(v) for v in shape); factors = tuple(int(f) for f in factors)
    n = auto_levels(shape, factors, min_size) if levels == "auto" else int(levels)
    if n < 0:
        raise ValueError("levels must be >= 0")
    if factors[0] > 1 and shards[0] % factors[0]:
        raise ValueError(f"shard depth {shards[0]} must be a multiple of the z factor {factors[0]}")
    if overwrite and os.path.isdir(path): shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    nt, w = _threads(threads, writers, max_threads)
    arrays = []; datasets = []
    cur = shape; cur_scale = list(scale) if scale else [1.0] * 3
    def level_codec(k):
        """B4D: averaging n = prod(factors)**k pixels divides the variance by n, which is the same as a camera with
        conversion conv/n and read noise rn*sqrt(n); with that model q keeps its meaning (fraction of the level's
        true noise) and the per-pixel bound holds at every level. Other codecs are unchanged."""
        if k == 0:
            return codec
        if pyramid_codec is not None:
            return pyramid_codec
        if not (hasattr(codec, "q") and hasattr(codec, "conversion")):
            return codec
        n_ = float(np.prod(factors)) ** k
        cfg = _b4d_codec_cfg(codec)
        cfg["conversion"] = cfg["conversion"] / n_; cfg["read_noise"] = cfg["read_noise"] * np.sqrt(n_)
        return cfg
    sy_k, sx_k = shards[1], shards[2]
    for k in range(n + 1):
        sz = shards[0]                                                                       # same depth at every level
        ch = tuple(min(c, max(1, s)) if s < c else c for c, s in zip(chunks, cur))           # chunks never larger than the level
        if sz % ch[0]:                                                                       # keep shard depth a chunk multiple
            sz = _ceil_div(sz, ch[0]) * ch[0]
        if level_shards is not None:
            sz, sy_k, sx_k = level_shards[k]
        elif k > 0 and shard_policy == "constant_bytes":
            # the level has prod(factors) fewer voxels: grow the xy footprint (y, then x) to keep bytes per shard up
            lost = int(np.prod(factors))
            while lost > 1:
                if sy_k * 2 <= _ceil_div(cur[1], ch[1]) * ch[1] and sy_k <= sx_k: sy_k *= 2; lost //= 2
                elif sx_k * 2 <= _ceil_div(cur[2], ch[2]) * ch[2]: sx_k *= 2; lost //= 2
                elif sy_k * 2 <= _ceil_div(cur[1], ch[1]) * ch[1]: sy_k *= 2; lost //= 2
                else: break
        sh = (sz, min(sy_k, _ceil_div(cur[1], ch[1]) * ch[1]), min(sx_k, _ceil_div(cur[2], ch[2]) * ch[2]))
        sh = tuple(_ceil_div(s, c) * c for s, c in zip(sh, ch))                             # shards are chunk multiples
        md = _metadata(cur, dtype, sh, ch, inner_codecs_for(level_codec(k), np.dtype(dtype).itemsize), 0, "/", dimension_names=axes)
        lp = os.path.join(path, str(k)); _write_metadata(lp, md)
        arrays.append(ZarrArray(lp, md, nt if k == 0 else max(8, nt // 2), sync, 1, sink, direct_io))
        datasets.append({"path": str(k), "coordinateTransformations": [{"type": "scale", "scale": [float(v) for v in cur_scale]}]})
        cur = tuple(_ceil_div(c, f) for c, f in zip(cur, factors)); cur_scale = [s * f for s, f in zip(cur_scale, factors)]
    unit_of = (lambda a: units.get(a) if isinstance(units, dict) else units)
    ax = [{"name": a, "type": ("time" if a == "t" else "channel" if a == "c" else "space"),
           **({"unit": unit_of(a)} if unit_of(a) else {})} for a in axes]
    ms_entry = {"name": name or os.path.basename(path.rstrip("/\\")), "axes": ax, "datasets": datasets}
    if n:
        ms_entry["type"] = "mean"
        ms_entry["metadata"] = {"description": "box mean (integer, rounded)", "method": "b4d.zarr_store.create_multiscale",
                                "version": "1", "factors": list(factors)}
    group = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [ms_entry]}}}
    with io.open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(group, f, indent=1)
    if calibration is not None:
        arrays[0].set_calibration(calibration, origin)
    ms = MultiscaleArray(path, arrays, factors, nt)
    ms._exec = ThreadPoolExecutor(w, thread_name_prefix="ms-writer")
    return ms


def open_multiscale(path, *, threads=None, sync=True, writers=1, max_threads=None) -> MultiscaleArray:
    """Open a group written by create_multiscale for more writes (threads / writers / max_threads as for create)."""
    with io.open(os.path.join(path, "zarr.json")) as f:
        g = json.load(f)
    msd = g["attributes"]["ome"]["multiscales"][0]
    factors = tuple(msd.get("metadata", {}).get("factors", (2, 2, 2)))
    nt, w = _threads(threads, writers, max_threads)
    arrays = [open(os.path.join(path, d["path"]), threads=nt, sync=sync) for d in msd["datasets"]]
    ms = MultiscaleArray(path, arrays, factors, nt); ms._exec = ThreadPoolExecutor(w, thread_name_prefix="ms-writer")
    return ms


# ------------------------------------------------------------------ open / create
def _write_metadata(path, metadata):
    os.makedirs(path, exist_ok=True)
    with io.open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(metadata, f, indent=1)


def _metadata(shape, dtype, shards, chunks, inner_codecs, fill_value=0, key_sep="/", dimension_names=None):
    return {"zarr_format": 3, "node_type": "array", "shape": [int(v) for v in shape], "data_type": str(np.dtype(dtype)),
            **({"dimension_names": list(dimension_names)} if dimension_names else {}),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [int(v) for v in shards]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": key_sep}},
            "fill_value": fill_value, "attributes": {},
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "chunk_shape": [int(v) for v in chunks], "codecs": inner_codecs,
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                "index_location": "end"}}]}


def inner_codecs_for(codec, typesize=2) -> list:
    """codec: b4d.zarr3.B4D instance or b4d config dict -> [b4d]; "raw"/"bytes"/None -> [bytes];
    numcodecs codec instance -> [bytes, <codec>]; list of dicts -> as given."""
    if codec is None or codec in ("raw", "bytes"):
        return [{"name": "bytes", "configuration": {"endian": "little"}}]
    if isinstance(codec, list):
        return codec
    if hasattr(codec, "q") and hasattr(codec, "conversion"):
        return [{"name": "b4d", "configuration": _b4d_codec_cfg(codec)}]
    if isinstance(codec, dict) and "q" in codec:
        return [{"name": "b4d", "configuration": dict(codec)}]
    cfg = codec.get_config() if hasattr(codec, "get_config") else dict(codec)
    cid = cfg.pop("id")
    if cid == "blosc":
        sh = cfg.get("shuffle", 1); sh = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(sh, sh)
        return [{"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "blosc", "configuration": {"cname": cfg.get("cname", "lz4"), "clevel": int(cfg.get("clevel", 5)), "shuffle": sh,
                                                    "typesize": typesize, "blocksize": int(cfg.get("blocksize", 0))}}]
    return [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": cid, "configuration": cfg}]


def _threads(threads, writers, max_threads):
    """threads = threads per block write; writers = blocks in flight; max_threads = total budget (overrides)."""
    w = max(1, int(writers))
    if max_threads is not None:
        return max(1, int(max_threads) // w), w
    return (threads or os.cpu_count() or 8), w


def create(path, shape, *, shards, chunks, dtype="uint16", codec=None, fill_value=0, key_sep="/",
           threads=None, sync=True, overwrite=True, writers=1, max_threads=None, sink="disk", direct_io=False,
           calibration=None, origin=(0, 0)) -> ZarrArray:
    """Create a sharded zarr v3 array written by the native writer. shards / chunks: (z, y, x) shard and inner chunk
    shape (shards a multiple of chunks). codec: b4d.zarr3.B4D (or its configuration dict), a numcodecs codec, or
    "raw"/None (bytes). max_threads caps the total native threads the writer may use (= threads per block write x blocks in
    flight); threads sets the per-write count directly. Only one of them is needed. sink: "disk" or "memory" (compress
    and assemble the shards, write nothing). calibration, origin: see ZarrArray.set_calibration."""
    import shutil
    _check_sink(sink)
    if overwrite and os.path.isdir(path):
        shutil.rmtree(path)
    md = _metadata(shape, dtype, shards, chunks, inner_codecs_for(codec, np.dtype(dtype).itemsize), fill_value, key_sep)
    _write_metadata(path, md)
    nt, w = _threads(threads, writers, max_threads)
    arr = ZarrArray(path, md, nt, sync, w, sink, direct_io)
    if calibration is not None:
        arr.set_calibration(calibration, origin)
    return arr


def open(path_or_spec, *, create: bool = False, threads=None, sync=True, overwrite=False, writers=1, max_threads=None,
         direct_io=False) -> ZarrArray:
    """Open by path or spec. spec: {"driver": "zarr3", "kvstore": {"driver": "file", "path": ...}, "metadata": {...}}.
    With create=True the metadata in the spec is written (shape, data_type, chunk_grid {"name": "regular",
    "configuration": {"chunk_shape": shards}}, codecs = [sharding_indexed{chunk_shape, codecs}]); overwrite=True first
    deletes an existing array at the path."""
    spec = path_or_spec
    if isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec); md_spec = None
    else:
        if spec.get("driver", "zarr3") != "zarr3":
            raise ValueError("only the zarr3 driver is supported")
        kv = spec["kvstore"]; path = kv["path"] if isinstance(kv, dict) else str(kv).replace("file://", "")
        md_spec = spec.get("metadata")
    if create:
        if md_spec is None:
            raise ValueError("create=True needs metadata in the spec")
        import shutil
        if overwrite and os.path.isdir(path):
            shutil.rmtree(path)
        sh = md_spec["codecs"][0]["configuration"]
        md = _metadata(md_spec["shape"], md_spec["data_type"], md_spec["chunk_grid"]["configuration"]["chunk_shape"],
                       sh["chunk_shape"], sh["codecs"], md_spec.get("fill_value", 0),
                       md_spec.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/"))
        _write_metadata(path, md)
    else:
        with io.open(os.path.join(path, "zarr.json")) as f:
            md = json.load(f)
    nt, w = _threads(threads, writers, max_threads)
    return ZarrArray(path, md, nt, sync, w, "disk", direct_io)


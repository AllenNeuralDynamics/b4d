"""Simulate a camera streaming frames at a fixed rate into the native B4D zarr writer, and report whether the rate
was sustained (zero dropped frames). This is the measuring instrument behind every writer number in docs/; the
auto-tuner (b4d-tune) drives it.

    python stream_sim.py --shape 10640,14192 --fps 110 --seconds 30 --sink memory --zero-copy \
        --threads 128 --block 64 --bank 64 --blocks 3 --ring 64 --chunks 32,64,64 --shards 64,1408,3584 --q 1
    python stream_sim.py --shape 2048,2048 --fps 2000 --block 512 --bank 512 --ring 512 --sink disk -o D:\data\b4d_out --verify

Stages, as in a real acquisition:

  camera  : a producer thread "exposes" one frame every 1/fps s into a ring of --ring slots and never waits: if the
            ring is full when a frame is due the frame is DROPPED and counted. Frames come from a bank of --bank
            distinct synthetic frames (a sphere phantom observed through the camera model: --conv/--offset/--rn,
            --spot-peak, --max-adu for the sensor's bit depth), generated once and cached on disk.
  pull    : frames are copied into the current block of --block frames (= one shard depth) with --copy-threads
            memcpy bands, or, with --zero-copy, the block *is* the bank: no per-frame copy, the DMA case.
  write   : arr[z0:z1].write(block): the C writer compresses every chunk with --threads threads, assembles the
            shards and writes them; --max-commits blocks may be in flight. --sink memory assembles the shards and
            discards them (CPU / pipeline capacity, no drive in the loop); --sink disk writes them (--direct-io
            bypasses the OS cache). --multiscale N adds an OME-NGFF pyramid; --gpu-encode cuda uses the CUDA kernel.

Output every --status seconds and at the end: produced / pulled / written / DROPPED, ring occupancy, blocks in
flight, per-block latency, raw and compressed MB/s, ratio, the writer's compress and IO seconds per block, and a
machine-readable `RESULT {...}` line. --verify reads a sample of windows back and checks the q/2 sigma bound.
Memory: --blocks x --block frames (+ the bank unless --zero-copy).
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ------------------------------------------------------------------ synthetic frame bank (cached on disk)
# Location: --cache-dir / $B4D_CACHE_DIR / %LOCALAPPDATA%\b4d\cache (Windows) / ~/.cache/b4d. Disable: --no-cache / B4D_NO_CACHE=1.
CACHE_ENABLED = os.environ.get("B4D_NO_CACHE") != "1"


def cache_dir():
    d = os.environ.get("B4D_CACHE_DIR")
    if not d:
        d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache"), "b4d", "cache") if os.name == "nt" \
            else os.path.join(os.path.expanduser("~/.cache"), "b4d")
    os.makedirs(d, exist_ok=True)
    return d


def cached_arrays(name, key, make):
    """make() -> dict of numpy arrays. Cached as an uncompressed .npz keyed by `name` and the parameter dict."""
    if not CACHE_ENABLED:
        return make()
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir(), f"{name}_{h}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    out = make()
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, path)
    print(f"[cache] generated and saved {name} -> {path} ({sum(v.nbytes for v in out.values()) / 1e6:.0f} MB)", flush=True)
    return out


def configure_cache(a):
    global CACHE_ENABLED
    if getattr(a, "no_cache", False):
        CACHE_ENABLED = False; os.environ["B4D_NO_CACHE"] = "1"
    if getattr(a, "cache_dir", None):
        os.environ["B4D_CACHE_DIR"] = a.cache_dir


def make_spots(shape, conv=2.0, offset=100.0, rn=3.0, seed=7, bg=5.0, peak=(10.0, 50.0), diam=(2.0, 20.0), fill=0.2, psf_fwhm=2.0, max_adu=65535):
    """Sphere phantom: solid spheres (diameter uniform in `diam` px, sub-voxel centres, per-sphere brightness
    uniform in `peak` = (min, max) electrons) are placed at random until the UNION of spheres covers `fill` of
    the voxels; the volume is then convolved with an isotropic Gaussian PSF of FWHM `psf_fwhm` px (2 px =
    Nyquist sampling), added to a `bg` electron background, and observed with Poisson + read noise. uint16 DN."""
    shape = tuple(int(v) for v in shape)
    peak = tuple(peak) if hasattr(peak, "__len__") else (0.2 * float(peak), float(peak))
    key = dict(kind="spheres", shape=shape, conv=conv, offset=offset, rn=rn, seed=seed, bg=bg, peak=list(peak), diam=list(diam), fill=fill, psf=psf_fwhm)
    if max_adu != 65535:                                                  # the default keeps the existing cache keys valid
        key["max_adu"] = max_adu
    return cached_arrays("spheres", key, lambda: {"img": _make_spheres(shape, conv, offset, rn, seed, bg, peak, diam, fill, psf_fwhm, max_adu)})["img"]


def _make_spheres(shape, conv, offset, rn, seed, bg, peak, diam, fill, psf_fwhm, max_adu=65535):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    vol = np.zeros(shape, np.float32)                                     # sphere brightness (0 = empty)
    total = float(nz) * ny * nx; target = fill * total
    covered = 0; n = 0
    while covered < target:
        d = rng.uniform(diam[0], diam[1]); r = d / 2.0
        amp = rng.uniform(peak[0], peak[1])
        cz, cy, cx = rng.uniform(0, nz), rng.uniform(0, ny), rng.uniform(0, nx)
        z0, z1 = max(0, int(np.floor(cz - r))), min(nz, int(np.ceil(cz + r)) + 1)
        y0, y1 = max(0, int(np.floor(cy - r))), min(ny, int(np.ceil(cy + r)) + 1)
        x0, x1 = max(0, int(np.floor(cx - r))), min(nx, int(np.ceil(cx + r)) + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        dz2 = ((np.arange(z0, z1) - cz) ** 2).astype(np.float32); dy2 = ((np.arange(y0, y1) - cy) ** 2).astype(np.float32); dx2 = ((np.arange(x0, x1) - cx) ** 2).astype(np.float32)
        mask = (dz2[:, None, None] + dy2[None, :, None] + dx2[None, None, :]) <= r * r
        if not mask.any():                                                # tiny sphere between voxel centres: mark the nearest voxel
            mask = np.zeros_like(mask); mask[min(int(round(cz)) - z0, mask.shape[0] - 1), min(int(round(cy)) - y0, mask.shape[1] - 1), min(int(round(cx)) - x0, mask.shape[2] - 1)] = True
        sub = vol[z0:z1, y0:y1, x0:x1]
        newly = int(np.count_nonzero(mask & (sub == 0)))
        sub[mask] = amp                                                   # later spheres overwrite earlier ones where they overlap
        covered += newly; n += 1
    sigma = psf_fwhm / 2.3548
    if sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma, mode="reflect", truncate=4.0)   # separable Gaussian PSF (isotropic)
    out = np.empty(shape, np.uint16)
    for z in range(nz):
        e = rng.poisson(vol[z] + bg).astype(np.float32) + rng.normal(0.0, rn, (ny, nx)).astype(np.float32)
        out[z] = np.clip(np.round(offset + conv * e), 0, max_adu)
    return out


def make_synthetic(shape, conv=2.0, offset=100.0, rn=3.0, seed=7, bg=5.0, peak=50.0):
    shape = tuple(int(v) for v in shape)
    key = dict(shape=shape, conv=conv, offset=offset, rn=rn, seed=seed, bg=bg, peak=peak)
    return cached_arrays("synthetic", key, lambda: {"img": _make_synthetic(shape, conv, offset, rn, seed, bg, peak)})["img"]


def _make_synthetic(shape, conv=2.0, offset=100.0, rn=3.0, seed=7, bg=5.0, peak=50.0):
    """Smooth background + 3-D Gaussian blobs + Poisson/read noise, in DN (uint16).
    bg   : background level in electrons (ramps from bg to 2*bg across x)
    peak : brightest blob amplitude in electrons above background (blobs span 0.2*peak .. peak)
    Defaults are dim: <= ~50 e- signal, i.e. SNR <= ~6 with 3 e- read noise."""
    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in shape)
    if len(shape) == 2:
        shape = (1,) + shape
    nz, h, w = shape
    truth = np.empty(shape, np.float32)
    truth[:] = (bg + bg * np.arange(w, dtype=np.float32) / w)[None, None, :]
    n_blobs = max(1, nz * h * w // 700_000)
    for _ in range(n_blobs):
        cz, cy, cx = rng.integers(0, nz), rng.integers(0, h), rng.integers(0, w)
        amp = rng.uniform(0.2 * peak, peak); sz, sxy = rng.uniform(2, 5), rng.uniform(1.5, 3)
        z0, z1 = max(0, cz - 12), min(nz, cz + 13); y0, y1 = max(0, cy - 8), min(h, cy + 9); x0, x1 = max(0, cx - 8), min(w, cx + 9)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        truth[z0:z1, y0:y1, x0:x1] += amp * np.exp(-((zz - cz) ** 2 / (2 * sz ** 2) + ((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sxy ** 2)))
    img = np.empty(shape, np.uint16)
    for z in range(nz):                                            # plane by plane to bound memory
        e = rng.poisson(truth[z]).astype(np.float32) + rng.normal(0, rn, (h, w)).astype(np.float32)
        img[z] = np.clip(offset + conv * e, 0, 65535)
    return img.reshape(shape[1:]) if nz == 1 else img


def err_stats(img, out, cam):
    """(mean, max) of |img-out|/sigma, computed per leading slice to bound memory."""
    img2 = img.reshape(-1, img.shape[-1]) if img.ndim < 3 else img
    out2 = out.reshape(img2.shape)
    tot, n, mx = 0.0, 0, 0.0
    for a_, b_ in zip(img2, out2):
        e = np.abs(a_.astype(np.float32) - b_) / cam.sigma(a_).astype(np.float32)
        tot += float(e.sum()); n += e.size; mx = max(mx, float(e.max()))
    return tot / n, mx


# ------------------------------------------------------------------ native writer streaming
def run_native(a, bank, ny, nx, chunks, shards, total_frames, frame_bytes, path, cam, B4D):
    """Single-process streaming with the native zarr writer: camera thread -> ring -> pull into numpy blocks
    -> arr[z0:z1].write(block) (C thread pool compresses, assembles shards, writes files)."""
    import b4d.zarr_store as zs
    from b4d.hardware import auto_threads
    threads = a.threads or auto_threads()
    if a.codec == "b4d":
        codec = B4D(q=a.q, mode=a.mode, camera=cam, predictor=a.predictor, entropy=a.entropy, dither=a.dither)
    elif a.codec == "raw":
        codec = "raw"
    else:
        import numcodecs
        codec = numcodecs.Blosc(cname=a.codec, clevel=a.clevel, shuffle=numcodecs.Blosc.BITSHUFFLE)
    sink = "disk" if a.sink == "disk" else "memory"
    gpu_enc = None
    if a.gpu_encode:
        if a.codec != "b4d":
            raise SystemExit("--gpu-encode needs the b4d codec")
        if a.gpu_encode not in ("cuda", "emu"):
            raise SystemExit("--gpu-encode must be 'cuda' (CUDA kernel) or 'emu' (its CPU emulation)")
        from b4d.cuda_encoder import CudaTileEncoder, MultiGpuEncoder
        gkw = dict(q=a.q, mode=a.mode, backend=a.gpu_encode, predictor=a.predictor, entropy=a.entropy, batch_tiles=a.gpu_batch_tiles, slots=a.gpu_slots,
                   tile_copy=not a.gpu_no_tile_copy, sm_limit=a.gpu_sm, dither=a.dither)
        gdev = [int(v) for v in str(a.gpu_devices).split(",") if v.strip()]
        if len(gdev) > 1:
            gpu_enc = MultiGpuEncoder(cam, devices=gdev, split=a.gpu_split or "blocks", **gkw)
            print(f"gpu encode on {len(gdev)} devices {gdev}: " + ("every block split into bands of tile rows, one per device" if a.gpu_split
                  else "blocks dispatched to the device with the fewest in flight"), flush=True)
            if not a.gpu_split and a.max_commits < len(gdev) * a.gpu_slots:
                print(f"WARNING: --max-commits {a.max_commits} < devices x slots = {len(gdev) * a.gpu_slots}: the devices cannot all overlap", flush=True)
        else:
            gpu_enc = CudaTileEncoder(cam, device=gdev[0] if gdev else 0, **gkw)
        gpu_enc.prepare((a.block, ny, nx), tuple(chunks), a.dtype)
    if a.multiscale:
        lv = "auto" if str(a.multiscale).lower() == "auto" else int(a.multiscale)
        fac = tuple(int(v) for v in a.ms_factors.split(","))
        pc = None                                                       # b4d: same codec family, per-level noise model
        if a.ms_codec != "b4d":
            import numcodecs
            pc = "raw" if a.ms_codec == "raw" else (numcodecs.Zstd(level=a.clevel) if a.ms_codec == "zstd" else numcodecs.Blosc(cname=a.ms_codec, clevel=a.clevel, shuffle=numcodecs.Blosc.BITSHUFFLE))
        arr = zs.create_multiscale(path, (total_frames, ny, nx), dtype=a.dtype, shards=shards, chunks=chunks, codec=codec, levels=lv, factors=fac,
                                   threads=threads, sync=not a.no_sync, writers=a.max_commits, sink=sink, pyramid_codec=pc, direct_io=a.direct_io)
        ms_desc = f"; multiscale {arr.nlevels - 1} level(s), factors {fac}, pyramid codec {a.ms_codec}"
    else:
        arr = zs.create(path, (total_frames, ny, nx), dtype=a.dtype, shards=shards, chunks=chunks, codec=codec,
                        threads=threads, sync=not a.no_sync, writers=a.max_commits, sink=sink, direct_io=a.direct_io)
        ms_desc = ""
    if gpu_enc is not None:
        # --gpu-levels N: only levels < N on the GPU, the coarser ones on the native CPU encoder (byte-identical: the
        # kernel matches the C encoder), so a pyramid level is not uploaded again after the CPU downsampled it
        gl = (arr.levels if a.multiscale else [arr])
        for i, lv_ in enumerate(gl):
            if a.gpu_levels < 0 or i < a.gpu_levels:
                lv_.encoder = gpu_enc.encode_tiles
        if a.multiscale and 0 <= a.gpu_levels < len(gl):
            ms_desc += f"; levels >= {a.gpu_levels} encoded on the CPU"
    print(f"native writer: 1 process, {threads} native threads per block write, {a.max_commits} block(s) in flight; codec {a.codec}; "
          f"sink {'disk' if a.sink == 'disk' else 'null (shards assembled and discarded: CPU capacity only)'}{ms_desc}"
          + ("; zero-copy: frames land in the writer's buffers, no copy stage" if a.zero_copy else "")
          + (f"; chunks encoded on {getattr(gpu_enc, 'devices', None) or gpu_enc.device} (CUDA kernel), shards assembled natively" if gpu_enc else ""), flush=True)
    if a.zero_copy:
        if a.bank != a.block:
            raise SystemExit("--zero-copy needs --bank equal to --block (the bank stands in for the camera's DMA target)")
        blocks = [bank] * a.blocks           # the camera 'writes' into these; the data is already there, so no copy is made
    else:
        blocks = [np.zeros((a.block, ny, nx), a.dtype) for _ in range(a.blocks)]
        for b in blocks:
            for k in range(a.block): b[k].fill(0)
    if gpu_enc is not None and hasattr(gpu_enc, "register"):            # page-lock the writer's blocks now (~4 s per 31.5 GB): done
        t_reg = time.perf_counter(); n_reg = gpu_enc.register(blocks)    # lazily it fell into the first measured blocks (drops)
        print(f"page-locked {n_reg} writer block buffer(s) for the GPU upload in {time.perf_counter() - t_reg:.1f} s", flush=True)
    ring = queue.Queue()                                                  # batches of (first_frame, count); capacity counted in frames
    ring_frames = [0]; ring_lock = threading.Lock()
    stats = dict(produced=0, dropped=0, pulled=0, ring_max=0, frames_written=0, lat=[], compressed=0)
    frame_ids = []                                                        # camera frame index of each stored z (drops shift them)
    free = queue.Queue(); [free.put(i) for i in range(a.blocks)]
    pool = ThreadPoolExecutor(a.copy_threads); band = -(-ny // a.copy_threads)
    if a.zero_copy:
        def copy_frame(src, dst):            # DMA already put the frame where the writer reads it
            pass
        def copy_slab(first, m, dst, zoff):
            pass
    elif a.copy_chunk_mb > 0:
        # The row bands below are 32 pool.map tasks per batch, each an np.copyto of ~320 KB for the 8-bit
        # cameras (kinetix, gsprint): Python dispatch and the GIL, not memcpy, set the copy rate (13 / 7.8 GB/s with
        # 32 threads while memcpy does 57). A batch is one contiguous run in the bank and in the block, so it is
        # copied as byte ranges of >= copy_chunk_mb each, at most copy_threads of them.
        chunk = int(a.copy_chunk_mb * 2 ** 20)
        def _copy_flat(src, dst):
            s = src.reshape(-1).view(np.uint8); d = dst.reshape(-1).view(np.uint8); nb = s.size
            k = max(1, min(a.copy_threads, nb // max(1, chunk)))
            if k == 1:
                np.copyto(d, s); return
            step = -(-nb // k)
            list(pool.map(lambda i: np.copyto(d[i * step:(i + 1) * step], s[i * step:(i + 1) * step]), range(k)))
        def copy_frame(src, dst):
            _copy_flat(src, dst)
        def copy_slab(first, m, dst, zoff):  # frames first..first+m-1 (bank-wrapped) -> dst[zoff:zoff+m], whole runs
            while m > 0:
                b0 = first % a.bank; n = min(m, a.bank - b0)
                _copy_flat(bank[b0:b0 + n], dst[zoff:zoff + n])
                first += n; zoff += n; m -= n
    else:
        def copy_frame(src, dst):
            list(pool.map(lambda k: np.copyto(dst[k * band:(k + 1) * band], src[k * band:(k + 1) * band]), range(a.copy_threads)))
        def copy_slab(first, m, dst, zoff):  # frames first..first+m-1 (bank-wrapped) -> dst[zoff:zoff+m], rows split over the copy threads
            while m > 0:
                b0 = first % a.bank; n = min(m, a.bank - b0); src = bank[b0:b0 + n]; d = dst[zoff:zoff + n]
                list(pool.map(lambda k: np.copyto(d[:, k * band:(k + 1) * band], src[:, k * band:(k + 1) * band]), range(a.copy_threads)))
                first += n; zoff += n; m -= n
    aborted = [False]
    def camera():
        """time-driven: every tick, all frames that have become due are offered as one batch (a camera delivers
        frames at its rate regardless of how often the host looks); frames beyond the ring's capacity are dropped.
        --abort-drops N: once N frames have been dropped the camera stops (the run is already 'not sustained'; this
        keeps a failing probe from spending its whole length in the overlapped, all-cores-busy regime)"""
        t0 = time.perf_counter(); produced = 0
        tick = min(0.002, 0.5 / a.fps)
        while produced < total_frames and not aborted[0]:
            now = time.perf_counter()
            due = min(total_frames, int((now - t0) * a.fps + 1e-9))
            n = due - produced
            if n <= 0:
                time.sleep(max(0.0, min(tick, (produced + 1) / a.fps - (now - t0)))); continue
            with ring_lock:
                take = min(n, a.ring - ring_frames[0])
                if take > 0:
                    ring.put((produced, take)); ring_frames[0] += take
                stats["ring_max"] = max(stats["ring_max"], ring_frames[0])
            stats["dropped"] += n - take; produced += n; stats["produced"] = produced
            if a.abort_drops and stats["dropped"] >= a.abort_drops:
                aborted[0] = True
        ring.put(None)
    # warm-up blocks (not counted): the same path as the run, so pools, arenas, buffers and clocks are at steady state
    nwarm = 0 if a.no_warmup else max(1, a.warmup_blocks)
    for _ in range(nwarm):
        bid = free.get()
        for i in range(a.block): copy_frame(bank[i % a.bank], blocks[bid][i])
        arr[0:a.block].write(blocks[bid]).result(); free.put(bid)
    if a.multiscale:
        arr.reset_pyramid()                     # warm-up blocks must not leave planes in the level accumulators
    arr.stats.update(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0, t_compress=0.0, t_io=0.0)   # per-block times exclude the warm-up
    if gpu_enc is not None:
        gpu_enc.stats.update(blocks=0, upload_s=0.0, kernel_s=0.0, download_s=0.0, host_s=0.0, host_tiles=0)
    if a.multiscale:
        arr.stats["downsample_seconds"] = 0.0
        for lv_ in arr.levels: lv_.stats.update(blocks=0, raw_bytes=0, written_bytes=0, seconds=0.0, t_compress=0.0, t_io=0.0)
    print(f"warm-up: {nwarm} block(s) done (excluded from statistics)\n", flush=True)
    inflight = {}
    errors = []                                                           # exceptions raised by block writes
    # A block write can also never finish, with no exception (2-GPU imx811 copy mode past the WDDM page-locked
    # limit: an upload event that never completes, GPUs idle), and then the run waited in free.get() or close() for ever.
    # Watchdog: blocks in flight (or the writer closing) and no write completed for --stall-timeout s -> stacks, FAILED, exit 4.
    progress = [time.perf_counter()]; closing = [False]
    def watchdog():
        while True:
            time.sleep(min(5.0, a.stall_timeout / 4))
            idle = time.perf_counter() - progress[0]
            if (inflight or closing[0]) and idle > a.stall_timeout:
                import faulthandler as _fh, sys as _sys
                _fh.dump_traceback(file=_sys.stderr, all_threads=True)
                print(f"FAILED: stalled: no block write completed for {idle:.0f} s (--stall-timeout {a.stall_timeout:g}); "
                      f"{len(inflight)} block(s) in flight{', writer closing' if closing[0] else ''}", flush=True)
                _sys.stderr.flush(); os._exit(4)
    if a.stall_timeout > 0:
        threading.Thread(target=watchdog, daemon=True).start()
    def on_done(bid, t0, n):
        # A write that raised (CUDA out of memory, copy-mode 2-GPU imx811) used to raise here before free.put,
        # so the block never came back and the main loop waited in free.get() for ever (9 h in one run). Now the block always
        # returns, the error is kept and the camera stops; the run then ends with FAILED and exit status 3.
        def cb(fut):
            try:
                c = fut.result()
                stats["lat"].append(time.perf_counter() - t0); stats["frames_written"] += n; stats["compressed"] += c
            except BaseException as e:                                    # noqa: BLE001 - reported and turned into the exit status
                errors.append(e); aborted[0] = True
            finally:
                inflight.pop(bid, None); progress[0] = time.perf_counter(); free.put(bid)
        return cb
    def submit(z0_, blk, bid_, n_):
        stats["submitted"] = stats.get("submitted", 0) + 1
        if a.inject_write_error and stats["submitted"] >= a.inject_write_error:   # tests: this and every later timed write fails
            from concurrent.futures import Future
            f_ = Future(); f_.set_exception(RuntimeError(f"injected write error (block {stats['submitted']})"))
        elif a.inject_write_stall and stats["submitted"] >= a.inject_write_stall:   # tests: this and every later write never finishes
            from concurrent.futures import Future
            f_ = Future()
        else:
            f_ = arr[z0_:z0_ + blk.shape[0]].write(blk)
        f_.add_done_callback(on_done(bid_, time.perf_counter(), n_))
    t_start = time.perf_counter(); threading.Thread(target=camera, daemon=True).start()
    next_status = t_start + a.status
    bid = free.get(); zfill = 0; z0 = 0
    def status():
        el = time.perf_counter() - t_start; lat = stats["lat"]
        print(f"  t={el:6.1f}s produced {stats['produced']:5d}  pulled {stats['pulled']:5d}  written {stats['frames_written']:5d}  DROPPED {stats['dropped']:4d}  "
              f"ring {ring_frames[0]:3d}/{a.ring} (max {stats['ring_max']})  blocks in flight {len(inflight)}  free {free.qsize()}  "
              f"block latency {np.mean(lat) if lat else 0:4.1f}s (max {max(lat) if lat else 0:4.1f})", flush=True)
    while not errors:
        item = ring.get()
        if item is None: break
        first, count = item
        with ring_lock: ring_frames[0] -= count
        while count > 0 and not errors:
            m = min(count, a.block - zfill)
            copy_slab(first, m, blocks[bid], zfill); frame_ids.extend(range(first, first + m))
            zfill += m; stats["pulled"] += m; first += m; count -= m
            if zfill == a.block:
                inflight[bid] = True
                submit(z0, blocks[bid], bid, zfill)
                z0 += zfill; zfill = 0; bid = free.get()
        if time.perf_counter() >= next_status:
            status(); next_status = time.perf_counter() + a.status
    if zfill and not errors:
        inflight[bid] = True
        n = zfill if z0 + zfill >= total_frames else min(a.block, total_frames - z0)   # pad to a whole shard unless at the array edge
        submit(z0, blocks[bid][:n], bid, zfill)
    if errors:
        aborted[0] = True                                                 # camera thread stops at its next tick
    t_c = time.perf_counter()
    if aborted[0] and not errors:
        print(f"ABORTED after {time.perf_counter() - t_start:.1f} s: {stats['dropped']} frames dropped (>= --abort-drops {a.abort_drops}); "
              f"the camera stopped early, statistics cover the frames offered so far", flush=True)
    print("closing writer ...", flush=True)
    progress[0] = time.perf_counter(); closing[0] = True
    import faulthandler, sys as _sys
    faulthandler.dump_traceback_later(120, repeat=False, file=_sys.stderr)   # if close hangs: stacks of every thread after 2 min
    if a.multiscale: arr.close(verbose=True)
    else: arr.close()
    faulthandler.cancel_dump_traceback_later()
    print(f"writer closed in {time.perf_counter() - t_c:.1f} s", flush=True)
    if errors:
        print(f"FAILED: {len(errors)} block write(s) raised after {time.perf_counter() - t_start:.1f} s; first: "
              f"{type(errors[0]).__name__}: {errors[0]}", flush=True)
        raise SystemExit(3)
    t_end = time.perf_counter(); status()
    raw = stats["frames_written"] * frame_bytes; el = t_end - t_start; stored = stats["compressed"]
    print(f"\nacquisition {el:.1f} s: offered {a.fps * frame_bytes / 1e6:,.0f} MB/s, written {raw / 1e6 / el:,.0f} MB/s raw ({stored / 1e6 / el:,.0f} MB/s to disk), ratio {raw / max(stored, 1):.2f}x")
    print(f"frames: produced {stats['produced']}, DROPPED {stats['dropped']} ({100 * stats['dropped'] / max(stats['produced'], 1):.2f}%), written {stats['frames_written']}")
    st = arr.stats
    if a.multiscale:                                                     # roll up the level arrays' native stats
        lv0 = arr.levels[0].stats
        st = dict(st, t_compress=sum(l.stats.get("t_compress", 0.0) for l in arr.levels), t_io=sum(l.stats.get("t_io", 0.0) for l in arr.levels),
                  groups_mask=lv0.get("groups_mask", 0))
        lw = st.get("level_wait_seconds", 0.0)
        fb = st.get("fused_blocks", 0)
        print(f"multiscale: {arr.nlevels - 1} extra level(s); level-1 planes fused into the compression pass for {fb}/{st['blocks']} blocks; "
              f"ordered pyramid step {(st['downsample_seconds'] - lw) / max(st['blocks'], 1):.2f} s per block "
              f"(+ {lw / max(st['blocks'], 1):.2f} s waiting for level writes behind the drive); "
              f"levels 1+ add {100 * (sum(l.stats['written_bytes'] for l in arr.levels[1:]) / max(lv0['written_bytes'], 1)):.0f}% to the stored bytes")
    if a.direct_io:
        lv_stats = [l.stats for l in arr.levels] if a.multiscale else [arr.stats]
        du = sum(s_.get("direct_used", 0) for s_ in lv_stats); dfb = sum(s_.get("direct_fallback", 0) for s_ in lv_stats)
        print(f"direct IO: {du} shard files written unbuffered, {dfb} fell back to buffered writes" + (" (volume does not support direct IO)" if dfb and not du else ""))
    if gpu_enc is not None:
        gs = gpu_enc.stats; nb = max(gs["blocks"], 1)
        if "kernel_s" in gs:
            print(f"gpu encode ({getattr(gpu_enc, 'devices', None) or gpu_enc.device}, CUDA kernel): per block upload+layout {gs['upload_s'] / nb:.4f} s [{gs.get('upload_mode', '-')}], "
                  f"kernel {gs['kernel_s'] / nb:.4f} s, download {gs['download_s'] / nb:.4f} s, host mapping {gs['host_s'] / nb:.4f} s; "
                  f"tiles finished on the host: {gs['host_tiles']}")
            if "blocks_per_device" in gs:
                print(f"gpu blocks per device (incl. warm-up resets): {gs['blocks_per_device']}")
    groups = bin(st.get("groups_mask", 0)).count("1")
    print(f"native writer: compress {st.get('t_compress', 0) / max(st['blocks'], 1):.4f} s + shard IO {st.get('t_io', 0) / max(st['blocks'], 1):.4f} s per block; "
          f"worker threads observed in {groups} processor group(s)" + (" (Windows: threads confined to one group!)" if groups == 1 and os.name == "nt" and os.cpu_count() > 64 else ""))
    print(f"native writer: {st['blocks']} blocks, {st['seconds'] / max(st['blocks'], 1):.4f} s per block inside write (compress + shard IO) "
          f"= {st['raw_bytes'] / 1e6 / max(st['seconds'], 1e-9):,.0f} MB/s raw while writing; block latency mean {np.mean(stats['lat']):.1f} s, max {max(stats['lat']):.1f} s "
          f"(fill time {a.block / a.fps:.1f} s; must stay below {(a.blocks - 1) * a.block / a.fps:.1f} s)")
    print("SUSTAINED: no drops" if stats["dropped"] == 0 else "NOT SUSTAINED: frames dropped")
    if os.environ.get("B4D_SIM_LATS"):                     # profiling: every block's write latency (s), in order
        print("LATS " + json.dumps([round(float(v), 4) for v in stats["lat"]]), flush=True)
    print("RESULT " + json.dumps(dict(codec=a.codec, writer="native", sink=a.sink, fps=a.fps, threads=threads, copy_threads=a.copy_threads, copy_chunk_mb=a.copy_chunk_mb, gpu_levels=a.gpu_levels,
                                     blocks=a.blocks, block=a.block, ring=a.ring, produced=stats["produced"], dropped=stats["dropped"], written=stats["frames_written"],
                                     raw_mbs=raw / 1e6 / el, disk_mbs=stored / 1e6 / el, ratio=raw / max(stored, 1), ring_max=stats["ring_max"], inflight_max=a.max_commits,
                                     lat_mean=float(np.mean(stats["lat"])), lat_max=float(max(stats["lat"])), fill_time=a.block / a.fps, sustained=stats["dropped"] == 0,
                                     aborted=aborted[0])), flush=True)
    if a.verify and a.sink != "disk":
        print("verify: skipped (null sink)")
    elif a.verify:
        verify_sample(os.path.join(path, "0") if a.multiscale else path, bank, a, cam, len(frame_ids), frame_ids)
    t_x = time.perf_counter()
    del blocks, bank; import gc; gc.collect()
    print(f"buffers released in {time.perf_counter() - t_x:.1f} s", flush=True)
    if not (a.keep or a.out):
        shutil.rmtree(path, ignore_errors=True)

def verify_sample(array_path, bank, a, cam, n_stored, frame_ids=None, win=1024, nwin=3):
    """read a few win x win windows of a few frames back (whole frames through zarr-python would decode tens of
    thousands of chunks per frame) and check the error bound / losslessness"""
    import zarr
    import b4d.zarr3  # noqa: F401
    t0 = time.perf_counter()
    z = zarr.open_array(array_path, mode="r"); ny, nx = z.shape[1], z.shape[2]
    win = min(win, ny, nx)
    frames = sorted(set([0, n_stored - 1] + list(np.linspace(0, n_stored - 1, 4).astype(int))))
    rng = np.random.default_rng(3); worst = 0.0; n = 0
    for f in frames:
        cam_idx = frame_ids[f] if frame_ids is not None else f            # after drops, stored z != camera frame index
        for _ in range(nwin):
            y0 = int(rng.integers(0, max(1, ny - win + 1))); x0 = int(rng.integers(0, max(1, nx - win + 1)))
            back = z[f, y0:y0 + win, x0:x0 + win]; ref = bank[cam_idx % a.bank][y0:y0 + win, x0:x0 + win]
            if a.codec == "b4d" and a.q > 0:
                worst = max(worst, err_stats(ref[None], back[None], cam)[1])
            else:
                worst = max(worst, 0.0 if np.array_equal(ref, back) else float("inf"))
            n += 1
    bound = a.q / 2 if (a.codec == "b4d" and a.q > 0) else 0
    print(f"verify: {n} windows of {win}x{win} from {len(frames)} frames read back in {time.perf_counter() - t0:.1f} s; "
          f"max |error| {worst:.2f} sigma (bound {bound:g}) -> {'OK' if worst <= 0.55 * a.q + 0.2 else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ main


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--shape", default="10640,14192", help="frame Y,X")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seconds", type=float, default=30.0, help="simulated acquisition length (rounded to whole blocks)")
    p.add_argument("--min-blocks", type=int, default=2, help="run at least this many blocks however short --seconds is")
    p.add_argument("--block", type=int, default=32, help="frames per writer block = shard depth")
    p.add_argument("--blocks", type=int, default=3, help="writer blocks in the pool (>= 2; more = deeper buffering)")
    p.add_argument("--ring", type=int, default=16, help="camera ring buffer size in frames")
    p.add_argument("--bank", type=int, default=None, help="distinct synthetic frames to cycle through (default: --block, so every batch is the same N frames; cached on disk per shape x N)")
    p.add_argument("--chunks", default=None, help="Z,Y,X (default block,128,128)")
    p.add_argument("--shards", default=None, help="Z,Y,X (default block,1408,3584)")
    p.add_argument("--copy-threads", type=int, default=8, help="parallel memcpy bands per frame in the pull stage")
    p.add_argument("--copy-chunk-mb", type=float, default=0, help="0 = row bands per frame (one task per copy thread); N = each batch of frames copied as contiguous byte ranges of >= N MB on the copy threads (fewer, larger tasks)")
    p.add_argument("--codec", default="b4d", choices=["b4d", "raw", "lz4", "zstd"], help="chunk codec (raw/lz4/zstd use the same writer path with zarr's bytes/Blosc codecs)")
    p.add_argument("--clevel", type=int, default=5, help="Blosc compression level for lz4/zstd")
    p.add_argument("--q", type=float, default=1.0); p.add_argument("--mode", type=int, default=2)
    p.add_argument("--predictor", default="above"); p.add_argument("--entropy", default="rice")
    p.add_argument("--dither", action="store_true", help="b4d subtractive dither (format v8)")
    p.add_argument("--conv", type=float, default=2.0); p.add_argument("--offset", type=float, default=100.0); p.add_argument("--rn", type=float, default=3.0)
    p.add_argument("--vst", choices=["variance", "std", "anscombe"], default=os.environ.get("B4D_SIM_VST", "variance"),
                   help="camera curve: the exact VST (variance objective, the b4d default; or std) or the Anscombe curve "
                        "(env B4D_SIM_VST sets the default)")
    p.add_argument("--bg", type=float, default=5.0); p.add_argument("--peak", type=float, default=50.0)
    p.add_argument("--phantom", default="spheres", choices=["spheres", "blobs"],
                   help="spheres: solid spheres of diameter --diam px filling --fill of the volume, convolved with a --psf FWHM Gaussian (default); "
                        "blobs: the smooth blob field of the other benches")
    p.add_argument("--diam", default="2,20", help="spheres: diameter range in px (uniform)")
    p.add_argument("--fill", type=float, default=0.2, help="spheres: fraction of voxels covered by the union of spheres")
    p.add_argument("--psf", type=float, default=2.0, help="spheres: isotropic Gaussian PSF FWHM in px (2 = Nyquist)")
    p.add_argument("--spot-peak", default="10,50", help="spheres: brightness range in electrons above background, uniform per sphere (min,max)")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint8"], help="frame and array dtype (uint8 for 8-bit sensors, with --max-adu 255)")
    p.add_argument("--max-adu", type=int, default=65535, help="spheres: clip the phantom at the camera's full well in ADU (4095 for 12-bit, 255 for 8-bit sensors)")
    p.add_argument("-o", "--out", default=None); p.add_argument("--keep", action="store_true")
    p.add_argument("--no-sync", action="store_true"); p.add_argument("--status", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=None, help="compression threads per block write (default: physical cores, b4d.hardware.auto_threads)")
    p.add_argument("--direct-io", action="store_true", help="native writer: unbuffered write-through shard files (Windows NO_BUFFERING / Linux O_DIRECT)")
    p.add_argument("--gpu-encode", default=None,
                   help="native writer: encode the chunks on the GPU. 'cuda' = the CUDA kernel (CuPy); 'emu' = the kernel source run on the "
                        "CPU (tests). Shard assembly and IO stay native; output byte-identical")
    p.add_argument("--gpu-levels", type=int, default=-1, help="multiscale + CUDA: levels < N encoded on the GPU, the rest on the CPU (-1 = all on the GPU)")
    p.add_argument("--gpu-devices", default="0", help="CUDA kernel path: comma list of device ids; several = whole blocks dispatched across the GPUs (set --max-commits and --gpu-slots so max-commits = devices x slots)")
    p.add_argument("--gpu-split", choices=["rows"], default=None, help="several --gpu-devices: share every block out as bands of tile rows, one per device (each device holds only its band: room for more slots), instead of whole blocks per device")
    p.add_argument("--gpu-sm", type=int, default=0, help="CUDA kernel path: run the kernels on this many SMs only (persistent mode; 0 = all, the normal launch)")
    p.add_argument("--gpu-batch-tiles", type=int, default=2048, help="CUDA kernel path: tiles per launch; device memory per slot ~ block + 2.5 x tiles x tile bytes. Reduce if the GPU has little memory")
    p.add_argument("--gpu-no-tile-copy", action="store_true", help="CUDA kernel path: address tiles in place in the uploaded block (saves 0.5 GB per slot, ~3x slower kernel)")
    p.add_argument("--gpu-blocks-in-flight", "--gpu-slots", dest="gpu_slots", type=int, default=2,
                   help="CUDA kernel path: blocks resident on the device at once, each with its own stream and buffers (~ block + 1.6 GB); "
                        "should equal --max-commits. 3 hides more latency if memory allows")
    p.add_argument("--zero-copy", action="store_true",
                   help="native writer: the camera delivers frames straight into the writer's block buffers (as a DMA engine would): "
                        "no per-frame memcpy, no copy threads. Modelled by compressing the frame bank in place; needs --bank == --block")
    p.add_argument("--multiscale", default=None, help="native writer: number of pyramid levels (integer; 'auto' allowed); off by default")
    p.add_argument("--ms-factors", default="2,2,2", help="native writer: per-axis box size per level; default 2,2,2 (isotropic), 1,2,2 for xy only")
    p.add_argument("--ms-codec", default="b4d", choices=["b4d", "zstd", "lz4", "raw"],
                   help="native writer: codec for pyramid levels >= 1. b4d (default): B4D with each level's own noise model "
                        "(q keeps its meaning per level); zstd/lz4/raw: standard codecs so neuroglancer/viv can show the pyramid")
    p.add_argument("--sink", default="disk", choices=["disk", "memory"],
                   help="disk: shard files (default). memory: shards are assembled and discarded (no storage in the loop): CPU/pipeline capacity independent of the drive")
    p.add_argument("--no-cache", action="store_true"); p.add_argument("--cache-dir", default=None)
    p.add_argument("--verify", action="store_true", help="after the run, read back a sample of frames and check the WNL error bound")
    p.add_argument("--no-warmup", action="store_true", help="skip the full-size warm-up block (default: one block is run through the writers before timing)")
    p.add_argument("--warmup-blocks", type=int, default=1, help="full-size blocks to run through the writer before timing (3-4 settles clocks, pools and buffers at the edge)")
    p.add_argument("--max-commits", type=int, default=2, help="block writes in flight (the pool needs --blocks >= max-commits + 1)")
    p.add_argument("--inject-write-error", type=int, default=0, help="testing: the Nth and every later timed block write raise (the run must end FAILED, exit 3)")
    p.add_argument("--inject-write-stall", type=int, default=0, help="testing: the Nth and every later timed block write never finish (the watchdog must end the run, exit 4)")
    p.add_argument("--stall-timeout", type=float, default=600.0, help="end the run FAILED (exit status 4, stacks on stderr) when blocks are in flight "
                                                                       "(or the writer is closing) and no block write completes for this many seconds; 0 = off")
    p.add_argument("--abort-drops", type=int, default=0, help="native writer: stop the camera once this many frames have been dropped (0 = run the full length). "
                        "A dropping run is 'not sustained' either way; stopping early keeps the machine out of the all-cores-busy overlapped regime")
    a = p.parse_args()
    configure_cache(a)

    ny, nx = (int(v) for v in a.shape.split(","))
    if a.bank is None:
        a.bank = a.block
    chunks = tuple(int(v) for v in a.chunks.split(",")) if a.chunks else (a.block, 128, 128)
    shards = tuple(int(v) for v in a.shards.split(",")) if a.shards else (a.block, 1408, 3584)
    if shards[0] != a.block or any(s % c for s, c in zip(shards, chunks)):
        raise SystemExit("shard depth must equal --block, and shards must be multiples of chunks")
    # whole blocks only, and at least --min-blocks of them (short/slow runs are extended rather than refused)
    n_blocks = max(a.min_blocks, int(round(a.fps * a.seconds / a.block)))
    total_frames = n_blocks * a.block
    frame_bytes = ny * nx * np.dtype(a.dtype).itemsize
    out_dir = a.out or os.path.join(os.getcwd(), "stream_sim_out"); os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stream.zarr"); shutil.rmtree(path, ignore_errors=True)

    print(f"sink: {a.sink}{' (shards assembled and discarded: CPU/pipeline capacity, no drive in the loop)' if a.sink == 'memory' else ''}")
    print(f"codec: {a.codec}" + (f" (q={a.q:g}, mode {a.mode}, {a.predictor}/{a.entropy}{', dither' if a.dither else ''})" if a.codec == "b4d" else (f" (clevel {a.clevel}, bitshuffle)" if a.codec != "raw" else "")))
    print(f"frame {ny}x{nx} {a.dtype} = {frame_bytes / 1e6:.0f} MB; {a.fps:g} fps = {a.fps * frame_bytes / 1e6:,.0f} MB/s offered; "
          f"{total_frames} frames in {total_frames / a.fps:.0f} s; block {a.block} frames ({a.block * frame_bytes / 1e9:.1f} GB) x {a.blocks}; ring {a.ring}")
    print(f"frame bank: {a.bank} frames of {ny}x{nx} ({a.bank * frame_bytes / 1e9:.1f} GB; generated once, then loaded from the cache) ...", flush=True)
    if a.phantom == "spheres":
        dm = tuple(float(v) for v in a.diam.split(",")); pk = tuple(float(v) for v in a.spot_peak.split(","))
        print(f"  spheres, diameter {dm[0]:g}-{dm[1]:g} px, union fill {a.fill:.0%}, brightness {pk[0]:g}-{pk[1]:g} e- (uniform per sphere), "
              f"PSF FWHM {a.psf:g} px, on {a.bg:g} e- background")
        bank = make_spots((a.bank, ny, nx), a.conv, a.offset, a.rn, seed=11, bg=a.bg, peak=pk, diam=dm, fill=a.fill, psf_fwhm=a.psf, max_adu=a.max_adu)
    else:
        bank = make_synthetic((a.bank, ny, nx), a.conv, a.offset, a.rn, seed=11, bg=a.bg, peak=a.peak)
    if a.dtype == "uint8":                                                # an 8-bit sensor's frames, stored as 8-bit
        if bank.max() > 255:
            raise SystemExit("--dtype uint8 needs a bank within 0-255: pass --max-adu 255")
        bank = bank.astype(np.uint8)

    import b4d
    from b4d.zarr3 import B4D

    cam = b4d.Camera(a.conv, a.offset, a.rn, exact_vst=False if a.vst == "anscombe" else a.vst)
    run_native(a, bank, ny, nx, chunks, shards, total_frames, frame_bytes, path, cam, B4D)

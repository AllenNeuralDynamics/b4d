"""Measure the sustained write (and read) speed of a drive with the writer's own IO pattern: shard-sized files,
written whole, buffered + flushed or unbuffered write-through, from one or several threads, for long enough to
get past any write cache (consumer NVMe drives collapse after tens of GB).

    python disk_speed.py D:\\                        # 64 GB of 256 MB files, direct + buffered, read back
    python disk_speed.py D:\\ --total-gb 200 --file-mb 128 --threads 4 --mode direct
    python disk_speed.py /data --total-gb 32 --no-read

Reports MB/s per file over time (so a cache that empties shows up as a knee), the sustained rate after the first
--warmup-gb, the worst 10-second window, and a RESULT line. Compare the sustained write MB/s with the
"compressed GB/s at the recommended rate" that b4d-tune prints for the camera: the drive must sustain it
with margin, pyramid levels included (+15% for 2x2x2 levels).

Direct IO: Windows FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH (sector-aligned buffers), Linux O_DIRECT +
fsync; this is what the native writer does with direct_io=True: the drive DMAs each shard straight from the
assembly buffer, nothing is left in the OS page cache. Buffered: ordinary writes + flush + fsync per file, the
writer's fallback mode: the OS copies the data into its page cache first and drains it later, which costs a second
copy, caps the rate (3.4 vs 8.0 GB/s on the Micron 7450 of this server) and fills the standby list with every byte
written (after ~1 TB that is what slowed pinning the camera buffers). Use direct for acquisition on local drives;
buffered only where direct is refused (network shares, some USB/filesystems; the writer falls back by itself).
Neither is 'zero-copy': that term is the camera DMA writing frames straight into the block the compressor reads.
Files are deleted at the end (or kept with --keep).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import shutil
import sys
import threading
import time

import numpy as np

SECTOR = 4096


class DirectFile:
    """whole-file unbuffered write-through writer (Windows CreateFile flags or Linux O_DIRECT)"""

    def __init__(self, path: str, size: int):
        self.path, self.size = path, size
        if os.name == "nt":
            k = ctypes.windll.kernel32
            GENERIC_WRITE, CREATE_ALWAYS = 0x40000000, 2
            flags = 0x20000000 | 0x80000000 | 0x08000000  # NO_BUFFERING | WRITE_THROUGH | SEQUENTIAL_SCAN
            k.CreateFileW.restype = ctypes.c_void_p
            self.h = k.CreateFileW(path, GENERIC_WRITE, 0, None, CREATE_ALWAYS, flags, None)
            if self.h in (None, ctypes.c_void_p(-1).value):
                raise OSError(f"CreateFileW failed ({ctypes.GetLastError()})")
            self.k = k
        else:
            self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_DIRECT", 0), 0o644)

    def write(self, mv: memoryview) -> None:
        if os.name == "nt":
            written = ctypes.c_ulong(0)
            buf = (ctypes.c_char * len(mv)).from_buffer(mv)
            if not self.k.WriteFile(
                ctypes.c_void_p(self.h), buf, len(mv), ctypes.byref(written), None
            ) or written.value != len(mv):
                raise OSError(f"WriteFile failed ({ctypes.GetLastError()}): {written.value}/{len(mv)}")
        else:
            off = 0
            while off < len(mv):
                off += os.write(self.fd, mv[off:])

    def close(self) -> None:
        if os.name == "nt":
            self.k.CloseHandle(ctypes.c_void_p(self.h))
            if self.size % SECTOR:  # unbuffered writes are whole sectors: trim the tail to the true size
                with open(self.path, "r+b") as f:
                    f.truncate(self.size)
        else:
            os.fsync(self.fd)
            os.close(self.fd)


def aligned_buffer(nbytes: int) -> memoryview:
    """page-aligned, non-trivial data (compressors and drives both treat zeros specially)"""
    n = (nbytes + SECTOR - 1) // SECTOR * SECTOR
    m = mmap.mmap(-1, n)
    a = np.frombuffer(m, dtype=np.uint32)
    a[:] = np.random.default_rng(1).integers(0, 2**32, a.size, dtype=np.uint32)  # incompressible
    return memoryview(m)[:n]


def write_pass(
    root: str,
    mode: str,
    file_bytes: int,
    total_bytes: int,
    threads: int,
    warmup_bytes: int,
    status_every: float,
    keep: bool,
):
    nfiles = max(1, total_bytes // file_bytes)
    buf = aligned_buffer(file_bytes)
    lock = threading.Lock()
    done = []  # (t_end, bytes)
    next_idx = [0]
    t0 = time.perf_counter()
    paths = []

    def worker(k):
        while True:
            with lock:
                i = next_idx[0]
                next_idx[0] += 1
            if i >= nfiles:
                return
            p = os.path.join(root, f"shard_{mode}_{i:05d}.bin")
            ts = time.perf_counter()
            if mode == "direct":
                f = DirectFile(p, file_bytes)
                f.write(buf)
                f.close()
            else:
                with open(p, "wb", buffering=0) as f:
                    f.write(buf)
                    f.flush()
                    os.fsync(f.fileno())
            with lock:
                done.append((time.perf_counter(), file_bytes, time.perf_counter() - ts))
                paths.append(p)

    ths = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(threads)]
    for t in ths:
        t.start()
    last = t0
    shown = 0
    while any(t.is_alive() for t in ths):
        time.sleep(0.2)
        now = time.perf_counter()
        if now - last >= status_every:
            with lock:
                n = len(done)
                recent = [d for d in done if d[0] > last]
            mbs = sum(b for _, b, _ in recent) / 1e6 / (now - last)
            print(
                f"  t={now - t0:6.1f}s  {n * file_bytes / 1e9:7.1f} GB written  "
                f"{mbs:8,.0f} MB/s over the last {now - last:.0f} s",
                flush=True,
            )
            last = now
            shown += 1
    t_end = time.perf_counter()
    done.sort()
    el = t_end - t0
    total = len(done) * file_bytes
    # sustained rate after the warm-up bytes; worst 10 s window
    warm_i = min(len(done), max(1, warmup_bytes // file_bytes))
    t_warm = done[warm_i - 1][0] if warm_i < len(done) else t0
    sustained = (
        (total - warm_i * file_bytes) / 1e6 / max(t_end - t_warm, 1e-9) if warm_i < len(done) else total / 1e6 / el
    )
    ends = np.array([d[0] for d in done]) - t0
    worst = None
    if el > 20:
        for w0 in np.arange(0, el - 10, 2.0):
            nb = np.count_nonzero((ends > w0) & (ends <= w0 + 10)) * file_bytes
            worst = nb / 1e6 / 10 if worst is None else min(worst, nb / 1e6 / 10)
    per_file = [d[2] for d in done]
    res = dict(
        mode=mode,
        threads=threads,
        file_mb=file_bytes / 1e6,
        files=len(done),
        gb=total / 1e9,
        seconds=el,
        mbs_overall=total / 1e6 / el,
        mbs_sustained=sustained,
        mbs_worst_10s=worst,
        file_s_median=float(np.median(per_file)),
        file_s_max=float(max(per_file)),
    )
    print(
        f"{mode:8s} {threads} thread(s): {total / 1e9:.1f} GB in {el:.0f} s = {res['mbs_overall']:,.0f} MB/s overall, "
        f"{sustained:,.0f} MB/s after the first "
        f"{warm_i * file_bytes / 1e9:.0f} GB"
        + (f", worst 10 s window {worst:,.0f} MB/s" if worst is not None else "")
        + f"; per {file_bytes / 1e6:.0f} MB file median {res['file_s_median']:.3f} s, max {res['file_s_max']:.3f} s",
        flush=True,
    )
    return res, paths


def read_pass(paths, threads: int):
    """read the files back (sequential, whole files) with a cold-ish cache: reads in file order written"""
    if not paths:
        return None
    lock = threading.Lock()
    idx = [0]
    nbytes = [0]
    t0 = time.perf_counter()

    def worker():
        buf = bytearray(64 << 20)
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(paths):
                return
            with open(paths[i], "rb", buffering=0) as f:
                while True:
                    n = f.readinto(buf)
                    if not n:
                        break
                    with lock:
                        nbytes[0] += n

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.perf_counter() - t0
    print(
        f"read back: {nbytes[0] / 1e9:.1f} GB in {el:.0f} s = {nbytes[0] / 1e6 / el:,.0f} MB/s ({threads} thread(s))",
        flush=True,
    )
    return dict(gb=nbytes[0] / 1e9, seconds=el, mbs=nbytes[0] / 1e6 / el, threads=threads)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("drive", help="directory on the drive to test (a subfolder b4d_disk_speed is created there)")
    p.add_argument(
        "--total-gb",
        type=float,
        default=64,
        help="bytes to write per mode (go past the drive's write cache: 64+ GB for consumer NVMe)",
    )
    p.add_argument(
        "--file-mb",
        type=float,
        default=256,
        help="file size = the shard size the writer will produce (compressed shard, 50-250 MB)",
    )
    p.add_argument("--mode", default="both", choices=["direct", "buffered", "both"])
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="concurrent file writers (the native writer writes shards from several threads)",
    )
    p.add_argument(
        "--warmup-gb", type=float, default=8, help="bytes excluded from the sustained figure (cache / first-touch)"
    )
    p.add_argument("--status", type=float, default=5.0)
    p.add_argument("--no-read", action="store_true", help="skip the read-back pass")
    p.add_argument("--keep", action="store_true", help="leave the files on the drive")
    return p


def cleanup(root, paths):
    for pth in paths:
        try:
            os.remove(pth)
        except OSError:
            pass
    try:
        os.rmdir(root)
    except OSError:
        pass


def summary(results):
    print("\nsummary:")
    for mode in ("direct", "buffered"):
        if mode in results:
            r = results[mode]
            worst = f" (worst 10 s {r['mbs_worst_10s']:,.0f})" if r["mbs_worst_10s"] else ""
            print(f"  {mode:8s} write: sustained {r['mbs_sustained']:,.0f} MB/s{worst}")
    if results.get("read"):
        print(f"  read: {results['read']['mbs']:,.0f} MB/s")


def main(argv=None):
    a = build_parser().parse_args(argv)
    root = os.path.join(a.drive, "b4d_disk_speed")
    os.makedirs(root, exist_ok=True)
    usage = shutil.disk_usage(root)
    need = a.total_gb * 1e9 * (2 if a.mode == "both" else 1)
    print(
        f"drive {a.drive}: {usage.free / 1e9:,.0f} GB free of {usage.total / 1e9:,.0f}; "
        f"writing {a.total_gb:g} GB per mode in {a.file_mb:g} MB files, "
        f"{a.threads} thread(s), modes {a.mode}",
        flush=True,
    )
    if usage.free < need * 1.05:
        raise SystemExit(f"not enough free space: need {need / 1e9:.0f} GB")
    file_bytes = int(a.file_mb * 1e6) // SECTOR * SECTOR
    results = {}
    all_paths = []
    modes = ("direct", "buffered") if a.mode == "both" else (a.mode,)
    for mode in modes:
        try:
            res, paths = write_pass(
                root, mode, file_bytes, int(a.total_gb * 1e9), a.threads, int(a.warmup_gb * 1e9), a.status, a.keep
            )
        except OSError as e:
            print(f"{mode}: not supported here ({e})", flush=True)
            continue
        results[mode] = res
        all_paths += paths
        if not a.no_read and mode == modes[-1]:
            results["read"] = read_pass(paths, a.threads)
    if not a.keep:
        cleanup(root, all_paths)
    summary(results)
    print("RESULT " + json.dumps(dict(drive=a.drive, file_mb=a.file_mb, threads=a.threads, **results)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

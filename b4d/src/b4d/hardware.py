"""What the machine offers, derived at run time and overridable, so no compute path hard-codes a particular server.

* Compression threads (:func:`auto_threads`): the physical cores. The B4D encoder is compute-bound and two
  hyperthreads share one core's vector units; on the reference system (AMD EPYC 9754, 128 cores / 256 threads) the measured
  optimum is 128 (the tuning measurements, docs/WRITER_DESIGN.md, tried 96, 160 and 192). Override: the ``threads`` argument or ``$B4D_THREADS``.
* Logical CPUs (:func:`logical_cpus`): affinity-aware on Linux, ``os.cpu_count()`` elsewhere.
* Host RAM (:func:`host_memory`, :func:`pool_budget_bytes`): read without importing ``psutil``. The block pool of a
  source may use at most ``$B4D_POOL_FRACTION`` (default 0.5) of the available RAM; :func:`max_block_frames` turns
  that into the deepest block a camera's frame size allows.

Nothing here changes the writer's measured defaults on a machine like the reference one; it only stops a smaller
machine from inheriting them.
"""

from __future__ import annotations

import os
import sys

POOL_FRACTION = 0.5  # of the available RAM that the camera block pool may take


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    try:
        return int(v) if v else None
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    v = os.environ.get(name)
    try:
        return float(v) if v else None
    except ValueError:
        return None


def logical_cpus() -> int:
    """Logical CPUs this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def physical_cores() -> int | None:
    """Physical core count, or ``None`` when the platform does not say."""
    try:
        if sys.platform == "win32":
            import struct
            from ctypes import byref, create_string_buffer, windll, wintypes

            k32 = windll.kernel32
            relation_core = 0  # RelationProcessorCore
            length = wintypes.DWORD(0)
            k32.GetLogicalProcessorInformationEx(relation_core, None, byref(length))
            buf = create_string_buffer(length.value)
            if not k32.GetLogicalProcessorInformationEx(relation_core, buf, byref(length)):
                return None
            count = offset = 0
            while offset < length.value:  # records: DWORD Relationship, DWORD Size, union
                relationship, size = struct.unpack_from("II", buf, offset)
                count += relationship == relation_core
                offset += size
            return count or None
        if sys.platform == "darwin":
            from ctypes import CDLL, byref, c_int, c_size_t

            n, size = c_int(0), c_size_t(4)
            if CDLL(None).sysctlbyname(b"hw.physicalcpu", byref(n), byref(size), None, 0) == 0:
                return n.value or None
            return None
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            siblings = 0
            for part in f.read().strip().split(","):
                lo, _, hi = part.partition("-")
                siblings += (int(hi) - int(lo) + 1) if hi else 1
        return (os.cpu_count() or 1) // max(1, siblings)
    except Exception:
        return None


def auto_threads(requested: int | None = None) -> int:
    """Compression threads per block write: ``requested`` if > 0, else ``$B4D_THREADS``, else the physical cores
    (capped at the logical CPUs), else all logical CPUs."""
    if requested and requested > 0:
        return int(requested)
    env = _env_int("B4D_THREADS")
    if env and env > 0:
        return env
    logical = logical_cpus()
    return max(1, min(physical_cores() or logical, logical))


def thread_candidates(base: int | None = None) -> list[int]:
    """Thread counts worth measuring around ``base`` (default :func:`auto_threads`): the base first, then 0.75x,
    1.25x and 1.5x of it, within the logical CPUs. On the reference machine: 128, 96, 160, 192."""
    base = auto_threads(base)
    cap = logical_cpus()
    out = [base]
    for f in (0.75, 1.25, 1.5):
        t = max(1, round(base * f))
        if t <= cap and t not in out:
            out.append(t)
    return out


def host_memory() -> tuple[int, int] | tuple[None, None]:
    """``(total, available)`` host RAM in bytes, or ``(None, None)`` when it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MemoryStatus()
            st.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None, None
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemTotal"])
    except Exception:
        return None, None


def pool_budget_bytes() -> int | None:
    """Bytes of RAM a camera block pool may take (``$B4D_POOL_FRACTION`` of the available RAM, default 0.5);
    ``None`` when the memory cannot be read."""
    _, avail = host_memory()
    if avail is None:
        return None
    return int(avail * (_env_float("B4D_POOL_FRACTION") or POOL_FRACTION))


def max_block_frames(frame_bytes: int, nblocks: int = 3) -> int | None:
    """Deepest block (frames) for which ``nblocks`` blocks fit the pool budget; ``None`` when unknown."""
    budget = pool_budget_bytes()
    if budget is None:
        return None
    return int(budget // (max(1, nblocks) * max(1, frame_bytes)))


def describe() -> dict:
    """The machine as a small dict (stored next to tuning results so a recommendation says where it was measured)."""
    total, avail = host_memory()
    return dict(
        platform=sys.platform,
        logical_cpus=logical_cpus(),
        physical_cores=physical_cores(),
        auto_threads=auto_threads(),
        ram_gb=round(total / 1e9, 1) if total else None,
        ram_available_gb=round(avail / 1e9, 1) if avail else None,
    )

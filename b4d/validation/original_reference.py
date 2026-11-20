"""Reference material of the original B3D (github.com/balintbalazs/B3D, commit c1fa7b1) for the validation suite:

* the demo HDF5 pair shipped with the repository (uncompressed data + the same data compressed by the original filter
  in lossless, mode 1 and mode 2 at q = 1) -> 24 reference chunk bitstreams;
* the original's win64 GPU binary (B3D_cudaCompress.dll + the CUDA 10 runtime it ships with), driven through ctypes
  with CuPy device buffers: a live original encoder/decoder for arbitrary data (needs an NVIDIA GPU).

Files are taken from a local clone (``B4D_ORIGINAL_REPO``) or downloaded once into ``%LOCALAPPDATA%/b4d/validation``
and verified by sha256.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

COMMIT = "c1fa7b1e5228dd3c41901c5c284448f2a78f32d8"
RAW = f"https://raw.githubusercontent.com/balintbalazs/B3D/{COMMIT}/"
FILES = {
    "B3D_demo_drosophila_compressed.h5": "b7cf254352dbeed503bcda7aa29173f47a25523e5473dd53cef0b77459a4959f",
    "B3D_demo_drosophila_uncompressed.h5": "b3b6a4f5124063858736989ef8608c53e8527f117004d860bbbe9bb1f30ef09c",
    "win64/B3D_cudaCompress.dll": "237b0478f9ac4ad6157fef4a93289f17a43351ec8001466a218348a2c3087aaa",
    "win64/cudart64_100.dll": "7ca2ea3675f211078a7b0921bdfb49d633f0adbddcbc80f7faf5fcab136eb413",
}
#: the demo file's datasets -> (q, mode, conversion DN/e-, offset DN, read noise e-) from their cd_values
DEMO_CASES = {
    "B3D_Lossless": dict(q=0.0, mode=1, conv=1.0, offset=0.0, rn=0.0),
    "B3D_Mode1_1.00": dict(q=1.0, mode=1, conv=2.184, offset=0.0, rn=1.6),
    "B3D_Mode2_1.00": dict(q=1.0, mode=2, conv=2.184, offset=0.0, rn=1.6),
}


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / "b4d" / "validation" / "original_b3d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def reference_file(rel: str) -> Optional[Path]:
    """Path to one of FILES (local clone first, then the cache, then a download); None if unavailable."""
    repo = os.environ.get("B4D_ORIGINAL_REPO")
    if repo and (Path(repo) / rel).exists():
        return Path(repo) / rel
    dst = cache_dir() / Path(rel).name
    if dst.exists() and _sha256(dst) == FILES[rel]:
        return dst
    try:
        urllib.request.urlretrieve(RAW + rel, dst)
    except Exception:
        return None
    if _sha256(dst) != FILES[rel]:
        dst.unlink(missing_ok=True)
        return None
    return dst


def demo_chunks():
    """(data (8, 1600, 768) uint16, {dataset name: [8 chunk bitstreams]}) or None when the files are unavailable."""
    try:
        import h5py
    except ImportError:
        return None
    fc, fu = reference_file("B3D_demo_drosophila_compressed.h5"), reference_file("B3D_demo_drosophila_uncompressed.h5")
    if fc is None or fu is None:
        return None
    with h5py.File(fu, "r") as f:
        data = f["Data"][:]
    chunks = {}
    with h5py.File(fc, "r") as f:
        for name in DEMO_CASES:
            d = f[name]
            chunks[name] = [d.id.read_direct_chunk((z, 0, 0))[1] for z in range(d.shape[0])]
    return data, chunks


class StdVectorU32(C.Structure):  # MSVC release std::vector<unsigned int>: first, last, end
    _fields_ = [("first", C.c_void_p), ("last", C.c_void_p), ("end", C.c_void_p)]


class OriginalB3D:
    """The original GPU codec (its own binary) on the local NVIDIA device. One instance per chunk element count."""

    _dll = None

    @classmethod
    def available(cls) -> bool:
        try:
            cls._load()
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            return False

    @classmethod
    def _load(cls):
        if cls._dll is not None:
            return cls._dll
        if os.name != "nt":
            raise RuntimeError("the original binary is Windows-only")
        dll_path = reference_file("win64/B3D_cudaCompress.dll")
        rt_path = reference_file("win64/cudart64_100.dll")
        if dll_path is None or rt_path is None:
            raise RuntimeError("original binaries unavailable")
        os.add_dll_directory(str(dll_path.parent))
        C.CDLL(str(rt_path))
        dll = C.CDLL(str(dll_path))
        f = {}
        f["create"] = getattr(dll, "?createInstance@cudaCompress@@YAPEAVInstance@1@HIIII@Z")
        f["create"].restype = C.c_void_p
        f["create"].argtypes = [C.c_int, C.c_uint, C.c_uint, C.c_uint, C.c_uint]
        f["destroy"] = getattr(dll, "?destroyInstance@cudaCompress@@YAXPEAVInstance@1@@Z")
        f["destroy"].argtypes = [C.c_void_p]
        ll_args = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_size_t,
            C.c_int,
        ]
        lossy_args = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_size_t,
            C.c_float,
            C.c_float,
            C.c_int,
            C.c_float,
            C.c_float,
        ]
        ns = "cudaCompress@@YAX"
        vec = "AEAV?$vector@IV?$allocator@I@std@@@std@@"
        exports = (
            ("compressLL", f"?compressImageLL@{ns}PEAVInstance@1@{vec}PEAF22PEAGHHH_KH@Z", ll_args),
            ("compress", f"?compressImage@{ns}PEAVInstance@1@{vec}PEAFPEAM3PEAGHHH_KMMHMM@Z", lossy_args),
            ("decompressLL", f"?decompressImageLL@{ns}PEAVInstance@1@{vec}PEAF22PEAGHHH_KH@Z", ll_args),
            ("decompress", f"?decompressImage@{ns}PEAVInstance@1@{vec}PEAFPEAM3PEAGHHH_KMMHMM@Z", lossy_args),
        )
        for name, mangled, args in exports:
            fn = getattr(dll, mangled)
            fn.restype = None
            fn.argtypes = args
            f[name] = fn
        cls._dll = f
        return f

    def __init__(self, elem_count: int, device: int = 0):
        import cupy as cp

        self.f = self._load()
        self.cp = cp
        self.n = int(elem_count)
        # GPUResources: blockCountMax 1, elemCountPerBlockMax = elemCount * 2, codingBlockSize 0 (-> 128), log2 = 16
        self.inst = self.f["create"](device, 1, self.n * 2, 0, 16)
        if not self.inst:
            raise RuntimeError("createInstance failed")
        self.d_image = cp.zeros(self.n * 2, dtype=cp.int16)
        self.d_buffer = cp.zeros(self.n, dtype=cp.float32)
        self.d_scratch = cp.zeros(self.n, dtype=cp.float32)
        self.d_symbols = cp.zeros(self.n, dtype=cp.uint16)

    def _dims(self, shape):
        shp = tuple(shape)
        return ((1,) + shp) if len(shp) == 2 else shp

    def compress(
        self,
        frame_u16: np.ndarray,
        q: float = 0.0,
        mode: int = 1,
        conv: float = 1.0,
        bg: float = 0.0,
        rn: float = 0.0,
        tile: int = 24,
    ) -> bytes:
        cp = self.cp
        a = np.ascontiguousarray(frame_u16, dtype=np.uint16)
        sz, sy, sx = self._dims(a.shape)
        assert sz * sy * sx == self.n
        self.d_image[: self.n] = cp.asarray(a.reshape(-1).view(np.int16))
        self.d_symbols.fill(0)
        cp.cuda.runtime.deviceSynchronize()
        vec = StdVectorU32()
        ptrs = (
            int(self.d_image.data.ptr),
            int(self.d_buffer.data.ptr),
            int(self.d_scratch.data.ptr),
            int(self.d_symbols.data.ptr),
        )
        if q > 0:
            self.f["compress"](
                self.inst, C.byref(vec), *ptrs, sx, sy, sz, mode, float(q), float(bg), tile, float(conv), float(rn)
            )
        else:
            self.f["compressLL"](self.inst, C.byref(vec), *ptrs, sx, sy, sz, mode, tile)
        cp.cuda.runtime.deviceSynchronize()
        return C.string_at(vec.first, vec.last - vec.first)

    def decompress(
        self,
        payload: bytes,
        shape,
        q: float = 0.0,
        mode: int = 1,
        conv: float = 1.0,
        bg: float = 0.0,
        rn: float = 0.0,
        tile: int = 24,
    ) -> np.ndarray:
        cp = self.cp
        sz, sy, sx = self._dims(shape)
        words = np.frombuffer(payload, dtype="<u4")
        buf = (C.c_uint32 * len(words))(*words.tolist())
        end = C.addressof(buf) + 4 * len(words)
        vec = StdVectorU32(C.addressof(buf), end, end)
        self.d_symbols.fill(0)
        self.d_image.fill(0)
        cp.cuda.runtime.deviceSynchronize()
        ptrs = (
            int(self.d_image.data.ptr),
            int(self.d_buffer.data.ptr),
            int(self.d_scratch.data.ptr),
            int(self.d_symbols.data.ptr),
        )
        if q > 0:
            self.f["decompress"](
                self.inst, C.byref(vec), *ptrs, sx, sy, sz, mode, float(q), float(bg), tile, float(conv), float(rn)
            )
        else:
            self.f["decompressLL"](self.inst, C.byref(vec), *ptrs, sx, sy, sz, mode, tile)
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(self.d_image[: self.n]).view(np.uint16).reshape(tuple(shape))

    def close(self):
        if getattr(self, "inst", None):
            self.f["destroy"](self.inst)
            self.inst = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

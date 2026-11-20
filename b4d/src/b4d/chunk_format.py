"""Helpers of the CUDA encoder: the mode-2 quantiser table and the Blosc2 zero-run chunk, laid out exactly as
b4d_blosc2.c produces them."""

from __future__ import annotations

import struct

import numpy as np

# ---- constants mirroring b4d_blosc2.c -------------------------------------
_B4D_VERSION = 3
_CODEC_ID = 200


def _vst_fwd(x: np.ndarray, conv: float, offset: float, rn: float) -> np.ndarray:
    """Same arithmetic, same order, as vst_fwd() in b4d_blosc2.c (double precision)."""
    from . import _anscombe
    return _anscombe(x, conv, offset, rn * rn + 0.375)


def _build_lut(meta: int, conv: float, offset: float, rn: float) -> np.ndarray:
    """Mode-2 quantiser table: s(x) - smin, as lut16_get() builds it (rint = round-half-even)."""
    q = (meta >> 2) * 0.25
    x = np.arange(65536, dtype=np.float64)
    T = _vst_fwd(x, conv, offset, rn)
    smin = np.rint(_vst_fwd(np.array([0.0]), conv, offset, rn) / q)[0]
    s = np.rint(T / q) - smin
    s = np.clip(s, 0, 65535)
    return s.astype(np.int32)


def _blosc2_header(nbytes: int, typesize: int, blocksize: int, cbytes: int, meta: int, blosc2_flags: int = 0) -> bytes:
    hdr = struct.pack(
        "<BBBBiii", 5, _B4D_VERSION, 0xD5, typesize, nbytes, blocksize, cbytes
    )  # versionlz = codec version
    hdr += bytes(6) + bytes([_CODEC_ID, meta]) + bytes(6) + bytes([0, blosc2_flags])
    assert len(hdr) == 32
    return hdr


def _zero_chunk(nbytes, typesize, blocksize, meta) -> bytes:
    return _blosc2_header(nbytes, typesize, blocksize, 32, meta, blosc2_flags=0x10)  # special value 1: run of zeros

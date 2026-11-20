"""The original B3D codec (Balazs et al. 2017, github.com/balintbalazs/B3D) as implemented in b4d_original.c.

Two levels:

* ``b4d.compress(arr, predictor="b3d", ...)`` / ``b4d.decompress`` and the native writer use the original pipeline
  inside the usual B4DP + Blosc2 container (format v5). ``entropy`` is ignored. ``q`` is a multiple of 0.25 (the
  container's quantiser field), ``mode`` 1 or 2 as in the original (``q = 0`` is its lossless mode).
* :func:`encode` / :func:`decode` here work on one 2-D image and return / take the *bare* bitstream, i.e. exactly the
  bytes the original HDF5 filter (``H5Z_FILTER_B3D``, id 32016) stores for a chunk. That is what the validation suite
  compares against the original repository's demo file and its GPU binary.

The original GPU build used ``--use_fast_math``: its VST table (``(2 sqrt(max(x-offset,0)/conv + rn^2) - 2 rn) / q``
for the 65536 pixel values; approximate division and square root) differs from IEEE single precision in about 17% of
the entries by one ulp, which flips the quantiser's rounding roughly once per million pixels in the lossy modes
(lossless is unaffected). With an NVIDIA GPU
present, :func:`gpu_vst_table` reproduces the original's table exactly and :func:`register_vst_table` registers it
process-wide, after which every encoder call with those parameters (including the native writer) is byte-identical
to the original.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import Camera, _b4d

TILE = 24  #: the original filter's default tile size (cd_values[5])

_VST_SRC = r"""
extern "C" __global__ void vst_q(const float* in, float* out, int num, float offset, float conversion, float sigma,
                                 float invq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < num) {
        // cudaCompress util::vst as the CUDA 10 build evaluates it under --use_fast_math: approximate division and
        // square root, the add NOT fused (checked against the original binary on three camera models)
        float t = __fadd_rn(__fdividef(fmaxf(in[x] - offset, 0), conversion), sigma * sigma);
        float v = 2 * sqrtf(t) - 2 * sigma;
        out[x] = v * invq;                                   // util::multiply(1/quantStep)
    }
}
"""


def _params(camera: Optional[Camera]):
    cam = camera or Camera()
    return float(cam.conversion), float(cam.offset), float(cam.read_noise)


def encode(
    img: np.ndarray,
    q: float = 0.0,
    mode: int = 1,
    camera: Optional[Camera] = None,
    tile: int = TILE,
    vst_table: Optional[np.ndarray] = None,
) -> bytes:
    """Bare original bitstream of a 2-D uint16 image (a 3-D chunk is passed as rows = ny*nz, like the original filter).
    mode 1 or 2 (ignored when q == 0 -> lossless)."""
    a = np.ascontiguousarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.uint16)
    if a.dtype != np.uint16:
        raise TypeError("img must be uint8 or uint16")
    a2 = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(1, -1)
    conv, offset, rn = _params(camera)
    tab = None if vst_table is None else np.ascontiguousarray(vst_table, dtype=np.float32)
    return _b4d.orig_encode(
        a2, int(a2.shape[1]), int(a2.shape[0]), int(mode), float(q), conv, offset, rn, int(tile), tab
    )


def decode(
    payload: bytes, shape, q: float = 0.0, mode: int = 1, camera: Optional[Camera] = None, tile: int = TILE
) -> np.ndarray:
    """Inverse of :func:`encode`: uint16 array of `shape` (the last axis is the image width)."""
    shape = tuple(int(s) for s in shape)
    width = shape[-1]
    height = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    conv, offset, rn = _params(camera)
    out = _b4d.orig_decode(payload, width, height, int(mode), float(q), conv, offset, rn, int(tile))
    return np.frombuffer(bytes(out), dtype=np.uint16).reshape(shape)


def vst_table(q: float, camera: Optional[Camera] = None) -> np.ndarray:
    """The IEEE single-precision table the C encoder uses when no GPU table is registered."""
    conv, offset, rn = _params(camera)
    return np.frombuffer(_b4d.orig_vst_table(conv, offset, rn, float(q)), dtype=np.float32).copy()


def gpu_vst_table(q: float, camera: Optional[Camera] = None) -> np.ndarray:
    """The original GPU build's table: the same expression compiled with --use_fast_math on the local NVIDIA device
    (CuPy + NVRTC). Raises ImportError without CuPy / a device."""
    import cupy as cp

    conv, offset, rn = _params(camera)
    mod = cp.RawModule(code=_VST_SRC, options=("--use_fast_math",))
    xs = cp.arange(65536, dtype=cp.float32)
    out = cp.empty(65536, cp.float32)
    mod.get_function("vst_q")(
        (256,),
        (256,),
        (
            xs,
            out,
            np.int32(65536),
            np.float32(offset),
            np.float32(conv),
            np.float32(rn),
            np.float32(np.float32(1.0) / np.float32(q)),
        ),
    )
    return cp.asnumpy(out)


def register_vst_table(q: float, camera: Optional[Camera], table: Optional[np.ndarray]) -> None:
    """Process-wide override for (camera, q): every encoder call with these parameters uses `table` (None clears)."""
    conv, offset, rn = _params(camera)
    tab = None if table is None else np.ascontiguousarray(table, dtype=np.float32)
    _b4d.orig_register_vst_table(conv, offset, rn, float(q), tab)

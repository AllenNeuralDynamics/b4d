"""
b4d — B³D noise-aware microscopy image compression (CPU, Blosc2-based).

    import b4d, numpy as np
    img = np.load("stack.npy")                       # uint8 / uint16, 2-D or 3-D

    cam  = b4d.Camera(conversion=2.0, offset=100, read_noise=3.0)
    blob = b4d.compress(img, camera=cam, threads=8)   # defaults (m1_a3): q=1, mode=1, predictor="above3", entropy="rice"
    blob = b4d.compress(img, q=0)                     # lossless
    out  = b4d.decompress(blob, threads=8)            # numpy array, original shape/dtype

    b4d.save("stack.b4d", img, camera=cam)
    img2 = b4d.load("stack.b4d")

Zarr: `b4d.B4DCodec` is a numcodecs codec (zarr v2 format);
      `b4d.zarr3.B4D` is a zarr v3 ArrayBytesCodec (supports sharding).

Reference: Balázs et al. (2017) bioRxiv 10.1101/164624.
"""
from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

import os as _os
if _os.name == "nt":                      # let the extension find blosc2.dll (Python 3.8+ ignores PATH for extension DLLs)
    for _d in (_os.environ.get("B4D_DLL_DIR"),
               _os.path.join(_os.environ.get("BLOSC2_DIR", ""), "bin"),
               _os.path.join(_os.environ.get("CONDA_PREFIX", ""), "Library", "bin"),
               _os.path.join(_os.environ.get("VCPKG_ROOT", ""), "installed", "x64-windows", "bin")):
        if _d and _os.path.isdir(_d):
            try:
                _os.add_dll_directory(_d)
            except (AttributeError, OSError):
                pass

try:
    from . import _b4d
except ImportError as exc:                # pragma: no cover
    raise ImportError(
        "b4d._b4d extension not found. If you are running from the source tree, "
        "cd elsewhere (the un-built source package shadows the installed one), "
        "or build in place with `pip install -e .`") from exc

try:
    from importlib.metadata import PackageNotFoundError as _PNF, version as _pkg_version
    __version__ = _pkg_version("b4d")
except _PNF:                              # pragma: no cover - not installed (source tree on sys.path)
    __version__ = "0.1.0"                 # keep equal to pyproject.toml

__all__ = ["Camera", "PixelCalibration", "compress", "decompress", "compress_tiles", "save", "load", "exact_vst",
           "VST_MAX_ERR", "PREDICTORS", "ENTROPY_CODERS", "MODES", "CODEC_ID", "B4DCodec"]

CODEC_ID = _b4d.CODEC_ID
PREDICTORS = {"left_above": 0, "above": 1, "mean3d": 2, "b3d": 3, "above3": 4}
ENTROPY_CODERS = {"huffman": 0, "rice": 1, "ans": 2}
MODES = (0, 1, 2)

_MAGIC = b"B4DP"
_VERSION = 1
_DTYPES = {np.dtype("uint8"): 0, np.dtype("uint16"): 1}
_DTYPES_INV = {v: k for k, v in _DTYPES.items()}
_DEFAULT_MAX_CHUNK = 1 << 29        # 512 MiB per Blosc2 chunk (limit is ~2 GiB)


@dataclass(frozen=True)
class Camera:
    """Camera noise model used by the lossy modes.

    conversion : gain in DN per electron (e.g. 0.5 for 2 e-/DN)
    offset     : baseline / dark offset in DN
    read_noise : read noise in electrons RMS
    exact_vst  : "variance" (default), "std", True (= "variance") or False. Code through a curve fitted to this noise
                 model (format v9 / v11) instead of the generalised Anscombe
                 2 sqrt(e + rn^2 + 3/8), which is one sigma per unit only approximately near the offset (up to ~8 % off for
                 real cameras), so q means a different fraction of the noise in the dark than in the light.
                 True / "variance": a step of q adds q^2/12 of each pixel's own variance at every signal (as a noise
                 measurement on decoded data sees it); "std": std(T(X)) = 1 at every signal (the classic exact VST). The
                 two differ near the offset. The curve's description (200 bytes) travels in every chunk, so readers need
                 nothing else; b4d.exact_vst(camera) shows it. Modes 1 and 2.
    sigma(x)^2 = conversion * (x - offset) + (conversion * read_noise)^2   [DN^2]
    """
    conversion: float = 1.0
    offset: float = 0.0
    read_noise: float = 0.0
    exact_vst: Union[bool, str] = "variance"

    def sigma(self, x) -> np.ndarray:
        """Per-pixel noise standard deviation in DN for pixel values x."""
        e = np.maximum((np.asarray(x, dtype=np.float64) - self.offset) / self.conversion, 0.0)
        return self.conversion * np.sqrt(e + self.read_noise ** 2)


_VST_CACHE: dict = {}
_VST_LOCK = threading.Lock()


def _vst_key(cam: "Camera"):
    return (float(np.float32(cam.conversion if cam.conversion > 0 else 1.0)), float(np.float32(cam.offset)), float(np.float32(cam.read_noise)))


_VST_OBJECTIVES = {"std": 0, "variance": 1}


def _vst_objective(cam: "Camera") -> int:
    v = cam.exact_vst
    if v is True:
        return 1
    if isinstance(v, str) and v in _VST_OBJECTIVES:
        return _VST_OBJECTIVES[v]
    raise ValueError(f"exact_vst must be False, True, 'variance' or 'std', not {v!r}")


def _vst_entry(cam: "Camera"):
    """(capsule, max |residual|) of the camera's exact VST, fitted once per camera model, objective and process"""
    key = _vst_key(cam) + (_vst_objective(cam),)
    with _VST_LOCK:
        e = _VST_CACHE.get(key)
        if e is None:
            e = _VST_CACHE[key] = _b4d.vst_new(*key)
    return e


VST_MAX_ERR = 0.01      # a fit that leaves a larger residual is not used (Anscombe instead, with a warning)
_VST_WARNED: set = set()


def _vst_of(cam: Optional["Camera"]):
    """the exact VST capsule for the C codec, or None (Anscombe). A camera whose noise is well under one ADU (e.g. 0.33 ADU)
    has no curve that meets the objective: the fit is left far off (0.75 for the variance objective) and coded through it
    the data grew 30 %, so such a fit falls back to Anscombe."""
    if cam is None or getattr(cam, "exact_vst", False) in (False, None):
        return None
    try:
        cap, err = _vst_entry(cam)
    except ValueError:                   # no curve at all for this model (e.g. read noise 0, the default Camera())
        cap, err = None, float("inf")
    if err > VST_MAX_ERR:
        key = _vst_key(cam) + (_vst_objective(cam),)
        if key not in _VST_WARNED:
            _VST_WARNED.add(key)
            import warnings
            warnings.warn(f"b4d: no exact VST meets its objective for {cam} (residual {err:.3g} > {VST_MAX_ERR}); "
                          "coding with the Anscombe curve", RuntimeWarning, stacklevel=3)
        return None
    return cap


def _anscombe(x, conv, offset, c):
    """The generalised Anscombe T(x), as vst_fwd() in b4d_blosc2.c (double): 2 sqrt(e + c) for e = (x - offset) / conv >= 0,
    continued along its tangent below the offset; c = rn^2 + 3/8. The one copy of the formula on the Python side."""
    e = (x - offset) / conv
    return np.where(e >= 0, 2.0 * np.sqrt(np.maximum(e, 0.0) + c), 2.0 * np.sqrt(c) + e / np.sqrt(c))


def exact_vst(camera: "Camera", signals=None) -> dict:
    """The exact VST of a camera model (camera.exact_vst picks the objective; False is taken as True): its knots (lam_max in
    electrons, gains g), the table T(x) for x = 0..65535, the largest residual the fit left, and std(T(X) | signal) for the
    curve and for Anscombe at the given signals (electrons; default 0..400 e-)."""
    if camera.exact_vst in (False, None):
        import dataclasses
        camera = dataclasses.replace(camera, exact_vst=True)
    cap, err = _vst_entry(camera)
    lam, g = _b4d.vst_knots(cap)
    key = _vst_key(camera)
    lams = np.ascontiguousarray(np.concatenate([[0.0], np.geomspace(0.02, 400, 60)]) if signals is None else signals, dtype=np.float64)
    return dict(lam_max=lam, g=np.array(g), max_err=err, table=np.frombuffer(_b4d.vst_table(cap), np.float64).copy(), signals=lams,
                std_exact=np.frombuffer(_b4d.vst_std(cap, *key, lams.tobytes()), np.float64).copy(),
                std_anscombe=np.frombuffer(_b4d.vst_std(None, *key, lams.tobytes()), np.float64).copy())


class PixelCalibration:
    """Per-pixel values of one camera for the encoder (tools/calibrate_camera.py measures them).

    The camera model stays global and every pixel keeps its T(x). A short list of pixels that are noisier than the model gets
    a coarser step, in proportion to that pixel's own noise at the predicted level, and travels in each chunk (format v7), so
    decoding needs nothing but the chunk, at the speed of an ordinary one for all other pixels. Quieter pixels stay at the
    camera's step: a finer step there costs bits and preserves noise.

    read_noise_dn : (rows, cols) temporal sd in the dark, DN. Listed when at least min_raise x the camera model's noise.
    offset        : (rows, cols) mean dark level, DN, optional. Pixels at least offset_limit sigma from the sensor's level are
                    listed with that difference, removed before prediction and restored on decoding.
    response      : (rows, cols) flat-field response around 1, optional. Only used with flatten.
    skip          : a listed pixel whose step reaches skip x q is left out of the prediction of the row below; 0 = never.
    flatten       : encoder-only flat fielding from offset and response. No map is needed to decode, but the chunks then decode
                    to the corrected image, not to the raw values, so it is off unless asked for. Costs a dense pass per block.
    """
    TIER_RATIO = 1.35

    def __init__(self, read_noise_dn: np.ndarray, camera: Camera, *, offset: Optional[np.ndarray] = None,
                 response: Optional[np.ndarray] = None, min_raise: float = 1.5, skip: float = 1.25,
                 offset_limit: float = 1.0, flatten: bool = False):
        rn = np.asarray(read_noise_dn, dtype=np.float64)
        if rn.ndim != 2:
            raise ValueError("read_noise_dn must be (rows, cols)")
        self.camera = camera; self.shape = rn.shape; self.flatten = bool(flatten)
        min_raise, skip, offset_limit = float(min_raise), float(skip), float(offset_limit)
        self.min_raise, self.skip, self.offset_limit = min_raise, skip, offset_limit
        if min_raise <= 1:
            raise ValueError("min_raise must be above 1")
        conv = float(camera.conversion); c = float(camera.read_noise) ** 2 + 0.375
        # noise in T units at zero signal, against the model's 1: 1 + extra / C
        extra = np.maximum((rn / conv) ** 2 + 0.375 - c, 0.0)
        s0 = np.sqrt(1.0 + extra / c)
        # tier 0 is "no extra noise" (a listed offset only); the others are geometric, each at the middle of its range
        edges = min_raise * self.TIER_RATIO ** np.arange(7)
        tier = np.where(s0 >= edges[0], np.searchsorted(edges, s0, side="right") + 1, 0).astype(np.uint8)
        extra_var = [0.0] + [float(c * ((e * self.TIER_RATIO ** 0.5) ** 2 - 1.0)) for e in edges]
        delta = flat_a = flat_b = None
        if offset is not None:
            off = np.asarray(offset, dtype=np.float64)
            if off.shape != rn.shape:
                raise ValueError("offset and read_noise_dn differ in shape")
            ref = float(np.median(off))
            # T of a pixel in the dark, against the sensor's level: exact for a hot pixel, whatever its distance
            cam_off = float(camera.offset)
            d = _anscombe(off, conv, cam_off, c) - _anscombe(np.full_like(off, ref), conv, cam_off, c)
            if self.flatten:
                flat_a = np.clip(np.rint(d * 256), -32768, 32767).astype(np.int16)
            else:
                far = np.abs(d) >= offset_limit
                tier = np.where(far & (tier == 0), 1, tier).astype(np.uint8)
                delta = np.where(far, np.clip(np.rint(d * 256), -32768, 32767), 0).astype(np.int16)
        if self.flatten and response is not None:
            resp = np.asarray(response, dtype=np.float64)
            if resp.shape != rn.shape:
                raise ValueError("response and read_noise_dn differ in shape")
            flat_b = np.clip(np.rint((1.0 / np.maximum(resp, 1e-3) - 1.0) / 2.0 * 32768), -4095, 4095).astype(np.int16)
        self.listed = float((tier != 0).mean())
        self.tier = tier; self.delta = delta
        self._capsule = _b4d.pixmap_new(rn.shape[0], rn.shape[1], tier.tobytes(), None if delta is None else delta.tobytes(),
                                        None if flat_a is None else flat_a.tobytes(), None if flat_b is None else flat_b.tobytes(),
                                        extra_var, skip)

    @classmethod
    def load(cls, path, camera: Camera, **kw) -> "PixelCalibration":
        """from the .npz written by tools/calibrate_camera.py"""
        with np.load(path, allow_pickle=False) as z:
            if kw.get("flatten") and "response" in z:
                kw.setdefault("response", z["response"])
            return cls(z["read_noise_dn"], camera, offset=z["offset"], **kw)

    def _check(self, mode, q, predictor, camera: Camera, shape, origin):
        pred = _option(predictor, PREDICTORS, "predictor")
        if float(q) <= 0 or (int(mode), pred) not in ((1, PREDICTORS["above3"]), (2, PREDICTORS["above"])):
            raise ValueError("a per-pixel calibration needs q > 0 and mode=1 with predictor='above3', or mode=2 with predictor='above'")
        if int(mode) == 2 and self.flatten:
            raise ValueError("flatten needs mode=1")
        if (camera.conversion, camera.offset, camera.read_noise) != (self.camera.conversion, self.camera.offset, self.camera.read_noise):
            raise ValueError("the calibration was built for another camera model")
        y0, x0 = int(origin[0]), int(origin[1])
        if y0 < 0 or x0 < 0 or y0 + shape[0] > self.shape[0] or x0 + shape[1] > self.shape[1]:
            raise ValueError(f"image {tuple(shape)} at {(y0, x0)} does not fit the calibration map {self.shape}")
        return y0, x0


def _meta(mode: int, q: float) -> int:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    qi = int(round(q * 4))
    if qi < 0 or qi > 63:
        raise ValueError("q must be in [0, 15.75] (steps of 0.25 sigma)")
    if qi == 0:
        mode = 0
    if mode and abs(q * 4 - qi) > 1e-6:
        raise ValueError("q must be a multiple of 0.25")
    return (qi << 2) | mode


def _option(value: Union[str, int], table: dict, name: str) -> int:
    if isinstance(value, str):
        try:
            return table[value.lower()]
        except KeyError:
            raise ValueError(f"{name} must be one of {list(table)}") from None
    if value in table.values():
        return int(value)
    raise ValueError(f"{name} must be one of {list(table)}")


def _default_blocksize(row_bytes: int, nbytes: int, target: int = 1 << 20) -> int:
    """Whole rows per block, ~1 MiB."""
    if row_bytes <= 0:
        return 0
    rows = max(1, target // row_bytes)
    return int(min(rows * row_bytes, nbytes))


def _check_dither(dither: Union[bool, str], mode: int, q: float, predictor: Union[str, int], calibration=None) -> int:
    """0 off, 1 every pixel, 2 the calibration's listed pixels only (the codec's b4d_params.dither)"""
    if not dither:
        return 0
    if dither not in (True, 1, "listed", "all"):
        raise ValueError("dither must be False, True (every pixel) or 'listed' (the calibration's listed pixels only)")
    if int(mode) not in (1, 2) or float(q) <= 0:
        raise ValueError("dither needs mode 1 or 2 and q > 0")
    if _option(predictor, PREDICTORS, "predictor") == PREDICTORS["b3d"]:
        raise ValueError("dither is not available with predictor='b3d'")
    if dither == "listed":
        if calibration is None:
            raise ValueError("dither='listed' needs a per-pixel calibration")
        return 2
    return 1


Dither = Union[bool, Literal["listed"]]


def compress(arr: np.ndarray, *, q: float = 1.0, mode: int = 1,
             camera: Optional[Camera] = None,
             predictor: Union[str, int] = "above3",
             entropy: Union[str, int] = "rice",
             threads: int = 1, blocksize: Optional[int] = None,
             row_width: Optional[int] = None,
             max_chunk_bytes: int = _DEFAULT_MAX_CHUNK, dither: Dither = False,
             calibration: Optional[PixelCalibration] = None, origin: Sequence[int] = (0, 0)) -> bytes:
    """Compress a uint8/uint16 array (any dimensionality) to a self-describing blob.

    The defaults are the m1_a3 configuration: q=1, mode=1, predictor="above3", entropy="rice".

    q          : quantisation step in units of the noise sigma; 0 = lossless.
    mode       : 1 = quantise prediction residuals (best ratio),
                 2 = quantise pixels before prediction (localisation microscopy).
    camera     : Camera noise model (required for q > 0 to be meaningful).
    predictor  : "left_above" (best ratio on structured data), "above" (row-parallel, fastest; the acquisition
                 default) or "mean3d" (rounded mean of the 3 above-row neighbours and the previous plane's 3x3 patch:
                 noise-free prediction, row-parallel; format v4; planes = arr.shape[-2] rows for 3-D input), "above3" (rounded
                 mean of the 3 above-row neighbours only; with mode=1 it reproduces the raw histogram; format v4), or
                 "b3d": the B3D codec of Balazs et al. 2017 end to end (24x24-tile predictor, its VST and
                 quantiser, run-length + Huffman; `entropy` is ignored; format v5; see b4d.b3d).
    entropy    : "ans" (smallest, adaptive contexts), "huffman", or "rice" (fastest write).
    threads    : Blosc2 threads for this call.
    row_width  : image width for 2-D prediction; defaults to arr.shape[-1].
    dither     : seeded subtractive dither (modes 1 and 2, q > 0; format v8): True dithers every pixel, "listed" only the
                 calibration's listed pixels. Removes the signal-dependent mean bias of the quantiser, for about q^2/12 more
                 variance and a larger output. The per-pixel bound is unchanged. Off by default.
    calibration: PixelCalibration of this camera (mode=1 with predictor="above3", format v7, or mode=2 with predictor="above",
                 format v8, when the image has listed pixels).
    origin     : (row, col) of arr's first pixel on the calibration map. Each piece is then one Blosc2 block, so threads does
                 not speed it up; compress_tiles and the native zarr writer are the fast paths. A piece needs enough planes to
                 pay for its list (1/400 of its raw bytes at most; otherwise it is written without one).
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _DTYPES:
        raise TypeError(f"unsupported dtype {arr.dtype}; use uint8 or uint16")
    if arr.ndim == 0:
        raise ValueError("scalar input")
    cam = camera or Camera()
    meta = _meta(mode, q)
    dither = _check_dither(dither, mode, q, predictor, calibration)
    pred = _option(predictor, PREDICTORS, "predictor")
    ent = _option(entropy, ENTROPY_CODERS, "entropy")
    ts = arr.dtype.itemsize
    rw = arr.shape[-1] if (row_width is None and arr.ndim >= 2) else int(row_width or 0)
    row_bytes = (rw or arr.shape[-1]) * ts
    # mean3d: planes are the second-to-last axis; chunks and Blosc2 blocks hold whole planes so the 3-D taps stay valid
    ph = int(arr.shape[-2]) if (pred == PREDICTORS["mean3d"] and arr.ndim >= 3 and row_width is None) else 0
    my0 = mx0 = 0; map_rows = 0
    vst = _vst_of(cam) if (pred != PREDICTORS["b3d"] and meta & 3) else None     # lossless (mode 0) never uses the curve
    if calibration is not None:
        if arr.ndim < 2 or row_width is not None:
            raise ValueError("a per-pixel calibration needs a 2-D or 3-D image and no row_width")
        my0, mx0 = calibration._check(mode, q, predictor, cam, arr.shape[-2:], origin)
        map_rows = int(arr.shape[-2])

    # split along the leading axis so every piece fits one Blosc2 chunk
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(1, -1)
    rows_per_chunk = max(1, min(flat.shape[0], max_chunk_bytes // max(row_bytes, 1)))
    if ph or map_rows:
        rows_per_chunk = max(ph or map_rows, (rows_per_chunk // (ph or map_rows)) * (ph or map_rows))
    pieces = []
    for r0 in range(0, flat.shape[0], rows_per_chunk):
        piece = flat[r0:r0 + rows_per_chunk]
        bs = _default_blocksize(row_bytes, piece.nbytes) if blocksize is None else int(blocksize)
        if pred == PREDICTORS["b3d"] and blocksize is None:
            bs = piece.nbytes  # one block per piece: the block payload is the original filter's bitstream of the whole piece
        if ph and bs:
            pb = ph * row_bytes; bs = max(1, bs // pb) * pb; bs = min(bs, piece.nbytes)
        if calibration is not None:     # one block per piece: the codec is told where the block's rows sit, not where a block starts
            pieces.append(_b4d.compress(piece, ts, meta, rw, pred, ent, float(cam.conversion), float(cam.offset), float(cam.read_noise),
                                        1, piece.nbytes, map_rows, dither, calibration._capsule, my0, mx0, vst=vst))
            continue
        pieces.append(_b4d.compress(piece, ts, meta, rw, pred, ent,
                                    float(cam.conversion), float(cam.offset), float(cam.read_noise),
                                    int(threads), bs, ph, dither, vst=vst))

    header = struct.pack("<4sBBB", _MAGIC, _VERSION, _DTYPES[arr.dtype], arr.ndim)
    header += struct.pack(f"<{arr.ndim}Q", *arr.shape)
    header += struct.pack("<I", len(pieces))
    header += struct.pack(f"<{len(pieces)}I", *(len(p) for p in pieces))
    return header + b"".join(pieces)


def compress_tiles(arr: np.ndarray, tile: Sequence[int], *, q: float = 1.0, mode: int = 1,
                   camera: Optional[Camera] = None, predictor: Union[str, int] = "above3",
                   entropy: Union[str, int] = "rice", threads: int = 1, blocksize: Optional[int] = None,
                   dither: Dither = False, calibration: Optional[PixelCalibration] = None, origin: Sequence[int] = (0, 0)) -> list:
    """Compress a 3-D uint8/uint16 array as independent (tz, ty, tx) tiles in ONE call (no per-tile Python
    work; the GIL is released for the whole loop). Returns a list of blobs in C order over the tile grid,
    each identical to compress(arr[tile], ...) with row_width = tile width. Edge tiles are partial.
    This is the fast path for writing many small zarr chunks from a large frame."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 3 or arr.dtype not in _DTYPES:
        raise TypeError("compress_tiles expects a 3-D uint8/uint16 array")
    tz, ty, tx = (int(v) for v in tile)
    cam = camera or Camera()
    meta = _meta(mode, q)
    dither = _check_dither(dither, mode, q, predictor, calibration)
    pred = _option(predictor, PREDICTORS, "predictor"); ent = _option(entropy, ENTROPY_CODERS, "entropy")
    ts = arr.dtype.itemsize
    bs = _default_blocksize(tx * ts, tz * ty * tx * ts) if blocksize is None else int(blocksize)   # clamped per tile in C
    extra = (None, 0, 0)
    if calibration is not None:
        extra = (calibration._capsule, *calibration._check(mode, q, predictor, cam, arr.shape[-2:], origin))
    extra = extra + (_vst_of(cam) if (pred != PREDICTORS["b3d"] and meta & 3) else None,)
    chunks = _b4d.compress_tiles(arr, *arr.shape, tz, ty, tx, ts, meta, pred, ent,
                                 float(cam.conversion), float(cam.offset), float(cam.read_noise), int(threads), bs, dither, *extra)
    # wrap each Blosc2 chunk in the B4DP container so blobs decode with decompress()/the zarr codecs
    nz, ny, nx = arr.shape
    out = []
    k = 0
    for bz in range(-(-nz // tz)):
        for by in range(-(-ny // ty)):
            for bx in range(-(-nx // tx)):
                shp = (min(tz, nz - bz * tz), min(ty, ny - by * ty), min(tx, nx - bx * tx))
                c = chunks[k]; k += 1
                hdr = struct.pack("<4sBBB", _MAGIC, _VERSION, _DTYPES[arr.dtype], 3) + struct.pack("<3Q", *shp) + struct.pack("<II", 1, len(c))
                out.append(hdr + c)
    return out


def _parse_header(buf):
    mv = memoryview(buf)
    magic, ver, dcode, ndim = struct.unpack_from("<4sBBB", mv, 0)
    if magic != _MAGIC:
        raise ValueError("not a B4D blob")
    if ver != _VERSION:
        raise ValueError(f"unsupported B4D blob version {ver}")
    pos = 7
    shape = struct.unpack_from(f"<{ndim}Q", mv, pos); pos += 8 * ndim
    (npieces,) = struct.unpack_from("<I", mv, pos); pos += 4
    sizes = struct.unpack_from(f"<{npieces}I", mv, pos); pos += 4 * npieces
    return _DTYPES_INV[dcode], tuple(shape), sizes, pos, mv


def decompress(buf, *, threads: int = 1, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Decompress a blob from compress(); returns a writable array."""
    dtype, shape, sizes, pos, mv = _parse_header(buf)
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.dtype != dtype or out.shape != shape or not out.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous array with matching shape/dtype")
    flat = out.reshape(-1)
    off = 0
    for sz in sizes:
        chunk = _b4d.decompress(mv[pos:pos + sz], int(threads))
        piece = np.frombuffer(chunk, dtype=dtype)
        flat[off:off + piece.size] = piece
        off += piece.size
        pos += sz
    if off != n:
        raise ValueError("blob does not match its declared shape")
    return out


def save(path, arr: np.ndarray, **kwargs) -> int:
    """compress(arr, **kwargs) to a file (same keywords and defaults); returns the number of bytes written."""
    blob = compress(arr, **kwargs)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def load(path, *, threads: int = 1) -> np.ndarray:
    """decompress() a file written by save()."""
    with open(path, "rb") as f:
        return decompress(f.read(), threads=threads)


# ---------------------------------------------------------------------------
# numcodecs / zarr integration (optional)
# ---------------------------------------------------------------------------
try:
    from numcodecs.abc import Codec as _Codec
    from numcodecs.registry import register_codec as _register_codec
except ImportError:                    # pragma: no cover
    _Codec = None

if _Codec is not None:
    class B4DCodec(_Codec):
        """numcodecs codec: zarr.create(..., compressor=b4d.B4DCodec(camera=...)); defaults as b4d.compress (m1_a3).

        Shape/dtype travel inside each encoded chunk, so no `dtype` argument is needed. dither=True dithers every pixel
        (there is no calibration here).
        """
        codec_id = "b4d"

        def __init__(self, *, q: float = 1.0, mode: int = 1, camera: Optional[Union[Camera, dict]] = None,
                     predictor: str = "above3", entropy: str = "rice", threads: int = 1, dither: bool = False):
            if isinstance(camera, dict):
                camera = Camera(**camera)
            self.q = float(q); self.mode = int(mode)
            self.camera = camera or Camera()
            self.predictor = predictor; self.entropy = entropy; self.threads = int(threads)
            _meta(self.mode, self.q)   # validate early
            self.dither = bool(_check_dither(dither, self.mode, self.q, predictor))   # every pixel (no calibration here)

        def encode(self, buf):
            arr = np.asarray(buf)
            if arr.dtype not in _DTYPES:
                raise TypeError(f"B4DCodec supports uint8/uint16 arrays, got {arr.dtype}")
            return compress(arr, q=self.q, mode=self.mode, camera=self.camera,
                            predictor=self.predictor, entropy=self.entropy, threads=self.threads, dither=self.dither)

        def decode(self, buf, out=None):
            arr = decompress(buf, threads=self.threads)
            if out is None:
                return arr
            from numcodecs.compat import ndarray_copy
            return ndarray_copy(arr, out)

        def get_config(self):
            cam = {"conversion": self.camera.conversion, "offset": self.camera.offset, "read_noise": self.camera.read_noise,
                   "exact_vst": self.camera.exact_vst}          # always stored: a config without it (older format) means False
            cfg = {"id": self.codec_id, "q": self.q, "mode": self.mode, "camera": cam,
                   "predictor": self.predictor, "entropy": self.entropy, "threads": self.threads}
            if self.dither:
                cfg["dither"] = True
            return cfg

        @classmethod
        def from_config(cls, config):
            cfg = dict(config); cfg.pop("id", None)
            if "nthreads" in cfg:          # older .zarray files store the thread count as "nthreads"
                n = cfg.pop("nthreads")
                cfg.setdefault("threads", n)
            if isinstance(cfg.get("camera"), dict):   # older configs have no exact_vst: Anscombe
                cfg["camera"] = {"exact_vst": False, **cfg["camera"]}
            return cls(**cfg)

        def __repr__(self):
            return (f"B4DCodec(q={self.q}, mode={self.mode}, camera={self.camera}, predictor={self.predictor!r}, "
                    f"entropy={self.entropy!r}, threads={self.threads}, dither={self.dither})")

    _register_codec(B4DCodec)
else:                                  # pragma: no cover
    B4DCodec = None

try:                                   # zarr v3 codec (needs zarr >= 3)
    from . import zarr3 as zarr3       # noqa: F401  (registers codec "b4d")
except Exception:                      # pragma: no cover
    zarr3 = None

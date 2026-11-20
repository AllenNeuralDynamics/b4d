"""Zarr v3 codec for B4D (an ArrayBytesCodec: replaces the default `bytes` serializer).

    import zarr, b4d
    from b4d.zarr3 import B4D
    z = zarr.create_array(store, shape=(512, 512, 512), chunks=(64, 64, 64), shards=(512, 512, 512),
                          dtype="uint16", serializer=B4D(camera=cam), compressors=None)

Defaults as b4d.compress (m1_a3: q=1, mode=1, predictor="above3", entropy="rice"). Registered under the codec name "b4d";
readers just need `import b4d` before opening. `threads` (Blosc2 threads per chunk) is a runtime setting and is not stored
in zarr.json.
"""
from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Literal

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

from . import Camera, compress, decompress, _meta, _check_dither, _vst_objective


def _check_vst(v):
    if v in (False, None):
        return False
    _vst_objective(Camera(exact_vst=v))   # validate
    return v


@dataclass(frozen=True)
class B4D(ArrayBytesCodec):
    q: float = 1.0
    mode: int = 1
    conversion: float = 1.0
    offset: float = 0.0
    read_noise: float = 0.0
    predictor: str = "above3"
    entropy: str = "rice"
    threads: int = 1
    dither: bool | Literal["listed"] = False      # True: every pixel; "listed": a calibration's listed pixels (native writer), else none
    exact_vst: bool | str = "variance"   # the camera's exact VST (format v9; True/"variance" or "std") instead of Anscombe

    is_fixed_size = False

    def __init__(self, *, q: float = 1.0, mode: int = 1, camera: Camera | dict | None = None,
                 conversion: float | None = None, offset: float | None = None, read_noise: float | None = None,
                 predictor: str = "above3", entropy: str = "rice", threads: int = 1, dither: bool | Literal["listed"] = False,
                 exact_vst: bool | str | None = None) -> None:
        if isinstance(camera, dict):
            camera = Camera(**camera)
        cam = camera or Camera(conversion if conversion is not None else 1.0,
                               offset if offset is not None else 0.0,
                               read_noise if read_noise is not None else 0.0)
        _meta(int(mode), float(q))  # validate
        for k, v in dict(q=float(q), mode=int(mode), conversion=float(cam.conversion), offset=float(cam.offset),
                         read_noise=float(cam.read_noise), predictor=predictor, entropy=entropy,
                         threads=int(threads), dither="listed" if dither == "listed" else bool(_check_dither(dither, int(mode), float(q), predictor)),
                         exact_vst=_check_vst(cam.exact_vst if exact_vst is None else exact_vst)).items():
            object.__setattr__(self, k, v)

    @property
    def camera(self) -> Camera:
        return Camera(self.conversion, self.offset, self.read_noise, self.exact_vst)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> "B4D":
        _, cfg = parse_named_configuration(data, "b4d", require_configuration=False)
        cfg = dict(cfg or {})
        cfg.setdefault("exact_vst", False)   # older metadata has no exact_vst: Anscombe
        return cls(**cfg)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        cfg: dict[str, JSON] = {
            "q": self.q, "mode": self.mode, "conversion": self.conversion, "offset": self.offset,
            "read_noise": self.read_noise, "predictor": self.predictor, "entropy": self.entropy}
        if self.dither:
            cfg["dither"] = self.dither
        cfg["exact_vst"] = self.exact_vst   # always stored: metadata without it (older format) means False
        return {"name": "b4d", "configuration": cfg}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> "B4D":
        dt = array_spec.dtype.to_native_dtype()
        if dt not in (np.dtype("uint8"), np.dtype("uint16")):
            raise TypeError(f"b4d codec supports uint8/uint16 arrays, got {dt}")
        return self

    def _encode_sync(self, arr):
        return compress(arr, q=self.q, mode=self.mode, camera=self.camera,
                        predictor=self.predictor, entropy=self.entropy, threads=self.threads, dither=self.dither is True)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        arr = np.ascontiguousarray(chunk_array.as_ndarray_like())
        # off the event loop: the C codec releases the GIL, so zarr's chunk-level
        # concurrency turns into real multi-core compression
        blob = await asyncio.to_thread(self._encode_sync, arr)
        return chunk_spec.prototype.buffer.from_bytes(blob)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        arr = await asyncio.to_thread(decompress, chunk_bytes.to_bytes(), threads=self.threads)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(arr.reshape(chunk_spec.shape))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("b4d", B4D)

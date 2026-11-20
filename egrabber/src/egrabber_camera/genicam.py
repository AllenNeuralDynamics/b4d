"""GenICam feature access on top of the ingest engine, shaped like the Python eGrabber SDK's remote module.

Host applications written against the Euresys Python SDK (``grabber.remote.get("Width", dtype=int)``,
``remote.set("ExposureTime", 1000.0)``) can talk to the camera through this engine instead, with the same calls:
the engine opens the device once and owns the frame grabber, so a second (Python SDK) grabber on the same camera is not
possible in the same process (the GenTL producer allows one ``GCInitLib`` per process).

    dev = IngestDevice("<serial>")               # opens the camera as configured (nothing is written), no buffer pool
    dev.fetch_remote("Width", int)               # 19200
    dev.fetch_remote("Width.Inc", int)           # 32            (".Min" / ".Max" / ".Inc" suffixes work)
    dev.remote.get("@ee TriggerSource", list)    # ["Software", "Line0", ...]   ("@ee" = enumeration entries)
    dev.set_remote("ExposureTime", 10_000.0)
    dev.ingest.alloc_pool(1, 8); dev.ingest.start(max_frames=32); ...

Values cross the C API as strings; ``dtype`` decides how they are read back (int/float/bool/str/list), as in the SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .ingest import Ingest

#: query commands of the SDK's ``get``: the type of their answer (the SDK looks the type of other features up)
_AT_DTYPES = {
    "@available": bool,
    "@readable": bool,
    "@writeable": bool,
    "@implemented": bool,
    "@command": bool,
    "@done": bool,
    "@features": list,
    "@!features": list,
    "@categories": list,
    "@!categories": list,
    "@ee": list,
    "@!ee": list,
    "@interfaces": list,
    "@selectors": list,
    "@declared": list,
}


def _to_int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:  # an IFloat feature read as an integer (the SDK rounds)
        return int(round(float(raw)))


def _infer(raw: str) -> Any:
    for conv in (int, float):
        try:
            return conv(raw)
        except ValueError:
            pass
    return raw


_CONVERTERS = {
    bool: lambda raw: raw.strip().lower() in ("1", "true"),
    int: _to_int,
    float: float,
    list: lambda raw: raw.split("\n") if raw else [],
    str: lambda raw: raw,
}


def convert(raw: str, dtype: Optional[type] = None, feature: str = "") -> Any:
    """Read a feature value returned as text as ``dtype`` (None: the type the ``@`` query answers with, else int, float
    or str by what the text looks like)."""
    if dtype is None and feature.startswith("@"):
        dtype = _AT_DTYPES.get(feature.split()[0])
    if dtype is None:
        return _infer(raw)
    return _CONVERTERS.get(dtype, lambda raw: raw)(raw)


class Remote:
    """The camera's GenICam features (``RemoteModule`` of the SDK): ``get`` and ``set`` by feature name."""

    def __init__(self, ingest: Ingest):
        self._ingest = ingest

    def get(self, feature: str, dtype: Optional[type] = None) -> Any:
        return convert(self._ingest.remote_get(feature), dtype, feature)

    def set(self, feature: str, value: Any) -> None:
        if isinstance(value, bool):
            value = int(value)
        self._ingest.remote_set(feature, repr(float(value)) if isinstance(value, float) else value)

    def __bool__(self) -> bool:
        return True


class Stream:
    """The grabber's data stream module (``StreamModule``): ``get`` only."""

    def __init__(self, ingest: Ingest):
        self._ingest = ingest

    def get(self, feature: str, dtype: Optional[type] = None) -> Any:
        return convert(self._ingest.stream_get(feature), dtype, feature)

    def __bool__(self) -> bool:
        return True


class IngestDevice:
    """Drop-in for the eGrabber ``EgrabberDevice`` helper (remote/stream access by name) over the ingest engine.

    ``ingest`` is the engine handle; acquisition goes through it (``alloc_pool`` / ``start`` / ``wait_block`` /
    ``release`` / ``stop`` / ``free_pool``)."""

    def __init__(self, serial: Optional[str] = None, **ingest_kwargs: Any):
        self.ingest = Ingest(serial=serial, nblocks=0, attach_only=True, **ingest_kwargs)
        self.remote = Remote(self.ingest)
        self.stream = Stream(self.ingest)
        self.logger = logging.getLogger("IngestDevice")  # EgrabberDevice has a .logger too (drop-in surface)

    def fetch_remote(self, feature: str, dtype: type) -> Any:
        value = self.remote.get(feature=feature, dtype=dtype)
        if value is None:
            raise RuntimeError(f"Failed to get remote property: {feature}")
        return value

    def set_remote(self, feature: str, value: Any) -> None:
        self.remote.set(feature=feature, value=value)

    def close(self) -> None:
        self.ingest.close()

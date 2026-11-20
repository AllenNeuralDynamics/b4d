"""Thin adapter over the DCAM-API 4 C interface (declarations in ``dcamapi.py``).

The adapter turns ``DCAMERR`` returns into values and ``DCAMError`` exceptions, names properties the way
DCAM does (``"EXPOSURE TIME"``, ``"OUTPUT TRIGGER SOURCE[1]"``), gives them a typed read/write (a mode
property reads and writes its value text, e.g. ``"EXTERNAL"``), and wraps the buffer (alloc, attach,
release, lockframe, copyframe), capture and wait calls. The block logic lives in ``source.py``.

DCAM is initialised once per process and never uninitialised: ``dcamapi_uninit`` closes every open
camera, and the camera list is fixed at ``dcamapi_init``.
"""

from __future__ import annotations

import ctypes as C
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from . import dcamapi as A

_sdk: SDK | None = None
_sdk_lock = threading.Lock()


class DCAMError(RuntimeError):
    def __init__(self, func: str, code: int):
        self.func = func
        self.code = int(code)
        self.name = A.err_name(self.code)
        super().__init__(f"{func} failed: {self.name} (0x{self.code & 0xFFFFFFFF:08X})")


@dataclass
class SDK:
    lib: Any
    device_count: int

    def ok(self, func: str, code: int) -> int:
        if code < 0:
            raise DCAMError(func, code)
        return code


def load(dll: str | None = None) -> SDK:
    """Load the runtime and initialise DCAM once per process (``dll`` or ``$DCAMAPI_DLL`` override the
    library path)."""
    global _sdk
    with _sdk_lock:
        if _sdk is not None:
            return _sdk
        lib = A.load_library(dll)
        init = A.DCAMAPI_INIT()
        code = lib.dcamapi_init(C.byref(init))
        if code < 0:
            raise DCAMError("dcamapi_init", code)
        _sdk = SDK(lib=lib, device_count=int(init.iDeviceCount))
        return _sdk


def sdk() -> SDK:
    return _sdk or load()


def _getstring(lib: Any, h: Any, which: int) -> str | None:
    buf = C.create_string_buffer(256)
    param = A.DCAMDEV_STRING(iString=which, text=C.cast(buf, C.c_char_p), textbytes=256)
    if lib.dcamdev_getstring(h, C.byref(param)) < 0:
        return None
    return buf.value.decode(errors="replace")


def parse_serial(camera_id: str | None) -> str:
    """``"S/N: 100001"`` -> ``"100001"``."""
    return (camera_id or "").removeprefix("S/N:").strip()


def enumerate_devices(dll: str | None = None) -> list[dict[str, str]]:
    """Cameras DCAM found at init: index, serial, model, bus, versions (read without opening them)."""
    s = load(dll)
    out = []
    for i in range(s.device_count):
        h = C.c_void_p(i)  # DCAM accepts a device index in place of a handle for dcamdev_getstring
        info = {k.name.lower(): _getstring(s.lib, h, k.value) or "" for k in A.IDSTR}
        info["index"] = str(i)
        info["serial"] = parse_serial(info["cameraid"])
        out.append(info)
    return out


def open_by_serial(serial: str | None = None, dll: str | None = None) -> Dcam:
    """Open the camera whose CAMERAID is ``S/N: <serial>``; the first camera when ``serial`` is None."""
    devices = enumerate_devices(dll)
    if not devices:
        raise RuntimeError("DCAM found no camera")
    if serial is None:
        return Dcam(int(devices[0]["index"]))
    for d in devices:
        if d["serial"] == str(serial):
            return Dcam(int(d["index"]))
    raise RuntimeError(f"serial number {serial} not found; DCAM reports {[d['serial'] for d in devices]}")


@dataclass
class Prop:
    """One DCAM property as the camera reports it now (access and limits can follow other settings)."""

    id: int
    name: str
    type: str  # mode, long, real
    attribute: int
    unit: str
    min: float
    max: float
    step: float
    default: float

    def has(self, flag: A.PROPATTR) -> bool:
        return bool(self.attribute & flag)

    @property
    def readable(self) -> bool:
        return self.has(A.PROPATTR.READABLE)

    @property
    def writable(self) -> bool:
        return self.has(A.PROPATTR.WRITABLE)

    @property
    def writable_busy(self) -> bool:
        """Writable while capturing."""
        return self.writable and self.has(A.PROPATTR.ACCESSBUSY)


class Dcam:
    """One open camera, opened by DCAM device index (``open_by_serial`` finds the index)."""

    def __init__(self, index: int):
        self.sdk = s = sdk()
        self.lib = s.lib
        self.index = int(index)
        param = A.DCAMDEV_OPEN(index=self.index)
        s.ok("dcamdev_open", self.lib.dcamdev_open(C.byref(param)))
        self.handle: Any = param.hdcam
        self.wait_handle: Any = None
        self.ids: dict[str, int] = {}
        self.names: dict[int, str] = {}
        self._attached: Any = None  # the pointer array handed to dcambuf_attach (kept alive while attached)
        self._discover()

    def _call(self, func: str, *args: Any) -> int:
        return self.sdk.ok(func, getattr(self.lib, func)(*args))

    def close(self) -> None:
        if self.handle is None:
            return
        if self.wait_handle is not None:
            self.lib.dcamwait_close(self.wait_handle)
            self.wait_handle = None
        self._call("dcamdev_close", self.handle)
        self.handle = None

    def string_info(self, which: str | A.IDSTR) -> str | None:
        """A device string (``"MODEL"``, ``"CAMERAID"``, ``"BUS"``, ...); None if the camera has none."""
        idstr = A.IDSTR[which] if isinstance(which, str) else which
        return _getstring(self.lib, self.handle, idstr.value)

    # ---- properties ---------------------------------------------------------------------
    def _discover(self) -> None:
        """Map every supported property name to its id, array elements included (``NAME[k]``)."""
        pid = C.c_int32(0)
        found: list[int] = []
        while self.lib.dcamprop_getnextid(self.handle, C.byref(pid), A.PROPOPTION.SUPPORT) >= 0 and pid.value:
            found.append(pid.value)
        for p in found:
            self._add(p)
            a = self._attr(p)
            if a.iProp_NumberOfElement and a.iPropStep_Element and a.iProp_ArrayBase == p:
                count = C.c_double(0)
                if self.lib.dcamprop_getvalue(self.handle, a.iProp_NumberOfElement, C.byref(count)) >= 0:
                    for k in range(1, int(count.value)):
                        self._add(p + k * a.iPropStep_Element)

    def _add(self, pid: int) -> None:
        buf = C.create_string_buffer(64)
        if self.lib.dcamprop_getname(self.handle, pid, buf, 64) < 0:
            return
        name = buf.value.decode(errors="replace")
        self.ids[name] = pid
        self.names[pid] = name

    def has(self, prop: str) -> bool:
        return prop in self.ids

    def prop_id(self, prop: str | int) -> int:
        if isinstance(prop, int):
            return prop
        try:
            return self.ids[prop]
        except KeyError:
            raise DCAMError(f"property {prop!r}", A.DCAMERR.INVALIDPROPERTYID) from None

    def _attr(self, pid: int) -> A.DCAMPROP_ATTR:
        attr = A.DCAMPROP_ATTR(iProp=pid)
        self._call("dcamprop_getattr", self.handle, C.byref(attr))
        return attr

    def prop(self, prop: str | int) -> Prop:
        pid = self.prop_id(prop)
        a = self._attr(pid)
        return Prop(
            id=pid,
            name=self.names.get(pid, hex(pid)),
            type=A.PROPTYPES.get(a.attribute & A.PROPTYPE_MASK, "none"),
            attribute=a.attribute & 0xFFFFFFFF,
            unit=A.PROPUNITS.get(a.iUnit, ""),
            min=a.valuemin,
            max=a.valuemax,
            step=a.valuestep,
            default=a.valuedefault,
        )

    def get_value(self, prop: str | int) -> float:
        pid = self.prop_id(prop)
        v = C.c_double(0)
        self._call("dcamprop_getvalue", self.handle, pid, C.byref(v))
        return v.value

    def set_value(self, prop: str | int, value: float) -> float:
        """Write and return the value the camera kept (DCAM rounds to the property's step)."""
        pid = self.prop_id(prop)
        v = C.c_double(float(value))
        self._call("dcamprop_setgetvalue", self.handle, pid, C.byref(v), 0)
        return v.value

    def text(self, prop: str | int, value: float) -> str | None:
        """Value text of a mode value, or None if it has none."""
        pid = self.prop_id(prop)
        buf = C.create_string_buffer(64)
        vt = A.DCAMPROP_VALUETEXT(iProp=pid, value=float(value), text=C.cast(buf, C.c_char_p), textbytes=64)
        if self.lib.dcamprop_getvaluetext(self.handle, C.byref(vt)) < 0:
            return None
        return buf.value.decode(errors="replace")

    def entries(self, prop: str | int) -> dict[str, int]:
        """Value text -> value of every value a mode property offers now, in order."""
        pid = self.prop_id(prop)
        a = self._attr(pid)
        out: dict[str, int] = {}
        v = C.c_double(a.valuemin)
        while True:
            t = self.text(pid, v.value)
            if t is not None:
                out[t] = int(v.value)
            if self.lib.dcamprop_queryvalue(self.handle, pid, C.byref(v), A.PROPOPTION.NEXT) < 0:
                break
        return out

    def get(self, prop: str | int) -> Any:
        """Current value: str for a mode property (its value text), int for a long, float for a real."""
        p = self.prop(prop)
        v = self.get_value(p.id)
        if p.type == "mode":
            return self.text(p.id, v) or str(int(v))
        if p.type == "long":
            return int(v)
        return v

    def set(self, prop: str | int, value: Any) -> Any:
        """Write a value (a mode takes its value text or its number) and return what the camera kept."""
        p = self.prop(prop)
        if p.type == "mode" and isinstance(value, str):
            options = self.entries(p.id)
            if value not in options:
                raise DCAMError(f"set {p.name}={value!r} (options {list(options)})", A.DCAMERR.INVALIDVALUE)
            value = options[value]
        kept = self.set_value(p.id, value)
        if p.type == "mode":
            return self.text(p.id, kept) or str(int(kept))
        return int(kept) if p.type == "long" else kept

    # ---- buffers and capture ------------------------------------------------------------
    def alloc(self, frames: int) -> None:
        """Allocate ``frames`` buffer frames in DCAM; frame k of a capture lands in buffer ``k % frames``."""
        self._call("dcambuf_alloc", self.handle, int(frames))

    def attach(self, pointers: Sequence[int]) -> None:
        """Hand DCAM caller-owned host memory as its buffer frames (``dcambuf_attach``, ATTACHKIND_FRAME).

        ``pointers[k]`` is the address of buffer frame k; each must hold ``BUFFER FRAMEBYTES`` bytes. Frame k of
        a capture lands in ``pointers[k % len(pointers)]``. The memory must stay valid until ``release``.
        """
        n = len(pointers)
        arr = (C.c_void_p * n)(*[int(p) for p in pointers])
        param = A.DCAMBUF_ATTACH(iKind=A.ATTACHKIND_FRAME, buffer=C.cast(arr, C.POINTER(C.c_void_p)), buffercount=n)
        self._call("dcambuf_attach", self.handle, C.byref(param))
        self._attached = arr

    def attach_primary(self, kind: int, address: int, count: int) -> None:
        """Hand DCAM one contiguous metadata array (``ATTACHKIND_PRIMARY_TIMESTAMP``: ``count`` DCAM_TIMESTAMP,
        ``ATTACHKIND_PRIMARY_FRAMESTAMP``: ``count`` int32); DCAM writes frame k's stamp into element ``k % count``.
        The memory must stay valid until ``release(kind)``.

        On the ORCA-Flash4.0 V3 (DCAM-API 26.11) lockframe is refused with attached
        frames, but these metadata attaches work next to ATTACHKIND_FRAME (as in the SDK's attach_metadatabuf sample).
        """
        ptr = C.c_void_p(int(address))
        param = A.DCAMBUF_ATTACH(iKind=int(kind), buffer=C.cast(C.pointer(ptr), C.POINTER(C.c_void_p)), buffercount=int(count))
        self._call("dcambuf_attach", self.handle, C.byref(param))

    def release(self, kind: int = A.ATTACHKIND_FRAME) -> None:
        """Release the buffer frames (allocated or attached), or one metadata attach kind; capture must be stopped."""
        self._call("dcambuf_release", self.handle, int(kind))
        if kind == A.ATTACHKIND_FRAME:
            self._attached = None

    def lockframe(self, index: int) -> A.DCAMBUF_FRAME:
        """Pointer, geometry and stamps of buffer frame ``index`` (-1: the newest)."""
        frame = A.DCAMBUF_FRAME(iFrame=index)
        self._call("dcambuf_lockframe", self.handle, C.byref(frame))
        return frame

    def copyframe(self, index: int, dst: int, rowbytes: int, width: int, height: int) -> A.DCAMBUF_FRAME:
        """Copy buffer frame ``index`` into ``dst`` (``rowbytes`` per row); returns its stamps."""
        frame = A.DCAMBUF_FRAME(iFrame=index, buf=dst, rowbytes=rowbytes, width=width, height=height)
        self._call("dcambuf_copyframe", self.handle, C.byref(frame))
        return frame

    def transferinfo(self) -> tuple[int, int]:
        """(newest buffer index, frames captured since start)."""
        info = A.DCAMCAP_TRANSFERINFO(iKind=A.TRANSFERKIND_FRAME)
        self._call("dcamcap_transferinfo", self.handle, C.byref(info))
        return info.nNewestFrameIndex, info.nFrameCount

    def status(self) -> A.CAPSTATUS:
        s = C.c_int32(0)
        self._call("dcamcap_status", self.handle, C.byref(s))
        return A.CAPSTATUS(s.value)

    def start(self, sequence: bool = True) -> None:
        if self.wait_handle is None:
            param = A.DCAMWAIT_OPEN(hdcam=self.handle)
            self._call("dcamwait_open", C.byref(param))
            self.wait_handle = param.hwait
        self._call("dcamcap_start", self.handle, A.CAPSTART_SEQUENCE if sequence else A.CAPSTART_SNAP)

    def stop(self, settle_s: float = 1.0) -> None:
        """Stop capturing and wait (up to ``settle_s``) until DCAM leaves BUSY: ``dcamcap_stop`` returns
        before the capture has wound down, and buffer calls made in between fail with BUSY (measured with the earlier driver
        on an ORCA-Flash4.0 V3)."""
        self._call("dcamcap_stop", self.handle)
        deadline = time.perf_counter() + settle_s
        while self.status() == A.CAPSTATUS.BUSY and time.perf_counter() < deadline:
            time.sleep(0.001)

    def fire_trigger(self) -> None:
        self._call("dcamcap_firetrigger", self.handle, 0)

    def wait(self, events: int, timeout_ms: int) -> int:
        """Block until one of ``events`` happens; returns the events that did. Raises DCAMError TIMEOUT or
        ABORT (``abort_wait`` from another thread)."""
        if self.wait_handle is None:
            raise DCAMError("dcamwait_start", A.DCAMERR.INVALIDWAITHANDLE)
        param = A.DCAMWAIT_START(eventmask=int(events), timeout=int(timeout_ms))
        self._call("dcamwait_start", self.wait_handle, C.byref(param))
        return param.eventhappened

    def abort_wait(self) -> None:
        if self.wait_handle is not None:
            self.lib.dcamwait_abort(self.wait_handle)

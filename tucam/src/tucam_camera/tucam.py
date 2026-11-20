"""Thin adapter over Tucsen's TUCam SDK V3 Python binding (the ``TUC_*`` API).

The binding (``include/python/TucApi.py``, ``TucDefine.py``) and the runtime (``TUCam.dll`` plus the GenTL
producers in ``gentl/``) come with Tucsen's SDK installer ("TUCamSDK V3"; 3.0.12.0 RC was used by the earlier
internal driver this adapter is adapted from) and are not redistributed here. Point ``sdk_dir`` (or the ``TUCAM_ROOT``
environment variable) at the SDK install root, the folder holding ``include/`` and ``runtime/``; without either,
the installer's default folders in ``DEFAULT_ROOTS`` are tried.

The adapter turns the binding's ``(status, value)`` returns into values and exceptions, gives GenICam nodes a typed
read/write, and wraps the buffer calls (announce / queue / revoke / buffer info / buffer callback) that
``source.py`` uses. Camera logic lives in ``source.py``.

SDK facts this module relies on (observed with that earlier driver, not re-verified here):

* The SDK is initialised once per process (a second ``TucInitialize`` is refused) and terminated at interpreter exit
  by :func:`terminate`, registered with ``atexit`` by :func:`load`. Without
  ``TucTerminate`` every process that had initialised the SDK (3.0.12.0 RC, Aries 6504) crashed after Python's own
  shutdown with an access violation (exit code 0xC0000005), even with no camera opened; with it, exit code 0.
* ``TucDeviceEnumerate`` closes every open camera, so :func:`devices` enumerates once per process, before any camera
  is opened, and cameras are then opened by serial number.
* The transport producers are loaded with plain ``LoadLibrary`` from ``TUCAM_GENTL64_PATH``, so that variable and
  ``PATH`` are pointed at this SDK's runtime folder.
"""

from __future__ import annotations

import atexit
import ctypes as C
import importlib
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

ENV_ROOT = "TUCAM_ROOT"
DEFAULT_ROOTS = (r"C:\Program Files\Tucsen\TUCamSDK V3", r"C:\Program Files\Tucsen\TUCamSDK V3 RC")

# GenICam PFNC pixel format codes as the SDK reports them in TUC_BUFFER_INFO.pixelFormat
PFNC_MONO8 = 0x01080001
PFNC_MONO12 = 0x01100005  # 12 bits in a little-endian 16-bit container
PFNC_MONO16 = 0x01100007
PFNC_MONO12_PACKED = 0x010C0047  # 2 px in 3 bytes: needs unpacking (not zero copy). PFNC names this code
# Mono12p, but the Aries 6504 frames carrying it use the GigE Mono12Packed bit order (checked on the camera)

#: PixelFormat node entry -> (PFNC code, numpy dtype name of the delivered container); None for packed formats
PIXEL_FORMATS: Dict[str, Tuple[int, Optional[str]]] = {
    "Mono8": (PFNC_MONO8, "uint8"),
    "Mono12": (PFNC_MONO12, "uint16"),
    "Mono16": (PFNC_MONO16, "uint16"),
    "Mono12Packed": (PFNC_MONO12_PACKED, None),
}

_sdk: Optional["SDK"] = None
_sdk_lock = threading.Lock()
_devices: Optional[List[Dict[str, str]]] = None
_open_cams: "weakref.WeakSet[TUCam]" = weakref.WeakSet()  # cameras open now (closed by terminate())
_atexit_registered = False


class TUCamError(RuntimeError):
    def __init__(self, func: str, status: int):
        self.func = func
        self.status = int(status)
        super().__init__(f"{func} failed: {status_name(self.status)} ({self.status})")


def status_name(status: int) -> str:
    names = _sdk.status_names if _sdk else {}
    return names.get(int(status), "UNKNOWN")


def resolve_root(sdk_dir: Union[str, Path, None] = None) -> Path:
    """Return the SDK install root: the explicit argument, else ``$TUCAM_ROOT``, else the installer default."""
    candidates = [Path(p) for p in (sdk_dir, os.environ.get(ENV_ROOT), *DEFAULT_ROOTS) if p]
    for d in candidates:
        if (d / "include" / "python" / "TucApi.py").is_file() and (d / "runtime" / "windows" / "x64").is_dir():
            return d
    raise FileNotFoundError(
        f"TUCam SDK V3 not found (looked in {[str(c) for c in candidates]}); pass sdk_dir or set {ENV_ROOT} "
        "to the SDK install root (the folder holding include/python and runtime/windows/x64)"
    )


@dataclass
class SDK:
    """The loaded binding: ``api`` is a ``TucApi.TucApi``, ``D`` the ``TucDefine`` module."""

    root: Path
    api: Any
    D: ModuleType
    status_names: Dict[int, str]

    def ok(self, func: str, status: int) -> None:
        if status != self.D.TUC_STATUS.TUC_STATUS_OK:
            raise TUCamError(func, status)

    def value(self, func: str, ret: tuple) -> Any:
        self.ok(func, ret[0])
        return ret[1]


def _status_names(D) -> Dict[int, str]:
    return {v: k[len("TUC_STATUS_"):] for k, v in vars(D.TUC_STATUS).items() if k.startswith("TUC_STATUS_")}


def load(sdk_dir: Union[str, Path, None] = None, log_level: str = "WARNING") -> SDK:
    """Load the binding and runtime and initialise the SDK once per process. :func:`terminate` runs at interpreter exit
    (``atexit``), after every camera."""
    global _sdk, _atexit_registered
    with _sdk_lock:
        if _sdk is not None:
            return _sdk
        root = resolve_root(sdk_dir)
        runtime = root / "runtime" / "windows" / "x64"
        os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "gentl"), os.environ.get("PATH", "")])
        os.environ["TUCAM_GENTL64_PATH"] = str(runtime / "gentl") + os.sep
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(runtime))
        include = str(root / "include" / "python")
        if include not in sys.path:
            sys.path.insert(0, include)
        D = importlib.import_module("TucDefine")
        api = importlib.import_module("TucApi").TucApi(dll_path=str(runtime / "TUCam.dll"))
        sdk_ = SDK(root=root, api=api, D=D, status_names=_status_names(D))
        status = api.TucInitialize(getattr(D.TUC_LOG_LEVEL, f"TUC_LOG_LEVEL_{log_level}"))
        if status != D.TUC_STATUS.TUC_STATUS_ALREADY_INITIALIZED:
            sdk_.ok("TucInitialize", status)
        _sdk = sdk_
        if not _atexit_registered:
            atexit.register(terminate)
            _atexit_registered = True
        return sdk_


def terminate() -> Optional[int]:
    """Close every camera still open and terminate the SDK (``TucTerminate``); returns its status, None if no SDK was
    loaded. Registered with ``atexit`` by :func:`load`; the next :func:`sdk` call loads and initialises it again."""
    global _sdk, _devices
    with _sdk_lock:
        s = _sdk
        if s is None:
            return None
        for cam in list(_open_cams):
            try:
                cam.close()
            except Exception:  # noqa: BLE001 - best effort at shutdown
                pass
        _sdk, _devices = None, None
        fn = getattr(s.api, "TucTerminate", None)
        return int(fn()) if fn is not None else None


def install(api: Any, D: ModuleType, root: Union[str, Path] = "") -> SDK:
    """Use an already-constructed binding (``api``, ``D``) instead of loading one from disk. For tests (a fake SDK)
    and for hosts that load the binding themselves. Resets the cached device list."""
    global _sdk, _devices
    with _sdk_lock:
        _sdk = SDK(root=Path(root), api=api, D=D, status_names=_status_names(D))
        _devices = None
        return _sdk


def sdk() -> SDK:
    return _sdk or load()


# --------------------------------------------------------------------------------------
# GenICam nodes
# --------------------------------------------------------------------------------------
TYPES = {2: "int", 3: "bool", 4: "command", 5: "float", 6: "str", 7: "register", 9: "enum"}  # TUC_FEATURE_TYPE
ACCESS = {0: "NI", 1: "NA", 2: "WO", 3: "RO", 4: "RW"}  # TUC_FEATURE_ACCESS


@dataclass
class Node:
    """One GenICam node as the SDK reports it now (access and limits follow the camera's mode)."""

    name: str
    type: str  # int, float, bool, str, enum, command, register, category
    access: str  # NI, NA, WO, RO, RW
    unit: str = ""
    description: str = ""

    @property
    def readable(self) -> bool:
        return self.access in ("RO", "RW")

    @property
    def writable(self) -> bool:
        return self.access in ("WO", "RW")


def _dec(b) -> str:
    return b.decode(errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)


def enumerate_devices() -> List[Dict[str, str]]:
    """Enumerate cameras now. The SDK closes every open camera when it enumerates: use :func:`devices` (cached)."""
    s = sdk()
    count = s.value("TucDeviceEnumerate", s.api.TucDeviceEnumerate())
    out = []
    for i in range(count):
        info = s.value("TucGetDeviceInfo", s.api.TucGetDeviceInfo(i))
        out.append(
            {
                f: _dec(getattr(info, f))
                for f in ("serialNumber", "modelName", "vendorName", "userDefinedName", "transportLayerType")
            }
        )
    return out


def devices(sdk_dir: Union[str, Path, None] = None) -> List[Dict[str, str]]:
    """Cameras the SDK found, enumerated once per process (before any camera is opened)."""
    global _devices
    if _sdk is None:
        load(sdk_dir)
    with _sdk_lock:
        if _devices is None:
            _devices = enumerate_devices()
        return list(_devices)


class TUCam:
    """One open camera (``TUC_OPEN_POLICY_EXCLUSIVE``, opened by serial number)."""

    def __init__(self, serial: str):
        self.sdk = s = sdk()
        D = s.D
        self.serial = serial
        self.handle: Optional[int] = s.value(
            "TucDeviceOpenEx",
            s.api.TucDeviceOpenEx(
                serial, D.TUC_DEVICE_ID_TYPE.TUC_DEVICE_ID_SERIAL, D.TUC_OPEN_POLICY.TUC_OPEN_POLICY_EXCLUSIVE
            ),
        )
        self._types: Dict[str, str] = {}  # node types never change
        _open_cams.add(self)

    @property
    def status_ok(self) -> int:
        return self.sdk.D.TUC_STATUS.TUC_STATUS_OK

    def close(self) -> None:
        if self.handle is not None:
            h, self.handle = self.handle, None
            _open_cams.discard(self)
            self.sdk.ok("TucDeviceClose", self.sdk.api.TucDeviceClose(h))

    # ---- nodes -------------------------------------------------------------------------
    def node(self, name: str) -> Node:
        info = self.sdk.value(
            f"TucGetFeatureInfoByName({name})", self.sdk.api.TucGetFeatureInfoByName(self.handle, name)
        )
        t = TYPES.get(info.featureType, "category")
        self._types[name] = t
        return Node(
            name=name,
            type=t,
            access=ACCESS.get(info.accessMode, "NA"),
            unit=_dec(info.unit),
            description=_dec(info.description),
        )

    def node_type(self, name: str) -> str:
        return self._types.get(name) or self.node(name).type

    def get(self, name: str) -> Any:
        """Current value: int, float, bool or str (an enum gives its symbolic entry)."""
        a, t = self.sdk.api, self.node_type(name)
        h = self.handle
        if t == "enum":
            # TucGetFeatureEnumInfo does not check readability: an unavailable enum has no current entry
            current = _dec(self.sdk.value(f"get {name}", a.TucGetFeatureEnumInfo(h, name)).currentSymbolic)
            if not current:
                raise TUCamError(f"get {name}", self.sdk.D.TUC_STATUS.TUC_STATUS_FEATURE_NOT_READABLE)
            return current
        getter = {
            "int": a.TucGetFeatureInt64,
            "float": a.TucGetFeatureFloat,
            "bool": a.TucGetFeatureBool,
            "str": a.TucGetFeatureString,
        }.get(t)
        if getter is None:
            raise TypeError(f"{name} is a {t} node and has no value")
        return self.sdk.value(f"get {name}", getter(h, name))

    def try_get(self, name: str, default: Any = None) -> Any:
        """``get`` that returns ``default`` when the node is missing or unavailable in the current mode."""
        try:
            return self.get(name)
        except (TUCamError, TypeError):
            return default

    def limits(self, name: str) -> Tuple[float, float, float]:
        """(minimum, maximum, increment) of an int or float node; increment 0 means no step."""
        a, t = self.sdk.api, self.node_type(name)
        if t == "int":
            i = self.sdk.value(f"TucGetFeatureInt64Info({name})", a.TucGetFeatureInt64Info(self.handle, name))
        elif t == "float":
            i = self.sdk.value(f"TucGetFeatureFloatInfo({name})", a.TucGetFeatureFloatInfo(self.handle, name))
        else:
            raise TypeError(f"{name} is a {t} node and has no limits")
        return i.minimum, i.maximum, i.increment

    def entries(self, name: str) -> List[str]:
        """Entries the enum offers in the camera's current mode."""
        e = self.sdk.value(f"TucGetFeatureEnumInfo({name})", self.sdk.api.TucGetFeatureEnumInfo(self.handle, name))
        return [_dec(e.pEntries[i].symbolic) for i in range(e.entryCount)]

    def set(self, name: str, value: Any) -> None:
        a, t, h = self.sdk.api, self.node_type(name), self.handle
        if t == "enum":
            entry = self.sdk.value(
                f"TucGetFeatureEnumEntryByName({name}, {value})", a.TucGetFeatureEnumEntryByName(h, name, str(value))
            )
            self.sdk.ok(f"set {name}={value}", a.TucSetFeatureEnumValue(h, name, entry.value))
        elif t == "int":
            self.sdk.ok(f"set {name}={value}", a.TucSetFeatureInt64(h, name, round(value)))
        elif t == "float":
            self.sdk.ok(f"set {name}={value}", a.TucSetFeatureFloat(h, name, float(value)))
        elif t == "bool":
            self.sdk.ok(f"set {name}={value}", a.TucSetFeatureBool(h, name, bool(value)))
        elif t == "str":
            self.sdk.ok(f"set {name}={value}", a.TucSetFeatureString(h, name, str(value)))
        elif t == "command":
            self.execute(name)
        else:
            raise TypeError(f"{name} is a {t} node and cannot be set")

    def execute(self, name: str) -> None:
        self.sdk.ok(f"execute {name}", self.sdk.api.TucExecuteFeatureCommand(self.handle, name))

    # ---- buffers and acquisition ------------------------------------------------------
    def required_size(self, unpacked: bool = False) -> int:
        """Bytes one announced buffer must hold for the current settings (includes chunk data when chunk mode is on)."""
        K = self.sdk.D.TUC_BUFFER_SIZE_KIND
        kind = K.TUC_BUFSIZE_DATA_UNPACKED if unpacked else K.TUC_BUFSIZE_ANNOUNCE
        return int(
            self.sdk.value("TucBuffersGetRequiredSize", self.sdk.api.TucBuffersGetRequiredSize(self.handle, kind))
        )

    def set_manual_queue(self, manual: bool) -> None:
        P = self.sdk.D.TUC_BUFFER_QUEUE_POLICY
        policy = P.TUC_BUFFER_QUEUE_POLICY_MANUAL if manual else P.TUC_BUFFER_QUEUE_POLICY_AUTO
        self.sdk.ok("TucBufferSetQueuePolicy", self.sdk.api.TucBufferSetQueuePolicy(self.handle, policy))

    def announce(self, address: int, size: int) -> int:
        """Announce ``size`` bytes of host memory at ``address`` as an acquisition buffer; returns its handle."""
        return self.sdk.value(
            "TucBufferAnnounce", self.sdk.api.TucBufferAnnounce(self.handle, C.c_void_p(address), size)
        )

    def revoke(self, buffer: int) -> None:
        self.sdk.ok("TucBufferRevoke", self.sdk.api.TucBufferRevoke(self.handle, buffer))

    def queue(self, buffer: int) -> int:
        """Return a delivered buffer to the input pool (MANUAL policy); returns the status, so callers on a hot path
        can decide what a failure means."""
        return self.sdk.api.TucBufferQueue(self.handle, buffer)

    def buffer_info(self, buffer: int) -> Tuple[int, Any]:
        """(status, TUC_BUFFER_INFO) for a delivered buffer."""
        return self.sdk.api.TucBufferGetInfo(self.handle, buffer)

    def make_buffer_callback(self, fn) -> Any:
        """Wrap ``fn(device, buffer, context)`` in the binding's callback type. Keep a reference to the result for as
        long as it is registered: the SDK holds a raw pointer."""
        return self.sdk.D.TUC_BUFFER_CALLBACK(fn)

    def register_buffer_callback(self, callback: Any) -> None:
        self.sdk.ok("TucRegisterBufferCallback", self.sdk.api.TucRegisterBufferCallback(self.handle, callback, None))

    def unregister_buffer_callback(self) -> None:
        self.sdk.ok("TucUnregisterBufferCallback", self.sdk.api.TucUnregisterBufferCallback(self.handle))

    def start(self, frame_count: int = 0) -> None:
        """Start acquisition; ``frame_count`` 0 runs until stopped."""
        flags = self.sdk.D.TUC_ACQ_START_FLAGS.TUC_ACQ_START_FLAGS_DEFAULT
        self.sdk.ok("TucAcquisitionStart", self.sdk.api.TucAcquisitionStart(self.handle, flags, int(frame_count)))

    def stop(self) -> None:
        flags = self.sdk.D.TUC_ACQ_STOP_FLAGS.TUC_ACQ_STOP_FLAGS_DEFAULT
        self.sdk.ok("TucAcquisitionStop", self.sdk.api.TucAcquisitionStop(self.handle, flags))

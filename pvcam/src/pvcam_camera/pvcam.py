"""ctypes binding to the PVCAM runtime (pvcam64.dll on Windows, libpvcam.so on Linux).

Only the calls a continuous, zero-copy acquisition needs are bound: library init/uninit, camera enumeration and
open/close, parameter get/set/enumerate, ``pl_exp_setup_cont`` / ``pl_exp_start_cont`` / ``pl_exp_stop_cont``,
``pl_exp_check_cont_status``, ``pl_exp_get_oldest_frame_ex`` / ``pl_exp_unlock_oldest_frame`` and the error text.

The declarations below were written from the PVCAM SDK 3.10.1.1 header (``pvcam.h`` / ``master.h``); the header is
not shipped with this package. Every PVCAM function returns ``rs_bool`` (``unsigned short``, 0 = failure) and sets
``pl_error_code()``; :class:`Pvcam` turns a failure into :class:`PvcamError` carrying ``pl_error_message`` text.

:class:`Pvcam` is the Python-level interface :class:`pvcam_camera.PvcamSource` drives. A test double only needs the
same methods (see ``tests/fake_pvcam.py``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from collections import namedtuple
from typing import List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------------------------- constants (pvcam.h)
CAM_NAME_LEN = 32
ERROR_MSG_LEN = 255

# PL_PARAM_ATTRIBUTES
ATTR_CURRENT, ATTR_COUNT, ATTR_TYPE, ATTR_MIN, ATTR_MAX = 0, 1, 2, 3, 4
ATTR_DEFAULT, ATTR_INCREMENT, ATTR_ACCESS, ATTR_AVAIL, ATTR_LIVE = 5, 6, 7, 8, 9
# PL_PARAM_ACCESS
ACC_READ_ONLY, ACC_READ_WRITE, ACC_EXIST_CHECK_ONLY, ACC_WRITE_ONLY = 1, 2, 3, 4

# parameter data types (TYPE_*), encoded in bits 24..31 of a parameter id
TYPE_INT16, TYPE_INT32, TYPE_FLT64, TYPE_UNS8, TYPE_UNS16, TYPE_UNS32, TYPE_UNS64, TYPE_ENUM = 1, 2, 4, 5, 6, 7, 8, 9
TYPE_BOOLEAN, TYPE_INT8, TYPE_CHAR_PTR, TYPE_VOID_PTR = 11, 12, 13, 14
TYPE_VOID_PTR_PTR, TYPE_INT64, TYPE_FLT32 = 15, 16, 19
CLASS0, CLASS2, CLASS3 = 0, 2, 3


def _pid(cls: int, typ: int, n: int) -> int:
    return (cls << 16) + (typ << 24) + n


PARAM_CAM_INTERFACE_TYPE = _pid(CLASS0, TYPE_ENUM, 10)
PARAM_CHIP_NAME = _pid(CLASS2, TYPE_CHAR_PTR, 129)
PARAM_VENDOR_NAME = _pid(CLASS2, TYPE_CHAR_PTR, 131)
PARAM_PRODUCT_NAME = _pid(CLASS2, TYPE_CHAR_PTR, 132)
PARAM_CAMERA_PART_NUMBER = _pid(CLASS2, TYPE_CHAR_PTR, 133)
PARAM_PAR_SIZE = _pid(CLASS2, TYPE_UNS16, 57)
PARAM_SER_SIZE = _pid(CLASS2, TYPE_UNS16, 58)
PARAM_READOUT_TIME = _pid(CLASS2, TYPE_FLT64, 179)  # header: "type is incorrectly defined. The actual type is UNS32"
PARAM_CAM_FW_VERSION = _pid(CLASS2, TYPE_UNS16, 532)
PARAM_HEAD_SER_NUM_ALPHA = _pid(CLASS2, TYPE_CHAR_PTR, 533)
PARAM_EXPOSURE_MODE = _pid(CLASS2, TYPE_ENUM, 535)
PARAM_EXPOSE_OUT_MODE = _pid(CLASS2, TYPE_ENUM, 560)
PARAM_BIT_DEPTH = _pid(CLASS2, TYPE_INT16, 511)
PARAM_BIT_DEPTH_HOST = _pid(CLASS2, TYPE_INT16, 551)
PARAM_IMAGE_FORMAT = _pid(CLASS2, TYPE_ENUM, 248)
PARAM_IMAGE_FORMAT_HOST = _pid(CLASS2, TYPE_ENUM, 552)
PARAM_IMAGE_COMPRESSION_HOST = _pid(CLASS2, TYPE_ENUM, 253)
PARAM_GAIN_INDEX = _pid(CLASS2, TYPE_INT16, 512)
PARAM_SPDTAB_INDEX = _pid(CLASS2, TYPE_INT16, 513)
PARAM_GAIN_NAME = _pid(CLASS2, TYPE_CHAR_PTR, 514)
PARAM_SPDTAB_NAME = _pid(CLASS2, TYPE_CHAR_PTR, 515)
PARAM_READOUT_PORT = _pid(CLASS2, TYPE_ENUM, 247)
PARAM_PIX_TIME = _pid(CLASS2, TYPE_UNS16, 516)
PARAM_EXP_RES = _pid(CLASS3, TYPE_ENUM, 2)
PARAM_EXP_RES_INDEX = _pid(CLASS3, TYPE_UNS16, 4)
PARAM_EXPOSURE_TIME = _pid(CLASS3, TYPE_UNS64, 8)
PARAM_CIRC_BUFFER = _pid(CLASS3, TYPE_BOOLEAN, 299)
PARAM_FRAME_BUFFER_SIZE = _pid(CLASS3, TYPE_UNS64, 300)
PARAM_METADATA_ENABLED = _pid(CLASS3, TYPE_BOOLEAN, 168)
PARAM_CENTROIDS_ENABLED = _pid(CLASS3, TYPE_BOOLEAN, 170)

# PL_IMAGE_FORMATS
PL_IMAGE_FORMAT_MONO16, PL_IMAGE_FORMAT_BAYER16, PL_IMAGE_FORMAT_MONO8, PL_IMAGE_FORMAT_BAYER8 = 0, 1, 2, 3
PL_IMAGE_COMPRESSION_NONE = 0

# PL_EXPOSURE_MODES (legacy and extended) and PL_EXPOSE_OUT_MODES
TIMED_MODE = 0
EXT_TRIG_INTERNAL = (7 + 0) << 8
EXT_TRIG_TRIG_FIRST = (7 + 1) << 8
EXT_TRIG_EDGE_RISING = (7 + 2) << 8
EXT_TRIG_SOFTWARE_EDGE = (7 + 5) << 8
EXPOSE_OUT_FIRST_ROW = 0

# PL_EXP_RES_MODES
EXP_RES_ONE_MILLISEC, EXP_RES_ONE_MICROSEC, EXP_RES_ONE_SEC = 0, 1, 2

# PL_IMAGE_STATUSES
READOUT_NOT_ACTIVE, EXPOSURE_IN_PROGRESS, READOUT_IN_PROGRESS, READOUT_COMPLETE, READOUT_FAILED = 0, 1, 2, 3, 4
FRAME_AVAILABLE = READOUT_COMPLETE

# PL_CIRC_MODES
CIRC_NONE, CIRC_OVERWRITE, CIRC_NO_OVERWRITE = 0, 1, 2

# PL_CCS_ABORT_MODES, PL_OPEN_MODES
CCS_NO_CHANGE, CCS_HALT = 0, 1
OPEN_EXCLUSIVE = 0

#: pl_exp_start_cont takes the buffer size as uns32: the circular buffer is at most this many bytes
MAX_CONT_BUFFER_BYTES = 0xFFFFFFFF


# ----------------------------------------------------------------------------------------------- C types
class PVCAM_FRAME_INFO_GUID(ctypes.Structure):
    _fields_ = [("f1", ctypes.c_uint32), ("f2", ctypes.c_uint16), ("f3", ctypes.c_uint16), ("f4", ctypes.c_uint8 * 8)]


class FRAME_INFO(ctypes.Structure):
    """pvcam.h FRAME_INFO (natural alignment; the header's #pragma pack(1) covers the metadata structs only)."""

    _fields_ = [
        ("FrameInfoGUID", PVCAM_FRAME_INFO_GUID),
        ("hCam", ctypes.c_int16),
        ("FrameNr", ctypes.c_int32),  # 1-based
        ("TimeStamp", ctypes.c_int64),  # EOF; header: "precision of 0.1ms"
        ("ReadoutTime", ctypes.c_int32),
        ("TimeStampBOF", ctypes.c_int64),
    ]


class rgn_type(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint16) for n in ("s1", "s2", "sbin", "p1", "p2", "pbin")]


#: one region: serial (x) first/last pixel and binning, parallel (y) first/last row and binning (inclusive bounds)
Region = namedtuple("Region", "s1 s2 sbin p1 p2 pbin")

#: what get_oldest_frame returns besides the address: the FRAME_INFO fields, as Python ints
FrameRecord = namedtuple("FrameRecord", "frame_nr timestamp timestamp_bof readout_time")


class PvcamError(RuntimeError):
    def __init__(self, function: str, code: int, message: str):
        super().__init__(f"{function} failed: {message} (code {code})")
        self.function, self.code, self.message = function, int(code), message


_rs_bool = ctypes.c_uint16
_I16P, _U16P, _U32P = ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32)
_I32P, _RGNP = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(rgn_type)

#: (name, restype, argtypes) of every bound function
_PROTOTYPES = [
    ("pl_pvcam_get_ver", _rs_bool, [_U16P]),
    ("pl_pvcam_init", _rs_bool, []),
    ("pl_pvcam_uninit", _rs_bool, []),
    ("pl_cam_get_total", _rs_bool, [_I16P]),
    ("pl_cam_get_name", _rs_bool, [ctypes.c_int16, ctypes.c_char_p]),
    ("pl_cam_open", _rs_bool, [ctypes.c_char_p, _I16P, ctypes.c_int16]),
    ("pl_cam_close", _rs_bool, [ctypes.c_int16]),
    ("pl_error_code", ctypes.c_int16, []),
    ("pl_error_message", _rs_bool, [ctypes.c_int16, ctypes.c_char_p]),
    ("pl_get_param", _rs_bool, [ctypes.c_int16, ctypes.c_uint32, ctypes.c_int16, ctypes.c_void_p]),
    ("pl_set_param", _rs_bool, [ctypes.c_int16, ctypes.c_uint32, ctypes.c_void_p]),
    (
        "pl_get_enum_param",
        _rs_bool,
        [ctypes.c_int16, ctypes.c_uint32, ctypes.c_uint32, _I32P, ctypes.c_char_p, ctypes.c_uint32],
    ),
    ("pl_enum_str_length", _rs_bool, [ctypes.c_int16, ctypes.c_uint32, ctypes.c_uint32, _U32P]),
    ("pl_create_frame_info_struct", _rs_bool, [ctypes.POINTER(ctypes.POINTER(FRAME_INFO))]),
    ("pl_release_frame_info_struct", _rs_bool, [ctypes.POINTER(FRAME_INFO)]),
    (
        "pl_exp_setup_cont",
        _rs_bool,
        [ctypes.c_int16, ctypes.c_uint16, _RGNP, ctypes.c_int16, ctypes.c_uint32, _U32P, ctypes.c_int16],
    ),
    ("pl_exp_start_cont", _rs_bool, [ctypes.c_int16, ctypes.c_void_p, ctypes.c_uint32]),
    ("pl_exp_check_cont_status", _rs_bool, [ctypes.c_int16, _I16P, _U32P, _U32P]),
    (
        "pl_exp_get_oldest_frame_ex",
        _rs_bool,
        [ctypes.c_int16, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(FRAME_INFO)],
    ),
    ("pl_exp_unlock_oldest_frame", _rs_bool, [ctypes.c_int16]),
    ("pl_exp_stop_cont", _rs_bool, [ctypes.c_int16, ctypes.c_int16]),
    ("pl_exp_abort", _rs_bool, [ctypes.c_int16, ctypes.c_int16]),
]

_SCALAR_CTYPES = {
    TYPE_INT8: ctypes.c_int8,
    TYPE_UNS8: ctypes.c_uint8,
    TYPE_INT16: ctypes.c_int16,
    TYPE_UNS16: ctypes.c_uint16,
    TYPE_INT32: ctypes.c_int32,
    TYPE_UNS32: ctypes.c_uint32,
    TYPE_INT64: ctypes.c_int64,
    TYPE_UNS64: ctypes.c_uint64,
    TYPE_FLT32: ctypes.c_float,
    TYPE_FLT64: ctypes.c_double,
    TYPE_ENUM: ctypes.c_int32,
    TYPE_BOOLEAN: _rs_bool,
}
# attributes whose data type does not follow the parameter's type
_ATTR_CTYPES = {
    ATTR_COUNT: ctypes.c_uint32,
    ATTR_TYPE: ctypes.c_uint16,
    ATTR_ACCESS: ctypes.c_uint16,
    ATTR_AVAIL: _rs_bool,
    ATTR_LIVE: _rs_bool,
}


def param_type(param_id: int) -> int:
    """TYPE_* of a parameter id (bits 24..31)."""
    return (int(param_id) >> 24) & 0xFF


def library_candidates() -> List[str]:
    """Names/paths tried by :func:`find_library`, in order. ``PVCAM_LIBRARY`` (a full path) goes first."""
    env = os.environ.get("PVCAM_LIBRARY")
    names = [env] if env else []
    if sys.platform.startswith("win"):
        names += ["pvcam64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 else "pvcam32.dll"]
    else:
        names += ["libpvcam.so", "libpvcam.so.2"]
        found = ctypes.util.find_library("pvcam")
        if found:
            names.append(found)
    return names


def find_library(path: Optional[str] = None):
    """Load the PVCAM runtime. Returns the ctypes library object; raises OSError listing what was tried."""
    loader = ctypes.WinDLL if sys.platform.startswith("win") else ctypes.CDLL  # PV_DECL is __stdcall on Windows
    tried = []
    for name in [path] if path else library_candidates():
        try:
            return loader(name)
        except OSError as e:
            tried.append(f"{name}: {e}")
    raise OSError(
        "PVCAM runtime not found (install the PVCAM runtime from Teledyne Photometrics, or set PVCAM_LIBRARY to the "
        "full path of pvcam64.dll / libpvcam.so). Tried: " + "; ".join(tried)
    )


class Pvcam:
    """The PVCAM runtime, with Python-level calls. One instance per process is enough (:meth:`shared`);
    :meth:`init` / :meth:`uninit` are reference counted so several sources can share it."""

    _shared: Optional["Pvcam"] = None
    _shared_lock = threading.Lock()

    def __init__(self, library: Optional[str] = None):
        self.lib = find_library(library)
        for name, restype, argtypes in _PROTOTYPES:
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        self._refs = 0
        self._lock = threading.Lock()
        self._frame_info = {}  # hcam -> POINTER(FRAME_INFO) from pl_create_frame_info_struct

    @classmethod
    def shared(cls, library: Optional[str] = None) -> "Pvcam":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(library)
            return cls._shared

    # ------------------------------------------------------------ errors
    def error_text(self, code: Optional[int] = None) -> Tuple[int, str]:
        c = int(self.lib.pl_error_code()) if code is None else int(code)
        buf = ctypes.create_string_buffer(ERROR_MSG_LEN + 1)
        self.lib.pl_error_message(c, buf)
        return c, buf.value.decode("ascii", "replace")

    def _check(self, ok, function: str) -> None:
        if not ok:
            code, msg = self.error_text()
            raise PvcamError(function, code, msg)

    # ------------------------------------------------------------ library
    def init(self) -> None:
        with self._lock:
            if self._refs == 0:
                self._check(self.lib.pl_pvcam_init(), "pl_pvcam_init")
            self._refs += 1

    def uninit(self) -> None:
        with self._lock:
            if self._refs == 0:
                return
            self._refs -= 1
            if self._refs == 0:
                self._check(self.lib.pl_pvcam_uninit(), "pl_pvcam_uninit")

    def version(self) -> str:
        v = ctypes.c_uint16()
        self._check(self.lib.pl_pvcam_get_ver(ctypes.byref(v)), "pl_pvcam_get_ver")
        # high byte major, high nibble of the low byte minor, low nibble trivial (pvcam.h pl_pvcam_get_ver)
        return f"{v.value >> 8}.{(v.value >> 4) & 0xF}.{v.value & 0xF}"

    # ------------------------------------------------------------ cameras
    def camera_names(self) -> List[str]:
        n = ctypes.c_int16()
        self._check(self.lib.pl_cam_get_total(ctypes.byref(n)), "pl_cam_get_total")
        out = []
        for i in range(n.value):
            buf = ctypes.create_string_buffer(CAM_NAME_LEN)
            self._check(self.lib.pl_cam_get_name(i, buf), "pl_cam_get_name")
            out.append(buf.value.decode("ascii", "replace"))
        return out

    def open(self, name: str) -> int:
        h = ctypes.c_int16()
        self._check(self.lib.pl_cam_open(name.encode("ascii"), ctypes.byref(h), OPEN_EXCLUSIVE), "pl_cam_open")
        fi = ctypes.POINTER(FRAME_INFO)()
        self._check(self.lib.pl_create_frame_info_struct(ctypes.byref(fi)), "pl_create_frame_info_struct")
        self._frame_info[h.value] = fi
        return h.value

    def close(self, hcam: int) -> None:
        fi = self._frame_info.pop(hcam, None)
        if fi is not None:
            self.lib.pl_release_frame_info_struct(fi)
        self._check(self.lib.pl_cam_close(hcam), "pl_cam_close")

    # ------------------------------------------------------------ parameters
    def _attr_ctype(self, param_id: int, attr: int):
        if attr in _ATTR_CTYPES:
            return _ATTR_CTYPES[attr]
        if param_id == PARAM_READOUT_TIME:
            return ctypes.c_uint32
        t = param_type(param_id)
        if t not in _SCALAR_CTYPES:
            raise TypeError(f"parameter 0x{param_id:08x} has type {t}, not a scalar")
        return _SCALAR_CTYPES[t]

    def get_param(self, hcam: int, param_id: int, attr: int = ATTR_CURRENT):
        if param_type(param_id) == TYPE_CHAR_PTR and attr in (ATTR_CURRENT, ATTR_DEFAULT):
            n = max(int(self.get_param(hcam, param_id, ATTR_COUNT)), 256)
            buf = ctypes.create_string_buffer(n)
            self._check(self.lib.pl_get_param(hcam, param_id, attr, buf), "pl_get_param")
            return buf.value.decode("ascii", "replace")
        ct = self._attr_ctype(param_id, attr)
        # 8 zeroed bytes whatever the type: PARAM_READOUT_TIME is declared FLT64 but written as UNS32
        raw = (ctypes.c_uint8 * 8)()
        self._check(self.lib.pl_get_param(hcam, param_id, attr, ctypes.byref(raw)), "pl_get_param")
        v = ct.from_buffer(raw).value
        if attr in (ATTR_AVAIL, ATTR_LIVE):
            return bool(v)
        if param_type(param_id) == TYPE_BOOLEAN and attr in (ATTR_CURRENT, ATTR_DEFAULT, ATTR_MIN, ATTR_MAX):
            return bool(v)
        return v

    def available(self, hcam: int, param_id: int) -> bool:
        try:
            return bool(self.get_param(hcam, param_id, ATTR_AVAIL))
        except PvcamError:
            return False

    def writable(self, hcam: int, param_id: int) -> bool:
        return self.available(hcam, param_id) and self.get_param(hcam, param_id, ATTR_ACCESS) in (
            ACC_READ_WRITE,
            ACC_WRITE_ONLY,
        )

    def set_param(self, hcam: int, param_id: int, value) -> None:
        if param_type(param_id) == TYPE_CHAR_PTR:
            buf = ctypes.create_string_buffer(str(value).encode("ascii"))
            self._check(self.lib.pl_set_param(hcam, param_id, buf), "pl_set_param")
            return
        v = self._attr_ctype(param_id, ATTR_CURRENT)(value)
        self._check(self.lib.pl_set_param(hcam, param_id, ctypes.byref(v)), "pl_set_param")

    def enum_values(self, hcam: int, param_id: int) -> List[Tuple[int, str]]:
        """[(value, description)] of an enumerated parameter."""
        n = int(self.get_param(hcam, param_id, ATTR_COUNT))
        out = []
        for i in range(n):
            ln = ctypes.c_uint32()
            self._check(self.lib.pl_enum_str_length(hcam, param_id, i, ctypes.byref(ln)), "pl_enum_str_length")
            buf = ctypes.create_string_buffer(max(int(ln.value), 1))
            val = ctypes.c_int32()
            self._check(
                self.lib.pl_get_enum_param(hcam, param_id, i, ctypes.byref(val), buf, max(int(ln.value), 1)),
                "pl_get_enum_param",
            )
            out.append((val.value, buf.value.decode("ascii", "replace")))
        return out

    # ------------------------------------------------------------ continuous acquisition
    def setup_cont(self, hcam: int, regions: Sequence[Region], exp_mode: int, exposure: int, buffer_mode: int) -> int:
        """pl_exp_setup_cont; returns the bytes of one frame in the pixel stream (``exp_bytes``)."""
        arr = (rgn_type * len(regions))(*[rgn_type(*r) for r in regions])
        nbytes = ctypes.c_uint32()
        self._check(
            self.lib.pl_exp_setup_cont(
                hcam, len(regions), arr, int(exp_mode), int(exposure), ctypes.byref(nbytes), int(buffer_mode)
            ),
            "pl_exp_setup_cont",
        )
        return int(nbytes.value)

    def start_cont(self, hcam: int, address: int, size: int) -> None:
        """pl_exp_start_cont into the caller's circular buffer at ``address`` (``size`` bytes, a multiple of the
        frame size, at most MAX_CONT_BUFFER_BYTES)."""
        if not 0 < size <= MAX_CONT_BUFFER_BYTES:
            raise ValueError(f"circular buffer of {size} bytes: pl_exp_start_cont takes 1..{MAX_CONT_BUFFER_BYTES}")
        self._check(self.lib.pl_exp_start_cont(hcam, ctypes.c_void_p(address), size), "pl_exp_start_cont")

    def check_cont_status(self, hcam: int) -> Tuple[int, int, int]:
        """(status PL_IMAGE_STATUSES, bytes arrived this time through the buffer, times through the buffer)"""
        st, b, n = ctypes.c_int16(), ctypes.c_uint32(), ctypes.c_uint32()
        self._check(
            self.lib.pl_exp_check_cont_status(hcam, ctypes.byref(st), ctypes.byref(b), ctypes.byref(n)),
            "pl_exp_check_cont_status",
        )
        return int(st.value), int(b.value), int(n.value)

    def get_oldest_frame(self, hcam: int) -> Tuple[int, FrameRecord]:
        """pl_exp_get_oldest_frame_ex: (address of the oldest unretrieved frame, its FRAME_INFO). The frame stays
        locked until :meth:`unlock_oldest_frame`."""
        fi = self._frame_info[hcam]
        p = ctypes.c_void_p()
        self._check(self.lib.pl_exp_get_oldest_frame_ex(hcam, ctypes.byref(p), fi), "pl_exp_get_oldest_frame_ex")
        f = fi.contents
        return int(p.value or 0), FrameRecord(int(f.FrameNr), int(f.TimeStamp), int(f.TimeStampBOF), int(f.ReadoutTime))

    def unlock_oldest_frame(self, hcam: int) -> None:
        self._check(self.lib.pl_exp_unlock_oldest_frame(hcam), "pl_exp_unlock_oldest_frame")

    def stop_cont(self, hcam: int, cam_state: int = CCS_HALT) -> None:
        self._check(self.lib.pl_exp_stop_cont(hcam, cam_state), "pl_exp_stop_cont")

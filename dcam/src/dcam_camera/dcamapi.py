"""ctypes declarations for the DCAM-API 4 C interface (``dcamapi.dll`` / ``libdcamapi.so``).

Written against ``dcamapi4.h`` and ``dcamprop.h`` from Hamamatsu's DCAM-SDK 4 (v25.0.5.6964). Only the
calls this package uses are declared. Hamamatsu's code is not redistributed: the runtime library comes with
the DCAM-API driver installer (``C:\\Windows\\System32\\dcamapi.dll`` on Windows), and nothing from the SDK
is needed at run time.

Structures follow the header's ``#pragma pack(8)``, which is ctypes' natural alignment on 64-bit.

Buffer attach: ``ATTACHKIND_FRAME`` (``DCAMBUF_ATTACH.buffer`` is an array of host pointers, one per buffer frame)
for the images, and the metadata kinds ``ATTACHKIND_[PRIMARY_]TIMESTAMP`` / ``ATTACHKIND_[PRIMARY_]FRAMESTAMP``, into
which DCAM writes each frame's timestamp / framestamp. No
device-memory (GPU) attach kind exists in the DCAM-API.
"""

from __future__ import annotations

import ctypes as C
import os
import platform
from enum import IntEnum, IntFlag

ENV_DLL = "DCAMAPI_DLL"

i32 = C.c_int32
HDCAM = C.c_void_p
HDCAMWAIT = C.c_void_p


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
class DCAMERR(IntEnum):
    """Error codes (``DCAMERR``). Failures are negative as int32; success is positive."""

    BUSY = -2147483391  # 0x80000101
    NOTREADY = -2147483389  # 0x80000103
    NOTSTABLE = -2147483388  # 0x80000104
    UNSTABLE = -2147483387  # 0x80000105
    NOTBUSY = -2147483385  # 0x80000107
    EXCLUDED = -2147483376  # 0x80000110
    COOLINGTROUBLE = -2147482878  # 0x80000302
    NOTRIGGER = -2147482877  # 0x80000303
    TEMPERATURE_TROUBLE = -2147482876  # 0x80000304
    TOOFREQUENTTRIGGER = -2147482875  # 0x80000305
    ABORT = -2147483390  # 0x80000102
    TIMEOUT = -2147483386  # 0x80000106
    LOSTFRAME = -2147482879  # 0x80000301
    MISSINGFRAME_TROUBLE = -2147479802  # 0x80000f06
    INVALIDIMAGE = -2147482847  # 0x80000321
    NORESOURCE = -2147483135  # 0x80000201
    NOMEMORY = -2147483133  # 0x80000203
    NOMODULE = -2147483132  # 0x80000204
    NODRIVER = -2147483131  # 0x80000205
    NOCAMERA = -2147483130  # 0x80000206
    NOGRABBER = -2147483129  # 0x80000207
    FAILOPENBUS = -2130702335  # 0x81001001
    FAILOPENCAMERA = -2113925119  # 0x82001001
    DEVICEPROBLEM = -2113925118  # 0x82001002
    INVALIDCAMERA = -2147481594  # 0x80000806
    INVALIDHANDLE = -2147481593  # 0x80000807
    INVALIDPARAM = -2147481592  # 0x80000808
    INVALIDVALUE = -2147481567  # 0x80000821
    OUTOFRANGE = -2147481566  # 0x80000822
    NOTWRITABLE = -2147481565  # 0x80000823
    NOTREADABLE = -2147481564  # 0x80000824
    INVALIDPROPERTYID = -2147481563  # 0x80000825
    NEWAPIREQUIRED = -2147481562  # 0x80000826
    WRONGHANDSHAKE = -2147481561  # 0x80000827
    NOPROPERTY = -2147481560  # 0x80000828
    INVALIDCHANNEL = -2147481559  # 0x80000829
    INVALIDVIEW = -2147481558  # 0x8000082a
    INVALIDSUBARRAY = -2147481557  # 0x8000082b
    ACCESSDENY = -2147481556  # 0x8000082c
    NOVALUETEXT = -2147481555  # 0x8000082d
    WRONGPROPERTYVALUE = -2147481554  # 0x8000082e
    DISHARMONY = -2147481552  # 0x80000830
    FRAMEBUNDLESHOULDBEOFF = -2147481550  # 0x80000832
    INVALIDFRAMEINDEX = -2147481549  # 0x80000833
    INVALIDSESSIONINDEX = -2147481548  # 0x80000834
    NOCORRECTIONDATA = -2147481544  # 0x80000838
    CHANNELDEPENDENTVALUE = -2147481543  # 0x80000839
    VIEWDEPENDENTVALUE = -2147481542  # 0x8000083a
    NODEVICEBUFFER = -2147481541  # 0x8000083b
    REQUIREDSNAP = -2147481540  # 0x8000083c
    LESSSYSTEMMEMORY = -2147481537  # 0x8000083f
    NOTSUPPORT = -2147479805  # 0x80000f03
    FAILREADCAMERA = -2097147902  # 0x83001002
    FAILWRITECAMERA = -2097147901  # 0x83001003
    UNKNOWNMSGID = -2147481599  # 0x80000801
    UNKNOWNSTRID = -2147481598  # 0x80000802
    UNKNOWNPARAMID = -2147481597  # 0x80000803
    INVALIDWAITHANDLE = -2080366591  # 0x84002001
    NEWRUNTIMEREQUIRED = -2080366590  # 0x84002002
    UNREACH = -2147479807  # 0x80000f01
    UNLOADED = -2147479804  # 0x80000f04
    NOCONNECTION = -2147479801  # 0x80000f07
    NOTIMPLEMENT = -2147479806  # 0x80000f02
    NONE = 0
    SUCCESS = 1


def err_name(code: int) -> str:
    try:
        return DCAMERR(int(code)).name
    except ValueError:
        return f"0x{int(code) & 0xFFFFFFFF:08X}"


class IDSTR(IntEnum):
    BUS = 0x04000101
    CAMERAID = 0x04000102
    VENDOR = 0x04000103
    MODEL = 0x04000104
    CAMERAVERSION = 0x04000105
    DRIVERVERSION = 0x04000106
    MODULEVERSION = 0x04000107
    DCAMAPIVERSION = 0x04000108
    CAMERA_SERIESNAME = 0x0400012C


class PROPATTR(IntFlag):
    """``DCAMPROP_ATTR`` bits of ``DCAMPROP_ATTR.attribute`` (the low nibble is the value type)."""

    HASRANGE = 0x80000000
    HASSTEP = 0x40000000
    HASDEFAULT = 0x20000000
    HASVALUETEXT = 0x10000000
    HASCHANNEL = 0x08000000
    AUTOROUNDING = 0x00800000
    STEPPING_INCONSISTENT = 0x00400000
    DATASTREAM = 0x00200000
    HASRATIO = 0x00100000
    VOLATILE = 0x00080000
    WRITABLE = 0x00020000
    READABLE = 0x00010000
    HASVIEW = 0x00008000
    ACCESSREADY = 0x00002000
    ACCESSBUSY = 0x00001000
    ACTION = 0x00000400
    EFFECTIVE = 0x00000200


PROPTYPE_MASK = 0x0000000F
PROPTYPES = {1: "mode", 2: "long", 3: "real"}  # DCAMPROP_TYPE_MODE / LONG / REAL
PROPUNITS = {1: "s", 2: "°C", 3: "K", 4: "m/s", 5: "1/s", 6: "°", 7: "µm"}  # DCAMPROP_UNIT


class PROPOPTION(IntEnum):
    PRIOR = -16777216  # 0xFF000000
    NEXT = 0x01000000
    SUPPORT = 0x00000000
    ARRAYELEMENT = 0x00000004


class PIXELTYPE(IntEnum):
    MONO8 = 0x00000001
    MONO16 = 0x00000002
    MONO12 = 0x00000003
    MONO12P = 0x00000005
    NONE = 0x00000000


class CAPSTATUS(IntEnum):
    ERROR = 0x0000
    BUSY = 0x0001
    READY = 0x0002
    STABLE = 0x0003
    UNSTABLE = 0x0004


class WAITEVENT(IntFlag):
    TRANSFERRED = 0x0001
    FRAMEREADY = 0x0002
    CYCLEEND = 0x0004
    EXPOSUREEND = 0x0008
    STOPPED = 0x0010


CAPSTART_SEQUENCE = -1
CAPSTART_SNAP = 0
ATTACHKIND_FRAME = 0
ATTACHKIND_TIMESTAMP = 1  # pointer array, one DCAM_TIMESTAMP per buffer frame
ATTACHKIND_FRAMESTAMP = 2  # pointer array, one int32 per buffer frame
ATTACHKIND_PRIMARY_TIMESTAMP = 3  # one contiguous DCAM_TIMESTAMP array (buffer = pointer to its address)
ATTACHKIND_PRIMARY_FRAMESTAMP = 4  # one contiguous int32 array
TRANSFERKIND_FRAME = 0
WAIT_TIMEOUT_INFINITE = -2147483648  # 0x80000000


# --------------------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------------------
class _Sized(C.Structure):
    """A DCAM structure whose first member is its own byte size."""

    def __init__(self, **kw):
        super().__init__(**kw)
        setattr(self, self._fields_[0][0], C.sizeof(self))


class DCAM_GUID(C.Structure):
    _fields_ = [("Data1", C.c_uint32), ("Data2", C.c_ushort), ("Data3", C.c_ushort), ("Data4", C.c_ubyte * 8)]


class DCAMAPI_INIT(_Sized):
    _fields_ = [
        ("size", i32),
        ("iDeviceCount", i32),
        ("reserved", i32),
        ("initoptionbytes", i32),
        ("initoption", C.POINTER(i32)),
        ("guid", C.POINTER(DCAM_GUID)),
    ]


class DCAMDEV_OPEN(_Sized):
    _fields_ = [("size", i32), ("index", i32), ("hdcam", HDCAM)]


class DCAMDEV_STRING(_Sized):
    _fields_ = [("size", i32), ("iString", i32), ("text", C.c_char_p), ("textbytes", i32)]


class DCAMPROP_ATTR(_Sized):
    _fields_ = [
        ("cbSize", i32),
        ("iProp", i32),
        ("option", i32),
        ("iReserved1", i32),
        ("attribute", i32),
        ("iGroup", i32),
        ("iUnit", i32),
        ("attribute2", i32),
        ("valuemin", C.c_double),
        ("valuemax", C.c_double),
        ("valuestep", C.c_double),
        ("valuedefault", C.c_double),
        ("nMaxChannel", i32),
        ("iReserved3", i32),
        ("nMaxView", i32),
        ("iProp_NumberOfElement", i32),
        ("iProp_ArrayBase", i32),
        ("iPropStep_Element", i32),
    ]


class DCAMPROP_VALUETEXT(_Sized):
    _fields_ = [("cbSize", i32), ("iProp", i32), ("value", C.c_double), ("text", C.c_char_p), ("textbytes", i32)]


class DCAMBUF_ATTACH(_Sized):
    _fields_ = [("size", i32), ("iKind", i32), ("buffer", C.POINTER(C.c_void_p)), ("buffercount", i32)]


class DCAM_TIMESTAMP(C.Structure):
    _fields_ = [("sec", C.c_uint32), ("microsec", i32)]


class DCAMCAP_TRANSFERINFO(_Sized):
    _fields_ = [("size", i32), ("iKind", i32), ("nNewestFrameIndex", i32), ("nFrameCount", i32)]


class DCAMBUF_FRAME(_Sized):
    _fields_ = [
        ("size", i32),
        ("iKind", i32),
        ("option", i32),
        ("iFrame", i32),
        ("buf", C.c_void_p),
        ("rowbytes", i32),
        ("type", i32),
        ("width", i32),
        ("height", i32),
        ("left", i32),
        ("top", i32),
        ("timestamp", DCAM_TIMESTAMP),
        ("framestamp", i32),
        ("camerastamp", i32),
    ]


class DCAMWAIT_OPEN(_Sized):
    _fields_ = [("size", i32), ("supportevent", i32), ("hwait", HDCAMWAIT), ("hdcam", HDCAM)]


class DCAMWAIT_START(_Sized):
    _fields_ = [("size", i32), ("eventhappened", i32), ("eventmask", i32), ("timeout", i32)]


# --------------------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------------------
_P = C.POINTER
PROTOTYPES: dict[str, list] = {
    "dcamapi_init": [_P(DCAMAPI_INIT)],
    "dcamapi_uninit": [],
    "dcamdev_open": [_P(DCAMDEV_OPEN)],
    "dcamdev_close": [HDCAM],
    "dcamdev_getstring": [HDCAM, _P(DCAMDEV_STRING)],
    "dcamprop_getattr": [HDCAM, _P(DCAMPROP_ATTR)],
    "dcamprop_getvalue": [HDCAM, i32, _P(C.c_double)],
    "dcamprop_setvalue": [HDCAM, i32, C.c_double],
    "dcamprop_setgetvalue": [HDCAM, i32, _P(C.c_double), i32],
    "dcamprop_queryvalue": [HDCAM, i32, _P(C.c_double), i32],
    "dcamprop_getnextid": [HDCAM, _P(i32), i32],
    "dcamprop_getname": [HDCAM, i32, C.c_char_p, i32],
    "dcamprop_getvaluetext": [HDCAM, _P(DCAMPROP_VALUETEXT)],
    "dcambuf_alloc": [HDCAM, i32],
    "dcambuf_attach": [HDCAM, _P(DCAMBUF_ATTACH)],
    "dcambuf_release": [HDCAM, i32],
    "dcambuf_lockframe": [HDCAM, _P(DCAMBUF_FRAME)],
    "dcambuf_copyframe": [HDCAM, _P(DCAMBUF_FRAME)],
    "dcamcap_start": [HDCAM, i32],
    "dcamcap_stop": [HDCAM],
    "dcamcap_status": [HDCAM, _P(i32)],
    "dcamcap_transferinfo": [HDCAM, _P(DCAMCAP_TRANSFERINFO)],
    "dcamcap_firetrigger": [HDCAM, i32],
    "dcamwait_open": [_P(DCAMWAIT_OPEN)],
    "dcamwait_close": [HDCAMWAIT],
    "dcamwait_start": [HDCAMWAIT, _P(DCAMWAIT_START)],
    "dcamwait_abort": [HDCAMWAIT],
}


def default_library() -> str:
    """``$DCAMAPI_DLL``, else the DCAM-API runtime's standard name for this platform."""
    if env := os.environ.get(ENV_DLL):
        return env
    return "dcamapi.dll" if platform.system() == "Windows" else "/usr/local/lib/libdcamapi.so"


def load_library(path: str | None = None) -> C.CDLL:
    """Load the DCAM-API runtime and declare the prototypes (every call returns a ``DCAMERR`` as int32)."""
    path = path or default_library()
    lib = C.WinDLL(path) if platform.system() == "Windows" else C.CDLL(path)
    for name, argtypes in PROTOTYPES.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = i32
    return lib

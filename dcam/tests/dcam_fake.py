"""A fake DCAM-API runtime for hardware-free tests of dcam_camera (adapted from an earlier internal driver's test fake).

``FakeDcamLib`` stands in for the ctypes library ``dcamapi.load_library`` returns: it has the same C
function names, takes the same ctypes arguments (structures by reference, string buffers) and returns
DCAMERR codes. It models one ORCA-Flash4.0 V3 (C13440-20CU) with the behaviour measured on the real
camera: mode value texts, ranges and steps; NOTSTABLE for layout settings while a buffer is allocated;
BUSY for settings without ACCESSBUSY during capture; SUBARRAY only applying with SUBARRAY MODE ON;
frame k of a capture landing in buffer frame ``k % slots``, overwriting old frames; a framestamp from ``stamp``;
``lockframe`` copying into one staging buffer; MONO12 packed 2 px in 3 bytes.

Frames are produced deterministically, not by a clock: each ``dcamwait_start`` on FRAMEREADY produces
``frames_per_wait`` frames in INTERNAL trigger mode, ``fire`` and ``dcamcap_firetrigger`` produce frames
on demand, and ``produce(n, skip=...)`` lets a test play the camera (``skip`` frames are lost before
DCAM: the framestamp advances, DCAM's frame count does not). Pixel values encode the framestamp so a
test can tell which frame it got: ``pixel(y, x) = (stamp * 7 + y + x) % 4096``.

Attached buffers (``dcambuf_attach``, ATTACHKIND_FRAME): the fake honours them. ``produce`` writes each frame's
bytes into the attached pointer ``count % slots`` with ``ctypes.memmove``, so a test sees exactly what a camera
writing into caller memory would leave there, including overwrites. ``lockframe`` on an attached buffer returns
the attached pointer and the frame's stamps, or fails with ``lockframe_attach_error`` when set (with the earlier
driver the real ORCA-Flash4.0 V3 refused lockframe with attached buffers; the error code the camera
returned is not recorded, so the fake uses NOTSUPPORT by default).

Metadata attach: ``ATTACHKIND_PRIMARY_TIMESTAMP`` / ``ATTACHKIND_PRIMARY_FRAMESTAMP``
arrays are honoured like the real ORCA-Flash4.0 V3 does: ``produce`` writes frame k's timestamp (sec = 1000 + stamp,
the same values lockframe returns) and framestamp into element ``count % buffercount``. ``meta_attach_error``
makes the attach fail; ``dcambuf_release(kind)`` releases only that kind.
"""

# The fake functions keep the C signatures (handle, option, ... arguments it does not need).
# ruff: noqa: ARG002

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dcam_camera import dcamapi

E = dcamapi.DCAMERR
PA = dcamapi.PROPATTR
R, W, RD, BZ = PA.READABLE, PA.WRITABLE, PA.ACCESSREADY, PA.ACCESSBUSY
TYPE = {"mode": 1, "long": 2, "real": 3}
UNIT = {"s": 1, "C": 2, "um": 7}


@dataclass
class P:
    id: int
    name: str
    type: str
    value: float
    min: float
    max: float
    step: float = 0.0
    texts: dict[int, str] = field(default_factory=dict)
    writable: bool = True
    ready: bool = False  # ACCESSREADY: writable with a buffer allocated
    busy: bool = False  # ACCESSBUSY: writable while capturing
    unit: str = ""
    elements: int = 0  # >0: array base with this many elements; NUMBER-OF id given by count_id
    count_id: int = 0
    step_element: int = 0


def _modes(*names: str, start: int = 1) -> dict[int, str]:
    return {start + i: n for i, n in enumerate(names)}


def orca_properties() -> list[P]:
    """The C13440-20CU property set in AREA mode (values measured on the camera, trimmed)."""
    on_off = {1: "OFF", 2: "ON"}
    pol = {1: "NEGATIVE", 2: "POSITIVE"}
    props = [
        P(
            0x400210,
            "SENSOR MODE",
            "mode",
            1,
            1,
            16,
            texts={1: "AREA", 12: "PROGRESSIVE", 14: "SPLIT VIEW", 16: "DUAL LIGHT SHEET"},
        ),
        P(0x400110, "READOUT SPEED", "long", 2, 1, 2, 1),
        P(0x400130, "READOUT DIRECTION", "mode", 5, 5, 5, texts={5: "DIVERGE"}, writable=False),
        P(0x420120, "COLORTYPE", "mode", 1, 1, 1, texts={1: "B/W"}, writable=False),
        P(0x420130, "BIT PER CHANNEL", "long", 16, 12, 16, 4),
        P(
            0x100110,
            "TRIGGER SOURCE",
            "mode",
            1,
            1,
            4,
            texts={1: "INTERNAL", 2: "EXTERNAL", 3: "SOFTWARE", 4: "MASTER PULSE"},
            ready=True,
            busy=True,
        ),
        P(0x100210, "TRIGGER MODE", "mode", 1, 1, 6, texts={1: "NORMAL", 6: "START"}, ready=True),
        P(0x100120, "TRIGGER ACTIVE", "mode", 1, 1, 3, texts={1: "EDGE", 2: "LEVEL", 3: "SYNCREADOUT"}, ready=True),
        P(0x1F0300, "TRIGGER GLOBAL EXPOSURE", "mode", 3, 3, 5, texts={3: "DELAYED", 5: "GLOBAL RESET"}),
        P(0x100220, "TRIGGER POLARITY", "mode", 1, 1, 2, texts=pol, ready=True, busy=True),
        P(0x100230, "TRIGGER CONNECTOR", "mode", 2, 2, 2, texts={2: "BNC"}, writable=False),
        P(0x100240, "TRIGGER TIMES", "long", 1, 1, 10000, 1, ready=True, busy=True),
        P(0x100260, "TRIGGER DELAY", "real", 0, 0, 10, 1e-6, ready=True, busy=True, unit="s"),
        P(0x100270, "INTERNAL TRIGGER HANDLING", "mode", 1, 1, 1, texts={1: "SHORTER EXPOSURE TIME"}, writable=False),
        P(0x200310, "SENSOR TEMPERATURE", "real", -10, -50, 100, 1, writable=False, unit="C"),
        P(0x200340, "SENSOR COOLER STATUS", "mode", 2, 1, 3, texts={1: "OFF", 2: "READY", 3: "BUSY"}, writable=False),
        P(0x1F0110, "EXPOSURE TIME", "real", 0.01, 0.001003669, 10.000004211, 1e-8, ready=True, busy=True, unit="s"),
        P(0x470010, "DEFECT CORRECT MODE", "mode", 2, 1, 2, texts=on_off, ready=True, busy=True),
        P(
            0x470030,
            "HOT PIXEL CORRECT LEVEL",
            "mode",
            1,
            1,
            3,
            texts={1: "STANDARD", 2: "MINIMUM", 3: "AGGRESSIVE"},
            ready=True,
            busy=True,
        ),
        P(0x401110, "BINNING", "mode", 1, 1, 4, texts={1: "1x1", 2: "2x2", 4: "4x4"}),
        P(0x402110, "SUBARRAY HPOS", "long", 0, 0, 2044, 4),
        P(0x402120, "SUBARRAY HSIZE", "long", 2048, 4, 2048, 4),
        P(0x402130, "SUBARRAY VPOS", "long", 0, 0, 2044, 4),
        P(0x402140, "SUBARRAY VSIZE", "long", 2048, 4, 2048, 4),
        P(0x402150, "SUBARRAY MODE", "mode", 1, 1, 2, texts=on_off),
        P(0x380510, "INTENSITY LUT MODE", "mode", 1, 1, 2, texts={1: "THROUGH", 2: "PAGE"}, ready=True),
        P(0x380520, "INTENSITY LUT PAGE", "mode", 1, 1, 1, writable=False),
        P(0x400620, "EXTRACTION MODE", "mode", 1, 1, 1, texts={1: "OFF"}, writable=False),
        P(0x403010, "TIMING READOUT TIME", "real", 0.0249942857, 0, 0, writable=False, unit="s"),
        P(0x403020, "TIMING CYCLIC TRIGGER PERIOD", "real", 0, 0, 0, writable=False, unit="s"),
        P(0x403030, "TIMING MIN TRIGGER BLANKING", "real", 0, 0, 0, writable=False, unit="s"),
        P(0x403050, "TIMING MIN TRIGGER INTERVAL", "real", 0.0249942857, 0, 0, writable=False, unit="s"),
        P(0x403090, "TIMING GLOBAL EXPOSURE DELAY", "real", 0.0149965714, 0, 0.0249943, writable=False, unit="s"),
        P(0x403060, "TIMING EXPOSURE", "mode", 3, 1, 3, texts={1: "AFTER READOUT", 3: "ROLLING"}, writable=False),
        P(0x403070, "TIMING INVALID EXPOSURE PERIOD", "real", 0.0149965714, 0, 0.0149966, writable=False, unit="s"),
        P(0x403810, "INTERNAL FRAME RATE", "real", 40.009144947, 0.1, 40.009144947, writable=False),
        P(0x403820, "INTERNAL FRAME INTERVAL", "real", 0.0249942857, 0.0249943, 10, writable=False, unit="s"),
        P(0x403840, "INTERNAL LINE SPEED", "real", 0.6670524691, 0.667052, 0.667052, writable=False),
        P(
            0x403850,
            "INTERNAL LINE INTERVAL",
            "real",
            9.74436090225564e-06,
            9.744e-06,
            9.744e-06,
            writable=False,
            unit="s",
        ),
        P(0x421010, "FRAMEBUNDLE MODE", "mode", 1, 1, 1, texts={1: "OFF"}, writable=False),
        P(0x421020, "FRAMEBUNDLE NUMBER", "long", 2, 2, 2, writable=False),
        P(0x421030, "FRAMEBUNDLE ROWBYTES", "long", 4096, 4096, 4096, 4096, writable=False),
        P(0x421040, "FRAMEBUNDLE FRAME STEP BYTES", "long", 8388608, 8388608, 8388608, writable=False),
        P(0x420210, "IMAGE WIDTH", "long", 2048, 4, 2048, 1, writable=False),
        P(0x420220, "IMAGE HEIGHT", "long", 2048, 4, 2048, 1, writable=False),
        P(0x420230, "IMAGE ROWBYTES", "long", 4096, 2, 4096, 2, writable=False),
        P(0x420240, "IMAGE FRAMEBYTES", "long", 8388608, 2, 8388608, 2, writable=False),
        P(0x420250, "IMAGE TOP OFFSET BYTES", "long", 0, 0, 0, writable=False),
        P(0x420270, "IMAGE PIXEL TYPE", "mode", 2, 1, 3, texts={1: "MONO8", 2: "MONO16", 3: "MONO12"}),
        P(0x420300, "IMAGE CAMERA STAMP", "long", 0, 0, 65535, 1),
        P(0x420330, "BUFFER ROWBYTES", "long", 4096, 2, 4096, 1, writable=False),
        P(0x420340, "BUFFER FRAMEBYTES", "long", 8388608, 2, 8388608, 1, writable=False),
        P(0x420350, "BUFFER TOP OFFSET BYTES", "long", 0, 0, 0, writable=False),
        P(0x420360, "BUFFER PIXEL TYPE", "mode", 2, 1, 3, texts={1: "MONO8", 2: "MONO16", 3: "MONO12"}, writable=False),
        P(0x420410, "RECORD FIXED BYTES PER FILE", "long", 256, 256, 256, 1, writable=False),
        P(0x420420, "RECORD FIXED BYTES PER SESSION", "long", 800, 0, 1104, 1, writable=False),
        P(0x420430, "RECORD FIXED BYTES PER FRAME", "long", 8388656, 0, 8388688, 1, writable=False),
        P(0x1C0010, "NUMBER OF OUTPUT TRIGGER CONNECTOR", "long", 3, 3, 3, writable=False),
        P(0x1C0190, "OUTPUT TRIGGER PRE HSYNC COUNT", "long", 0, 0, 2047, 1, ready=True, busy=True),
        P(
            0x1E0020,
            "MASTER PULSE MODE",
            "mode",
            1,
            1,
            3,
            texts={1: "CONTINUOUS", 2: "START", 3: "BURST"},
            ready=True,
            busy=True,
        ),
        P(
            0x1E0030,
            "MASTER PULSE TRIGGER SOURCE",
            "mode",
            1,
            1,
            2,
            texts={1: "EXTERNAL", 2: "SOFTWARE"},
            ready=True,
            busy=True,
        ),
        P(0x1E0040, "MASTER PULSE INTERVAL", "real", 0.1, 1e-5, 10, 1e-6, ready=True, busy=True, unit="s"),
        P(0x1E0050, "MASTER PULSE BURST TIMES", "long", 1, 1, 10000, 1, ready=True, busy=True),
        P(0xFF0010, "SYSTEM ALIVE", "mode", 2, 1, 2, texts={1: "OFFLINE", 2: "ONLINE"}, writable=False),
        P(0xFFE010, "CONVERSION FACTOR COEFF", "real", 0.45, 0, 255.99, 0.01, writable=False),
        P(0xFFE020, "CONVERSION FACTOR OFFSET", "real", 100, -65535, 65535, 1, writable=False),
        P(0x4201C0, "NUMBER OF VIEW", "long", 1, 1, 2, 1, writable=False),
        P(0x420810, "IMAGE DETECTOR PIXEL WIDTH", "real", 6.5, 6.5, 6.5, writable=False, unit="um"),
        P(0x420820, "IMAGE DETECTOR PIXEL HEIGHT", "real", 6.5, 6.5, 6.5, writable=False, unit="um"),
        P(0x420830, "IMAGE DETECTOR PIXEL NUM HORZ", "long", 2048, 2048, 2048, writable=False),
        P(0x420840, "IMAGE DETECTOR PIXEL NUM VERT", "long", 2048, 2048, 2048, writable=False),
        P(0x410A10, "TIME STAMP PRODUCER", "mode", 5, 5, 5, texts={5: "IMAGING DEVICE"}, writable=False),
        P(0x410A20, "FRAME STAMP PRODUCER", "mode", 5, 5, 5, texts={5: "IMAGING DEVICE"}, writable=False),
    ]
    # output trigger arrays: 3 elements, element step 0x100, count in NUMBER OF OUTPUT TRIGGER CONNECTOR
    arrays = [
        (0x1C0110, "SOURCE", "mode", 2, 2, 6, 0, {2: "READOUT END", 3: "VSYNC", 6: "TRIGGER"}, True, ""),
        (0x1C0120, "POLARITY", "mode", 1, 1, 2, 0, pol, True, ""),
        (0x1C0130, "ACTIVE", "mode", 1, 1, 1, 0, {1: "EDGE"}, False, ""),
        (0x1C0140, "DELAY", "real", 0, 0, 10, 1e-6, {}, True, "s"),
        (0x1C0150, "PERIOD", "real", 1e-6, 1e-6, 10, 1e-6, {}, True, "s"),
        (
            0x1C0160,
            "KIND",
            "mode",
            1,
            1,
            5,
            0,
            {1: "LOW", 2: "EXPOSURE", 3: "PROGRAMABLE", 4: "TRIGGER READY", 5: "HIGH"},
            True,
            "",
        ),
        (
            0x1C0170,
            "BASE SENSOR",
            "mode",
            16,
            1,
            16,
            0,
            {1: "VIEW 1", 2: "VIEW 2", 15: "ANY VIEW", 16: "ALL VIEWS"},
            True,
            "",
        ),
    ]
    for base, what, t, v, lo, hi, st, texts, wr, unit in arrays:
        for k in range(3):
            props.append(
                P(
                    base + k * 0x100,
                    f"OUTPUT TRIGGER {what}[{k}]",
                    t,
                    v,
                    lo,
                    hi,
                    st,
                    dict(texts),
                    wr,
                    ready=True,
                    busy=True,
                    unit=unit,
                    elements=3 if k == 0 else 0,
                    count_id=0x1C0010,
                    step_element=0x100,
                )
            )
    return props


ARRAY_ELEMENT_IDS = {p.id for p in orca_properties() if "[" in p.name and not p.name.endswith("[0]")}
PIXEL_BYTES = {1: 1.0, 2: 2.0, 3: 1.5}  # IMAGE PIXEL TYPE -> bytes per pixel


def pack_mono12(img: np.ndarray) -> np.ndarray:
    """uint16 (h, w) 12-bit -> MONO12 bytes (h, w*3/2), DCAM layout."""
    p0, p1 = img[:, 0::2].astype(np.uint16), img[:, 1::2].astype(np.uint16)
    return (
        np.stack([p0 >> 4, (p0 & 0xF) | ((p1 & 0xF) << 4), p1 >> 4], axis=-1).astype(np.uint8).reshape(img.shape[0], -1)
    )


def pattern(stamp: int, h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return ((stamp * 7 + y + x) % 4096).astype(np.uint16)


def _obj(arg: Any) -> Any:
    return arg._obj if hasattr(arg, "_obj") else arg


def _handle(h: Any) -> int:
    v = h.value if isinstance(h, ctypes.c_void_p) else h
    return int(v or 0)


class FakeDcamLib:
    """See the module docstring. ``serials`` sets how many cameras the fake reports."""

    def __init__(self, serials: tuple[str, ...] = ("100001",)):
        self.serials = serials
        self.calls: list[str] = []
        self.initialised = False
        self.open_index: int | None = None
        self.handle = 0x1000
        self.props = {p.id: p for p in orca_properties()}
        self.by_name = {p.name: p for p in self.props.values()}
        # capture state
        self.slots = 0
        self.capturing = False
        self.busy_polls_after_stop = 0  # dcamcap_status reports BUSY this many times after a stop
        self._busy_left = 0
        self.count = 0  # DCAM frames captured since start
        self.stamp = 0  # camera frame counter (next framestamp)
        self.slot_stamp: list[int] = []
        self.frames_per_wait = 1
        self.wait_error: int | None = None  # force the next waits to fail with this code
        self.on_copy: Any = None  # called inside copyframe/lockframe (e.g. to overwrite the slot)
        self.aborted = False
        self.fired = 0
        self.staging = np.zeros(0, np.uint8)
        self.attached: list[int] | None = None  # pointers from dcambuf_attach
        self.lockframe_attach_error: int | None = None  # e.g. E.NOTSUPPORT: lockframe refused while attached
        self.lockframe_calls = 0
        self.meta: dict[int, tuple[int, int]] = {}  # attach kind -> (address, count) of an attached metadata array
        self.meta_attach_error: int | None = None  # e.g. E.NOTSUPPORT: metadata attach refused

    # ---- helpers ------------------------------------------------------------------------
    def p(self, name: str) -> P:
        return self.by_name[name]

    def val(self, name: str) -> float:
        return self.p(name).value

    def text(self, name: str) -> str:
        p = self.p(name)
        return p.texts.get(int(p.value), str(int(p.value)))

    def geometry(self) -> tuple[int, int, int]:
        """(height, width, rowbytes) of the current frame."""
        b = int(self.val("BINNING"))
        if self.text("SUBARRAY MODE") == "ON":
            w, h = int(self.val("SUBARRAY HSIZE")), int(self.val("SUBARRAY VSIZE"))
        else:
            w, h = 2048, 2048 if self.text("SENSOR MODE") in ("AREA", "PROGRESSIVE") else 1024
        w, h = w // b, h // b
        return h, w, int(w * PIXEL_BYTES[int(self.val("IMAGE PIXEL TYPE"))])

    def _derive(self) -> None:
        h, w, rb = self.geometry()
        px = int(self.val("IMAGE PIXEL TYPE"))
        for n, v in (
            ("IMAGE WIDTH", w),
            ("IMAGE HEIGHT", h),
            ("IMAGE ROWBYTES", rb),
            ("BUFFER ROWBYTES", rb),
            ("IMAGE FRAMEBYTES", rb * h),
            ("BUFFER FRAMEBYTES", rb * h),
            ("BUFFER PIXEL TYPE", px),
        ):
            self.p(n).value = v
        self.p("BIT PER CHANNEL").value = {1: 8, 2: 16, 3: 12}[px]
        mode = self.text("SENSOR MODE")
        self.p("NUMBER OF VIEW").value = 2 if mode in ("SPLIT VIEW", "DUAL LIGHT SHEET") else 1
        lightsheet = mode in ("PROGRESSIVE", "DUAL LIGHT SHEET")
        self.p("BINNING").texts = {1: "1x1"} if lightsheet else {1: "1x1", 2: "2x2", 4: "4x4"}
        if lightsheet and self.val("BINNING") != 1:  # the camera forces 1x1 in light-sheet readout
            self.p("BINNING").value = 1
            return self._derive()
        for n in ("INTERNAL FRAME RATE", "INTERNAL FRAME INTERVAL", "INTERNAL LINE INTERVAL", "INTERNAL LINE SPEED"):
            self.p(n).writable = lightsheet
        self.p("READOUT DIRECTION").writable = lightsheet
        li = self.p("INTERNAL LINE INTERVAL")  # measured: fixed in AREA, 9.744 µs .. 100 ms in light sheet
        li.max, li.step = (0.1, 3.759398496240602e-09) if lightsheet else (li.min, 0.0)
        rd = self.p("READOUT DIRECTION")
        rd.texts = {1: "FORWARD", 2: "BACKWARD"} if lightsheet else {5: "DIVERGE"}
        rd.min, rd.max = min(rd.texts), max(rd.texts)
        if int(rd.value) not in rd.texts:
            rd.value = rd.min
        if not lightsheet:
            readout = 0.025 * h * int(self.val("BINNING")) / 2048
            fr = self.p("INTERNAL FRAME RATE")
            fr.value = fr.max = 1 / max(self.val("EXPOSURE TIME"), readout)

    # ---- device -------------------------------------------------------------------------
    def dcamapi_init(self, param):
        self.calls.append("init")
        _obj(param).iDeviceCount = len(self.serials)
        self.initialised = True
        return 1

    def dcamapi_uninit(self):
        return 1

    def dcamdev_open(self, param):
        o = _obj(param)
        if not 0 <= o.index < len(self.serials):
            return E.INVALIDCAMERA
        self.open_index = o.index
        o.hdcam = self.handle
        self._derive()
        return 1

    def dcamdev_close(self, h):
        self.calls.append("close")
        self.open_index = None
        return 1

    def dcamdev_getstring(self, h, param):
        o = _obj(param)
        hv = _handle(h)
        index = self.open_index if hv == self.handle else hv
        assert index is not None
        strings = {
            dcamapi.IDSTR.BUS: "USB3",
            dcamapi.IDSTR.CAMERAID: f"S/N: {self.serials[index]}",
            dcamapi.IDSTR.VENDOR: "Hamamatsu",
            dcamapi.IDSTR.MODEL: "C13440-20CU",
            dcamapi.IDSTR.CAMERAVERSION: "4.31.B",
            dcamapi.IDSTR.DRIVERVERSION: "1.2.6.7243",
            dcamapi.IDSTR.MODULEVERSION: "26.11.4322.7243",
            dcamapi.IDSTR.DCAMAPIVERSION: "4.00",
            dcamapi.IDSTR.CAMERA_SERIESNAME: "ORCA-Flash4.0 V3",
        }
        s = strings.get(o.iString)
        if s is None:
            return E.UNKNOWNSTRID
        addr = ctypes.c_void_p.from_address(ctypes.addressof(o) + type(o).text.offset).value
        assert addr is not None
        ctypes.memmove(addr, s.encode() + b"\0", len(s) + 1)
        return 1

    # ---- properties ---------------------------------------------------------------------
    def _get(self, pid: int) -> P | None:
        return self.props.get(pid)

    def dcamprop_getattr(self, h, param):
        o = _obj(param)
        p = self._get(o.iProp)
        if p is None:
            return E.INVALIDPROPERTYID
        attr = TYPE[p.type] | R | dcamapi.PROPATTR.HASRANGE
        attr |= W if p.writable else 0
        attr |= RD if p.ready else 0
        attr |= BZ if p.busy else 0
        attr |= dcamapi.PROPATTR.HASVALUETEXT if p.texts else 0
        o.attribute = ctypes.c_int32(int(attr)).value
        o.iUnit = UNIT.get(p.unit, 0)
        o.valuemin, o.valuemax, o.valuestep, o.valuedefault = p.min, p.max, p.step, p.value
        if p.elements:
            o.iProp_NumberOfElement, o.iProp_ArrayBase, o.iPropStep_Element = p.count_id, p.id, p.step_element
        return 1

    def dcamprop_getvalue(self, h, pid, pvalue):
        p = self._get(pid)
        if p is None:
            return E.INVALIDPROPERTYID
        _obj(pvalue).value = p.value
        return 1

    def dcamprop_setgetvalue(self, h, pid, pvalue, option):
        p = self._get(pid)
        v = _obj(pvalue)
        if p is None:
            return E.INVALIDPROPERTYID
        if not p.writable:
            return E.NOTWRITABLE
        value = v.value
        if self.capturing and not p.busy:
            return E.BUSY
        if self.slots and not p.ready and value != p.value:
            return E.NOTSTABLE
        if p.type == "mode" and p.texts and int(value) not in p.texts:
            return E.INVALIDVALUE
        if not p.min <= value <= p.max:
            return E.OUTOFRANGE
        if p.type == "long" and p.step and (value - p.min) % p.step:
            return E.INVALIDPARAM
        if p.type == "real" and p.step:
            value = round(value / p.step) * p.step
        if p.name.startswith("SUBARRAY") and p.name != "SUBARRAY MODE" and self.text("SUBARRAY MODE") == "ON":
            hpos, hsize = self.val("SUBARRAY HPOS"), self.val("SUBARRAY HSIZE")
            vpos, vsize = self.val("SUBARRAY VPOS"), self.val("SUBARRAY VSIZE")
            new = {
                "SUBARRAY HPOS": "hpos",
                "SUBARRAY HSIZE": "hsize",
                "SUBARRAY VPOS": "vpos",
                "SUBARRAY VSIZE": "vsize",
            }[p.name]
            vals = {"hpos": hpos, "hsize": hsize, "vpos": vpos, "vsize": vsize, new: value}
            if vals["hpos"] + vals["hsize"] > 2048 or vals["vpos"] + vals["vsize"] > 2048:
                return E.INVALIDSUBARRAY
        p.value = value
        if p.name == "BIT PER CHANNEL":
            self.p("IMAGE PIXEL TYPE").value = {8: 1, 12: 3, 16: 2}[int(value)]
        self._derive()
        v.value = p.value
        return 1

    def dcamprop_queryvalue(self, h, pid, pvalue, option):
        p = self._get(pid)
        v = _obj(pvalue)
        if p is None:
            return E.INVALIDPROPERTYID
        if option == dcamapi.PROPOPTION.NEXT:
            if p.texts:
                nxt = [k for k in sorted(p.texts) if k > v.value]
                if not nxt:
                    return E.OUTOFRANGE
                v.value = nxt[0]
                return 1
            if not p.step or v.value + p.step > p.max:
                return E.OUTOFRANGE
            v.value += p.step
        return 1

    def dcamprop_getnextid(self, h, pprop, option):
        ids = sorted(i for i in self.props if i not in ARRAY_ELEMENT_IDS)
        o = _obj(pprop)
        nxt = [i for i in ids if i > o.value]
        if not nxt:
            return E.NOPROPERTY
        o.value = nxt[0]
        return 1

    def dcamprop_getname(self, h, pid, buf, size):
        p = self._get(pid)
        if p is None:
            return E.INVALIDPROPERTYID
        buf.value = p.name.encode()[: size - 1]
        return 1

    def dcamprop_getvaluetext(self, h, param):
        o = _obj(param)
        p = self._get(o.iProp)
        if p is None:
            return E.INVALIDPROPERTYID
        t = p.texts.get(int(o.value))
        if t is None:
            return E.NOVALUETEXT
        addr = ctypes.c_void_p.from_address(ctypes.addressof(o) + type(o).text.offset).value
        assert addr is not None
        ctypes.memmove(addr, t.encode() + b"\0", len(t) + 1)
        return 1

    # ---- buffers ------------------------------------------------------------------------
    def dcambuf_alloc(self, h, frames):
        if self.slots:
            return E.INVALIDPARAM  # already allocated
        self.calls.append(f"alloc {frames}")
        self.slots = int(frames)
        self.slot_stamp = [-1] * self.slots
        return 1

    def dcambuf_attach(self, h, param):
        o = _obj(param)
        if o.iKind in (dcamapi.ATTACHKIND_PRIMARY_TIMESTAMP, dcamapi.ATTACHKIND_PRIMARY_FRAMESTAMP):
            if self.meta_attach_error is not None:
                return self.meta_attach_error
            if self.capturing or o.iKind in self.meta or o.buffercount < 1 or not o.buffer[0]:
                return E.INVALIDPARAM
            self.meta[int(o.iKind)] = (int(o.buffer[0]), int(o.buffercount))
            self.calls.append(f"attach meta {o.iKind} {o.buffercount}")
            return 1
        if self.slots:
            return E.INVALIDPARAM  # already allocated or attached
        if o.iKind != dcamapi.ATTACHKIND_FRAME or o.buffercount < 1:
            return E.INVALIDPARAM
        self.attached = [int(o.buffer[i]) for i in range(o.buffercount)]
        if not all(self.attached):
            self.attached = None
            return E.INVALIDPARAM
        self.calls.append(f"attach {o.buffercount}")
        self.slots = int(o.buffercount)
        self.slot_stamp = [-1] * self.slots
        return 1

    def dcambuf_release(self, h, kind):
        if self.capturing:
            return E.BUSY
        if kind in (dcamapi.ATTACHKIND_PRIMARY_TIMESTAMP, dcamapi.ATTACHKIND_PRIMARY_FRAMESTAMP):
            if kind not in self.meta:
                return E.INVALIDPARAM
            del self.meta[kind]
            self.calls.append(f"release meta {kind}")
            return 1
        self.calls.append("release")
        self.slots = 0
        self.attached = None
        return 1

    def _frame_bytes(self, stamp: int) -> np.ndarray:
        h, w, _ = self.geometry()
        img = pattern(stamp, h, w)
        px = int(self.val("IMAGE PIXEL TYPE"))
        if px == 3:
            return pack_mono12(img)
        if px == 1:
            return (img & 0xFF).astype(np.uint8)
        return img

    def _locate(self, index: int) -> int | None:
        """Buffer slot of frame ``index`` (-1: the newest), None if it holds no frame."""
        if index < 0:
            index = (self.count - 1) % self.slots if self.slots and self.count else -1
        if not 0 <= index < self.slots or self.slot_stamp[index] < 0:
            return None
        return index

    def dcambuf_copyframe(self, h, pframe):
        f = _obj(pframe)
        if self._busy_left:
            return E.BUSY
        slot = self._locate(f.iFrame)
        if slot is None:
            return E.INVALIDFRAMEINDEX
        if self.on_copy:
            self.on_copy()
        hh, ww, _ = self.geometry()
        if (f.height, f.width) != (hh, ww) or int(self.val("IMAGE PIXEL TYPE")) == 3:
            return E.INVALIDPARAM
        data = self._frame_bytes(self.slot_stamp[slot])
        row = data.view(np.uint8).reshape(hh, -1)
        for y in range(hh):
            ctypes.memmove(f.buf + y * f.rowbytes, row[y].ctypes.data, row.shape[1])
        f.framestamp = self.slot_stamp[slot]
        f.timestamp.sec, f.timestamp.microsec = 1000 + self.slot_stamp[slot], 0
        return 1

    def dcambuf_lockframe(self, h, pframe):
        f = _obj(pframe)
        if self._busy_left:
            return E.BUSY
        slot = self._locate(f.iFrame)
        if slot is None:
            return E.INVALIDFRAMEINDEX
        if self.on_copy:
            self.on_copy()
        hh, ww, rb = self.geometry()
        self.lockframe_calls += 1
        if self.attached is not None:
            if self.lockframe_attach_error is not None:
                return self.lockframe_attach_error
            f.buf, f.rowbytes, f.width, f.height = self.attached[slot], rb, ww, hh
            f.type = int(self.val("IMAGE PIXEL TYPE"))
            f.framestamp = self.slot_stamp[slot]
            f.timestamp.sec, f.timestamp.microsec = 1000 + self.slot_stamp[slot], 0
            return 1
        self.staging = np.ascontiguousarray(self._frame_bytes(self.slot_stamp[slot]).view(np.uint8).reshape(hh, rb))
        f.buf, f.rowbytes, f.width, f.height = self.staging.ctypes.data, rb, ww, hh
        f.type = int(self.val("IMAGE PIXEL TYPE"))
        f.framestamp = self.slot_stamp[slot]
        f.timestamp.sec, f.timestamp.microsec = 1000 + self.slot_stamp[slot], 0
        return 1

    # ---- capture ------------------------------------------------------------------------
    def produce(self, n: int = 1, skip: int = 0) -> None:
        """The camera produces ``n`` frames DCAM receives, after losing ``skip`` before DCAM."""
        self.stamp += skip
        for _ in range(n):
            slot = self.count % self.slots
            self.slot_stamp[slot] = self.stamp
            if self.attached is not None:  # the camera writes into the caller's memory
                data = np.ascontiguousarray(self._frame_bytes(self.stamp)).view(np.uint8)
                ctypes.memmove(self.attached[slot], data.ctypes.data, data.nbytes)
            if dcamapi.ATTACHKIND_PRIMARY_TIMESTAMP in self.meta:
                addr, n = self.meta[dcamapi.ATTACHKIND_PRIMARY_TIMESTAMP]
                ts = dcamapi.DCAM_TIMESTAMP.from_address(addr + (self.count % n) * ctypes.sizeof(dcamapi.DCAM_TIMESTAMP))
                ts.sec, ts.microsec = 1000 + self.stamp, 0
            if dcamapi.ATTACHKIND_PRIMARY_FRAMESTAMP in self.meta:
                addr, n = self.meta[dcamapi.ATTACHKIND_PRIMARY_FRAMESTAMP]
                ctypes.c_int32.from_address(addr + (self.count % n) * 4).value = self.stamp
            self.count += 1
            self.stamp += 1

    def dcamcap_start(self, h, mode):
        if not self.slots:
            return E.NOTREADY
        if self.capturing:
            return E.BUSY
        self.calls.append("start")
        self.capturing = True
        self.count = self.stamp = 0
        self.slot_stamp = [-1] * self.slots
        self.aborted = False
        return 1

    def dcamcap_stop(self, h):
        self.calls.append("stop")
        self.capturing = False
        self._busy_left = self.busy_polls_after_stop
        return 1

    def dcamcap_status(self, h, pstatus):
        o = _obj(pstatus)
        if self._busy_left:
            self._busy_left -= 1
            o.value = dcamapi.CAPSTATUS.BUSY
        elif self.capturing:
            o.value = dcamapi.CAPSTATUS.BUSY
        else:
            o.value = dcamapi.CAPSTATUS.READY if self.slots else dcamapi.CAPSTATUS.UNSTABLE
        return 1

    def dcamcap_transferinfo(self, h, param):
        o = _obj(param)
        o.nFrameCount = self.count
        o.nNewestFrameIndex = (self.count - 1) % self.slots if self.count and self.slots else -1
        return 1

    def dcamcap_firetrigger(self, h, kind):
        if not self.capturing:
            return E.NOTBUSY
        self.fired += 1
        if self.text("TRIGGER SOURCE") == "SOFTWARE":
            self.produce(1)
        return 1

    def dcamwait_open(self, param):
        _obj(param).hwait = 0x2000
        _obj(param).supportevent = 0x1F
        return 1

    def dcamwait_close(self, hwait):
        return 1

    def dcamwait_start(self, hwait, param):
        o = _obj(param)
        if self.wait_error is not None:
            code, self.wait_error = self.wait_error, None
            return code
        if self.aborted:
            self.aborted = False
            return E.ABORT
        if self.capturing and self.text("TRIGGER SOURCE") == "INTERNAL" and self.frames_per_wait:
            self.produce(self.frames_per_wait)
            o.eventhappened = dcamapi.WAITEVENT.FRAMEREADY
            return 1
        return E.TIMEOUT

    def dcamwait_abort(self, hwait):
        self.aborted = True
        return 1

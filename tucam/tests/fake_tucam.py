"""A fake TUCam SDK V3 binding (the ``api`` object and ``TucDefine`` module) for tests without a camera.

It implements the calls ``tucam_camera.tucam`` makes: enumeration, open/close, GenICam nodes, required buffer size,
queue policy, buffer announce / queue / revoke / info, the buffer callback and acquisition start/stop. The fake
camera is driven from the test thread: ``FakeApi.produce(n)`` makes the camera emit n frames. Each frame takes the
oldest queued buffer (FIFO), its pixels are written into the announced memory with ``ctypes.memmove`` (followed by
``chunk_bytes`` of chunk trailer when chunk mode is on), and the registered callback is called synchronously. A
frame emitted while no buffer is queued is lost; the camera's frame counter advances either way.

Behaviour modelled on observations with an earlier internal driver: acquisition start puts every announced buffer in the input pool,
in announce order; with chunk mode on, ``frameId`` carries the camera's frame counter and ``timestamp`` the camera
clock in us; with it off both stay 0.
"""

import ctypes
from types import SimpleNamespace

import numpy as np

OK, ALREADY_INITIALIZED, NOT_READABLE, NOT_FOUND, ERROR = 0, 1, 2, 3, 4
INT, BOOL, CMD, FLOAT, STR, ENUM = 2, 3, 4, 5, 6, 9
RO, RW = 3, 4
PFNC = {"Mono8": 0x01080001, "Mono12": 0x01100005, "Mono16": 0x01100007, "Mono12Packed": 0x010C0047}


def pack_mono12packed(img):
    """uint16 12-bit pixels -> GigE Mono12Packed bytes (b0 = p0 >> 4, b1 = p0 & 0xF | (p1 & 0xF) << 4, b2 = p1 >> 4)"""
    p = img.reshape(-1, 2).astype(np.uint16)
    out = np.empty((p.shape[0], 3), np.uint8)
    out[:, 0] = p[:, 0] >> 4
    out[:, 1] = (p[:, 0] & 0xF) | ((p[:, 1] & 0xF) << 4)
    out[:, 2] = p[:, 1] >> 4
    return out.reshape(-1)


def _ns(**kw):
    return type("NS", (), kw)


D = SimpleNamespace(
    TUC_STATUS=_ns(
        TUC_STATUS_OK=OK,
        TUC_STATUS_ALREADY_INITIALIZED=ALREADY_INITIALIZED,
        TUC_STATUS_FEATURE_NOT_READABLE=NOT_READABLE,
        TUC_STATUS_NOT_FOUND=NOT_FOUND,
        TUC_STATUS_ERROR=ERROR,
    ),
    TUC_BUFFER_SIZE_KIND=_ns(TUC_BUFSIZE_ANNOUNCE=0, TUC_BUFSIZE_DATA_UNPACKED=1),
    TUC_BUFFER_QUEUE_POLICY=_ns(TUC_BUFFER_QUEUE_POLICY_AUTO=0, TUC_BUFFER_QUEUE_POLICY_MANUAL=1),
    TUC_DEVICE_ID_TYPE=_ns(TUC_DEVICE_ID_SERIAL=1),
    TUC_OPEN_POLICY=_ns(TUC_OPEN_POLICY_EXCLUSIVE=1),
    TUC_ACQ_START_FLAGS=_ns(TUC_ACQ_START_FLAGS_DEFAULT=0),
    TUC_ACQ_STOP_FLAGS=_ns(TUC_ACQ_STOP_FLAGS_DEFAULT=0),
    TUC_LOG_LEVEL=_ns(TUC_LOG_LEVEL_WARNING=2),
    # identity: exceptions raised in the callback reach the test (a ctypes callback would swallow them)
    TUC_BUFFER_CALLBACK=lambda fn: fn,
)


class FakeApi:
    HANDLE = 7

    def __init__(self, height=32, width=48, pixel_format="Mono16", chunk_bytes=128, serial="FAKE0001",
                 mode_formats=None):
        self.serial = serial
        # like the Aries 6504, PixelFormat (read-only) can follow SensorOperationMode,
        # e.g. mode_formats={"Dynamic": "Mono16", "Speed": "Mono12Packed"}
        self.mode_formats = mode_formats or {}
        self.chunk_bytes = chunk_bytes
        self.features = {
            "Width": [INT, RW, width],
            "Height": [INT, RW, height],
            "PixelFormat": [ENUM, RO, pixel_format, list(PFNC)],
            "ChunkModeActive": [BOOL, RW, False],
            "TriggerMode": [ENUM, RW, "Standard", ["FreeRunning", "Standard", "Software"]],
            "ExposureTime": [INT, RW, 1000],
            "AcquisitionFrameRate": [FLOAT, RW, 100.0],
            "SensorOperationMode": [ENUM, RW, "Dynamic", ["Dynamic", "Speed"]],
            "Dynamic": [ENUM, RW, "HDR", ["HighGain", "LowGain", "HDR"]],
            "Speed": [ENUM, RW, "LowGain", ["HighGain", "MidGain", "LowGain"]],
            "OverallSystemGainK": [FLOAT, RO, 0.5],
            "AverageBlackLevel": [FLOAT, RO, 100.0],
        }
        self.fps_max = 297.0
        self.open = False
        self.policy = 0
        self.announced = {}  # handle -> (address, size)
        self.fifo = []
        self.info = {}
        self.callback = None
        self.running = False
        self.frame_count = 0
        self.counter = 0  # the camera's frame counter
        self.emitted = 0
        self.lost = 0
        self._next_handle = 1000
        self.incomplete_next = 0
        self.offset_next = None

    # ---- enumeration and device
    def TucInitialize(self, _level):
        self.initialized = True
        return OK

    def TucTerminate(self):  # tucam.terminate() calls it
        self.initialized = False
        self.terminated = getattr(self, "terminated", 0) + 1
        return OK

    def TucDeviceEnumerate(self):
        return OK, 1

    def TucGetDeviceInfo(self, i):
        return OK, SimpleNamespace(
            serialNumber=self.serial.encode(),
            modelName=b"FakeCam 2048",
            vendorName=b"Tucsen",
            userDefinedName=b"",
            transportLayerType=b"U3V",
        )

    def TucDeviceOpenEx(self, serial, _id_type, _policy):
        if serial != self.serial:
            return NOT_FOUND, None
        self.open = True
        return OK, self.HANDLE

    def TucDeviceClose(self, _h):
        self.open = False
        return OK

    # ---- nodes
    def _f(self, name):
        return self.features.get(name)

    def TucGetFeatureInfoByName(self, _h, name):
        f = self._f(name)
        if f is None:
            return NOT_FOUND, None
        return OK, SimpleNamespace(featureType=f[0], accessMode=f[1], unit=b"", description=b"")

    def _get(self, name):
        f = self._f(name)
        return (NOT_FOUND, None) if f is None else (OK, f[2])

    TucGetFeatureInt64 = TucGetFeatureFloat = TucGetFeatureBool = TucGetFeatureString = (
        lambda self, _h, name: self._get(name)
    )

    def TucGetFeatureEnumInfo(self, _h, name):
        f = self._f(name)
        if f is None:
            return NOT_FOUND, None
        entries = [SimpleNamespace(symbolic=e.encode(), value=i) for i, e in enumerate(f[3])]
        return OK, SimpleNamespace(currentSymbolic=f[2].encode(), entryCount=len(entries), pEntries=entries)

    def TucGetFeatureEnumEntryByName(self, _h, name, entry):
        f = self._f(name)
        if f is None or entry not in f[3]:
            return NOT_FOUND, None
        return OK, SimpleNamespace(value=f[3].index(entry))

    def TucSetFeatureEnumValue(self, _h, name, value):
        f = self._f(name)
        if f[1] != RW:
            return ERROR
        f[2] = f[3][value]
        if name == "SensorOperationMode" and f[2] in self.mode_formats:
            self.features["PixelFormat"][2] = self.mode_formats[f[2]]
        return OK

    def _set(self, name, value):
        f = self._f(name)
        if f is None or f[1] != RW:
            return ERROR
        f[2] = value
        return OK

    TucSetFeatureInt64 = TucSetFeatureFloat = TucSetFeatureBool = TucSetFeatureString = (
        lambda self, _h, name, v: self._set(name, v)
    )

    def TucGetFeatureFloatInfo(self, _h, name):
        if name == "AcquisitionFrameRate":
            return OK, SimpleNamespace(minimum=1.0, maximum=self.fps_max, increment=0.0)
        return OK, SimpleNamespace(minimum=0.0, maximum=1e9, increment=0.0)

    def TucGetFeatureInt64Info(self, _h, name):
        return OK, SimpleNamespace(minimum=0, maximum=1 << 30, increment=1)

    # ---- buffers
    @property
    def bpp(self):
        return 1 if self.features["PixelFormat"][2] == "Mono8" else 2

    @property
    def image_bytes(self):
        h, w = self.features["Height"][2], self.features["Width"][2]
        if self.features["PixelFormat"][2] == "Mono12Packed":
            return h * w * 3 // 2
        return h * w * self.bpp

    def TucBuffersGetRequiredSize(self, _h, kind):
        return OK, self.image_bytes + (self.chunk_bytes if self.features["ChunkModeActive"][2] else 0)

    def TucBufferSetQueuePolicy(self, _h, policy):
        if self.running:
            return ERROR
        self.policy = policy
        return OK

    def TucBufferAnnounce(self, _h, address, size):
        if self.running or size < self.TucBuffersGetRequiredSize(_h, 0)[1]:
            return ERROR, None
        h = self._next_handle
        self._next_handle += 1
        self.announced[h] = (ctypes.c_void_p(address.value if hasattr(address, "value") else address).value, size)
        return OK, h

    def TucBufferRevoke(self, _h, buffer):
        if self.running or buffer not in self.announced:
            return ERROR
        self.announced.pop(buffer)
        if buffer in self.fifo:
            self.fifo.remove(buffer)
        return OK

    def TucBufferQueue(self, _h, buffer):
        if buffer not in self.announced or buffer in self.fifo:
            return ERROR
        self.fifo.append(buffer)
        return OK

    def TucBufferGetInfo(self, _h, buffer):
        return OK, self.info[buffer]

    def TucRegisterBufferCallback(self, _h, cb, _ctx):
        self.callback = cb
        return OK

    def TucUnregisterBufferCallback(self, _h):
        self.callback = None
        return OK

    def TucAcquisitionStart(self, _h, _flags, count):
        if self.running:
            return ERROR
        self.running = True
        self.frame_count = int(count)
        self.emitted = 0
        self.fifo = list(self.announced)  # every announced buffer goes to the input pool, in announce order
        return OK

    def TucAcquisitionStop(self, _h, _flags):
        if not self.running:
            return ERROR
        self.running = False
        return OK

    # ---- the camera
    def pattern(self, counter):
        h, w = self.features["Height"][2], self.features["Width"][2]
        dt = np.uint8 if self.bpp == 1 else np.uint16
        return ((np.arange(h * w, dtype=np.uint32) + counter * 7) % (256 if self.bpp == 1 else 4096)).astype(dt)

    def skip(self, n):
        """frames the camera emits that never reach the host although buffers are queued (lost before the host)"""
        self.counter += n
        self.emitted += n

    def produce(self, n):
        """emit n frames; returns how many reached a buffer"""
        got = 0
        for _ in range(n):
            if not self.running:
                break
            counter = self.counter
            self.counter += 1
            self.emitted += 1
            if self.frame_count and self.emitted >= self.frame_count:
                self.running = False  # the SDK stops by itself after frame_count frames
            if not self.fifo:
                self.lost += 1
                continue
            h = self.fifo.pop(0)
            addr, size = self.announced[h]
            chunk = bool(self.features["ChunkModeActive"][2])
            off = self.offset_next if self.offset_next is not None else 0
            self.offset_next = None
            img = self.pattern(counter)
            if self.features["PixelFormat"][2] == "Mono12Packed":
                img = pack_mono12packed(img)
            ctypes.memmove(addr + off, img.ctypes.data, img.nbytes)
            if chunk:
                trailer = np.full(self.chunk_bytes, 0xCC, np.uint8)
                ctypes.memmove(addr + off + img.nbytes, trailer.ctypes.data, self.chunk_bytes)
            bad = self.incomplete_next > 0
            self.incomplete_next -= int(bad)
            self.info[h] = SimpleNamespace(
                frameId=counter if chunk else 0,
                timestamp=(10_000 + counter * 3367) if chunk else 0,
                imageHeight=self.features["Height"][2],
                imageWidth=self.features["Width"][2],
                pixelFormat=PFNC[self.features["PixelFormat"][2]],
                imageOffset=off,
                isIncomplete=bad,
                imagePresent=True,
            )
            got += 1
            if self.callback is not None:
                self.callback(self.HANDLE, h, None)
        return got

"""A fake PVCAM runtime with the Python-level interface of pvcam_camera.pvcam.Pvcam.

It "DMAs" frames into the caller's circular buffer with ctypes.memmove, in slot order from the buffer start, and
models the two readings of CIRC_NO_OVERWRITE in pvcam.h:

- ``overwrite_locked=False`` (default): a locked or unretrieved slot is never written; a frame that finds the next
  slot occupied is dropped (FrameNr still advances, so the consumer sees a gap).
- ``overwrite_locked=True``: the frame is written anyway (the enum text: "the oldest frame in circular buffer is also
  replaced ... via DMA"), and get_oldest_frame reports an overrun if an unretrieved frame was replaced.

``require_unlock=True`` models a runtime that hands out the next frame only after the previous one is unlocked.

Frames are produced from a budget (``allow(n)``), a few per ``check_cont_status`` call, so tests are deterministic.
Pixel value of frame k (FrameNr) = k modulo the pixel range; timestamps are ``k * period_ticks`` (0.1 ms counts).
"""

import ctypes
import threading
from collections import deque

import numpy as np
from pvcam_camera import pvcam as pv
from pvcam_camera.pvcam import FrameRecord, PvcamError


class FakeCamera:
    def __init__(self, name, serial, shape, product="Kinetix"):
        self.name, self.serial = name, serial
        par, ser = shape
        self.params = {
            pv.PARAM_HEAD_SER_NUM_ALPHA: serial,
            pv.PARAM_PRODUCT_NAME: product,
            pv.PARAM_CHIP_NAME: "TMP-" + product,
            pv.PARAM_SER_SIZE: ser,
            pv.PARAM_PAR_SIZE: par,
            pv.PARAM_READOUT_PORT: 0,
            pv.PARAM_SPDTAB_INDEX: 0,
            pv.PARAM_GAIN_INDEX: 1,
            pv.PARAM_BIT_DEPTH: 12,
            pv.PARAM_BIT_DEPTH_HOST: 12,
            pv.PARAM_IMAGE_FORMAT: pv.PL_IMAGE_FORMAT_MONO16,
            pv.PARAM_IMAGE_FORMAT_HOST: pv.PL_IMAGE_FORMAT_MONO16,
            pv.PARAM_METADATA_ENABLED: True,  # on, as after a PyVCAM session: the source must switch it off
            pv.PARAM_CIRC_BUFFER: True,
            pv.PARAM_EXP_RES: pv.EXP_RES_ONE_MILLISEC,
            pv.PARAM_EXP_RES_INDEX: 0,
            pv.PARAM_EXPOSURE_TIME: 1,
            pv.PARAM_EXPOSURE_MODE: pv.EXT_TRIG_INTERNAL,
            pv.PARAM_EXPOSE_OUT_MODE: pv.EXPOSE_OUT_FIRST_ROW,
            pv.PARAM_READOUT_TIME: 11300,
            pv.PARAM_FRAME_BUFFER_SIZE: 0,
            pv.PARAM_CAM_FW_VERSION: 0x1E33,
        }
        self.read_only = {pv.PARAM_HEAD_SER_NUM_ALPHA, pv.PARAM_SER_SIZE, pv.PARAM_PAR_SIZE, pv.PARAM_READOUT_TIME}
        self.enums = {
            pv.PARAM_READOUT_PORT: [(0, "Sensitivity"), (1, "Speed"), (2, "Dynamic Range"), (3, "Sub-Electron")],
            pv.PARAM_EXP_RES: [(pv.EXP_RES_ONE_MILLISEC, "One Millisecond"), (pv.EXP_RES_ONE_MICROSEC, "One µs")],
            pv.PARAM_EXPOSURE_MODE: [(pv.EXT_TRIG_INTERNAL, "Internal"), (pv.EXT_TRIG_EDGE_RISING, "Edge Trigger")],
        }
        self.setup = None  # (regions, exp_mode, exposure, buffer_mode)
        self.frame_bytes = 0


class FakePvcam:
    METADATA_BYTES = 80  # extra bytes per frame while metadata is on (any non-zero value shows the check works)

    def __init__(
        self,
        shape=(24, 32),
        serials=("A00F000001", "A00F000002"),
        frames_per_poll=2,
        overwrite_locked=False,
        require_unlock=False,
        period_ticks=113,
        readout_ticks=113,
        metadata_writable=True,
    ):
        self.cams = {i: FakeCamera(f"PMPCIECam{i:02d}", s, shape) for i, s in enumerate(serials)}
        self.frames_per_poll = frames_per_poll
        self.overwrite_locked, self.require_unlock = overwrite_locked, require_unlock
        self.period_ticks, self.readout_ticks = period_ticks, readout_ticks
        self.metadata_writable = metadata_writable
        self.refs = 0
        self.inits = self.uninits = 0
        self.opened = {}
        self._lock = threading.Lock()
        self.budget = 0  # frames the "camera" may still produce
        self.running = False
        self.start_address = None
        self.start_size = None
        self.produced = self.dropped = self.written = self.overwritten_locked = self.unlock_calls = 0

    # ------------------------------------------------------------ library
    def init(self):
        if self.refs == 0:
            self.inits += 1
        self.refs += 1

    def uninit(self):
        if self.refs:
            self.refs -= 1
            if self.refs == 0:
                self.uninits += 1

    def version(self):
        return "3.10.1"

    def camera_names(self):
        return [c.name for c in self.cams.values()]

    def open(self, name):
        for h, c in self.cams.items():
            if c.name == name:
                if h in self.opened:
                    raise PvcamError("pl_cam_open", 1, "camera already open")
                self.opened[h] = c
                return h
        raise PvcamError("pl_cam_open", 2, f"no camera {name}")

    def close(self, hcam):
        self.opened.pop(hcam)

    # ------------------------------------------------------------ parameters
    def available(self, hcam, pid):
        return pid in self.cams[hcam].params

    def writable(self, hcam, pid):
        c = self.cams[hcam]
        if pid == pv.PARAM_METADATA_ENABLED and not self.metadata_writable:
            return False
        return pid in c.params and pid not in c.read_only

    def get_param(self, hcam, pid, attr=pv.ATTR_CURRENT):
        c = self.cams[hcam]
        if attr == pv.ATTR_AVAIL:
            return pid in c.params
        if pid not in c.params:
            raise PvcamError("pl_get_param", 3, "parameter not available")
        if attr == pv.ATTR_ACCESS:
            return pv.ACC_READ_WRITE if self.writable(hcam, pid) else pv.ACC_READ_ONLY
        if attr == pv.ATTR_COUNT:
            return len(c.enums.get(pid, [])) or 1
        if attr == pv.ATTR_MIN and pid == pv.PARAM_EXPOSURE_TIME:
            return 1
        if attr == pv.ATTR_DEFAULT and pid == pv.PARAM_FRAME_BUFFER_SIZE:
            return 16 * c.frame_bytes
        if pid == pv.PARAM_EXP_RES:
            return c.enums[pv.PARAM_EXP_RES][c.params[pv.PARAM_EXP_RES_INDEX]][0]
        return c.params[pid]

    def set_param(self, hcam, pid, value):
        c = self.cams[hcam]
        if not self.writable(hcam, pid):
            raise PvcamError("pl_set_param", 4, "parameter is read-only")
        c.params[pid] = value
        if pid == pv.PARAM_READOUT_PORT:  # Kinetix port table (REPORT.md): Speed port is 8-bit
            bits = 8 if value == 1 else (12 if value == 0 else 16)
            fmt = pv.PL_IMAGE_FORMAT_MONO8 if bits == 8 else pv.PL_IMAGE_FORMAT_MONO16
            c.params.update(
                {
                    pv.PARAM_BIT_DEPTH: bits,
                    pv.PARAM_BIT_DEPTH_HOST: bits,
                    pv.PARAM_IMAGE_FORMAT: fmt,
                    pv.PARAM_IMAGE_FORMAT_HOST: fmt,
                }
            )

    def enum_values(self, hcam, pid):
        return list(self.cams[hcam].enums[pid])

    # ------------------------------------------------------------ continuous acquisition
    def setup_cont(self, hcam, regions, exp_mode, exposure, buffer_mode):
        c = self.cams[hcam]
        (r,) = regions
        w, h = (r.s2 - r.s1 + 1) // r.sbin, (r.p2 - r.p1 + 1) // r.pbin
        bpp = 1 if c.params[pv.PARAM_IMAGE_FORMAT_HOST] == pv.PL_IMAGE_FORMAT_MONO8 else 2
        c.frame_bytes = w * h * bpp + (self.METADATA_BYTES if c.params[pv.PARAM_METADATA_ENABLED] else 0)
        c.setup = (list(regions), exp_mode, exposure, buffer_mode)
        self.dtype = np.uint8 if bpp == 1 else np.uint16
        self.mask = (1 << c.params[pv.PARAM_BIT_DEPTH_HOST]) - 1
        return c.frame_bytes

    def start_cont(self, hcam, address, size):
        c = self.cams[hcam]
        if size % c.frame_bytes:
            raise PvcamError("pl_exp_start_cont", 5, "buffer size is not a multiple of the frame size")
        self.hcam, self.fb = hcam, c.frame_bytes
        self.start_address, self.start_size, self.nslots = address, size, size // c.frame_bytes
        self.frame_nr = 0
        self.write_slot = 0
        self.laps = 0
        self.unretrieved = deque()  # (slot, frame_nr)
        self.locked = deque()  # (slot, frame_nr), oldest first
        self.slot_owner = {}  # slot -> "unretrieved" | "locked"
        self.overrun = False
        self.running = True

    def allow(self, n):
        with self._lock:
            self.budget += int(n)

    def _produce_one(self):
        self.frame_nr += 1
        self.produced += 1
        slot = self.write_slot
        owner = self.slot_owner.get(slot)
        if owner is not None and not self.overwrite_locked:
            self.dropped += 1  # no free slot: the frame is lost, FrameNr advances
            return
        if owner == "unretrieved":
            self.unretrieved = deque(x for x in self.unretrieved if x[0] != slot)
            self.overrun = True
        elif owner == "locked":
            self.overwritten_locked += 1
        pix = np.full(self.fb // np.dtype(self.dtype).itemsize, self.frame_nr & self.mask, self.dtype)
        ctypes.memmove(self.start_address + slot * self.fb, pix.ctypes.data, self.fb)
        self.written += 1
        if owner != "locked":
            self.slot_owner[slot] = "unretrieved"
            self.unretrieved.append((slot, self.frame_nr))
        else:  # overwritten while locked: the new frame is unretrieved in a slot that is still locked
            self.unretrieved.append((slot, self.frame_nr))
        self.write_slot = (slot + 1) % self.nslots
        if self.write_slot == 0:
            self.laps += 1

    def check_cont_status(self, hcam):
        if not self.running:
            return pv.READOUT_NOT_ACTIVE, 0, self.laps
        with self._lock:
            n = min(self.frames_per_poll, self.budget)
            self.budget -= n
        for _ in range(n):
            self._produce_one()
        status = pv.FRAME_AVAILABLE if (self.unretrieved or self.overrun) else pv.EXPOSURE_IN_PROGRESS
        if self.require_unlock and self.locked:
            status = pv.FRAME_AVAILABLE
        return status, self.write_slot * self.fb, self.laps

    def get_oldest_frame(self, hcam):
        if self.overrun:
            self.overrun = False
            raise PvcamError("pl_exp_get_oldest_frame_ex", 6, "circular buffer overrun")
        if self.require_unlock and self.locked:
            slot, nr = self.locked[0]
        else:
            if not self.unretrieved:
                raise PvcamError("pl_exp_get_oldest_frame_ex", 7, "no new frame")
            slot, nr = self.unretrieved.popleft()
            self.locked.append((slot, nr))
            self.slot_owner[slot] = "locked"
        ts = nr * self.period_ticks
        return self.start_address + slot * self.fb, FrameRecord(nr, ts, ts - self.readout_ticks, self.readout_ticks)

    def unlock_oldest_frame(self, hcam):
        if not self.locked:
            raise PvcamError("pl_exp_unlock_oldest_frame", 8, "no locked frame")
        self.unlock_calls += 1
        slot, _ = self.locked.popleft()
        if self.slot_owner.get(slot) == "locked" and not any(s == slot for s, _ in self.unretrieved):
            del self.slot_owner[slot]
        elif any(s == slot for s, _ in self.unretrieved):
            self.slot_owner[slot] = "unretrieved"

    def stop_cont(self, hcam, cam_state=pv.CCS_HALT):
        self.running = False

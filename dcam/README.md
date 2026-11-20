# dcam-camera

A camera driver for Hamamatsu cameras through **DCAM-API 4**. `dcam_camera.DcamSource` hands out **blocks of frames
in host memory that DCAM writes into directly** (`dcambuf_attach`), without copying. The package is a plain `ctypes`
binding and depends only on numpy.

```
camera ──DCAM──> attached frame k % (nblocks*F) = block (k//F) % nblocks, slot k % F ──(block complete)──> consumer
                    ^                                                                                    │
                    └── DCAM keeps writing round the frames; there is no queue: release(block) ends the hold ┘
```

## Requirements

- The DCAM-API runtime, installed by Hamamatsu's DCAM-API driver installer: `dcamapi.dll` (Windows,
  `C:\Windows\System32\dcamapi.dll`) or `libdcamapi.so` (Linux, `/usr/local/lib/libdcamapi.so`). `DcamSource(dll=...)`
  or the `DCAMAPI_DLL` environment variable override the path. The DCAM-SDK is not needed at run time, and no
  Hamamatsu files are included in this package.
- Python 3.10 or later and numpy.

## Install

```
pip install -e dcam
python dcam\examples\grab_frames.py --frames 64
```

`grab_frames.py` takes `--serial`, `--frames`, `--block`, `--blocks`, `--pixel-type MONO16|MONO8` and
`--exposure-us`.

DCAM is initialised once per process and never uninitialised (`dcamapi_uninit` closes every open camera), and the
camera list is fixed when DCAM is initialised. A camera can be opened by one application at a time: close HCImage
before opening it here.

## Usage

```python
from dcam_camera import DcamSource

with DcamSource(serial="<serial>", frames_per_block=64, nblocks=4,
                pixel_type="MONO16",            # MONO16 (default) or MONO8: zero copy
                roi=(0, 0, 2048, 1024),         # (x, y, w, h) in sensor pixels
                binning=1, exposure_us=10_000,
                trigger_source="INTERNAL",
                properties={"READOUT SPEED": 2}) as src:   # any DCAM property by name, written first
    print(src.info)                             # shape, dtype, fps, frame_bytes, extra
    src.start(max_frames=640)                   # attach the pool, start capture; stops itself after 640 frames
    while True:
        try:
            blk = src.wait_block(timeout_ms=1000)   # None on timeout
        except StopIteration:                       # stopped and drained
            break
        if blk is None:
            continue
        frames = blk.array                      # (nframes, H, W) view of the memory DCAM wrote into
        ok = src.overwritten(blk) == 0          # after processing: was the block overwritten while held?
        src.release(blk)
    print(src.stats())
```

### `DcamSource` parameters

All parameters after `serial` are keyword-only. Settings that change the frame layout (pixel type, binning, ROI,
sensor mode) belong in the constructor; `start()` refuses a layout that has changed since construction.

| parameter | default | meaning |
|---|---|---|
| `serial` | None | camera serial number; None opens the first camera |
| `frames_per_block` | 64 | frames per block |
| `nblocks` | 3 | blocks in the pool; at least 2 |
| `pixel_type` | `"MONO16"` | `IMAGE PIXEL TYPE`: `"MONO16"`, `"MONO8"` (zero copy) or `"MONO12"` (needs `mono12="copy"`) |
| `roi` | None | `(x, y, w, h)` in sensor pixels: `SUBARRAY HPOS/VPOS/HSIZE/VSIZE`; `SUBARRAY MODE` is switched on unless the ROI is the full sensor |
| `binning` | None | `BINNING` (for example 1, 2, 4) |
| `exposure_us` | 0 | `EXPOSURE TIME` in µs; 0 leaves it unchanged |
| `fps` | 0 | `INTERNAL FRAME RATE`; 0 leaves it unchanged (see below) |
| `trigger_source` | `"INTERNAL"` | `TRIGGER SOURCE`; the camera keeps this setting, so it is written on open. None leaves it unchanged |
| `trigger_polarity` | None | `TRIGGER POLARITY` |
| `properties` | None | dict of DCAM property names and values, written before everything else |
| `mono12` | `"refuse"` | `"refuse"` or `"copy"` for the packed MONO12 format |
| `stamps` | `"attach"` | how per-frame stamps are read: `"attach"`, `"lockframe"` or `"none"` (see below) |
| `prefault` | True | write the pool once before acquisition so its pages are resident |
| `buffers` | None | `nblocks` caller-owned host arrays used instead of an internal pool |
| `buffers_kind` | `"host"` | `"device"` raises `NotImplementedError` (see GPU memory) |
| `camera_model` | None | `(conversion DN/e-, offset DN, read noise e-)`; no models are built in |
| `dll` | None | path of the DCAM-API runtime |

`fps` writes `INTERNAL FRAME RATE`, which DCAM makes writable only in some sensor modes (on the ORCA-Flash4.0 V3,
for example, PROGRESSIVE and DUAL LIGHT SHEET). In AREA mode the exposure sets the rate and `fps=` raises
`ValueError`. Sensor modes that deliver more than one view per frame (`NUMBER OF VIEW` > 1) are refused.

`buffers=[...]` takes `nblocks` C-contiguous host arrays of the source's dtype with shape
`(>= frames_per_block, H, W)`; the first `frames_per_block` frames of each are used.

### Members

The source has `info`, `frames_per_block`, `nblocks`, `start(max_frames)`, `wait_block(timeout_ms)`,
`release(block)`, `stop()`, `stats()`, `close()` and `block_array(i)`. In addition:

- `overwritten(blk)`: the number of frames of a held block that DCAM has already overwritten (see Losses).
- `fire_trigger()`: `dcamcap_firetrigger`, for `trigger_source="SOFTWARE"`.
- `camera`: the `Dcam` adapter, for property access by name, for example `camera.get("SENSOR TEMPERATURE")`,
  `camera.set("TRIGGER POLARITY", "POSITIVE")`, `camera.entries("BINNING")`.

Blocks have `array`, `full_array`, `frames` (per-frame table), `seq` (delivered-block count), `index` (pool block),
`nframes`, `partial`, and `gen` (acquisition block number: `seq` plus the blocks lost before delivery).

## Zero copy

1. `start()` allocates `nblocks` blocks of `frames_per_block × H × W` pixels, each aligned to 4096 bytes and,
   with `prefault=True`, written once with zeros; or it uses `buffers=`.
2. It builds `N = nblocks × frames_per_block` frame pointers, block after block, frame after frame
   (`block_base + slot × frame_bytes`), and calls `dcambuf_attach` with `ATTACHKIND_FRAME`. DCAM writes capture frame
   k into pointer `k % N`.
3. `wait_block` polls `dcamcap_transferinfo` (frames captured since start) and waits on the `FRAMEREADY | STOPPED`
   events in slices of at most 100 ms. When frames `g·F .. g·F+F-1` have been captured, it returns pool block
   `g % nblocks` as a numpy view. `blk.array[s].ctypes.data` is the pointer DCAM was given.
4. Zero copy needs a contiguous buffer layout: `BUFFER ROWBYTES = W × bytes per pixel`,
   `BUFFER FRAMEBYTES = ROWBYTES × H` and `BUFFER TOP OFFSET BYTES = 0`. Otherwise the constructor raises.

**MONO12** is packed (2 pixels in 3 bytes) and cannot be delivered without a copy. `pixel_type="MONO12"` raises by
default. With `mono12="copy"`, DCAM writes into a packed staging ring of N frames and each block is unpacked into the
uint16 pool at `wait_block` (numpy); `info.extra["copy"]` and `stats().extra["copy"]` are then True. The bit layout is
p0 = b0 << 4 | b1 & 0xF, p1 = b2 << 4 | b1 >> 4 (verified on the ORCA-Flash4.0 V3).

## Per-frame metadata

`blk.frames` has one row per frame:

| field | value |
|---|---|
| `frame_id` | camera framestamp |
| `timestamp_ns` | DCAM timestamp (seconds, µs), in ns |
| `host_ns` | `time.monotonic_ns()` when the source first saw the frame counted; its resolution is how often `wait_block`, `stats` and `release` poll |
| `size_filled` | `BUFFER FRAMEBYTES` |
| `incomplete` | 0 (DCAM exposes no per-frame completeness flag) |
| `slot` | position in the block |

How the stamps are read is set by `stamps`:

- `"attach"` (default): at `start()` the source attaches an N-element `DCAM_TIMESTAMP` array
  (`ATTACHKIND_PRIMARY_TIMESTAMP`) and an N-element int32 array (`ATTACHKIND_PRIMARY_FRAMESTAMP`) next to the N
  frames, and DCAM fills element `k % N` with frame k's stamps. There is no per-frame call. If DCAM refuses the
  attach, the source falls back to `"lockframe"` and records the error in `stats().extra["stamps_error"]`.
  `stats().extra["stamps_unwritten"]` counts frames whose element DCAM had not written; their stamps fall back to the
  DCAM frame index and host time.
- `"lockframe"`: one `dcambuf_lockframe` call per frame. `stats().extra["lockframe_foreign_pointer"]` counts calls
  that returned an address other than the attached frame's.
- `"none"`, or when `dcambuf_lockframe` fails: `frame_id` is the DCAM frame index and `timestamp_ns` is `host_ns`.
  Frames first seen in the same poll share a timestamp, and frames lost before DCAM cannot be detected.

`stats().extra["stamps"]` reports the method in use. On the ORCA-Flash4.0 V3, `dcambuf_lockframe` and
`dcambuf_copyframe` are refused (`NOTREADY`) when frames are attached, so stamps are read from the attached metadata
arrays.

## Losses

DCAM has no buffer queue and no back-pressure: it writes round the N attached frames and overwrites old frames
whether or not the consumer still holds them; `release()` only ends the hold. Acquisition block g lives in pool
block `g % nblocks`, which DCAM begins to overwrite with frame `(g + nblocks) × F`. The source compares DCAM's
captured-frame count `c` with that frame number (while capturing, the frame at index `c` is taken as in flight):

| case | detection | accounting |
|---|---|---|
| block overwritten before delivery | `c + 1 > (g + nblocks) × F` at `wait_block` | not returned; its frames are counted in `lost` (`extra["lost_overwritten"]`, `extra["lost_blocks_before_delivery"]`) |
| block overwritten while held | `overwritten(blk) = min(nframes, c + 1 − (g + nblocks) × F)`, checked at `release` | those frames move from `delivered` to `lost`; `extra["overrun"] = True`, `extra["overrun_seqs"]` |
| frames lost before DCAM | framestamp gap: `missed = max(framestamp − stamp0 − k)` | `lost` (`extra["missed"]`) |
| DCAM wait events LOSTFRAME / MISSINGFRAME_TROUBLE | `dcamwait_start` return code | `extra["dcam_wait_events"]` |

`lost = lost_overwritten + missed`. A block can be held for `(nblocks − 1) × frames_per_block / fps` seconds without
loss; for example, 4.8 s with 4 blocks of 64 frames at 40 fps. A block returned by `wait_block` can still be
overwritten while it is being processed, so a consumer that must not commit corrupted data checks
`src.overwritten(blk) == 0` after processing and before committing.

`stamp0` (the framestamp minus the DCAM index of the first frame read) absorbs a start-up offset: on the
ORCA-Flash4.0 V3 the camera sometimes runs several frames before DCAM receives the first one (14 frames has been
observed), and that first frame then carries a framestamp greater than 0.

DCAM has no frame-count capture mode. With `start(max_frames=n)` the source calls `dcamcap_stop` once the count
reaches `n`; DCAM may capture a few more frames before it stops, and those are not delivered. After `stop()` or
`max_frames`, the last block is delivered with `partial=True` and `nframes < frames_per_block`; then `wait_block`
raises `StopIteration` and the attachment is released (`dcambuf_release`).

## GPU memory

DCAM attaches frames by host pointer (`ATTACHKIND_FRAME`); the API as bound here has no device-memory attach kind,
so GPUDirect RDMA is not available and `buffers_kind="device"` raises `NotImplementedError` (`GPU_STATUS` holds the
message). A GPU consumer can page-lock the host blocks returned by `block_array(i)` and copy them to the device.

## Tested hardware

Hamamatsu ORCA-Flash4.0 V3 (C13440-20CU) over USB3, DCAM-API 26.11.7243, at 2048 × 2048 (standard and slow scan) and
at a 1024 × 1024 ROI. The source sustains the camera's rate with continuous framestamps and no lost frames; with
attached stamps, timestamps step 24.994 to 24.995 ms at 40.009 fps.

## Use with b4d

`DcamSource` implements the b4d `BlockSource` contract (`b4d.ingest`) without importing b4d, so it can be passed to
`b4d.ingest.BlockRecorder`, which compresses the blocks straight out of the attached memory:

```
pip install -e "dcam[b4d]"
python dcam\examples\record_with_b4d.py --seconds 30
b4d-record --source dcam_camera.source:DcamSource --source-arg serial=<serial> --seconds 60 --q 1 --out D:\data\run1
```

The contract tests check the source against b4d's protocol definition (`b4d/src/b4d/ingest/protocol.py` in a b4d
checkout next to this package) and are skipped when it is not found.

## Tests

```
python -m pytest dcam/tests -q
```

The tests need no camera: `tests/dcam_fake.py` stands in for the DCAM-API runtime (the property set and access rules
of the ORCA-Flash4.0 V3, frame count, framestamps, MONO12 packing) and honours `dcambuf_attach`: each frame is written
with `ctypes.memmove` into attached pointer `count % N`, including overwrites. Frames are produced on demand, so
every count is exact. Covered: contract shape, structure sizes, zero copy (returned arrays are the attached memory,
page alignment), per-frame rows, overrun of a held block, blocks lost before delivery, missed frames with a start-up
offset, attached metadata stamps (wrap, missed frames, release), fallback from attached stamps to lockframe and to
host time, the final partial block (after `max_frames` and after `stop()`), restart, caller buffers, device buffers
raising, MONO8, and the MONO12 copy path.

## Package layout

```
src/dcam_camera/
  dcamapi.py    ctypes declarations of the DCAM-API 4 C interface (structures, constants, prototypes)
  dcam.py       adapter: DCAMError, open_by_serial, typed property access by name, attach/release,
                lockframe/copyframe, capture start/stop, transferinfo, wait events
  source.py     DcamSource
examples/       grab_frames.py, record_with_b4d.py
tests/          dcam_fake.py, test_contract.py, test_source.py
```

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

## License

MIT; see [LICENSE](LICENSE).

# egrabber-camera

A camera driver for Euresys Coaxlink CoaXPress frame grabbers, built on the Euresys eGrabber SDK. A C++ acquisition
engine writes frames by DMA into a pool of pinned host memory, and a Python package (`egrabber_camera`) hands that
memory out as **blocks of frames**, without copying. The package depends only on numpy.

```
camera (CXP) ──DMA──> block k, slot i ──(block complete)──> consumer
                         ^                                      │
                         └──────── slots re-queued ◄── release(block)
```

## Requirements

- Windows, with a Euresys Coaxlink frame grabber and the eGrabber SDK installed (default path
  `C:\Program Files\Euresys\eGrabber`; set `EURESYS_EGRABBER_ROOT` for another location). The eGrabber C++ API is
  header-only over the GenTL producer `coaxlink.cti`, which is loaded at run time.
- To build the engine: `clang-cl` on `PATH`, with an MSVC toolchain and Windows SDK that it can detect.
- Python 3.10 or later and numpy. CuPy (`pip install -e "egrabber[gpu]"`, `cupy-cuda12x`) is optional.

## Build and install

```
python egrabber\build.py            # builds bin\egrabber_ingest.dll and bin\eg_grab.exe; --debug for a debug build
pip install -e egrabber             # or build a wheel after build.py; the wheel then contains the DLL
```

`build.py` also copies the DLL into `src/egrabber_camera/`, so an editable install finds it and a wheel built
afterwards ships it. The DLL is looked up (`egrabber_camera.find_dll`) in this order: an explicit `dll=` argument,
the installed package, the source checkout's `egrabber/bin/`, then the path in the `EGRABBER_INGEST_DLL`
environment variable.

## Quick start

```
egrabber\bin\eg_grab.exe --seconds 20 --block 64 --blocks 3      # C++ engine only: rate, timestamps, losses
egrabber-grab --seconds 10                                       # Python, through EGrabberSource
egrabber-grab --info                                             # print camera information and exit
python egrabber\examples\grab_frames.py --frames 64
```

`--serial <serial>` selects a camera in each tool; without it the first camera found is used. `egrabber-grab` also
takes `--fps`, `--exposure-us`, `--block`, `--blocks`, `--frames`, `--save <file.npy>` (first frame) and
`--large-pages`.

## Usage

```python
from egrabber_camera import EGrabberSource

with EGrabberSource(serial="<serial>", frames_per_block=64, nblocks=3) as src:
    print(src.info)                                  # name, serial, shape, dtype, fps, frame_bytes, camera_model
    src.start(max_frames=640)                        # stops by itself after 640 frames
    while True:
        try:
            blk = src.wait_block(timeout_ms=1000)    # None on timeout
        except StopIteration:                        # stopped and drained
            break
        if blk is None:
            continue
        frames = blk.array                           # (nframes, H, W) view over the DMA memory
        ids, ts = blk.frames["frame_id"], blk.frames["timestamp_ns"]
        src.release(blk)                             # the block's slots go back to the grabber
    print(src.stats())
```

### `EGrabberSource` parameters

All parameters after `serial` are keyword-only.

| parameter | default | meaning |
|---|---|---|
| `serial` | None | camera `DeviceSerialNumber`; None opens the first camera found |
| `frames_per_block` | 64 | frames per block |
| `nblocks` | 3 | blocks in the pool |
| `fps` | 0 | `AcquisitionFrameRate`; 0 sets its maximum |
| `exposure_us` | 0 | `ExposureTime` in µs; 0 leaves it unchanged |
| `pixel_format` | `"Mono12"` | `PixelFormat` written at open; None leaves it unchanged |
| `large_pages` | False | allocate the pool with 2 MB large pages (needs the *Lock pages in memory* right; falls back to 4 KB pages) |
| `prefault_threads` | 16 | threads that touch the pool's pages before acquisition |
| `verbose` | False | engine log output |
| `dll` | None | path of `egrabber_ingest.dll` |
| `buffers` | None | `nblocks` caller-owned arrays announced to the grabber instead of an internal pool |
| `buffers_kind` | `"host"` | `"host"` or `"device"` (GPUDirect RDMA; Euresys supports it on Linux only, so it is refused on Windows) |
| `camera_model` | None | `(conversion DN/e-, offset DN, read noise e-)`; None takes the `CAMERA_MODELS` entry whose key prefixes the camera's model name |

### Members

```python
src.info                 # SourceInfo: name, serial, shape=(H, W), dtype, fps, frame_bytes, camera_model, extra
src.frames_per_block, src.nblocks
src.start(max_frames)    # pin the pool, announce every slot as a DMA target, start acquisition
blk = src.wait_block(timeout_ms)   # .array, .frames (per-frame table), .seq, .index, .nframes, .partial
src.release(blk)         # the block's memory may be refilled from now on
src.stats()              # SourceStats: delivered, lost, incomplete, blocks_completed, blocks_released,
                         # free_blocks, ready_blocks, fps_measured, extra
src.stop(); src.close()
src.block_array(i)       # the full array of pool block i (for example, to page-lock it for a GPU upload)
src.camera               # the lower-level Ingest object while started
```

A recording is complete when `lost` and `incomplete` both stay 0: `lost` counts frames the camera produced that
never reached a block (every block was held by the consumer), and `incomplete` counts frames that arrived damaged.

### Lower-level access

`Ingest` is the ctypes binding of the engine's C API (`src/cpp/egrabber_ingest.h`), with GenICam feature access
through `remote_get`, `remote_set` and `stream_get`. `IngestDevice` opens a camera as configured, without writing
any setting or allocating a pool, for host applications that own the camera settings or read one frame at a time:

```python
from egrabber_camera import IngestDevice

dev = IngestDevice("<serial>")
dev.remote.get("Width", int); dev.remote.set("ExposureTime", 10_000.0)   # same get/set shape as the Euresys Python SDK
dev.ingest.alloc_pool(frames_per_block=1, nblocks=16)                      # after ROI, binning and pixel format are set
dev.ingest.start(max_frames=32); blk = dev.ingest.wait_block(2000); dev.ingest.release(blk); dev.ingest.stop()
dev.ingest.start(max_frames=32)             # restartable; blocks still held by a consumer stay out of the queue
b, s = dev.ingest.next_slot()               # where the next frame will land (zero copy for frame-at-a-time hosts)
dev.ingest.wait_slot(b, s, 2000)            # wait for exactly that frame
dev.ingest.free_pool(); dev.close()
```

The `Ingest.alloc_pool` options after `nblocks` are keyword-only.

## How it works

The engine is in `src/cpp/egrabber_ingest.cpp`.

1. **Open.** `EGrabberDiscovery` lists the cameras and the one with the matching `DeviceSerialNumber` is opened as
   `EGrabber<CallbackOnDemand>`. `PixelFormat`, `AcquisitionFrameRate` and optionally `ExposureTime` are set, and
   the payload size is read.
2. **Pool.** `nblocks` regions of `frames_per_block × payload` bytes are allocated with `VirtualAlloc`, pre-faulted
   by `prefault_threads` threads, and announced with `announceAndQueue(UserMemoryArray(block, payload))`: one GenTL
   buffer per frame slot, queued in slot order, block after block. Announcing pins the memory for DMA.
3. **Acquisition thread.** A `processEvent<OneOf<NewBufferData, DataStreamData>>` loop wraps each new buffer without
   re-queueing it, derives its slot from `BUFFER_INFO_BASE`, and records `FRAMEID`, `TIMESTAMP_NS`, `SIZE_FILLED`
   and `IS_INCOMPLETE` in the block's frame table. When the last slot of a block arrives, the block becomes ready.
4. **Release.** `release(block)` re-queues the block's buffer range.
5. **Stop.** `stop()`, or reaching `max_frames`, stops the stream, drains the output queue and hands out the last
   partial block, if any (`partial=True`).
6. **Loss accounting.** The grabber does not report a frame it had no buffer for, so losses are detected as gaps in
   consecutive `TIMESTAMP_NS` (in units of the frame period) and in `FRAMEID`. A frame started while the queue ran
   dry is flagged incomplete. `eg_grab --hold-ms 1500` provokes this case on purpose.

The acquisition thread makes every grabber call except the buffer re-queue, which is thread-safe and called from
the consumer. The C API does not throw; `eg_last_error()` returns the last error message.

### Pinning large pools on Windows

The pool must be page-locked for DMA. When the Windows standby cache has grown to hundreds of GB and the free list
is near zero, pinning a pool of about 100 GB can take minutes instead of seconds. Emptying the standby list first (for example
with RAMMap's *Empty Standby List*) restores the normal pinning time. Granting the acquisition account the *Lock pages in memory* right avoids the problem.

## Tested hardware

| item | value |
|---|---|
| camera | Vieworks VPS-245MX2-M12H00 (Sony IMX811, rolling shutter) |
| frame grabber | Euresys Coaxlink Quad CXP-12 (PC3603, 1-camera firmware), link `CXP12_X4` |
| frame | 19,200 × 12,800 px, `PixelFormat = Mono12` (the only format this camera offers) |
| container | the grabber unpacks Mono12 to 16-bit words (`UnpackingMode = Lsb`): 491.52 MB per frame, 2 bytes per pixel |
| frame rate | 12.4381 fps maximum (`AcquisitionFrameRate.Max`), period 80.397 ms, 6,114 MB/s |
| camera model | gain 7.25 e-/ADU (conversion 0.1379 DN/e-), offset 2.05 ADU, read noise 6.89 e-, measured from dark frames; built into `CAMERA_MODELS` |
| storage | Micron 7450 PRO 12.8 TB (U.3 NVMe) |
| SDK | eGrabber 26.09.0.1 |

Measured on this system:

- Announcing a 94 GB pool (3 blocks of 64 frames) takes 4.6 to 11 s.
- The engine sustains the camera's maximum rate with 0 lost and 0 incomplete frames.
- Dark frames: offset 2.05 ADU, read noise 6.78 e-, dark current 2.6 e-/s, fixed pattern below 0.06 ADU.

## Use with b4d

`EGrabberSource` implements the b4d `BlockSource` contract (`b4d.ingest`) without importing b4d, so it can be passed
to `b4d.ingest.BlockRecorder`, which compresses each block in place:

```python
from b4d.ingest import BlockRecorder, RecorderConfig

with EGrabberSource(serial="<serial>") as src:
    with BlockRecorder(src, RecorderConfig(path=r"D:\data\run1\run.zarr", q=1, gpu="none")) as rec:
        rec.start(seconds=60)
        print(rec.report(rec.run()))
```

```
pip install -e "egrabber[b4d]"
python egrabber\examples\record_with_b4d.py --seconds 30
b4d-record --source egrabber_camera.source:EGrabberSource --source-arg serial=<serial> --seconds 60 --q 1 --out D:\data\run1
b4d-record --source egrabber_camera.source:EGrabberSource --seconds 60 --q 2 --gpu staged
b4d-record --source egrabber_camera.source:EGrabberSource --seconds 60 --multiscale 6
```

The recorder takes the camera model from `src.info.camera_model`; `--gain`, `--conv`, `--offset` and `--rn`
override it.

Measured on the tested hardware above (dark scene; other cameras, disks and scenes will differ): the camera's
maximum rate (12.438 fps, 6,114 MB/s raw) is recorded with zero lost frames at q 1 and q 2, in 60 s and 5-minute
runs, with 64-frame blocks: compression 0.57 s per block (5.1 s of frames), direct IO 1.35 s per block, block
latency about 2 s against a 10.3 s budget. 60 s runs with a 6-level pyramid and codec mode 2 `above`/`rice` gave
compression ratios of 5.26x (CPU, q 1, 1,162 MB/s), 7.78x (CPU, q 2, 786 MB/s), 5.43x (GPU staged, q 1) and 8.04x
(GPU staged, q 2).

## Tests

```
python -m pytest egrabber/tests -q
```

The unit tests run without a camera: contract shape, DLL discovery and GenICam feature access. A short live grab
runs when `EGRABBER_CAMERA_TESTS=1` is set and a camera is attached.

## Package layout

```
src/cpp/egrabber_ingest.h     C API of the engine
src/cpp/egrabber_ingest.cpp   the engine: EGrabber<CallbackOnDemand>, pinned block pool, acquisition thread
src/cpp/eg_grab.cpp           standalone C++ test of the engine (rate, timestamps, losses)
src/egrabber_camera/
  ingest.py                   Ingest, Block, Stats: ctypes over the DLL, numpy (or CuPy) views of the DMA memory
  source.py                   EGrabberSource
  genicam.py                  GenICam feature access (IngestDevice, remote.get / remote.set)
  cli.py                      the egrabber-grab command
examples/                     grab_frames.py, record_with_b4d.py
tests/                        unit tests, and a live grab when a camera is attached
build.py                      clang-cl build of the DLL and eg_grab.exe
```

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

## License

MIT; see [LICENSE](LICENSE).

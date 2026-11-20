# tucam-camera

A camera driver for Tucsen sCMOS cameras through Tucsen's **TUCam SDK V3**. `tucam_camera.TucamSource` hands out
**blocks of frames in host memory that the camera writes into**, without copying. The package is a thin adapter
over the SDK's own Python binding and depends only on numpy.

```
camera (USB3 Vision) ──SDK──> block k, slot i ──(last slot filled)──> consumer
                                 ^                                        │
                                 └──── block's slots re-queued ◄── release(block)
```

## Requirements

- The TUCam SDK V3, installed with Tucsen's installer. No Tucsen files are included in this package: `tucam.py`
  imports the SDK's Python binding (`include/python/TucApi.py`, `TucDefine.py`) and loads its runtime
  (`runtime/windows/x64/TUCam.dll`, `gentl/*.cti`) from the SDK install root. The root is, in order: the `sdk_dir=`
  argument, the `TUCAM_ROOT` environment variable, then `C:\Program Files\Tucsen\TUCamSDK V3` and
  `C:\Program Files\Tucsen\TUCamSDK V3 RC`. The runtime folder is put on `PATH` and `TUCAM_GENTL64_PATH` is pointed at
  its `gentl\` folder, from which the SDK loads its transport producers.
- Python 3.10 or later and numpy.
- The SDK allows one owner per camera: close every Tucsen application (SamplePro, Mosaic) before opening the camera.

## Install

```
pip install -e tucam
python tucam\examples\grab_frames.py --frames 64
```

`grab_frames.py` takes `--serial`, `--frames`, `--block`, `--sdk-dir` and `--no-chunk`.

The SDK is initialised once per process and terminated at interpreter exit (`tucam_camera.tucam.terminate`,
registered with `atexit`; without it the process can crash with 0xC0000005 on exit). `TucDeviceEnumerate` closes
every open camera, so `tucam_camera.devices()` enumerates once per process, before any camera is opened, and cameras
are then opened by serial number.

## Usage

```python
from tucam_camera import TucamSource

with TucamSource(serial="<serial>", frames_per_block=64, nblocks=3) as src:
    print(src.info)                  # shape, dtype, fps, frame_bytes, extra: pixel_format, frame_stride, contiguous ...
    src.start(max_frames=640)        # allocate and announce the pool, start; stops after 640 frames
    while True:
        try:
            blk = src.wait_block(timeout_ms=1000)   # None on timeout
        except StopIteration:                       # stopped and drained
            break
        if blk is None:
            continue
        frames = blk.array           # (nframes, H, W) view over the memory the camera wrote
        ids, ts = blk.frames["frame_id"], blk.frames["timestamp_ns"]
        src.release(blk)             # re-queues the block's slots to the SDK
    print(src.stats())
```

Members: `info`, `frames_per_block`, `nblocks`, `start(max_frames)`, `wait_block(timeout_ms)`, `release(block)`,
`stats()`, `stop()`, `close()`, `block_array(i)`, and `camera` (the open `TUCam`, for GenICam node access with `get`,
`set`, `limits`, `entries` and `execute`). Blocks have `array`, `full_array`, `frames`, `seq`, `index`, `nframes` and
`partial`.

### `TucamSource` parameters

All parameters after `serial` are keyword-only.

| parameter | default | meaning |
|---|---|---|
| `serial` | None | camera serial number; None opens the first camera found |
| `frames_per_block` | 64 | frames per block |
| `nblocks` | 3 | blocks in the pool |
| `fps` | 0 | `AcquisitionFrameRate`; 0 sets its maximum (Tucsen's own software leaves it near 100 Hz) |
| `exposure_us` | 0 | `ExposureTime` in whole µs; 0 leaves it unchanged |
| `chunk_mode` | True | `ChunkModeActive`: per-frame camera counter and timestamp, needed for loss counting |
| `free_run` | True | reset `TriggerMode` to FreeRunning on open |
| `stride_align` | 64 | frame stride granularity in bytes |
| `image_offset` | 0 | expected byte offset of the pixels inside a slot |
| `starve_margin` | 2 | queued-buffer count at or below which a counter gap counts as back-pressure overflow |
| `sdk_dir` | None | TUCam SDK V3 install root |
| `buffers` | None | `nblocks` caller-owned host arrays used instead of an internal pool |
| `buffers_kind` | `"host"` | `"device"` raises `NotImplementedError` (see GPU memory) |
| `camera_model` | None | `(conversion DN/e-, offset DN, read noise e-)`; the SDK provides no noise model |
| `nodes` | None | GenICam node values written in order at construction (see below) |
| `packed` | `"refuse"` | `"refuse"` or `"copy"` for Mono12Packed |
| `unpack_threads` | 8 | threads that unpack one frame with `packed="copy"` |

`nodes={...}` writes GenICam nodes in the given order at construction, before the frame layout is read: sensor and
gain mode, ROI and anything else the camera exposes. The camera keeps them after `close()`. For example, Speed mode
at a centred 1024 × 1024 ROI on the Aries 6504:

```python
TucamSource("<serial>", packed="copy",
            nodes={"SensorOperationMode": "Speed", "Speed": "HighGain",
                   "Width": 1024, "Height": 1024, "OffsetX": 512, "OffsetY": 512})
```

Reduce Width and Height before raising the offsets, and lower the offsets before restoring the full size. Change
the frame layout (ROI, binning, pixel format, chunk mode) before `start()`, which refuses a layout that has changed
since construction.

`info.extra` carries the camera's `OverallSystemGainK` (e-/ADU) and `AverageBlackLevel` (ADU) when they can be read.

## Zero copy

1. `start()` allocates `nblocks` blocks of `frames_per_block × frame_stride` bytes, each aligned to 4096 bytes and
   pre-faulted, or uses the caller's `buffers=[...]`.
2. The SDK queue policy is set to MANUAL and every frame slot of every block is announced as one acquisition buffer
   (`TucBufferAnnounce(block address + i × frame_stride, frame_stride)`). The buffer callback is registered and
   acquisition started; at start the SDK puts every announced buffer in its input pool.
3. Frames land directly in the slots. The callback, on the SDK's grab thread, only reads the buffer information
   (frame id, timestamp, geometry, format, flags), fills the block's frame table row and, when a block's last slot is
   filled, hands the block to `wait_block()`.
4. `release(block)` re-queues that block's slots (`TucBufferQueue`, in slot order). Until then the SDK has fewer
   buffers: real back-pressure.
5. `stop()`, or reaching `max_frames`, hands out the block being filled as a partial block (`partial=True`,
   `nframes < frames_per_block`); then `wait_block()` raises `StopIteration`. `close()` unregisters the callback,
   revokes every buffer, restores the AUTO queue policy and closes the camera.

`block.array`, `block.full_array` and `block_array(i)` are views over the announced memory. `blk.frames` is a copy of
the frame table rows, so a consumer may keep it after `release`.

The source assumes that the SDK fills queued buffers in queue order; `stats().extra["out_of_order"]` counts slots
that were not filled in order.

### Frame stride

The slot size is the SDK's required announce size (`TucBuffersGetRequiredSize`) rounded up to `stride_align`. With
chunk mode on, that size exceeds `H × W × bytes per pixel` by the chunk data, 128 bytes per frame on the Aries 6504.
For a 2048 × 2048 Mono16 frame: 8,388,608 bytes of pixels, stride 8,388,736 bytes. The block view is
`np.ndarray((n, H, W), strides=(frame_stride, W × bpp, bpp))`: **each frame's pixels are contiguous, but the frames
are not contiguous with each other** (`array.flags.c_contiguous` is False). `info.extra["frame_stride"]` and
`info.extra["contiguous"]` report this. A consumer that needs C-contiguous blocks must copy them, or use
`chunk_mode=False`, which makes the stride equal to the frame size and the blocks C-contiguous but disables loss
counting.

The pixels are expected at the start of the slot with the chunk data after them (`image_offset=0`). Each frame's
reported `imageOffset`, geometry and pixel format are checked, and a frame that disagrees is flagged incomplete and
counted in `stats().extra["layout_mismatch"]`.

### Caller buffers

`buffers=[...]` takes `nblocks` writable C-contiguous host arrays of at least `frames_per_block × frame_stride` bytes
each; with chunk mode off, a `(frames_per_block, H, W)` array qualifies. The source announces and revokes them but
does not free them.

## Pixel formats

Mono8 (uint8), Mono12 and Mono16 (both in a 16-bit container) are delivered as is, without a copy. Mono12Packed
(2 pixels in 3 bytes) needs unpacking, which is a per-frame copy:

- `packed="refuse"` (default): a packed format raises `ValueError` at construction.
- `packed="copy"`: the SDK is given a packed staging slot for every pool slot. The buffer callback unpacks each frame
  into its uint16 pool slot, on `unpack_threads` threads (row bands), before the frame is counted. Blocks are then
  C-contiguous and hold 12-bit values (0 to 4095). Staging slots go back to the SDK with their block in `release`,
  so back-pressure and loss counting work as without the copy. `info.extra["copy"]` and `stats().extra["copy"]` are
  True, and `stats().extra["unpack_ms_mean"]` and `stats().extra["unpack_ms_max"]` give the cost: 1.71 ms mean per
  2048 × 2048 frame with 8 threads on the Aries 6504, against a 3.37 ms frame period at 297 fps.

The bit layout is GigE-style Mono12Packed, p0 = b0 << 4 | b1 & 0xF, p1 = b2 << 4 | b1 >> 4, checked against frames
from the Aries 6504. The SDK labels the format with PFNC code 0x010C0047, which PFNC calls Mono12p (a different bit
order). `TucBufferCopyToMemory` copies the packed bytes unchanged, so it cannot unpack. The SDK's own conversion,
`TucBufferConvertToImage`, is not used: it converts the newest delivered frame rather than the buffer passed to it.

On the Aries 6504, `PixelFormat` is read-only and follows the sensor and gain mode:

| SensorOperationMode | gain mode | PixelFormat | maximum rate at 2048 × 2048 (rolling / global reset) |
|---|---|---|---|
| Dynamic | HDR | Mono16 | 170.5 / 167.6 fps |
| Dynamic | HighGain, LowGain | Mono12Packed | 170.5 / 167.6 fps |
| Speed | HighGain, MidGain, LowGain | Mono12Packed | 297.0 / 291.9 fps |

No mode on this camera delivers `Mono12` (12 bits in the low bits of 16), although the enumeration lists it, and the
camera has no node to choose packing or bit alignment.

## Losses

With chunk mode on, the SDK fills each buffer's `frameId` with the camera's frame counter and `timestamp` with the
camera clock in µs (`timestamp_ns` = timestamp × 1000; `TIMESTAMP_NS_PER_TICK` in `source.py` sets the scale). A
counter jump of k means k − 1 frames were lost:

| counter | meaning |
|---|---|
| `stats().extra["lost_overflow"]` | the jump came after the queued-buffer count had fallen to `starve_margin` or below: every block was held by the consumer (back-pressure) |
| `stats().extra["lost_before_host"]` | the jump came while buffers were queued: frames lost between the sensor and the host |
| `stats().lost` | the sum |
| `stats().incomplete` | frames delivered damaged (SDK status not OK, incomplete, no image, layout mismatch); they stay in their slot with `incomplete=1` |
| `stats().extra["pool_empty_events"]` | frames after which no buffer was queued |

Limits:

- Frames lost after the last delivered frame leave no counter jump and are not counted.
- With chunk mode off the frame ids stay 0 and nothing is counted (`extra["loss_counting"]` is False).
- In hardware-trigger mode, a trigger the camera never received is not a counter jump.
- The SDK's own drop counters (`TucAcquisitionGetStatistics`, buffer-underrun events) stay 0 on the Aries 6504 while
  frames are lost, so they are not used.
- With `max_frames`, the source ends when delivered plus lost frames reach `max_frames`, and stops the SDK in
  `stop()`.

The frame table (`blk.frames`) has one row per frame: `frame_id` (camera counter), `timestamp_ns` (camera clock),
`host_ns` (`time.monotonic_ns()` in the callback), `size_filled` (H × W × bpp for a good frame, 0 for a damaged one,
derived from the checked geometry), `incomplete` and `slot`.

## GPU memory

`TucBufferAnnounce` takes a host address and the SDK binding has no device-memory call, so GPUDirect RDMA is not
available: `buffers_kind="device"` raises `NotImplementedError`, and CuPy arrays in `buffers=` raise `TypeError`. A
GPU consumer can page-lock the host blocks returned by `block_array(i)` (for example with `cudaHostRegister`) and
upload them, using the frame pitch for strided blocks.

## Tested hardware

Tucsen Aries 6504 over USB-C, TUCam SDK V3 (`TUCam.dll` 3.0.12.0), Windows 11:

- Dynamic / HDR, Mono16, zero copy: 170.5 fps at 2048 × 2048, 1.43 GB/s raw, no lost frames.
- Speed / HighGain, Mono12Packed with `packed="copy"`: 297 fps at 2048 × 2048 (2.49 GB/s after unpacking) and
  587 fps at a 1024 × 1024 ROI, no lost frames.
- The Speed-mode rates above were reached on a 20 Gbit/s (USB 3.2 Gen 2x2) link, about 1.87 GB/s on the wire at
  297 fps; on a 10 Gbit/s link Speed mode reaches about 152 fps.
- Width, Height, OffsetX and OffsetY snap to multiples of 16; the exposure time snaps to whole line times. A GenICam
  node read takes about 1.3 ms over USB.
- 3 blocks of 64 full Mono16 frames hold 192 frames (1.61 GB), 1.13 s at 170 fps: the longest a consumer can hold
  all blocks before frames are lost. Raise `nblocks` or `frames_per_block` for longer stalls.

## Use with b4d

`TucamSource` implements the b4d `BlockSource` contract (`b4d.ingest`) without importing b4d, so it can be passed to
`b4d.ingest.BlockRecorder`:

```
pip install -e "tucam[b4d]"
python tucam\examples\record_with_b4d.py --seconds 30
b4d-record --source tucam_camera.source:TucamSource --source-arg serial=<serial> --seconds 60 --q 1 --out D:\data\run1
```

Pass the camera model with `camera_model=` or the recorder's `--gain`, `--offset` and `--rn` options. The contract
tests check the source against b4d's protocol definition when b4d is installed.

## Tests

```
python -m pytest tucam/tests -q
```

The tests need no camera: `tests/fake_tucam.py` implements the SDK calls the adapter makes (enumeration, nodes,
required size, queue policy, announce, queue, revoke, buffer information, the callback, start and stop). Its camera
writes each frame into the oldest queued buffer with `ctypes.memmove`, followed by 128 bytes of chunk data, and loses
frames when no buffer is queued. Covered: contract shape, zero copy (the block view shares memory with the pool and
each frame sits at the address announced for its slot), contiguous blocks with chunk mode off, re-queue on release,
back-pressure loss counting while blocks are held, losses before the host, frame table rows, layout mismatch, the
partial final block after `max_frames` and after `stop()`, caller buffers, device buffers raising, packed formats
refused by default, `nodes` written in order, the Mono12Packed bit layout, `packed="copy"` (unpacked contiguous pool,
1 and 3 threads, back-pressure losses still counted), `terminate()`, and `close()` revoking every buffer.

## Package layout

```
src/tucam_camera/
  tucam.py      thin adapter over the SDK's Python binding: SDK discovery and initialisation, device enumeration,
                GenICam nodes, buffer announce / queue / revoke / information / callback, acquisition start and stop
  source.py     TucamSource
examples/       grab_frames.py, record_with_b4d.py
tests/          fake_tucam.py, test_contract.py, test_source.py
```

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

## License

MIT; see [LICENSE](LICENSE).

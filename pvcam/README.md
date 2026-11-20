# pvcam-camera

A camera driver for Teledyne Photometrics cameras (designed for the **Kinetix**) through the PVCAM API.
`pvcam_camera.PvcamSource` hands out **blocks of frames that are views of the circular buffer PVCAM writes into**,
without copying. The binding is plain `ctypes` over the PVCAM runtime and depends only on numpy.

```
camera ──PVCAM driver──> circular buffer (one allocation, nblocks × frames_per_block slots)
                           block 0 | block 1 | block 2 ...  ──(block complete)──> consumer
                              ^                                                       │
                              └──── frames unlocked (pl_exp_unlock_oldest_frame) ◄── release(block)
```

The package does not use PyVCAM. PyVCAM allocates the circular buffer itself, runs it in `CIRC_OVERWRITE` mode and
hands out one-frame views without a lock, so frames cannot land in caller-owned block memory or be held against
overwrite. This package passes its own buffer to `pl_exp_start_cont` and keeps frames locked until their block is
released.

## Requirements

- The **PVCAM runtime** from Teledyne Photometrics, which installs `pvcam64.dll` (Windows) or `libpvcam.so`
  (`libpvcam.so.2`, Linux) and the camera's interface driver, for example the PCIe driver. The SDK headers are not
  needed. If the library is not on the loader path, set `PVCAM_LIBRARY` to its full path, or pass `library=`.
- Python 3.10 or later and numpy.

## Install

```
pip install -e pvcam
python pvcam\examples\grab_frames.py --frames 64 --port 1
```

`grab_frames.py` takes `--serial`, `--frames`, `--block`, `--blocks`, `--port` (readout port value) and
`--exposure-us`.

## Usage

```python
from pvcam_camera import PvcamSource

with PvcamSource(serial="<serial>", frames_per_block=64, nblocks=3, readout_port=1) as src:   # Kinetix Speed port, 8 bit
    print(src.info)                             # name, serial, shape, dtype, fps, frame_bytes, camera_model, extra
    src.start(max_frames=640)                   # allocate the circular buffer and start; stops after 640 frames
    while True:
        try:
            blk = src.wait_block(timeout_ms=1000)   # None on timeout
        except StopIteration:                       # stopped and drained
            break
        if blk is None:
            continue
        frames = blk.array                      # (nframes, H, W) view of the circular buffer
        ids = blk.frames["frame_id"]            # FrameNr
        src.release(blk)                        # the block's frames are unlocked; PVCAM may refill them
    print(src.stats())
```

Members: `info`, `frames_per_block`, `nblocks`, `start(max_frames)`, `wait_block(timeout_ms)`, `release(block)`,
`stats()` (delivered, lost, incomplete, blocks completed and released, free and ready blocks, `fps_measured`,
`extra`), `stop()`, `close()`, and `block_array(i)` (the full `(frames_per_block, H, W)` view of pool block i).
Blocks have `array`, `full_array`, `frames`, `frame_info`, `seq`, `index`, `nframes` and `partial`.

### `PvcamSource` parameters

All parameters after `serial` are keyword-only.

| parameter | default | meaning |
|---|---|---|
| `serial` | None | camera serial number; None opens the first camera |
| `frames_per_block` | 64 | frames per block |
| `nblocks` | 3 | blocks in the circular buffer |
| `camera_name` | None | PVCAM camera name, for example `PMPCIECam00` |
| `exposure_us` | None | exposure time in µs; None uses the camera minimum. The exposure resolution is switched to µs when the camera offers it |
| `readout_port` | None | readout port value |
| `speed_index` | None | readout speed index |
| `gain_index` | None | gain index. Port, speed and gain are set in that order, because setting the speed resets the gain |
| `roi` | None | `(x, y, w, h)` |
| `binning` | `(1, 1)` | `(serial binning, parallel binning)` |
| `exp_mode` | None | exposure mode; None uses `EXT_TRIG_INTERNAL` combined with the current expose-out mode when the camera lists it, else `TIMED_MODE` |
| `fps` | None | nominal rate reported in `info.fps`; None uses `1e6 / max(PARAM_READOUT_TIME µs, exposure µs)` |
| `buffer_mode` | `"no_overwrite"` | `"no_overwrite"` (`CIRC_NO_OVERWRITE`) or `"overwrite"` |
| `prefault` | True | touch each page of the buffer once before acquisition |
| `poll_interval_s` | 0.0005 | polling interval of the acquisition thread |
| `timestamp_unit_ns` | 100000 | nanoseconds per `FRAME_INFO.TimeStamp` count (PVCAM documents 0.1 ms precision) |
| `max_get_failures` | None | stop after this many consecutive failed frame retrievals |
| `buffers` | None | caller-owned circular buffer (see below) |
| `buffers_kind` | `"host"` | `"device"` raises `NotImplementedError` (see GPU memory) |
| `camera_model` | None | `(conversion DN/e-, offset DN, read noise e-)`; default from `CAMERA_MODELS` |
| `runtime` | None | a `pvcam.Pvcam`-like object to use instead of the PVCAM runtime (the tests pass a fake) |
| `library` | None | path of the PVCAM runtime library |

All camera parameters are read in the constructor and cached; no parameter is read during acquisition, because
reading a non-live PVCAM parameter during acquisition can stop a Kinetix acquisition.

### Kinetix readout ports

| port (value) | bit depth (host dtype) | readout (ms) | maximum frame rate at 3200 × 3200 (fps) | data rate (GB/s) |
|---|---|---|---|---|
| Sensitivity (0) | 12 (uint16) | 11.3 | 88.4 | 1.81 |
| Speed (1) | 8 (uint8) | 2.0 | 498 | 5.10 |
| Dynamic Range (2) | 16 (uint16) | 12.0 | 83.3 | 1.71 |
| Sub-Electron (3) | 16 (uint16) | 192 | 5.2 | 0.11 |

### Camera model

`CAMERA_MODELS[("kinetix", 1, 8)] = (0.851 DN/e-, 6.0 DN, 2.35 e-)` is applied when the camera's product or chip
name contains "Kinetix", the readout port is 1 (Speed) and the host bit depth is 8 (maximum 255 ADU). It is not
tied to a particular Speed-port gain index. For other ports `info.camera_model` is None; pass `camera_model=` with
a model measured for the port and gain in use.

### Buffer size limit

`pl_exp_start_cont` takes the buffer size as `uns32`, so the circular buffer holds at most 4,294,967,295 bytes:
419 full Kinetix frames at 8 bit (10.24 MB each) or 209 at 12 or 16 bit (20.48 MB each). A
`frames_per_block × nblocks` beyond that raises `ValueError` in the constructor.

## Zero copy

1. **One buffer.** The source allocates one contiguous buffer of `nblocks × frames_per_block` frame slots, aligned
   to 4096 bytes, and with `prefault=True` touches each page once. Pool block i is slots
   `[i × fpb, (i + 1) × fpb)`. The buffer has exactly `nblocks × fpb` slots, so a block never straddles the wrap.
2. **Setup.** `pl_exp_setup_cont` with one region and the `buffer_mode` (default `CIRC_NO_OVERWRITE`). `PARAM_METADATA_ENABLED` and
   `PARAM_CENTROIDS_ENABLED` are switched off, so a slot holds only pixels. The frame size returned by the setup call
   must equal `H × W × bytes per pixel` (from `PARAM_IMAGE_FORMAT_HOST`); otherwise the constructor raises, for
   example for metadata that cannot be switched off, a compressed host format or several ROIs.
3. **Start.** `pl_exp_start_cont(hcam, buffer_address, buffer_bytes)`: PVCAM writes into the package's memory.
4. **Acquisition thread.** One thread makes every PVCAM call. While `pl_exp_check_cont_status` reports
   `FRAME_AVAILABLE`, `pl_exp_get_oldest_frame_ex` returns the frame address and its `FRAME_INFO`. The slot is derived
   from the address and must equal `frames retrieved mod slots`; the block layout relies on PVCAM filling the buffer
   in order from its start, and this is checked for every frame (a mismatch stops the acquisition and sets
   `extra["error"]`). The frame is not unlocked. When a block's last slot arrives, the block becomes ready.
5. **Release.** `release(block)` marks the block, and the thread then calls `pl_exp_unlock_oldest_frame` once per
   frame of the block. PVCAM unlocks the oldest frame, so blocks are unlocked in fill order: a block released before
   an earlier one waits for it.
6. **Stop.** `stop()`, or reaching `max_frames`, takes the frames already available, calls
   `pl_exp_stop_cont(CCS_HALT)` and hands out the final partial block (`partial=True`).

### Locked frames

The PVCAM header documents `CIRC_NO_OVERWRITE` both as reporting an overrun error when the oldest frame is replaced
and, for `pl_exp_unlock_oldest_frame`, as halting acquisition while frames stay locked; it does not say whether
several frames may be locked at once. The source handles each runtime behaviour:

- **Locks honoured.** The driver cannot place a frame while the buffer is full of held blocks; the frame is dropped,
  and the next delivered frame shows a `FrameNr` gap, counted in `lost`. Held memory stays intact.
- **Overwrite despite the lock.** A new frame lands in a slot of a block the consumer still holds. The source detects
  this when it retrieves that frame, counts it in `incomplete` and `extra["overwritten_held_frames"]`, stops the
  acquisition and sets `extra["error"]`; the remaining blocks are delivered and `wait_block` then raises
  `StopIteration`. An overrun error from `pl_exp_get_oldest_frame_ex` is counted in `extra["get_frame_errors"]`
  (`extra["last_get_error"]` keeps the message); with `max_get_failures=n`, n consecutive failures stop the
  acquisition.
- **One locked frame at a time.** If PVCAM returns the same frame (address and `FrameNr`) again, it needs an unlock
  before it hands out the next frame. The source then unlocks each frame on retrieval
  (`extra["lock_mode"] == "unlock_on_retrieve"`), and held blocks are protected only by the overwrite check above.

### Caller buffers

`buffers=[array]` takes one C-contiguous host array of `nblocks × frames_per_block` frames
(`nblocks × fpb × frame_bytes` bytes). `buffers=[b0, ..., b(n-1)]` is accepted only if the arrays are adjacent pieces
of one allocation, because `pl_exp_start_cont` takes one pointer and one size.

## Losses and per-frame metadata

| field | meaning |
|---|---|
| `lost` | sum of the gaps in `FRAME_INFO.FrameNr` (1-based) between consecutive retrieved frames, plus `FrameNr - 1` before the first one: frames the camera produced that never reached a block. A gap after the last delivered frame cannot be seen |
| `incomplete` | frames detected as written into a block the consumer still held |
| `extra["frame_nr_backwards"]` | `FrameNr` went down or repeated (expected to stay 0) |
| `extra["error"]` | why the acquisition stopped early (overwrite, out-of-order slot, READOUT_FAILED, PVCAM error) |
| `fps_measured` | `(last FrameNr - first FrameNr) / TimeStamp span` |

A recording is complete when `lost` and `incomplete` stay 0 and `extra["error"]` is empty.

`block.frames` has one row per frame: `frame_id` = `FrameNr`; `timestamp_ns` = `FRAME_INFO.TimeStamp` (end of frame)
× `timestamp_unit_ns`; `host_ns` = `time.perf_counter_ns()` at retrieval; `size_filled` = frame bytes;
`incomplete` = 0; `slot` = position in the block. `block.frame_info` keeps `FrameNr`, `TimeStamp`, `TimeStampBOF` and
`ReadoutTime` as PVCAM delivered them. These timestamps come from the device driver, not the sensor, and fast sCMOS
cameras with an internal frame buffer may report identical start and end times; hardware timestamps exist only in
embedded metadata, which this package switches off.

## GPU memory

The PVCAM API (SDK 3.10.1.1 on Windows, 3.10.2.5 on Linux) has no GPU, CUDA, RDMA or device-memory entry point, and
`pl_exp_start_cont` takes a host pointer. GPUDirect RDMA is therefore not available: `buffers_kind="device"` raises
`NotImplementedError` (`pvcam_camera.GPU_STATUS` holds the message). A GPU consumer can page-lock the host blocks
returned by `block_array(i)` (for example with `cudaHostRegister`) and upload them.

## Testing

The package is tested against a simulated PVCAM runtime, `tests/fake_pvcam.py`, which has the same Python-level
interface as `pvcam.Pvcam` and writes frames into the caller's circular buffer with `ctypes.memmove`.

```
python -m pytest pvcam/tests -q
```

No camera or PVCAM installation is needed. Covered: contract shape, setup (metadata off, `CIRC_NO_OVERWRITE`, frame
size, exposure mode), open by serial, metadata that cannot be switched off, the Kinetix 8-bit camera model, the
`uns32` buffer limit, zero copy (block arrays share memory with the circular buffer at the expected addresses and
hold the pixels the fake wrote), caller buffers, per-frame rows and `FRAME_INFO` rows, the partial final block,
back-pressure and loss counting while blocks are held (held memory intact), unlock order, detection of an overwritten
held block, the one-locked-frame fallback, stop with a partial block, and `buffers_kind="device"`.

`test_live_camera_short_grab` runs a short acquisition when `PVCAM_CAMERA_TESTS=1` is set and a camera is attached.

## Use with b4d

`PvcamSource` implements the b4d `BlockSource` contract (`b4d.ingest`) without importing b4d, so it can be passed to
`b4d.ingest.BlockRecorder`, which compresses the blocks straight out of the circular buffer:

```
pip install -e "pvcam[b4d]"
python pvcam\examples\record_with_b4d.py --port 1 --seconds 30
b4d-record --source pvcam_camera.source:PvcamSource --source-arg serial=<serial> --seconds 60 --q 1 --out D:\data\run1
```

The contract tests check the source against `b4d.ingest` when it imports, or against b4d's protocol definition in a
b4d checkout next to this package.

## Package layout

```
src/pvcam_camera/
  pvcam.py      ctypes binding: load the runtime, pl_pvcam_init/uninit (reference counted), camera list, open and
                close, get/set/enumerate parameters, continuous acquisition, frame retrieval and unlock, error text.
                Declarations follow the PVCAM SDK 3.10.1.1 header (not shipped here).
  source.py     PvcamSource
examples/       grab_frames.py, record_with_b4d.py
tests/          fake_pvcam.py, test_contract.py
```

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

## License

MIT; see [LICENSE](LICENSE).

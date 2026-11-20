# b4d/tools — simulate a camera, find the best writer settings for it, check the drive, look at the result

Seven programs (not library code, not tests), and two helper modules.

| program | what it does |
|---|---|
| `stream_sim.py` | Simulates a camera at a fixed frame rate (sphere phantom through the camera model: `--shape`, `--conv`, `--offset`, `--rn`, `--max-adu`, `--spot-peak`) streaming into the native writer with the given settings (`--block`, `--chunks`, `--shards`, `--threads`, `--max-commits`, `--codec b4d|zstd|lz4|raw`, `--q`, `--predictor`, `--multiscale`, `--gpu-encode`) and sink (`--sink memory` = CPU/pipeline capacity, `--sink disk` with `--direct-io` = the drive in the loop). Reports whether the rate was sustained (zero dropped frames), ratio, MB/s, per-block compress and IO time, and a machine-readable `RESULT` line; `--verify` reads a sample back and checks the error bound. |
| `b4d-tune` (`src/b4d/tune.py`, installed with b4d; needs this `stream_sim.py`, found from a checkout or `--sim` / `$B4D_STREAM_SIM`) | Drives `stream_sim.py` to find, by measurement, the best settings for a camera on this machine: for each setting the capacity (highest sustained fps, bracketed and confirmed over 60 s), changing one parameter at a time in the order that mattered in the tuning campaign (block depth, chunk footprint, blocks in flight, threads, shard footprint) and keeping a change only if it gains 3%. Writes `results.csv`, logs and `recommendation.json` (settings, capacity, the recommended operating rate = capacity / 1.2, ratio, the compressed data rate the storage must sustain). Presets: `--camera imx411|imx811|kinetix|aries`, or describe any camera with `--shape --conv --offset --rn --max-adu --fps-start`. |
| `disk_speed.py` | Measures a drive the way the writer uses it: shard-sized files written whole, unbuffered write-through (`direct`, what `direct_io=True` does) and buffered + flushed, one or several threads, for enough GB to get past the write cache; reports the sustained MB/s, the worst 10 s window and the read-back speed. Compare with the compressed GB/s `b4d-tune` reports for the camera. |
| `calibrate_camera.py` | Measures a camera's per-pixel offset (DN), read noise (electrons) and gain (electrons per DN) from a dark capture and flat-field captures at several light levels, as the `.npz` that `b4d.PixelCalibration.load` and `b4d-record --calibration` read. Captures are stacks (.ims, .tif, .npy, .h5, .zarr; read a few frames at a time, so larger than memory) or mean/variance images saved by `capture_stats.py`. `python calibrate_camera.py --dark dark.npz --flat f1.npz ... f10.npz --dark-end dark_end.npz -o camera_cal.npz --max-ram 16`. |
| `capture_stats.py` | `FrameStats`: running mean, variance and maximum images of frames as they are captured (numpy only, for the capture PC), saved in the form `calibrate_camera.py` reads. |
| `stack_io.py` | Reads part of a (frames, rows, cols) stack without loading the rest (.ims, .tif/.tiff, .npy, .h5/.hdf5, .zarr); used by `calibrate_camera.py`. |
| `compare_camera_settings.py` | Chooses between camera settings (on-camera corrections on or off) from a dark and sample stacks for each: b4d's ratio with and without the per-pixel list at each q, decoded noise, pixels the setting changes, and whether it alters real signal. The best ratio wins, unless real signal or noise fidelity suffers. |
| `napari_visualization.py` | Opens a B4D zarr array or OME-Zarr pyramid in napari (lazy, chunk-on-demand decoding through dask). |
| `neuroglancer_visualization.py` | Serves a B4D dataset to neuroglancer (or any zarr v3 reader) as a virtual unsharded zarr v3 view, transcoding B4D chunks on demand in C. |

```
cd b4d
python tools/stream_sim.py --shape 10640,14192 --fps 110 --seconds 30 --sink memory --zero-copy --threads 128 \
    --block 64 --bank 64 --blocks 3 --ring 64 --chunks 32,64,64 --shards 64,1408,3584 --q 1 \
    --mode 1 --predictor above3          # stream_sim's own codec defaults are mode 2, above
b4d-tune --camera imx811 --q 1            # 1-3 h; --quick for a coarse pass; resumes from results.csv
b4d-tune --shape 2048,2048 --conv 3.94 --offset 100 --rn 0.71 --max-adu 65535 --fps-start 2000 --q 2 --name aries
python tools/disk_speed.py D:\ --total-gb 64 --threads 2                     # sustained write/read MB/s of the drive
python tools/compare_camera_settings.py --gain camera_cal.npz --setting off dark_off.tif before.tif after.tif --setting on dark_on.tif sample_on.tif
python tools/napari_visualization.py D:\data\b4d_out\stream.zarr
python tools/neuroglancer_visualization.py D:\data\b4d_out\stream.zarr --codec zstd --open
```

A real camera is tuned the same way: the settings `b4d-tune` recommends are the `b4d-record` / `RecorderConfig`
settings (`--block`, `--chunks`, `--shards`, `--threads`, `--max-commits`); `b4d-record --tuned recommendation.json` applies them (flags
given explicitly still win). Candidates (thread counts around the physical cores, block depths that fit the pool's RAM budget) come from
`b4d.hardware`, which also supplies the default `--threads`. `b4d-record` with the camera's
`BlockSource` replaces the simulator.

The tuning order and the rules of thumb come from a measurement campaign summarised in `docs/WRITER_DESIGN.md`.
Correctness checks live in `validation/` and `tests/`.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

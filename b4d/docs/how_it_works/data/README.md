# Benchmark tables of Figures 1 and 2

The result tables that `../make_intro_figs.py` reads to draw Figure 2 of the README (`figs/fig00b_write_rates.png`).
The script only reads them; it measures nothing.

```
python make_intro_figs.py                 # light theme, reads this directory
python make_intro_figs.py --theme dark    # figs/dark/
```

## How the runs were made

Every row is one run of `tools/stream_sim.py`: a simulated camera (its frame size, bit depth and noise model) streams
frames into the b4d writer, which compresses them (q = 1, closed loop, three-pixel-mean predictor, Rice coding,
64 × 64 × 64 chunks) and assembles shards into an in-memory sink (no disk). Frames come from a pre-filled frame bank
(zero copy) unless a table says they are copied.

Machine: AMD EPYC 9754 (128 cores), 2 × NVIDIA RTX A6000 (84 SMs each, PCIe Gen4 x16). Most runs had 8 DIMMs
installed; the Kinetix multiscale SM ladders (`ladder_gpu_kinetix_multiscale_*`) ran after the memory was raised to
12 DIMMs.

Cameras (`camera` column):

| camera | sensor | frame | bit depth |
|---|---|---|---|
| `imx411` | Sony IMX411 | 14192 × 10640 | 16 |
| `imx811` | Sony IMX811 | 19200 × 12800 | 16 |
| `gsprint4521` | GSPRINT4521 | 5120 × 4096 | 8 |
| `kinetix` | Teledyne Kinetix | 3200 × 3200 | 8 |

Two kinds of table:

- **Sweep** (`sweep_*`): the writer's capacity against the compute it may use (`units`: CPU cores, or SMs per GPU).
  At each unit count a saturated `probe` (the camera offers more than the writer can take, so frames drop) measures
  the capacity, then a `confirm` run of at least 30 s at 0.95 × that capacity must drop no frame (on the pinned CPU
  multiscale sweep, `confirm2` and `confirm3` step down to 0.88 and 0.80 × if it does). A `monitor` run repeats the
  confirmation for at least 60 s at the largest unit count. CPU runs are each held to N physical cores (both SMT
  threads of each).
- **Ladder** (`ladder_*`): the sustained frame rate at full size (128 cores, or 84 SMs per GPU unless the name says
  otherwise). The offered rate starts at a rate known to sustain and rises × 1.10 per 30 s run until frames drop (or
  falls by the same step until a run sustains); capacity is bracketed by the last sustained and the first dropping
  offer.

## Files

| file | kind | target | what |
|---|---|---|---|
| `sweep_cpu_cores.csv` | sweep | CPU | cores 4–128, multiscale off (Figure 2a) |
| `sweep_cpu_cores_multiscale.csv` | sweep | CPU | cores 4–128, multiscale on (2e) |
| `sweep_gpu_sms_1gpu.csv` | sweep | 1 GPU | SMs 1–84, multiscale off (2b) |
| `sweep_gpu_sms_1gpu_kinetix512.csv` | sweep | 1 GPU | as above, Kinetix with 512-frame blocks as in the ladders; replaces the Kinetix curve of the previous table |
| `sweep_gpu_sms_2gpu.csv` | sweep | 2 GPUs | SMs per GPU 1–84, multiscale off (2b, dashed) |
| `sweep_gpu_sms_1gpu_multiscale.csv` | sweep | 1 GPU | SMs 1–84, multiscale on (2f); its Kinetix rows are not plotted (see below) |
| `sweep_gpu_sms_2gpu_multiscale.csv` | sweep | 2 GPUs | SMs per GPU 8–84, multiscale on (2f, dashed); its Kinetix rows are not plotted |
| `ladder_gpu_kinetix_multiscale_1gpu_sm{32,48,64,84}.csv` | ladder | 1 GPU | Kinetix, multiscale on, at 32, 48, 64 and 84 SMs: the Kinetix points of 2f (12 DIMMs) |
| `ladder_gpu_kinetix_multiscale_2gpu_sm{32,48,64,84}.csv` | ladder | 2 GPUs | the same with two GPUs: the dashed Kinetix points of 2f (12 DIMMs) |
| `ladder_cpu_zero_copy.csv` | ladder | CPU | zero copy (2c) |
| `ladder_gpu_zero_copy.csv` | ladder | 1 and 2 GPUs | zero copy (2c; the reference of 2d and 2h) |
| `ladder_cpu_copy.csv` | ladder | CPU | frames copied from a ring buffer into the writer's blocks on 32 threads, in byte ranges of at least 8 MB (2d) |
| `ladder_cpu_zero_copy_paired.csv` | ladder | CPU | zero copy, run back to back with `ladder_cpu_copy.csv` on the same build: the CPU reference of 2d |
| `ladder_gpu_copy.csv` | ladder | 1 GPU | frames copied on 32 threads (2d) |
| `ladder_gpu_copy_split.csv` | ladder | 2 GPUs | frames copied on 32 threads, each frame split by rows across the two GPUs (2d) |
| `ladder_cpu_multiscale.csv` | ladder | CPU | multiscale on (2g, 2h) |
| `ladder_gpu_multiscale.csv` | ladder | 1 and 2 GPUs | multiscale on (2g, 2h) |

The Kinetix multiscale sweeps on the GPU are limited by their saturated probe (well below what the ladders sustain),
so the Kinetix points of 2f come from the ladders at 32, 48, 64 and 84 SMs instead.

Multiscale: levels halve x, y and z down to about 256 pixels, each level compressed with b4d; on the GPU the
full-resolution level is encoded on the GPU and the coarser levels on the CPU.

## Columns

Ladder tables have four extra leading columns (`ngpu`, `step`, `copy_threads`, `block`); the rest are common.

| column | meaning |
|---|---|
| `ngpu` | GPUs used (0 = CPU) |
| `step` | ladder step: offered rate = start × 1.10^step |
| `copy_threads` | threads copying frames into the writer's blocks (0 = zero copy) |
| `block` | frames per writer block (= shard depth) |
| `timestamp` | start of the run (local time, ISO 8601) |
| `target` | `cpu` or `gpu` (where the compression runs) |
| `camera` | simulated camera (table above) |
| `units` | CPU cores (CPU) or SMs per GPU the kernels may use (GPU, of 84) |
| `stage` | `probe`, `confirm`, `confirm2`, `confirm3`, `monitor` (sweeps) or `ladder` |
| `fps_offered` | frame rate the simulated camera offered |
| `seconds` | requested duration of the run (0 = a fixed number of blocks: the unpinned probes) |
| `frames` | frames written |
| `dropped` | frames dropped (the writer had no free block); 0 = sustained |
| `sustained` | 1 if no frame dropped |
| `camera_limited` | 1 if the simulator itself delivered less than 97 % of the offered rate |
| `write_s_per_block` | seconds per block inside the writer's `write()` (compression + shard assembly) |
| `compress_s_per_block` | of which compression |
| `io_s_per_block` | of which shard I/O (assembly into the sink) |
| `gpu_upload_s`, `gpu_kernel_s`, `gpu_download_s`, `gpu_host_s` | GPU runs: per-block seconds for upload + layout, kernels, download and host mapping |
| `host_tiles` | GPU runs: tiles finished on the host |
| `writer_fps` | writer capacity in frames per second |
| `writer_gbs` | writer capacity in GB/s of raw frames (writer_fps × frame bytes / 10⁹) |
| `raw_gbs_achieved` | raw GB/s actually streamed during the run |
| `ratio` | compression ratio (raw bytes / stored bytes) |
| `lat_mean` | mean block latency (s) |
| `groups` | CPU runs: Windows processor groups the worker threads ran in |
| `wall_s` | wall time of the run, including start-up (s) |

How the figure uses them: for a sweep, the plotted capacity at each unit count is `writer_gbs` of the `confirm` run
with zero drops where there is one, else of the best zero-drop probe (`monitor` rows are not plotted; a point whose
confirmation dropped frames is drawn open). For a ladder, the sustained rate of each (camera, `ngpu`) is `writer_fps`
of the highest offer with `dropped` = 0, capped at the lowest higher offer that dropped frames.

# Codec speed table (Figure 1)

`codec_speed.csv` is the table that `../make_scatter_fig.py` reads to draw Figure 1 of the README
(`figs/scatter_encdec_doc.png`). The script only reads it; it measures nothing.

```
python make_scatter_fig.py                # reads data/codec_speed.csv, writes figs/scatter_encdec_doc.png
```

## How the runs were made

The rows are an extract of the codec speed benchmark (not in this repository), which measured each codec's own
encoding and decoding speed, on the same chunks for every codec, from 1 to 128 cores.

- Data: three stacks of the IMX411 sensor, 256 frames × 512 × 512 pixels each (sensor rows 4096–4607, columns
  3584–4095): a dark recording (`dark_pre`) and two flat-field levels (`flat_3.1`, `flat_50`).
- Chunks: c × c × c voxels for c = 64, 128 and 256 (256, 32 and 4 chunks per stack). b4d, B3D, SZ3, ZFP and Blosc2
  code the whole 3-D chunk; the 2-D codecs (JPEG-LS, JPEG-XL, JPEG 2000) its c frames of c × c, one call each.
- Timing: codec calls only (wall-clock timer around each call), into preallocated, pre-touched buffers. Copying the
  chunk out of the stack and converting to and from noise units for the external codecs are timed separately and not
  included. b4d's own variance-stabilizing transform is part of b4d's work and is included.
- Cores: one single-threaded worker process per physical core, pinned, spread evenly over the CPU's L3 domains; all
  workers start together after their buffers, data and codec are warmed up, and encoding and decoding are separate
  phases. Machine: AMD EPYC 9754 (128 cores), Windows 11, High performance power plan, nothing else running. b4d built
  with clang-cl -O3.

Settings (`setting` column), each the q = 1 equivalent of its codec:

| setting | codec | parameters |
|---|---|---|
| `m1_a3_xv` | b4d | q = 1, mode 1, three-pixel-mean (`above3`) predictor, exact variance-stabilizing transform |
| `b3d_m1` | B3D | q = 1, mode 1 (b4d's implementation of the original B3D predictor) |
| `sz3_nu0.5` | SZ3 | absolute error bound 0.5 in noise units (= q/2 at q = 1) |
| `zfp_nu0.5` | ZFP | fixed-accuracy tolerance 0.5 in noise units |
| `jls_nu0.5` | JPEG-LS | near-lossless bound 0.5 in noise units (NEAR = 64 on the scaled integers, see below) |
| `jxl_a` | JPEG-XL | distance 0.085, effort 3, in noise units (tuned to b4d's q = 1 ratio) |
| `j2k_a` | JPEG 2000 | PSNR 65 dB, in noise units (tuned to b4d's q = 1 ratio) |
| `blosc2_lz4` | Blosc2 (lossless) | byte shuffle (2-byte items), LZ4, clevel 5 |
| `blosc2_zstd1`, `blosc2_zstd3` | Blosc2 (lossless) | byte shuffle (2-byte items), Zstd, clevel 1 or 3 |

Noise units: pixel values in ADU mapped by the sensor's variance-stabilizing transform (from its photon-transfer
calibration) so that the noise has unit standard deviation; the decoded values are mapped back to the nearest ADU
value. SZ3 and ZFP code the noise units as float32; the 2-D codecs code them as 16-bit integers
round((T − T₀) × 128), so the PSNR of JPEG 2000 and the distance of JPEG-XL are relative to the data's range.

## Columns

| column | meaning |
|---|---|
| `setting` | codec setting (table above) |
| `level` | stack: `dark_pre`, `flat_3.1` or `flat_50` |
| `size` | chunk edge c (voxels) |
| `n` | cores (worker processes): 1 or 128 |
| `ratio` | compression ratio (raw bytes / compressed bytes) over all chunks the workers coded |
| `codec_enc` | encoding speed: sum over workers of raw bytes / codec time, MB/s (10⁶ bytes/s) |
| `codec_dec` | decoding speed, the same for decoding, MB/s |

How the figure uses them: each point is the mean over the three stacks of `ratio` and of `codec_enc` (a, b) or
`codec_dec` (c, d) for one setting and chunk size; 1-core panels in MB/s, 128-core panels in GB/s.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

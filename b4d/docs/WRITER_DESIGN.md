<sub>B4D CODEC · ENGINEERING RECORD</sub>

# b4d Optimization History

<sub>Writer design record (`WRITER_DESIGN.md`) of the b4d codec and of the native zarr writer built around it.</sub>

Every layer of the b4d pipeline, from the noise model to the zarr writer: the variants considered, the options tried
within each, what changed, and how each one affects the speed, compression ratio or accuracy of b4d. This is the
engineering record of the codec and of the native zarr writer built around it.

Result tables under `docs/how_it_works/data/` ship with the repository (its README describes them). Paths `bench_scaling/...` and "the scaling report" below refer to the benchmark scripts and results of the separate validation workspace
(not in this repository; README, Reproducibility).

**Machine** AMD EPYC 9754, 128 cores / 256 threads, 768 GB RAM (8 × 96 GB DDR5-4800), Windows 11 &ensp;·&ensp; **GPUs** 2 × NVIDIA RTX A6000
(PCIe Gen4 x16, WDDM driver) &ensp;·&ensp; **Error unit** σ = camera noise (one unit of the variance-stabilizing
transform) &ensp;·&ensp; **Final build** codec campaign 2026-11-15, write-rate ladders 2026-11-16 build; clang-cl, format v10 / v11 (8 Rice streams)

> [!NOTE]
> **Defaults.** The default is mode 1, predictor `above3`, Rice coder, exact-VST table
> (`exact_vst="variance"`) at q = 1. The earlier acquisition default was mode 2, predictor `above`, Rice coder,
> generalized Anscombe transform; files written with it still decode. Some measurements below were made with the earlier
> default and are labelled as such.

---

<sub>OVERVIEW</sub>

## The pipeline, layer by layer

*Data flows from layer 01 (top) to layer 12 (bottom). Select a layer number to go to its table.*

**Legend** (used in every Change column): ✓ kept / default &ensp; ○ optional, off by default &ensp; ✕ rejected &ensp;
↷ superseded &ensp; … pending (untested or not installed)

| # | Layer | Now | Alternatives tried |
|---|---|---|---|
| [01](#l1) | **Noise model** | Gain · offset · read-noise model; its own model at each pyramid level | ✕ Per-pixel maps &ensp; ○ Per-pixel list |
| [02](#l2) | **VST and quantization** | Exact-VST table ("variance"), linear below the offset, q = 1, closed loop | ↷ Anscombe only &ensp; ✕ Clamp below offset &ensp; ✕ "std" objective &ensp; ○ Dither |
| [03](#l3) | **Predictor** | `above3` (mean of the 3 pixels above) | ↷ `above` &ensp; ○ `left_above` &ensp; ○ `mean3d` &ensp; ✕ Temporal |
| [04](#l4) | **Residuals** | int16 zigzag; raw-tile fallback; escapes | ↷ B3D mapping &ensp; ○ Lossless mode |
| [05](#l5) | **Entropy coder** | Rice, groups of 128, 8 streams (format v10 / v11), exact k planner, table decode | ○ Huffman &ensp; ○ rANS &ensp; ✕ Lockstep emit &ensp; ✕ Merged put |
| [06](#l6) | **Chunk geometry** | 64³ chunks; banded gather; shards of 50–400 MB | ✕ 128³ / 256³ &ensp; ↷ 32 × 64 × 64 |
| [07](#l7) | **Container** | B4DP → Blosc2 chunk → zarr v3 shard | ✕ HDF5 filter compatibility &ensp; ↷ Shared blosc1 context |
| [08](#l8) | **CPU implementation** | AVX2 C core + single-core patch; clang-cl -O3, portable (no -march) | ✕ AVX-512 &ensp; ✕ PGO &ensp; ✕ Fused decode &ensp; ✕ Core pinning &ensp; ✕ LTO |
| [09](#l9) | **GPU implementation** | One CUDA kernel, byte-identical to C; zero-copy sliced upload; 2-GPU dispatch | ↷ PyTorch prototype &ensp; ○ Copy mode on 2 GPUs (12–27 % below zero copy) &ensp; … TCC |
| [10](#l10) | **Writer / streaming** | Native C writer, one call per block, zero-copy ingest, direct I/O | ↷ zarr-python &ensp; ○ Frame copy (40–46 % below zero copy on the CPU) |
| [11](#l11) | **Multiscale** | 2 × 2 × 2 means; level 0 on the GPU, the other levels on the CPU | ✕ All levels on the GPU &ensp; ✕ Overlapped pyramid &ensp; … uint8 downsample path |
| [12](#l12) | **Versus other codecs** | The only lossy codec tested that keeps gain, read noise and offset | ↷ Compared: B3D, SZ3, ZFP, JPEG family, Blosc2 |

---

<sub>OVERVIEW</sub>

## What was tested, by kind

*The same experiments, grouped by what each was trying to improve. The tag is the layer number and links to it.*

### Fidelity and bias

*Does decoded data keep the sensor's statistics?*

- [01](#l1) Read noise inside the noise model.
- [01](#l1) Per-pixel maps rejected: hot pixels biased up to +23 DN.
- [02](#l2) Linear continuation below the offset instead of clamping.
- [02](#l2) Dither: bias 0.50 → 0.05 DN, for about 12 % less ratio.
- [03](#l3) `above3` is unbiased without dither; `above` is not.

### Compression ratio

*Fewer bits at the same error bound.*

- [02](#l2) Exact VST "variance": dark-frame ratio −9 % for an added variance of exactly q²/12.
- [03](#l3) Predictors at q = 1: `above3` 6.33, `left_above` 6.11, `above` 5.74.
- [05](#l5) Rice against Huffman and rANS: 5–10 % less ratio for 2–3 times the speed.
- [06](#l6) Chunk size: larger chunks gain 0.2–0.3 %.
- [04](#l4) Lossless mode: 2.56 against 2.63 for Blosc2 + Zstd.

### Algorithm restructuring

*Same output, less work per pixel.*

- [05](#l5) Branch-free k planner; table-driven decode.
- [08](#l8) Per-row LUT; rows streamed into the coder.
- [08](#l8) Fused decode: no gain, dropped.
- [09](#l9) GPU: residuals computed once; bit pack in registers.

### SIMD and instruction set

*Vectorizing the inner loops.*

- [08](#l8) AVX2 quantizer (+41 % at 1 thread).
- [08](#l8) AVX2 table gathers, encode and decode.
- [05](#l5) AVX2 vector Rice emit.
- [08](#l8) AVX-512: no difference on Zen 4.

### Memory and data movement

*Allocations, copies and cache footprint.*

- [08](#l8) Per-worker arenas; per-thread buffers kept between calls.
- [06](#l6) Banded gather: 2.7 → 14.7 GB/s.
- [10](#l10) Zero-copy ingest (a frame copy costs 40–46 % on the CPU at 128 cores).
- [09](#l9) Registered, sliced GPU upload: 11.8 → 21.5 GB/s.

### Parallelism and scale-out

*Using all CPU cores and both GPUs.*

- [08](#l8) Process/thread split, persistent pool, batched calls.
- [08](#l8) Core pinning and SMT: no gain.
- [09](#l9) 2 GPUs: 1.86–2.02 times one GPU; copy mode on 2 GPUs (frames split by rows) costs 12–27 %.
- [11](#l11) Pyramid split between GPU and CPU: within one ladder step (≤ 3.4 %) on one or two GPUs, against about −40 %
  with all levels on the GPU; on the CPU writer the pyramid costs 20–23 % at 128 cores.

### Build and compiler

*Same source, different code generation.*

- [08](#l8) MSVC /O2 /GL /LTCG (reference build).
- [08](#l8) clang-cl -O3 -mtune=znver4: encode +13–19 %.
- [08](#l8) PGO rejected; -march=znver4 not faster for b4d (portable build kept); AVX-512 kernels rejected; LTO no gain.

### Platform and measurement

*Limits outside the codec, and measurement method.*

- [09](#l9) WDDM page-locked memory limit; TCC not tried.
- [08](#l8) Power plan: Balanced caused phases at 0.3–0.6 times the throughput.
- [10](#l10) fps ladder; hang watchdogs in the simulator.

---

<a id="l1"></a><sub>LAYER 01</sub>

## Noise model

> **Now:** each camera is described by σ² = gain · (x − offset) + (gain · read noise)². Every pyramid level gets its own
> model, so the error bound means the same thing at every level.

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Global model** | Read noise inside the model | ✓ Kept | Without it, dark regions are coded finer than their noise and q means different things at different intensities, which lowers the ratio on dark data. At 5 e⁻ with a read noise of 3 e⁻, σ is 3.7 e⁻, not 2.2 e⁻. |
| | Per-level model in the pyramid | ✓ Kept | Averaging n pixels divides the variance by n, equivalent to a camera with gain / n and read noise · √n. Level k is quantized with that model, so q keeps its meaning and the error bound holds at every level. |
| | Model accuracy | ✓ Lesson | The ratio depends directly on the model. For the Sony IMX811 (VPS-245MX2), a campaign table of 12.30 e⁻/ADU underestimated the noise 1.7 times and made q = 1 effectively lossless on dark frames. The fitted model, gain 7.25 e⁻/ADU (conversion 0.1379), offset 2 ADU, read noise 6.89 e⁻, predicts the measured dark-frame noise (0.955 ADU) to 0.1 %. |
| **Per-pixel** | Per-pixel noise maps | ✕ Rejected | Slightly higher ratio, but hot pixels were biased up to +23 DN. Adding dither removed the bias but created thousands of false spikes. |
| | Per-pixel list of noisy pixels | ○ Optional | Coarser steps only for the noisiest ~1 % of pixels. Ratio +2.5 % at q = 1 and +13 % at q = 4; speed −4 to +6 %. CPU only. |
| **Edge cases** | Saturated pixels | ✓ Accepted | At q = 1 the top value decodes up to ~150 DN low, because the step is wide there. |
| | Defect detection on decoded data | ✓ Accepted | Pixels just under a spike or hot-pixel threshold can cross it after the added noise, so detectors need about 1 DN of margin. |

---

<a id="l2"></a><sub>LAYER 02</sub>

## Variance-stabilizing transform and quantization

> **Now:** a 65,536-entry table maps each sample to a noise-normalized value. The table is an exact VST fitted so that
> quantizing adds exactly q²/12 of variance; it is continued linearly below the offset. Quantization is closed loop at
> q = 1, with an error bound of 0.50σ.

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Transform shape** | Anscombe √ form, applied by table | ✓ Kept (base) | The +3/8 term keeps the variance near 1 at low counts. A table gives identical output on every compiler and on the GPU, with no per-pixel arithmetic. |
| | Linear below the offset (instead of clamping) | ✓ Kept | Clamping biased the means of dark regions up by ~0.4 read noise and discarded the information below the baseline. |
| | **Exact VST, "variance" objective** | **✓ Kept at q = 1** | **The Anscombe step on dark data is ~8 % wider than σ. The exact table adds exactly q²/12 of variance: dark-frame ratio −9 %, bright ratio unchanged. Encode speed unchanged; decode 15–18 % slower on one thread and equal at 32 threads.** |
| | Exact VST, "std" objective | ✕ Rejected | Targets a standard deviation of 1 instead of an exact added variance. |
| | Retuned q or a hybrid dark target | ✕ Closed | q comes in 0.25 steps, so it cannot match Anscombe exactly, and a larger q coarsens the bright signal. |
| **Quantizer** | Closed loop (mode 1) against open loop (mode 2) | ✓ Mode 1 | Closed loop bounds the error at 0.50 noise units against 5.88 open loop. With the `above` predictor it was ~15 % slower than mode 2 for a ratio within 2 % at q = 1; with `above3` it gives +14 % ratio at q = 1. |
| | Lossless (mode 0) | ○ Optional | 2.3–3 on dim data, about the same as Zstd with shuffle. |
| **Bias control** | No dither with the `above` predictor | ✕ Fails | Rounding leaves a −0.455 DN offset. In mode 2 the mean error at low ADU reaches about 0.5 DN at q = 2 on a camera with one DN of read noise. |
| | Subtractive dither | ○ Fallback | Maximum bias 0.50 → 0.05 DN, for ~12 % less ratio, encode 15–36 % slower and decode 13–24 % slower. Not needed with `above3`. |

<details>
<summary>Details and measurements</summary>

A camera pixel x (DN) relates to electrons by e = (x − offset) / conversion, with variance e + rn² (shot noise plus read
noise, in e⁻). The generalized Anscombe transform T(x) = 2 √(e + rn² + 3/8) makes the noise unit variance; one unit of T
is one σ. The decoder reconstructs the centre of the bin, so the error is bounded by q/2 σ in the transform domain (plus
half a DN from integer rounding); measured maximum |error| = 0.50σ at q = 1 in every test. The transform and quantization
are one 16-bit lookup table per (camera, q) pair, cached and shared across threads; the decoder uses the inverse table.
Mode 1 uses a 32-bit fixed-point table.

</details>

---

<a id="l3"></a><sub>LAYER 03</sub>

## Predictor

> **Now:** `above3` predicts each pixel as the mean of the three pixels above it, using reconstructed values. Sensor
> noise spreads the rounding offsets and acts as dither, so it stays unbiased without added dither. Prediction stays
> within a plane, so chunk depth is a storage choice only.

**Ratio by predictor at q = 1**<br>
<sub>Closed loop (mode 1).</sub>

| Predictor | Without dither | With dither |
|---|---|---|
| `above3` | 6.33 | 6.22 |
| `left_above` | 6.11 | 6.02 |
| `above` | 5.74 | 5.66 |

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **In-plane** | `above3` | ✓ Kept | Highest ratio; unbiased without dither; encodes 26–39 % faster than the dithered alternative. 6.7 on 488 nm light-sheet data at q = 1, mode 1 (`COMPARISON.md`). No left tap, so rows vectorize; the plane boundary is not a restart. Written as `mean3d` with plane height 0. |
| | `above` | ↷ Superseded | Lowest cost per pixel, but biased without dither. 5.7 at 1σ on the dim phantom. |
| | `left_above` | ○ Optional (archival) | +5–10 % ratio over `above` (6.1–6.3 on the dim phantom), but serial along the row: 5 % slower with Rice, up to 2 times slower under full load with the heavier coders. |
| **Across planes** | `mean3d` | ○ Optional | Rounded mean of the 3 pixels above and the previous plane's 3 × 3 patch, inside the block (format v4). +17–30 % on noise-dominated data (6.0 on the dim phantom at q = 1), at 2 times the single-thread encode cost of `above`. CPU and CUDA encoders, either coder. |
| | Temporal (previous frame) | ✕ Rejected | +2–7 % on realistic motion, but every frame depends on the previous one: no random access, no parallelism across frames, error propagation. |

---

<a id="l4"></a><sub>LAYER 04</sub>

## Residual handling

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Mapping** | int16 zigzag | ✓ Kept | Replaces the B3D mapping; one step before the coder. |
| | Raw-tile fallback | ✓ Kept | A tile whose residual leaves int16, or that does not compress, is stored raw. This bounds the worst case at the raw size. |
| | Tile independence | ✓ Verified | Splitting along tile edges gives the same bytes, with no bias at edges or corners. |
| **GPU** | Rare tiles finished on the host | ✓ Kept | Overflow, all-zero and incompressible tiles are flagged for the CPU. This happened 0 times in every run. |
| **Lossless** | b4d lossless | ○ Optional | 2.56 against 2.63 for Blosc2 + Zstd, so it does not reach the lossless ratio of Blosc2. |

---

<a id="l5"></a><sub>LAYER 05</sub>

## Entropy coder

> **Now:** Rice, in groups of 128 with a 4-bit k, sparse and all-zero group forms, and 8 interleaved streams (format v10,
> or v11 with the exact VST). It needs no tables or histogram pass. Huffman and rANS keep 4 streams; Rice blocks with 4
> streams (v3–v9) still decode.

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Coder** | Rice | ✓ Kept | 2–3 times the speed of the alternatives, for 5–10 % less ratio. Its cost scales with output bits, so brighter data encodes slower (52 MB/s per thread at 10–1,000 e⁻ peaks against 63 at 10–50 e⁻ in the early multi-process runs). |
| | Huffman (canonical, per block) | ○ Optional | ~+3 % ratio (~5 % at high q) at ~55 % of the encode speed. |
| | rANS, 3 contexts | ○ Optional (archive) | +1 % ratio at q = 1, up to +50 % at q = 4; ~50 % of the Rice encode speed. For lossless archives and pyramid levels written offline. |
| **Choosing k** | Branch-free planner, exact k − 1 / k / k + 1 check | ✓ Kept | The planner was 65 % of encode time. 190 → 310 MB/s per core, bit-identical (hash over 432 configurations). |
| **Writing bits** | Branchless 8-byte writer | ✓ Kept | Emit −20 % time, codec +4 %. |
| | Bit writer held in registers | ✓ Kept | Removes pointer aliasing: +5–10 % at 1 and 128 threads. |
| | One flush per 4 or 2 codes | ✓ Kept | +13–15 % when the codes fit in 64 bits. |
| | AVX2 vector emit (8 codes at once) | ✓ Kept | Emit 1.57 → 1.17 ns/px; one-core encode +14 %. |
| | 4 streams written in lockstep | ✕ Rejected | −10 %: the emitter is instruction-bound, not latency-bound. |
| | Unary + remainder in one put | ✕ Rejected (2 tests) | −9 % in one test, emit 1.57 → 1.93 ns/px in the other. |
| | 3-codes-per-flush tier | ✕ Rejected | +0.3 %, within noise. |
| | Byte-swapped 32-bit flush | ✕ Rejected | Neutral. |
| **Reading bits** | Branch-free refill, streams read in step | ✓ Kept | Decode +30–50 % per thread. |
| | Position-based reader | ✓ Kept | An 8-byte load and a leading-zero count per code: Rice decode 3.03 → 2.25 ns/px. |
| | Table decode, 10 / 11 / 12-bit tables | ✓ Kept (10 bits) | Rice decode 2.25 → 1.50 ns/px; one-core decode 663 → 883 MB/s. The three widths perform the same, so the smallest was kept. |
| **Format** | 8 Rice streams (format v10 / v11, 2026-11-15) | ✓ Kept | Decodes ~6.5 % faster per core than 4 streams (`src/c/b4d_blosc2.c`). Changed every compressed hash and no decoded array; the sizes of streams 0–2 are in the block header, those of streams 3–6 open the payload. |

<details>
<summary>Details and measurements</summary>

Rice details: a 4-bit code per group selects k (0..13), a sparse group (positions + Elias-gamma values, for groups with a
few non-zeros) or an all-zero group; runs of zero groups within a stream are one 7-bit token. One pass plans and one
emits; the planner picks k from the group mean and checks k − 1, k and k + 1 exactly. Huffman: per-block histogram,
canonical code lengths, escapes for large values, two passes plus table transmission. rANS: 64-bit state, static
per-block frequency tables normalized to 12 bits, up to three local-activity contexts used only when the estimated gain
exceeds 2 %, division-free encode through reciprocal tables.

**Coders measured on one core**<br>
<sub>Mode 2, `above` unless noted (earlier default, before the single-core patch).</sub>

| Coder | Ratio at 1σ | Ratio at 2σ | Encode (MB/s) | Decode (MB/s) |
|---|---|---|---|---|
| Rice | 5.7 | 8.1–8.4 | 310 (190 before the planner rewrite) | ~300 |
| Huffman | 5.9 | 8.5 | ~175 | ~250 |
| rANS (`left_above`) | 6.2–6.3 | 9.1 | ~115 | ~200 |

Per-stage profile of the Rice encoder, one core, 1 Mpixel block, before → after the planner work: LUT 0.5 → 0.5 ms,
residual 0.5 → 0.5 ms, plan 7.2 → 2.4 ms, emit 2.8 → 3.0 ms, total 11.0 → 6.5 ms.

</details>

---

<a id="l6"></a><sub>LAYER 06</sub>

## Chunk and tile geometry

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Chunk size** | 64³ (512 KB at uint16, fits the 1 MB L2 per core) | ✓ Kept | Uniform access from any axis, and the highest 128-core scaling efficiency. |
| | 128³ / 256³ | ✕ Rejected | Ratio +0.2–0.3 % (6.21 → 6.23 at q = 1). Encode efficiency at 128 cores in the final build: 0.59 (64³), 0.55 (128³), 0.44 (256³); decode 0.77, 0.42, 0.41. An earlier measurement (before the single-core patch) gave 0.55 → 0.25 / 0.21. A 128³ chunk wastes 127/128 of a decode when one plane is read. |
| | 32 × 64 × 64 | ↷ Superseded | 2.8 % (q = 2) to 7.6 % (q = 1) more writer throughput than 64³ on the Sony IMX411; 64³ was accepted for uniform access from any axis. |
| | Below 32³ | ✕ Rejected | Per-chunk overhead dominates. (16–32, 128, 128) suits readers that view xy planes only. |
| **Gather** | Bands of up to 16 tiles instead of per-tile | ✓ Kept | 64 × 64 tiles were gathered as 128-byte fragments 28 KB apart. A band (one chunk row by up to 16 chunks in x) is copied with one `memcpy` per (plane, row) of 2 KB and the tiles are cut from the L2-resident band: 2.7 → 14.7 GB/s. |
| | One-tile bands for large tiles | ✓ Kept (16-bit) | +0.7–1.4 % at 128 threads; 1 % slower for 8-bit, which keeps 4-tile bands. |
| **Blocks and shards** | Shards of 50–400 MB, at least 8 per block | ✓ Kept | 84 processes with 128-row shards wrote 3.7 MB files at 524 MB/s; this size target reaches 1,000+ MB/s on the same drive. The y/x shape of the shard has no effect (±8 % across ±5 times the footprint for all cameras). |
| | Block depth ≈ 0.5 s of frames | ✓ Kept | Doubling the block depth from 64 to 128 frames gave +31 % on the Sony IMX811. N ≈ fps × 0.5, rounded to a multiple of the chunk depth (and of 2^levels with a z-downsampled pyramid); block memory = N × frame bytes. |
| | Full-frame shards with the old task split | ✕ Rejected | Superseded by band tasks, which make parallelism independent of the shard geometry. |

<details>
<summary>Details and measurements</summary>

**Per-camera settings of the October 2026 tuning campaign**<br>
<sub>128 threads, one commit in flight, q = 2, dim data, earlier default codec, zero copy, in-memory sink.</sub>

| Camera | `--shape` | `--block` | `--ring` | `--shards` | Sustained frame rate (fps) | Pixel rate (Gpixel/s) | Raw rate (GB/s) | Ratio |
|---|---|---|---|---|---|---|---|---|
| Sony IMX411 | 10640,14192 | 64 | 64 | 64,1408,3584 | 185.7 | 28.0 | 56.1 | 7.44 |
| Sony IMX811 | 12800,19200 | 128 | 64 | 128,1280,3840 | 120.8 | 28.5 | 59.4 | 7.44 |
| Teledyne Kinetix | 3200,3200 | 256 | 1024 | 256,640,3200 | 2,617 | 26.8 | 53.6 | 7.50 |
| Tucsen Aries 6504 | 2048,2048 | 1024 | 2048 | 1024,512,2048 | 6,297 | 26.4 | 52.8 | 5.38 |

In that campaign the Teledyne Kinetix ran as a 16-bit container holding 8-bit values, limited by the Python loop; at
q = 1, capacity was 6–23 % lower and the ratio 4–5.3, with the same configuration. The final rates (q = 1, default codec)
are in [layer 10](#l10).

</details>

---

<a id="l7"></a><sub>LAYER 07</sub>

## Container

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Layout** | B4DP → Blosc2 chunk → zarr v3 shard | ✓ Kept | Each block header describes itself (mode, q, model, predictor, coder, VST curve). Overhead is 323 bytes per chunk (0.3 %). |
| | Format versions | ✓ Kept | Since 2026-11-15 every Rice block is written as v10 (8 streams), or v11 with the exact VST; readers from before v10 cannot decode them. Huffman and rANS blocks keep their versions, and every earlier version (v1–v9) still decodes. The other versions (v4 `mean3d`, v5 B3D, v8 dither, v7 noise list) are written only when their option is selected. |
| **Blosc in the writer** | Per-worker blosc2 context | ✓ Kept | c-blosc2's Blosc1-format entry point (`blosc1_compress`) sits behind a global mutex, which serialized all workers. Each worker owns a `blosc2_context`, and the writer rewrites the 32-byte-header chunk into the 16-byte-header Blosc1 layout (block starts shifted, compressed streams verbatim, run-length and all-zero cases re-emitted as an LZ4 block or a Zstd RLE frame), which numcodecs / zarr-python, tensorstore and neuroglancer decode. |
| | 16 KB blocks for bitshuffle + Zstd | ✓ Kept | At ~24 KB blocks, throughput fell from ~45 to 4–9 GB/s. |
| **Compatibility** | B3D HDF5 filter | ✕ Not supported | Different format. B3D is reproduced bit for bit only as a separate option, `predictor="b3d"`. |

<details>
<summary>Details and measurements</summary>

Each zarr chunk is a `B4DP` container: a 7-byte header (magic, version, dtype, ndim), the chunk shape, the number of pieces
and their sizes, then a Blosc2 chunk whose payload is the b4d bitstream. The zarr v3 codec entry is
`{"name": "b4d", "configuration": {q, mode, conversion, offset, read_noise, predictor, entropy, exact_vst}}`. The native
writer emits exactly the bytes `b4d.compress()` produces (verified byte for byte), so files written by either path are
interchangeable.

</details>

---

<a id="l8"></a><sub>LAYER 08</sub>

## CPU implementation and compilers

> **Now:** an AVX2 C core with cached tables, int32 fixed point, row-streamed mode 1 and per-worker arenas, plus the
> single-core patch. The recommended build is clang-cl with `B4D_CFLAGS="/clang:-O3 /clang:-mtune=native"` (`setup.py`
> adds `/arch:AVX2`, no -march, and drops `/GL` and `/LTCG`); it is bit-identical to MSVC. The final codec campaign
> (2026-11-15) used clang-cl 20.1.8 with `-O3 -mtune=znver4`.

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Parallelism** | Process/thread split (writer GIL) | ✓ Kept | Python writer at ~60 MB/s per thread with 12 threads per process: band copies serialized around the C call. 8 × 12 → 12 × 8: +35 % per thread; the native writer then removed the GIL from the path. |
| | Processor groups | ✓ Kept | Windows children inherit the parent's processor group, so all writer processes ran on 64 CPUs. `SetThreadGroupAffinity` per pool thread at creation: 128 threads run on 128 CPUs. |
| | Persistent shared pool | ✓ Kept | Two blocks in flight spawned 2 × 128 threads plus copy threads. A persistent pool with a job queue: compress 0.54 → ~0.40 s per block with overlap. |
| | Batched calls (one per worker) | ✓ Kept | 40.3 → 43.3 GB/s at 128 cores. |
| | Core pinning | ✕ Rejected (opt-in only) | Never faster; up to −8 % at 64–96 threads. |
| | SMT (192 / 256 threads in the writer) | ✕ Rejected | Compress 22–44 % slower. |
| **Memory** | Per-worker arenas instead of ~37,000 `malloc` calls | ✓ Kept | The Windows heap lock serialized 128 threads: 5–9 s → 0.66 s per block. |
| | Per-thread buffers kept between calls; contiguous tiles not copied | ✓ Kept | Overhead outside the codec 0.5 → 0.34 ns/px; one-core encode +8 % (+16 % on 256³). |
| **Algorithm** | Mode-1 per-row LUT | ✓ Kept | 18.5 → 31 GB/s at 128 threads. |
| | Mode-1 rows streamed into Rice | ✓ Kept | Removed a plateau between 112 and 128 threads: +14 %. |
| | LUT fused into the prediction loop | ✕ Rejected | Neutral. |
| | Fused reconstruction + inverse VST | ✕ Rejected | Bit-identical, but no gain at 1 or 128 cores. |
| **SIMD** | AVX2 quantizer (÷3 as a multiply) | ✓ Kept | +41 % at 1 thread; 35.9 → 48.2 GB/s at 128. |
| | AVX2 gather for the decode VST table | ✓ Kept | Inverse VST 1.62 → 0.41 ns/px; one-core decode +30 %. |
| | AVX2 gather for the encode LUT | ✓ Kept | 0.52 → 0.46 ns/px. |
| | /arch:AVX512 for the whole build | ✕ Rejected | No difference: Zen 4 splits 512-bit operations in two. |
| **Compiler** | MSVC /O2 /GL /LTCG | ↷ Reference | Reference build. |
| | clang-cl default / -O3 | ↷ Superseded | Both about the same as -mtune. An earlier test (before the single-core patch) gave +30 % encode per thread (440 → 571 MB/s at q = 2) and +24 % at 16 threads over MSVC, with an identical bitstream. |
| | **clang-cl -O3 -mtune=znver4** | **✓ Kept** | **Encode +13–19 % over MSVC, decode −1 to −2 %; bit-identical. At 128 cores, encode 75 against 66 GB/s (build sweep below).** |
| | Profile-guided optimization | ✕ Rejected | Trained on the phantom at several q: q = 2 encode −7 %, q = 1 unchanged, lossless +7 %, decode −5 to −9 %. |
| | AVX-512 kernels (planner, decoder) | ✕ Rejected | Byte-identical but slower: decode −14 to −24 %, encode −2 to −3 % (Zen 4 splits 512-bit operations). |
| | -march=znver4 | ✕ Not used | B3D +14–24 %, b4d ±2–4 % (noise); the portable build is kept. |
| | LTO (-flto=thin) | ✕ No gain | About 0 %. |

### Single-core profile

Per-stage timers showed that decode was not memory-bound on one core: Rice decoding and an inverse-VST loop dominated.
Every change kept the output bit-identical, checked by a 400-case fuzz and the full test suite on every build.

**Decode stage time, before and after (ns/px)**<br>
<sub>One core, 128³ chunks.</sub>

| Stage | Before (ns/px) | After (ns/px) |
|---|---|---|
| Rice decode | 3.03 | 1.50 |
| Inverse VST table | 1.62 | 0.41 |
| Reconstruction | 0.34 | 0.34 |
| Total | 4.99 | 2.25 |

**Decode speed, one change at a time (MB/s)**<br>
<sub>One core. The fused reconstruction was not kept.</sub>

| Step | Decode (MB/s) |
|---|---|
| Base | 400 |
| VST gather | 527 |
| Position-based reader | 663 |
| Fused reconstruction (not kept) | 629 |
| Table decode, 11 bits | 786 |
| Table decode, 10 bits | 883 |

**One-core speed by build (MB/s, minimum–maximum)**<br>
<sub>Range over 3 signal levels × 3 chunk sizes.</sub>

| Build | Encode (MB/s) | Decode (MB/s) |
|---|---|---|
| Base (MSVC) | 527–605 | 366–403 |
| MSVC + patch | 673–801 | 796–932 |
| clang default | 770–933 | 767–899 |
| clang -O3 | 760–934 | 751–903 |
| clang -O3 -mtune=znver4 | 787–951 | 788–917 |

### 128-core scaling of the three builds

Builds run back to back at each core count, median of 3 rounds, 128³ chunks, codec time only (4-stream format, before
v10). This sweep compares the builds with each other; the 128-core rates of the final build are in *Details and
measurements* below.

**Encode and decode rate by build (GB/s)**<br>
<sub>The optimized builds decode at ~2.25 times the base rate up to 64 cores.</sub>

| Build | Direction | 1 core (GB/s) | 32 cores (GB/s) | 64 cores (GB/s) |
|---|---|---|---|---|
| Base | Encode | 0.59 | 17.4 | 33.3 |
| MSVC + patch | Encode | 0.72 | 21.0 | 41.2 |
| clang + patch | Encode | 0.85 | 23.8 | 45.2 |
| Base | Decode | 0.39 | 12.3 | 24.7 |
| MSVC + patch | Decode | 0.87 | 27.9 | 55.7 |
| clang + patch | Decode | 0.88 | 27.8 | 55.4 |

> [!WARNING]
> **Measurement lesson.** Under the Balanced power plan, whole runs slowed to 0.3–0.6 times the throughput for minutes.
> Running the builds back to back at each core count, and later switching to the High performance plan, gave rounds
> that agree to within a few percent.

<details>
<summary>Details and measurements</summary>

**Final build (2026-11-15)**<br>
<sub>clang, 8 Rice streams, format v11 (exact VST), mode 1, `above3`, Rice. Codec time only, mean over a dark stack and
two flat-field stacks of the Sony IMX411 model. Source: the scaling report.</sub>

| q | Chunk | Ratio | Encode, 1 core (MB/s) | Encode, 128 cores (GB/s) | Encode efficiency at 128 | Decode, 1 core (MB/s) | Decode, 128 cores (GB/s) | Decode efficiency at 128 |
|---|---|---|---|---|---|---|---|---|
| 1 | 64³ | 6.21 | 803 | 60.4 | 0.59 | 852 | 84.5 | 0.77 |
| 1 | 128³ | 6.23 | 827 | 58.7 | 0.55 | 978 | 52.7 | 0.42 |
| 1 | 256³ | 6.23 | 949 | 53.5 | 0.44 | 1,011 | 53.3 | 0.41 |
| 2 | 64³ | 9.34 | 852 | 73.4 | 0.67 | 834 | 84.3 | 0.79 |
| 2 | 128³ | 9.38 | 891 | 71.7 | 0.63 | 965 | 50.4 | 0.41 |
| 2 | 256³ | 9.38 | 970 | 68.3 | 0.55 | 1,000 | 52.0 | 0.41 |

Decode on 128³ and 256³ chunks levels off at 50–53 GB/s at 128 cores, while 64³ chunks reach 84–85 GB/s, which suggests a
memory-bandwidth limit for the larger chunks.

</details>

---

<a id="l9"></a><sub>LAYER 09</sub>

## GPU (CUDA) implementation

> **Now:** one CUDA kernel with one thread block per tile and 8 Rice streams (format v10 / v11). It runs mode 1 with
> `above3` (the default) or `above`, and mode 2 / 0 with `above` or `mean3d`; `predictor="b3d"` runs on the C encoder
> only. Its output is byte-identical to the C encoder, checked on the first block of every recording. Blocks are
> page-locked in place and uploaded in slices that overlap the kernels. With 2 GPUs (`MultiGpuEncoder(cam, devices=(0, 1),
> split=...)`), `split="blocks"` (default) sends each block to the GPU with the fewest blocks in flight, and
> `split="rows"` shares each block out as bands of tile rows.

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Architecture** | PyTorch prototype → one CUDA kernel | ↷ Superseded | The prototype was bit-exact but ran many eager kernels. |
| **Kernel** | Gather into contiguous tiles | ✓ Kept | In-place addressing tripled the kernel time; the copy costs ~0.03 s per block. |
| | Residuals computed once, before the coder | ✓ Kept | 15.9 against 14.3 GB/s when recomputed. |
| | Bit pack accumulated in registers | ✓ Kept | +5–20 %. |
| | Second pass for k candidates | ✕ Rejected (reverted) | −3 to −5 %. |
| | 64³ tiles with dynamic shared memory | ✓ Kept | The shared-memory struct is 55 KB for the 262,144-element tile class, above the 48 KB static limit. That class is compiled with `-DB4DK_DYNSMEM` (dynamic shared memory sized by a `b4d_smem_bytes` kernel); smaller classes are unchanged. Output bit-identical (host against device-resident block: identical SHA-256 over 400 chunks). |
| **Host → device** | Upload in 64 slices overlapped with kernels | ✓ Kept | 17.8 GB/s against 16.5 with 16 slices; a block already in device memory reaches 19.9. |
| | Staged copy → zero-copy registered blocks | ✓ Kept | 11.8 → 21.5 GB/s per GPU for 16-bit data. In the final ladder one GPU sustains 21.6–22.1 GB/s for 16-bit frames, near the PCIe Gen4 x16 limit (31.5 GB/s per direction, theoretical); 8-bit frames (13.5–14.2 GB/s) are kernel-bound. |
| | Copy mode on 2 GPUs, frames split by rows | ○ Optional | Works; costs 12.0–27.3 % against 2-GPU zero copy (final ladders, Fig. 2 d): Sony IMX411 138.6 → 122.0 fps (12.0 %), Sony IMX811 83.6 → 73.6 fps (12.0 %), GSPRINT4521 1,369 → 996 fps (27.3 %), Teledyne Kinetix 2,448 → 1,780 fps (27.3 %). The zero-copy run used the default settings, so the loss also includes the difference between the two configurations. An earlier 2-GPU copy run without the row split scaled −6 to +15 % against 1 GPU; the largest frames exceeded the Windows driver's page-locked limit, because each registration counts once per GPU. |
| | Copy mode on 1 GPU | ○ Optional | Costs 3.2 % (one ladder step; final ladders, Fig. 2 d): Sony IMX411 73.3 → 70.9 fps, Sony IMX811 44.0 → 42.6, GSPRINT4521 678 → 656, Teledyne Kinetix 1,318 → 1,276. |
| **Device → host** | Output buffers: half the raw bytes for 8-bit tiles, a third for 16-bit; registered in pieces of 1 GiB; one DMA pass over every new page-locked buffer | ✓ Kept | 2026-11-19. Teledyne Kinetix 8-bit blocks compress to 0.337 of raw, just over the former third, so every pooled output buffer was regrown once (a new 2.5 GB registration) in the first blocks, and the first DMA through new page-locked memory runs at ~8 GB/s instead of ~27: the first blocks took 1–3 s longer and frames were dropped. Since driver 553.50 (2026-11-17), downloads into one registration of 2 GB or more also fail at random (`cudaErrorMemoryAllocation`, which then poisons the context), so downloads are split at the 1 GiB pieces. Warm-up blocks needed to hold the capacity: 1 GPU, Teledyne Kinetix multiscale at 1,273 fps, 4–6 → 1; 2 GPUs zero copy at 2,448 fps, 8 → 2; 2 GPUs multiscale still 4. Byte output unchanged. |
| | Device-resident blocks (CuPy arrays, `device_blocks=True`) | ✓ Kept | The C call receives a zero-length stand-in; the kernel gathers tiles from the caller's array. Needed for GPUDirect RDMA. |
| | GPUDirect RDMA | … Pending (Linux only) | Device-resident blocks work on Linux; the Windows driver refuses them. Not measured. |
| **Scale-out** | 2 GPUs | ✓ Kept | 1.86–2.02 times one GPU (zero copy): Sony IMX411 73.3 → 138.6 fps, Sony IMX811 44.0 → 83.6, GSPRINT4521 678 → 1,369, Teledyne Kinetix 1,318 → 2,448. Bytes identical to 1 GPU and to the CPU. |
| **Host threads** | CPU threads feeding one GPU | ✓ Measured | The 1-GPU rate is reached with 2–4 host threads (probe: Sony IMX411 21.1 GB/s at 2 threads, Teledyne Kinetix 10.4 GB/s at 2 threads); at 128 host threads the confirmed rates are 19.95 (Sony IMX411), 19.31 (Sony IMX811), 13.47 (GSPRINT4521) and 9.92 GB/s (Teledyne Kinetix). Two confirm points (Sony IMX811 at 16 and 32 host threads) fell to 1.2 and 2.9 GB/s and are not explained. |
| **Driver** | TCC mode | … Pending | Would remove the Windows driver's limits; one GPU drives the display. |

*Cost per pixel of GPUs and CPUs: `GPU_VS_CPU.md`. One CPU with 128 cores sustains 80–83 GB/s for 16-bit frames,
3.6–3.9 times one A6000 and 1.9–2.0 times two.*

<details>
<summary>Details and measurements</summary>

**Write rate of one GPU against streaming multiprocessors (SMs) (GB/s)**<br>
<sub>Zero copy, confirmed stage; `docs/how_it_works/data/sweep_gpu_sms_1gpu.csv`.</sub>

| SMs | Sony IMX411, 16-bit (GB/s) | Sony IMX811, 16-bit (GB/s) | GSPRINT4521, 8-bit (GB/s) | Teledyne Kinetix, 8-bit, 512-frame blocks (GB/s; `docs/how_it_works/data/sweep_gpu_sms_1gpu_kinetix512.csv`) |
|---|---|---|---|---|
| 4 | 2.01 | 2.03 | 0.93 | 0.95 |
| 8 | 3.40 | 3.55 | 1.81 | 1.76 |
| 16 | 6.53 | 5.91 | 3.35 | 3.22 |
| 32 | 12.26 | 12.52 | 6.41 | 5.63 |
| 48 | 17.35 | 17.61 | 9.13 | 7.31 |
| 64 | 19.88 | 19.00 | 11.32 | 9.11 |
| 84 | 20.08 | 19.72 | 13.33 | 9.53 |

The rate rises approximately linearly up to 48 of 84 SMs. For 16-bit frames it levels off at 19–20 GB/s from 64 SMs (21.6–22.1 GB/s in the ladders). At
32 SMs the pixel rates of 16-bit and 8-bit frames are equal (6.1 and 6.4 Gpixel/s). Two GPUs at 84 SMs each
(`docs/how_it_works/data/sweep_gpu_sms_2gpu.csv`): Sony IMX411 37.8, Sony IMX811 36.9, GSPRINT4521 19.7, Teledyne Kinetix
20.6 GB/s.

</details>

---

<a id="l10"></a><sub>LAYER 10</sub>

## Writer and streaming pipeline

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Writer** | Native C writer, one call per block | ✓ Kept | zarr-python reached 1–2 GB/s. The native writer sustains, on 128 cores with zero copy at q = 1 (2026-11-16 build): Sony IMX411 264 fps (79.7 GB/s), Sony IMX811 169 fps (83.2 GB/s), GSPRINT4521 2,032 fps (42.6 GB/s), Teledyne Kinetix 3,827 fps (39.2 GB/s): 2.0 to 42 times the maximum rate of each real camera (2026-11-08 build: 196, 125, 1,495 and 2,912 fps). |
| | Commits in flight: 1 against 2 (CPU) | ✓ Kept (1) | Less than 1.3 % apart; 1 avoids a machine-wide reduced-throughput state seen with 2 blocks in flight. |
| **Ingest** | Zero copy (the camera writes into writer blocks) | ✓ Kept | Copying each frame into the writer (32 copy threads, 8 MB pieces) costs 40–46 % at 128 cores: Sony IMX411 262 → 157 fps (40.1 %), Sony IMX811 170 → 99.3 fps (41.5 %), GSPRINT4521 1,992 → 1,082 fps (45.7 %), Teledyne Kinetix 3,815 → 2,102 fps (44.9 %). The copied rates (21–23 GB/s of 8-bit, 47–49 GB/s of 16-bit frames) are the same as with the 2026-11-08 build, which lost 4.5–15.7 % against its slower zero-copy rate: the copy, not the compression, sets the rate. An earlier build lost 27–42 % (October campaign: ~20 % at 105 → 80–85 fps). GPU costs: [layer 09](#l9). |
| | Direct I/O | ✓ Kept | Buffered I/O with fsync ran at about half the drive rate and filled the OS cache. With `direct_io=True` the shard is assembled into a sector-aligned buffer and written with one unbuffered write-through call, then truncated to its length; byte-identical, with fallback where direct I/O is unsupported. |
| **Storage** | Steady-state rate | ✓ Specification | Long runs failed after ~30 s on a consumer NVMe: its SLC cache (~28 GB) was exhausted and the post-cache rate was 0.3–0.6 GB/s. Storage must sustain raw rate / ratio for the whole acquisition (× 1.2 with a pyramid), rated at steady state. |
| **Testing** | fps ladder (× 1.10 steps, 30 s each, 0 drops) | ✓ Kept | Replaced a 3-times overload probe that varied by ±20 %. The capacity is the writer rate at the highest offered fps with zero dropped frames. |
| | Simulator watchdogs | ✓ Kept | A failed or stalled block write now ends the run with an error code instead of hanging for hours. |

<details>
<summary>Details and measurements</summary>

**Structure**

```
camera ──► block buffer (N frames, contiguous) ──► write(block)  [one C call, GIL released]
   gather bands ─► compress tiles into per-worker arenas ─► assemble shards + crc32c index ─► files
   └─► pyramid: box mean into level accumulators ─► (same path for each level when a shard depth is full)
```

`b4d.zarr_store` presents a spec-based API (`open(spec)`, `arr[z0:z1].write(block)` returning a Future,
`create_multiscale(...)`); `b4d_zarr.c` does everything after the call. The zarr `blosc` codec (lz4, lz4hc, zstd +
shuffle / bitshuffle) runs on the same pool as b4d; any other numcodecs codec can replace b4d (encoded in Python threads,
assembled and written natively). `MultiscaleArray` writes arrays with leading axes (OME `(c, z, y, x)`, chunk and shard
extent 1 on `c`): `ms[c, z0:z1].write()` writes channel `c` at every level. Its optional `on_written(level, z0, z1)` hook
reports which shard files are durable (level 0 per block, level k per shard-depth flush), for hosts that stage the files
elsewhere (for example an s5cmd upload).

Work decomposition: band tasks claimed from a shared atomic counter; per-worker arenas, blobs recorded as (worker, offset,
length), returned to a global cache; shards assembled as blobs in C order, then the index (16 bytes per chunk: offset,
length; 2⁶⁴ − 1 for absent chunks), then the crc32c of the index; chunks that overhang the array edge are padded with the
fill value. Files are written to `<key>.tmp`, flushed when `sync=True`, and renamed into place. The pool's capacity grows
on demand to 2 times the logical CPUs. `writers=2` lets block N + 1 compress while the files of block N are written, so the
block cycle is max(compress, I/O).

Real camera ingest (`egrabber/`, Euresys eGrabber, 2026-10-09): the writer's block buffers are announced to the Coaxlink as
DMA targets (one GenTL buffer per frame slot), so zero copy is the real data path; nothing in `b4d_zarr.c` changed.

**Final CPU and GPU ladders**<br>
<sub>128 cores, zero copy, q = 1, default codec, in-memory sink; `docs/how_it_works/data/ladder_cpu_zero_copy.csv`
(2026-11-16 build; the 2026-11-08 build reached 196.1, 124.9, 1,494.9 and 2,912.4 fps) and the GPU ladders (`docs/how_it_works/data/ladder_gpu_*.csv`); highest offered rate with zero dropped frames.</sub>

| Camera | Block (frames) | CPU, 128 cores: frame rate / write rate (fps / GB/s) | 1 GPU: frame rate / write rate (fps / GB/s) | 2 GPUs: frame rate / write rate (fps / GB/s) | Ratio |
|---|---|---|---|---|---|
| Sony IMX411 (10,640 × 14,192, 16-bit) | 64 | 264.0 / 79.7 | 73.3 / 22.1 | 138.6 / 41.9 | 6.10 |
| Sony IMX811 (12,800 × 19,200, 16-bit) | 128 (CPU), 64 (GPU) | 169.4 / 83.2 | 44.0 / 21.6 | 83.6 / 41.1 | 6.11 |
| GSPRINT4521 (4,096 × 5,120, 8-bit) | 512 | 2,031.7 / 42.6 | 677.6 / 14.2 | 1,369.0 / 28.7 | 3.00 |
| Teledyne Kinetix (3,200 × 3,200, 8-bit) | 256 (CPU), 512 (GPU) | 3,826.6 / 39.2 | 1,317.7 / 13.5 | 2,447.9 / 25.1 | 3.00 |

CPU core sweep (`docs/how_it_works/data/sweep_cpu_cores.csv`, each run held to N physical cores, 2026-11-16 build): per core at 4
cores 0.67–0.69 GB/s (16-bit) or 0.39 GB/s (8-bit), 0.33–0.39 Gpixel/s; efficiency against 32 times the 4-core rate is
97–107 % at 64 cores and 76–87 % at 128 cores. The 2026-11-08 build (`cpu_above3_results.csv`, threads not pinned):
0.29–0.30 Gpixel/s per core.

Earlier headline (October campaign, one process, 128 threads, q = 2, earlier default codec): 105 fps = 31.7 GB/s zero copy;
80–85 fps with the simulator's memcpy stage; ~45 fps with a 5-level pyramid (copy included); compressed output 3.9 / 3.0 /
2.0 GB/s.

**Tuning for a different camera, machine or storage**

Work from the data rate R = fps × pixels × bytes per pixel (GB/s), not from fps or frame size alone.

- **Buffers.** `blocks >= writers + 1` (3–4). Size the ring in time, 0.3–0.5 s of frames (32 frames at 85 fps, 128–256 at
  500 fps). Depth costs RAM and the crash-loss window, never throughput beyond absorbing start-up and stalls.
- **Threads.** `threads` = physical cores available to the writer (SMT adds 0–5 % and contention); `writers=2` (3 only for
  irregular I/O). Pyramid levels use half the level-0 count. If a copy stage exists, give it threads (32 in the final
  ladder); with DMA delivery, none.
- **Start-up.** Run 3–4 full-size blocks through the writer before the camera starts (`--warmup-blocks 4`); pre-fault
  every large buffer. With the GPU encoder, call `prepare()` and `register()` before the camera starts: they allocate,
  page-lock and run one DMA pass over every buffer (1 GPU and 2 GPUs in zero copy then hold their capacity after 1–2
  warm-up blocks; 2 GPUs with the pyramid still need 4).
- **Procedure on a new system.** (1) `--sink memory`, no pyramid, `threads` = physical cores, `writers=2`, 0.5 s blocks:
  run the fps ladder. (2) `--zero-copy` if frames can be DMA-delivered: the difference is the cost of the copy. (3) Add the
  pyramid: expect −20 to −23 % on the CPU at 128 cores (ladders; −26 to −31 % in the pinned core sweep) and less than one
  ladder step (≤ 3.4 %) with level 0 on the GPU. (4) `--sink disk` at 70 % of the
  CPU capacity for ≥ 5 minutes; if the committed latency rises, storage sets the rate. (5) Then set the geometry for the
  readers and confirm that the write rate is unchanged.

</details>

---

<a id="l11"></a><sub>LAYER 11</sub>

## Multiscale pyramid

| Variant | Option | Change | Description and impact on b4d |
|---|---|---|---|
| **Downsampling** | Cascaded 2 × 2 × 2 integer means | ✓ Kept | 1 + 1/8 + 1/64 + … = 1.14 passes in total, instead of one pass per level. |
| | Persistent accumulators, row-parallel downsampler, double buffering | ✓ Kept | 0.8 → 0.14 s per block (a 2.4 GB `np.empty` per block had cost first-touch faults at ~2 GB/s; the downsampler had been parallel over 16 output planes only); 5.6 GB/s per core. A full CPU pyramid costs 20–23 % of throughput at 128 cores (ladders; 26–31 % in the pinned core sweep) and 17–20 % more bytes; with level 0 on the GPU, at most one ladder step (≤ 3.4 %). |
| | Level 1 fused into the compression gather | ✓ Kept (CPU writer) | An early measurement found more CPU than the saved memory pass plus a single-threaded copy; the writer now fuses level 1 into the gather by default when level 0 is compressed on the CPU, and runs the separate box-mean pass when an external (GPU) encoder is set. |
| **Codec tables** | Open-addressing table caches | ✓ Kept (fix) | The codec's lookup-table caches were direct-mapped and keyed by camera model. The per-level models of a pyramid (conversion / 8^k, read noise × √8^k) collided, so their tables were rebuilt on every call and each evicted table was kept until exit. Sony IMX811 with the pyramid: 2.61 → 0.95 s per block, 3,067 → 0 dropped frames, resident memory flat instead of +150 GB in 60 s. All multiscale results before 2026-11-16 23:00 (CPU, and GPU runs whose levels ≥ 1 are coded on the CPU) were re-measured. |
| **Ordering** | Levels ≥ 2 derived on a pyramid thread | ✓ Kept | The ordered pyramid step derived levels ≥ 2 inline (0.02–0.06 s per 16-bit block, near memory bandwidth) and held the next block back. A full level-1 buffer is now handed to one pyramid thread; the buffer is reused only after its write and its derivation have finished. 4–7 % less time per block (Sony IMX811 0.951 → 0.889 s, Sony IMX411 0.309 → 0.295 s, Teledyne Kinetix 0.078 → 0.074 s); bit-identical. |
| **Placement** | Level 0 on the GPU, the other levels on the CPU | ✓ Kept | Costs against multiscale off (2026-11-17, `docs/how_it_works/data/ladder_*_multiscale.csv`): one ladder step (3.4 %) on one and two GPUs for every camera (Sony IMX411 73.3 → 70.8 fps on one GPU, 138.6 → 133.9 on two; GSPRINT4521 678 → 655 and 1,369 → 1,323; Teledyne Kinetix 1,318 → 1,273 and 2,448 → 2,365); Sony IMX811 44.0 → 42.5 on one GPU and 83.6 → 88.9 on two (one step higher). Runs before 2026-11-16 23:00 lost 2–13 % (table-cache collisions, fixed). Byte-identical to the all-CPU result. |
| | All levels on the GPU | ✕ Rejected | About −40 %. |
| | Pyramid overlapped with level 0 | ✕ Rejected (off) | No gain, and a hang on close. |
| **8-bit** | Level 1 fused for uint8 frames | ✓ Kept | The fused gather now also handles 8-bit frames (GSPRINT4521, Teledyne Kinetix); bit-identical to the separate box mean. |

<details>
<summary>Details and measurements</summary>

Levels are 2 × 2 × 2 (or 1 × 2 × 2) box means computed in C (row-parallel, integer, rounded), exact against a numpy
reference including odd edges. Each level has constant chunk and shard depth: downsampled planes accumulate across blocks
(double-buffered) until a shard depth is available, then are written on the level's own thread and cascaded; level k lags
level 0 by up to 2^k blocks, and `close()` flushes the remainder as padded shards. Output is OME-NGFF 0.5 (validated with
the `ome-zarr` reference reader). Viewers that decode in their own code (neuroglancer, viv) cannot read b4d chunks;
`pyramid_codec=Zstd(3)` keeps levels ≥ 1 readable there. Levels: halve until the smallest axis is ~256–512 pixels (5 for
10,640 × 14,192).

Level-1 fixes found on the real-camera runs (level 1 of the first three blocks was wrong in both the CPU and the GPU
datasets; level 0 was never affected):

1. `_reserve_l1()` handed out a fused level-1 destination even when level 0 is encoded by an external (GPU) encoder; the C
   writer never gathers bands in that case, so the reserved planes stayed zero. It now returns no reservation when an
   encoder is set.
2. `reset_pyramid()` (after warm-up blocks) reset the fill side of the level-1 accumulator but not the submission-side
   mirror, so the first level-1 shards carried stale warm-up planes. It now resets the mirror and first waits for every
   submitted block write.
3. `_submit_level()` advanced `gen_done` only on the overshoot path, so fused reservations stopped after two buffer
   generations (4 blocks). It now counts every submitted generation.

Verified with an emulation of the recorder sequence (warm-up, reset, 6 blocks; CPU fused and CUDA external encoder): every
level-1 plane equals the box mean of the frames, and the CPU and GPU level-1 arrays are identical.

**Cost of the pyramid on the CPU writer (2026-11-16)**<br>
<sub>128 cores, zero copy, q = 1, Anscombe curve, in-memory sink, 2 blocks in flight; write time per block from
`bench_scaling/profile_cpu_ms.py` (`data/profile_cpu_ms_{msfix,lutfix,cascade}_1116.csv`). The levels hold 14.3 % more
pixels than level 0 (1/7 of it), which sets the lower bound of the cost.</sub>

| Camera | Pyramid off (s) | Before the fixes (s) | Before the fixes, dropped frames | Cache fix (s) | Cache fix + pyramid thread (s) | Cost (%) |
|---|---|---|---|---|---|---|
| Teledyne Kinetix (8-bit, 256 frames) | 0.067 | 0.090 | 392 | 0.078 | 0.074 | +10 |
| Sony IMX411 (16-bit, 64 frames) | 0.245 | 0.335 | 235 | 0.309 | 0.295 | +20 |
| Sony IMX811 (16-bit, 128 frames) | 0.754 | 2.61 | 3,067 | 0.951 | 0.889 | +18 |

**Pinned core sweep, pyramid on and off (2026-11-17)**<br>
<sub>Each run held to N physical cores (both SMT threads; Windows CPU Sets, `bench_scaling/cpuset.py`), default threads,
saturated 20 s probe, 30 s confirmation with zero dropped frames (`sweep.py --pin`; `docs/how_it_works/data/sweep_cpu_cores.csv`,
`docs/how_it_works/data/sweep_cpu_cores_multiscale.csv`). Earlier multiscale sweeps counted threads, not cores (the levels' own threads
ran on idle cores) and timed only the level-0 write call; they are superseded.</sub>

| Cores | GSPRINT4521 off / on (GB/s) | Teledyne Kinetix off / on (GB/s) | Sony IMX411 off / on (GB/s) | Sony IMX811 off / on (GB/s) | Rate lost (%) |
|---|---|---|---|---|---|
| 4 | 1.57 / 1.00 | 1.54 / 1.01 | 2.68 / 2.25 | 2.78 / 2.17 | 16–36 |
| 32 | 12.33 / 8.83 | 12.33 / 9.48 | 24.06 / 17.28 | 24.06 / 16.22 | 23–33 |
| 128 | 40.22 / 28.26 | 37.34 / 27.79 | 74.89 / 52.52 | 76.28 / 52.64 | 26–31 |

Where the cost goes (`bench_scaling/ms_level_cost.py`: one process, synthetic 16-bit block 64 × 4096 × 4096, 4 levels,
128 threads; plain array 0.038 s per block, pyramid 0.054 s): the fused level-1 box mean costs about 3 % of the level-0
pass; alone, level 1 compresses at 320–340 Mpixel/s per thread, faster than level 0 (237). The loss comes from running
the levels next to level 0: concurrently (default, 64 threads per level write) both slow down on shared cores, level 0 by
25–33 %; serialised after each block with 128 threads, level 0 recovers but the level writes are small (levels 2–4 hold
under 2 % of the pixels and take about 7 % of the time) and the total is the same. Fewer level threads (16–24) gain
2–4 %. Pure extra-pixel work would be +9–14 %; closing the gap needs the level-1 tiles compressed in the same pool job as
level 0 and the small levels written less often (not done).

Two build details mattered: `setup.py` with `B4D_CC=clang-cl` kept MSVC's `/GL` and `/LTCG`, and the
linked module was 13–17 % slower than the benchmark build (now stripped, linked with `lld-link`); and the exact VST is
fitted when `BlockRecorder.start()` runs, before the source starts, because a 0.2 s fit on the first write overflowed a
3-block ring at 400 fps.

</details>

---

<a id="l12"></a><sub>LAYER 12</sub>

## Versus other codecs

> **Result:** b4d is the only lossy codec tested that keeps gain, read noise and offset. At 128 cores and q = 1 it
> encodes 9.0–9.3 times faster than SZ3 and 6.9–11.4 times faster than B3D (CPU reimplementation), and decodes 8.0–9.1
> times faster than SZ3 and 6.1–7.3 times faster than B3D.

**Ratio, speed and statistics after decoding, final build**<br>
<sub>64³ chunks, codec time only, mean over a dark stack and two flat-field stacks of the Sony IMX411 model
(the scaling report). Ranges span the settings of each codec. The statistics column is
from the earlier comparison and was not re-measured.</sub>

| Codec | Ratio | Encode, 1 core (MB/s) | Encode, 128 cores (GB/s) | Decode, 128 cores (GB/s) | Statistics after decoding |
|---|---|---|---|---|---|
| **b4d, q = 1 (`m1_a3_xv`)** | **6.21** | **803** | **60.4** | **84.5** | **Read noise +0.1 %, gain −0.1 %, mean error +0.002σ** |
| b4d, q = 2 (`m1_a3_xv_q2`) | 9.34 | 852 | 73.4 | 84.3 | Gain +0.3 %, read noise +0.4 % |
| B3D, q = 1, modes 1 and 2 (CPU) | 6.68–6.75 | 60–90 | 5.3–8.8 | 11.5–13.9 | ✕ Fails: read noise of dark frames collapses; offset +1.7 ADU |
| SZ3, noise-scaled bound | 6.29–9.46 | 140–143 | 6.5 | 9.3–9.5 | Gain −5.6 to −18 % |
| SZ3, fixed bound | 3.90–5.74 | 134–140 | 6.5–6.7 | 9.9–10.5 | Unbiased, read noise +4.8 to +20 % |
| ZFP | 1.66–2.39 | 126–139 | 13.9–16.5 | 10.4–14.0 | Near-exact; ratio 1.7–2.4 |
| JPEG-LS | 5.26–7.92 | 20–21 | 2.3–2.4 | 2.5–2.6 | Gain −5.6 % |
| JPEG-XL | 5.88–9.40 | 4 | 0.1 | 2.1–2.2 | Gain +9.6 to +28 % |
| JPEG 2000 | 5.49–7.32 | 9 | 1.0–1.1 | 3.6–5.2 | Gain −5 to +5 % |
| Blosc2 + Zstd level 1 (lossless) | 2.63 | 942 | 96.0 | 190.7 | Lossless |
| Blosc2 + LZ4 (lossless) | 1.80 | 1,978 | 170.9 | 395.5 | Lossless |
| b4d lossless | 2.56 | 720 | 35.6 | 98.2 | Lossless |

<details>
<summary>Details and measurements</summary>

On a light-sheet tissue crop (`final_1115b\scaling_report_tissue.md`), b4d at q = 1 gives a ratio of 5.95 at 781 MB/s per
core, 53.5 GB/s encode and 84.0 GB/s decode at 128 cores.

</details>

---

<sub>CONSTRAINTS</sub>

## Limits

- Blocks must start on shard boundaries and cover whole shards except at the array edge; multiscale writes must be whole
  frames.
- No temporal prediction; z chunking is a storage decision.
- b4d chunks are readable by zarr-python (`import b4d.zarr3`) and this package. Neuroglancer and viv need a native codec
  implementation (or `tools/neuroglancer_visualization.py`, which transcodes on the fly); stores written with raw / Zstd /
  Blosc and pyramid levels written with `pyramid_codec` open in all of them.
- Linux: same tuning rules; `O_DIRECT` instead of `NO_BUFFERING`, no processor groups. Not measured here.

---

<sub>SEE ALSO</sub>

## Related documents

- `GPU_VS_CPU.md`: the cost comparison of GPUs and CPUs.
- `COMPARISON.md`: the comparison of `above3` with B3D and dither.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

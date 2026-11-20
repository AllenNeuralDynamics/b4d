# `above3` compared with B3D and dither

`above3` is mode 1 with a prediction from the three pixels above (`x − 1`, `x`, `x + 1`). This page compares it with the
original B3D pipeline and with seeded dither on real light-sheet data, for accuracy and speed.

## Summary

- **Against dither**: at `q = 1` the histogram distance, bias and error are the same, the ratio is 16% higher (6.75
  against 5.81) and encoding is 26 to 39% faster. At `q = 2` the ratio is 24% higher and encoding 9 to 16% faster, but
  the histogram distance is slightly worse (0.109 against 0.092).
- **Against B3D**: at `q = 1` the ratio is about the same (6.75 against 6.73), the mean bias is 20 times smaller (0.024
  against 0.496 DN), the RMS error is a third lower, and the histogram distance is 7 times smaller. Encoding on the CPU is
  3 to 4 times faster on the Apple silicon Mac. On an AMD EPYC 9754 (Sony IMX411 dark and flat-field stacks,
  `q = 1`) it is 8.9 to 13.4 times faster on one core and 6.9 to 11.4 times faster on 128 cores; on
  those stacks B3D's ratio is 8 to 9% higher (6.68 to 6.75 against 6.21) and B3D does not keep the read noise of dark
  frames (see [x86 speed](#x86-speed)).
- **Against `above`** (mode 2, predicted from the one pixel above): encoding is on par at `q = 1` and 14 to 20% slower at `q = 2`.
  Decoding is 24 to 30% slower. The ratio is 14% higher at `q = 1`, 8% at `q = 2`, and the histogram distance falls from
  about 0.8 to 0.026 at `q = 1`.
- **At `q = 4`** `above3` no longer behaves like noise: its errors correlate between neighbouring rows (0.10). Dither's do
  not.

## Setup

- **Data**: 30 chunks of 128 × 256 × 256 uint16 pixels from `s3://aind-open-data/853731_2026-09-18_14-52-22/SPIM`, 488 nm
  channel, tiles 0, 2, 5, 7, 9, 10, 12, 15, 17 and 19. Each tile gives a bright, a mid and a dark chunk, picked near the 99.9th,
  50th and 3rd percentiles of its multiscale level 8. Chunk medians are 36 to 74 DN, with a typical chunk at 39 DN, so most pixels are close to the
  camera offset.
- **Camera**: IMX411, 0.77 e⁻/ADU, offset 34 ADU, read noise 3.48 e⁻: `b4d.Camera(1 / 0.77, 34.0, 3.48)`. Measured noise is
  0.94 to 1.12 times this model (median 0.97).
- **Machine**: Apple silicon Mac (arm64). x86 speeds, on other data, are in [x86 speed](#x86-speed).
- **Reproduce**: `validation/compare_codecs.py`.

| Name | Settings |
|---|---|
| `above3` | `mode=1, predictor="above3"`, `entropy="rice"` unless stated |
| dither | `mode=2, predictor="above", entropy="rice", dither=True` |
| B3D m1, B3D m2 | `mode=1` or `2`, `predictor="b3d"`: the reimplementation of B3D in this repository, byte-identical to the original |
| `above` | `mode=2, predictor="above", entropy="rice"`: prediction from the one pixel above, for reference |

The dither row is the `dither=True` option in this repository. It is not Dotphoton's Jetraw, which was not tested.

## Accuracy

Medians over the 30 chunks.

| `q` | Method | Ratio | Mean bias (DN) | Largest local bias (DN) | RMS error (DN) | Histogram distance | Largest error correlation |
|---|---|---|---|---|---|---|---|
| 1 | `above3` | 6.75 | 0.024 | 0.029 | 1.54 | 0.026 | 0.000 |
| 1 | `above3`, `ans` | 6.81 | 0.024 | 0.029 | 1.54 | 0.026 | 0.000 |
| 1 | dither | 5.81 | 0.022 | 0.026 | 1.55 | 0.026 | 0.000 |
| 1 | B3D m1 | 6.73 | 0.496 | 0.979 | 2.33 | 0.181 | 0.041 |
| 1 | B3D m2 | 6.62 | 0.271 | 0.699 | 2.21 | 0.820 | 0.043 |
| 1 | `above` | 5.93 | 0.054 | 0.115 | 1.47 | 0.803 | 0.001 |
| 2 | `above3` | 10.01 | 0.075 | 0.086 | 3.07 | 0.109 | 0.003 |
| 2 | `above3`, `ans` | 10.76 | 0.075 | 0.086 | 3.07 | 0.109 | 0.003 |
| 2 | dither | 8.06 | 0.093 | 0.105 | 3.07 | 0.092 | 0.000 |
| 2 | B3D m1 | 10.13 | 0.758 | 1.312 | 3.62 | 0.221 | 0.020 |
| 2 | B3D m2 | 8.97 | 0.541 | 0.785 | 3.34 | 0.904 | 0.017 |
| 2 | `above` | 9.26 | 0.211 | 0.372 | 3.00 | 0.903 | 0.001 |
| 4 | `above3` | 22.34 | 0.170 | 0.306 | 5.09 | 0.258 | 0.100 |
| 4 | `above3`, `ans` | 33.97 | 0.170 | 0.306 | 5.09 | 0.258 | 0.100 |
| 4 | dither | 10.24 | 0.342 | 0.384 | 6.19 | 0.270 | 0.000 |
| 4 | B3D m1 | 35.57 | 0.421 | 0.889 | 5.45 | 0.281 | 0.147 |
| 4 | B3D m2 | 21.83 | 1.828 | 2.353 | 6.22 | 0.948 | 0.026 |
| 4 | `above` | 17.53 | 1.119 | 2.001 | 4.71 | 0.923 | 0.015 |

- **Histogram distance** is the fraction of pixels whose decoded value would have to change for the decoded histogram to match
  the raw one (`0.5 · Σ|raw − decoded| / N`). 0 is identical.
- **Largest local bias** is the worst mean error over 8 × 8 patches, binned by raw DN, for bins of at least 500 patches.
- **Largest error correlation** is the worst correlation of the error between neighbouring pixels along x, y or z.

### Segmentation

Dice between the raw and decoded images after a 5 × 5 × 5 box filter and an Otsu threshold, on the 5 chunks with structure
(99th percentile at least 100 DN). This is a simple proxy, not a segmentation pipeline.

| `q` | Method | Dice, raw image's threshold on both (worst, median) | Dice, each image's own threshold (worst, median) |
|---|---|---|---|
| 1 | `above3` | 0.9808, 0.9986 | 0.9448, 0.9980 |
| 1 | dither | 0.9810, 0.9987 | 0.9807, 0.9979 |
| 1 | B3D m1 | 0.9802, 0.9987 | 0.9616, 0.9922 |
| 2 | `above3` | 0.9613, 0.9972 | 0.9302, 0.9973 |
| 2 | dither | 0.9613, 0.9971 | 0.9414, 0.9970 |
| 2 | B3D m1 | 0.9604, 0.9973 | 0.9604, 0.9930 |
| 4 | `above3` | 0.9103, 0.9925 | 0.9095, 0.9918 |
| 4 | dither | 0.9188, 0.9940 | 0.9159, 0.9924 |
| 4 | B3D m1 | 0.9183, 0.9936 | 0.9189, 0.9922 |

With a shared threshold the three are equal at `q = 1` and `q = 2`. The worst-case differences with separate thresholds come
from one low-contrast chunk (foreground 76 DN, background 66 DN), where Otsu's threshold moves by up to 0.5 DN between
images. With so few chunks the separate-threshold column mostly measures that instability, so it says little about any one method.

## Speed

MB/s of raw data, best of 3. All 30 chunks (503 MB) go through `compress_tiles` as independent tiles, as the writer does; decoding
runs the same tiles on a thread pool. Percentages are relative to `above`.

| Threads | `q` | Method | Encode (MB/s) | Encode vs `above` (%) | Decode (MB/s) | Decode vs `above` (%) |
|---|---|---|---|---|---|---|
| 1 | 1 | `above3` | 683 | 101 | 516 | 74 |
| 1 | 1 | `above3`, `ans` | 336 | 50 | 277 | 40 |
| 1 | 1 | dither | 492 | 73 | 533 | 76 |
| 1 | 1 | B3D m1 | 165 | 24 | 254 | 36 |
| 1 | 1 | B3D m2 | 226 | 34 | 243 | 35 |
| 1 | 1 | `above` | 674 | – | 698 | – |
| 1 | 2 | `above3` | 700 | 86 | 580 | 76 |
| 1 | 2 | `above3`, `ans` | 398 | 49 | 275 | 36 |
| 1 | 2 | dither | 603 | 74 | 562 | 74 |
| 1 | 2 | B3D m1 | 207 | 25 | 384 | 51 |
| 1 | 2 | B3D m2 | 242 | 30 | 311 | 41 |
| 1 | 2 | `above` | 814 | – | 759 | – |
| 8 | 1 | `above3` | 2,883 | 100 | 2,724 | 71 |
| 8 | 1 | `above3`, `ans` | 1,515 | 53 | 1,412 | 37 |
| 8 | 1 | dither | 2,281 | 79 | 2,669 | 70 |
| 8 | 1 | B3D m1 | 695 | 24 | 1,281 | 34 |
| 8 | 1 | B3D m2 | 935 | 32 | 962 | 25 |
| 8 | 1 | `above` | 2,877 | – | 3,822 | – |
| 8 | 2 | `above3` | 3,009 | 80 | 2,892 | 70 |
| 8 | 2 | `above3`, `ans` | 1,833 | 49 | 1,518 | 37 |
| 8 | 2 | dither | 2,760 | 73 | 2,940 | 71 |
| 8 | 2 | B3D m1 | 1,002 | 27 | 2,017 | 49 |
| 8 | 2 | B3D m2 | 1,287 | 34 | 1,705 | 41 |
| 8 | 2 | `above` | 3,769 | – | 4,150 | – |

The B3D rows are the CPU reimplementation. The original B3D runs on a GPU, and its GPU speed is not compared here.

## x86 speed

One AMD EPYC 9754 (128 cores), 64³ chunks, codec time only, mean over a dark stack and two flat-field stacks of the Sony
IMX411 camera model. This is a different dataset from the light-sheet chunks above, and `above3` here uses the default
exact variance-stabilizing table.

| `q` | Method | Ratio | Encode, 1 core (MB/s) | Encode, 128 cores (GB/s) | Decode, 1 core (MB/s) | Decode, 128 cores (GB/s) |
|---|---|---|---|---|---|---|
| 1 | `above3` | 6.21 | 803 | 60.4 | 852 | 84.5 |
| 1 | B3D m1 | 6.68 | 60 | 5.3 | 133 | 13.9 |
| 1 | B3D m2 | 6.75 | 90 | 8.8 | 104 | 11.5 |
| 2 | `above3` | 9.34 | 852 | 73.4 | 834 | 84.3 |
| 2 | B3D m1 | 11.18 | 69 | 5.9 | 165 | 15.1 |
| 2 | B3D m2 | 12.18 | 150 | 15.2 | 188 | 22.3 |

On these stacks `above3` encodes 8.9 to 13.4 times faster than B3D on one core at `q = 1` (5.7 to 12.3 times at `q = 2`) and
6.9 to 11.4 times faster on 128 cores (4.8 to 12.4 times at `q = 2`); it decodes 6.1 to 7.3 times faster on 128 cores at
`q = 1`. B3D's ratio is 8 to 9% higher at `q = 1` and 20 to 30% higher at `q = 2`; on these stacks B3D collapses the read
noise of dark frames and shifts the offset by +1.7 ADU ([WRITER_DESIGN.md, layer 12](WRITER_DESIGN.md#l12)).

## Limits

- One dataset, one channel and one camera. The result relies on real sensor noise spreading the rounding offsets, so it needs
  noise near the camera model and weakens as `q` rises. With the three taps of `above3` the histogram distance matches dither at `q = 1` and starts to fall behind at `q = 2`.
- `above3` runs on the CPU and on the GPU encoder (mode 1, byte-identical to the C encoder). The GPU speed was not measured on this dataset; write rates are in Table 2 of the [README](../README.md#overview).
- The speeds in [Speed](#speed) were measured through `compress_tiles` on one machine, not in a full recording.
- The first row of each plane is predicted from the last row of the previous plane, which costs some ratio on those rows.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

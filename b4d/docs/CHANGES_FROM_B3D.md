# Algorithmic changes from B3D

B4D keeps the idea of B3D (Balázs et al. 2017, [github.com/balintbalazs/B3D](https://github.com/balintbalazs/B3D)):
transform each pixel so that its noise has unit variance, quantise in units of that noise, predict from
neighbouring pixels, and entropy-code the residuals. This page lists where the algorithm differs. The original
pipeline itself remains available as `predictor="b3d"`.

| stage | B3D | B4D |
|---|---|---|
| Variance-stabilising transform | `T(x) = 2·sqrt(max(x − offset, 0)/conv + rn²) − 2·rn`, float32 | A transform fitted to the camera's noise model (default `exact_vst="variance"`: a step q adds q²/12 of each pixel's variance at every signal level), or the generalized Anscombe form `T(x) = 2·sqrt(e + rn² + 3/8)` with `e = (x − offset)/conv`; continued linearly below the offset, evaluated in double precision into an integer lookup table |
| Quantisation step | `q` in units of the transform (σ = 1) | same meaning, `q` in multiples of 0.25 |
| Mode 1 (quantise the residual) | closed loop, float residual rounded half away from zero | closed loop in fixed point (1/256 DN) |
| Mode 2 (quantise, then predict) | round `T/q` to uint16, then predict | same, through the integer table; optional seeded subtractive dither (`dither=True`) |
| Lossless mode | prediction of raw pixels | same, selected as `q = 0` |
| Predictor | `(left + above) >> 1` on 24 × 24 tiles; tile edges use left or above only | selectable per dataset, no tiling: `above3` (default), `above`, `left_above`, `mean3d` |
| Residual to symbol | `2·abs(v) + (v >> 15)` | zigzag |
| Entropy coder | zero run-length (runs ≤ 255) followed by two canonical Huffman tables per chunk | selectable per dataset: adaptive Golomb-Rice (default), canonical Huffman with zero-run symbols, or rANS |
| Third dimension | a 3-D chunk is coded as one 2-D image of `ny·nz` rows | planes are kept; `mean3d` predicts across them |
| Multiscale pyramid | none | each level of means has its own noise model: `conv/n`, `rn·√n` for a mean of `n` pixels |
| Container | HDF5 filter chunk; parameters live in the filter settings | self-describing header (magic `B4DP`, version, dtype, shape) around Blosc2 chunks; every block records its mode, step, predictor and coder |

## Transform

The added 3/8 is the Anscombe term. It keeps the variance of the transformed value close to one at low photon
counts, where the B3D form understates the noise. Below the camera offset the transform continues linearly with matched slope instead of
clamping, so negative read-noise excursions are coded symmetrically and dark regions are not biased upward. Because the forward transform is a table indexed by pixel
value, its result does not depend on the floating-point behaviour of the compiler or device.

By default the table is fitted numerically to the camera model (format v9, and v11 with eight Rice streams; its
200-byte description travels in each chunk), so that quantization adds q²/12 of the noise variance at every signal
level, also where the Anscombe form deviates (low counts). `exact_vst=False` selects the Anscombe table.

## Dither (optional)

Rounding `T/q` puts bin edges at fixed values, so at low ADU, where a bin spans only a few DN, whole DN levels map to
one bin centre. The decoded mean then shifts by a signal-dependent amount of up to about half a DN (0.5 σ at `q = 2` on
a camera with one DN of read noise). This is deterministic, so averaging frames does not remove it.

`dither=True` (modes 1 and 2, off by default) applies subtractive dither (Roberts 1962; Schuchman 1964). Each pixel's code is
`floor(T/q + r)` with `r` uniform on `[0, 1)` in steps of 1/256, and the decoder takes `(code − r + 1/2)·q` before
inverting the transform. The error bound stays `q/2`, and the error no longer depends on the signal.

- `r` comes from a splitmix64 stream keyed by a 32-bit seed stored in each block. The seed is a hash of the block's
  pixels, so output is repeatable and independent of thread count, while the error differs between blocks.
- The transform table holds `T/q` in units of 1/256 instead of rounded codes. The decoder inverts the transform per pixel, because
  a code no longer identifies one value.
- Dithered blocks are format version 8 (version 6, the earlier mode-2 dither, still decodes). Blocks written without
  dither are byte-identical to before.

Measured on simulated frames with `above` and `rice`, 0 to 1000 e⁻, at two camera models (1 and 0.74 DN of read noise):

| | without dither | with dither |
|---|---|---|
| largest mean bias (DN) | 0.50 | 0.05 |
| decoded / raw variance, `q = 1` | 0.77 to 2.01 | 1.02 to 1.14 |
| decoded / raw variance, `q = 2` | 0.88 to 1.73 | 1.32 to 1.69 |
| compression ratio, `q = 1` | 5.56 to 5.98 | 5.59 to 5.89 |
| compression ratio, `q = 2` | 7.89 to 9.22 | 7.70 to 7.96 |

Single-thread encoding is 15 to 36% slower (`rice` loses its fast path) and decoding 13 to 24% slower. The CUDA encoder
dithers with the Rice coder (mode 2 `above`, mode 1 `above3` and `above`); it computes each tile's seed on the device,
and its bytes match the C encoder's.

## Per-pixel calibration (optional)

The camera model is one curve, but on the sensor measured here the noisiest 1% of pixels have 3 to 8 times the median variance. Those pixels
keep being coded to a step meant for quieter ones, and their noise costs bits. `PixelCalibration` (`mode=1` with `above3`, or `mode=2` with `above`; off by default)
gives them a step in proportion to their own noise:

- Measured per-pixel read noise in T units is `S² = 1 + extra / max(T²/4, C)`, where `extra` is the pixel's read variance above the
  model's in e⁻² and `C = rn² + 3/8`. `T` comes from the prediction, so the decoder computes the same step and the step falls back to `q`
  as signal rises and shot noise takes over.
- Only pixels with `S ≥ 1.5` at zero signal are listed. Quieter pixels stay at the camera's step, since a finer step costs bits to keep noise.
- A listed pixel whose step reaches `1.25 q` is left out of the prediction of the row below: its noise would otherwise spread to
  three neighbours.
- Hot pixels: the difference between the pixel's dark level and the sensor's is removed before prediction and restored on decoding.
- The list is sparse and in the block: about one byte per listed pixel per chunk, seven noise tiers. A block whose list would exceed 1/400 of
  its raw size is written as an ordinary stream, and so is a block with nothing to list, byte for byte as before. Blocks with a list are
  format version 7 (mode 1) or 8 (mode 2).
- The vectorised row passes run over every pixel as before, then the listed pixels are redone one by one, from the table of steps. The
  cost follows the length of the list, not the width of the row.

Measured with a map from half of each chunk's planes, on the other half (64 planes), on 6 chunks used for choosing the limits and 9 others. The
data had the camera's defect and hot-pixel corrections on.

| 32×64×64 chunks, `rice` | `q = 1` | `q = 2` | `q = 4` |
|---|---|---|---|
| ratio without, chosen on / other chunks | 6.58 / 6.51 | 9.78 / 9.70 | 19.33 / 18.16 |
| gain, chosen on / other chunks (%) | +2.5 / +2.1 | +1.4 / +1.2 | +13.0 / +10.4 |
| same, 64×128×128 chunks (%) | +2.9 / +3.0 | +2.0 / +2.0 | +15.1 / +13.7 |
| decoded/raw noise variance, listed pixels / the rest | 1.03–1.09 / 1.09–1.12 | 1.45–1.62 / 1.36–1.45 | 0.61–1.32 / 1.11–1.57 |

The rest of the pixels keep their error and noise to three digits at `q = 1` and `q = 2`. Encoding and decoding speed match a build without
the list, within the spread of the measurement (-4% to +6% across coders, `q` and 1 or 8 threads, arm64). The dense `flatten` option is
encoder-only and costs a pass per block; it decodes to the corrected image, not to the raw values. The GPU encoder does not implement any
of this.

## Predictors

- **`above`**: the pixel in the previous row. Each row depends only on the row before it, so a row is one
  vectorised pass in both the encoder and the decoder. The first row of a block uses the left neighbour.
- **`left_above`**: `(left + above) >> 1`, as in B3D but without the 24 × 24 tiling.
- **`above3`**: the rounded mean of the three pixels above (`x − 1`, `x`, `x + 1`), written as a `mean3d` block with
  no planes, so it needs no decoder change. In mode 1 the prediction sits between the quantiser's grid points by an amount
  set by the sensor noise, so each pixel is rounded on its own grid. That reproduces the raw histogram as dither does,
  without added noise and without the coding cost. Measurements: `COMPARISON.md`.
- **`mean3d`**: the rounded mean of up to 12 causal neighbours: the three pixels above (`x − 1`, `x`, `x + 1`) in
  the current plane and the 3 × 3 patch around the same position in the previous plane. Averaging many neighbours
  gives a prediction with less noise than any single neighbour, which matters when the data is noise-dominated. It
  has no left tap, so rows still vectorise.

In lossy mode every predictor runs on reconstructed values, so the error bound of `q/2` in transform units holds
for each pixel and does not accumulate along the prediction chain.

## Multiscale pyramid

A level made by averaging `n` pixels has its variance divided by `n`, which is the noise of a camera with conversion
`conv/n` and read noise `rn·√n`. Each level is coded with that model, so `q` keeps its meaning (a fraction of that
level's own noise) and the per-pixel bound holds at every level.

## Entropy coders

Rice splits a block into eight interleaved streams (format v10, or v11 with the fitted transform; four-stream blocks of
formats v3–v9 still decode), Huffman and rANS into four, so that a decoder can run independent chains.

- **Adaptive Golomb-Rice (`rice`)**: residuals are coded in groups of 128. A 4-bit code per group selects the Rice
  parameter `k`, or marks the group as sparse (positions plus Elias-gamma values) or all zero. Runs of zero groups
  are coded as a single token. There is no code table to build or transmit.
- **Huffman (`huffman`)**: one canonical Huffman table per block, with zero-run symbols and escapes for large values.
- **rANS (`ans`)**: interleaved range-ANS with per-block frequency tables and up to three local-activity contexts, chosen by
  the magnitude of the residuals just coded.

## Unchanged

The meaning of the two lossy modes, of the step `q` (`q = 1` bounds the error at half the pixel's noise standard
deviation), and of the camera model (conversion, offset, read noise) is the same as in B3D.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

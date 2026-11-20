/*
 * b4d_blosc2.h — B³D microscopy compression as a Blosc2 user-defined codec.
 *
 * Reference: Balázs et al. (2017), "A real-time compression library for
 * microscopy images", bioRxiv 10.1101/164624.
 *
 * Pipeline (per Blosc2 block):
 *   lossless (mode 0): integer prediction -> residuals -> RLE -> Huffman
 *   mode 1  : Anscombe VST -> closed-loop prediction with residual quantisation
 *             (step q, in units of sigma) -> RLE -> Huffman
 *   mode 2  : Anscombe VST -> quantise pixels (step q) -> lossless prediction
 *             -> RLE -> Huffman   (better for localisation microscopy)
 *
 * Usage:
 *   b4d_register();                         // once per process
 *   b4d_set_camera(conv_dn_per_e, offset_dn, read_noise_e);
 *   b4d_set_row_width(width);               // 0 = 1-D prediction only
 *   cparams.compcode      = B4D_CODEC_ID;
 *   cparams.compcode_meta = b4d_meta(1, 1.0f);   // mode 1, q = 1 sigma
 *   cparams.filters[BLOSC2_MAX_FILTERS-1] = BLOSC_NOFILTER;  // no shuffle!
 *   cparams.typesize = 2;                   // uint8 or uint16 only
 *
 * The meta byte encodes: bits 0-1 = mode (0/1/2), bits 2-7 = q in 0.25 sigma
 * units (0 .. 15.75). Camera parameters and row width are taken from the
 * process-wide settings at *compression* time and stored inside each block,
 * so decompression is fully self-contained.
 */
#ifndef B4D_BLOSC2_H
#define B4D_BLOSC2_H

#include <stdint.h>
#include "blosc2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define B4D_CODEC_ID   200       /* must lie in 160..255 (user codec range) */
#define B4D_CODEC_NAME "b4d"
#define B4D_VERSION    3   /* 3: rANS coder with contexts; v1/v2 streams still decode. Written for the legacy predictors. */
#define B4D_VERSION_MAX 11 /* 4: adds the mean3d predictor (header carries the plane height); 5: the original B3D pipeline
                              (B4D_PRED_ORIGINAL); 6: seeded subtractive dither in mode 2; 7: per-pixel noise list in mode 1 (b4d_pixmap);
                              8: dither in mode 1 too, dither of the listed pixels only, the noise list in mode 2 (above predictor),
                              and an order-free dither seed (a GPU computes it in parallel). v6 still decodes; dither now writes v8.
                              9: the exact VST (b4d_vst): its description follows the fixed header, before the noise list.
                              10: Rice with 8 streams (the sizes of streams 3..6 open the payload), without the exact VST;
                              11: the same with the exact VST (its description as in v9). Every Rice block is written as v10 / v11;
                              Huffman / rANS keep their versions, and v3..v9 Rice blocks (4 streams) still decode.
                              Each other version is written only when its option is selected. */

/* Register the codec with Blosc2. Returns 0 on success. */
int b4d_register(void);

/* Camera noise model used by the lossy modes.
 *   conversion : camera gain in DN per electron   (default 1.0)
 *   offset     : camera baseline in DN            (default 0.0)
 *   read_noise : read noise in electrons RMS      (default 0.0)
 */
void b4d_set_camera(float conversion, float offset, float read_noise);

/* Row width (in elements) for 2-D prediction; 0 disables (1-D prediction).
 * For best results choose a Blosc2 blocksize that is a multiple of
 * row_width * typesize so rows never straddle blocks. */
void b4d_set_row_width(uint32_t row_width);

/* Predictor (process-wide, stored per block):
 *   B4D_PRED_LEFT_ABOVE : (left + above)/2 — best ratio, serial within a row
 *   B4D_PRED_ABOVE      : above only — every row is one vectorised pass
 *                         (first row uses left). Recommended for mode 1. */
#define B4D_PRED_LEFT_ABOVE 0
#define B4D_PRED_ABOVE      1
/*   B4D_PRED_MEAN3D     : rounded mean of up to 12 causal neighbours: the three above-row neighbours (x-1, x, x+1)
 *                         of the current plane and the 3x3 patch of the previous plane (same (y, x) +- 1), inside the
 *                         block only (plane height = plane_rows, planes are consecutive runs of plane_rows rows; the
 *                         first row of the block uses the left predictor). No left tap: rows stay vectorised in the
 *                         encoder AND the decoder. Format v4. Noise-free prediction for noise-dominated data:
 *                         ~20% fewer bits lossless and at q <= 1.5 than `above` on this project's cameras. */
#define B4D_PRED_MEAN3D     2
/*   B4D_PRED_ORIGINAL   : the original B3D codec (Balazs 2017) end to end, bit-identical to the HDF5 filter's chunk
 *                         bytes: 24x24-tile (left+above)/2 predictor, its float VST and quantiser, run-length +
 *                         Huffman (cudaCompress). The entropy setting is ignored. Format v5; see b4d_original.c. */
#define B4D_PRED_ORIGINAL   3
/*   B4D_PRED_ABOVE3     : rounded mean of the three above-row neighbours (x-1, x, x+1). An encoder setting, not a new
 *                         stream type: blocks are written as B4D_PRED_MEAN3D with plane height 0 (the block is one plane),
 *                         so they are format v4 and need no decoder change. The first row of a plane is predicted from
 *                         the last row of the previous plane. */
#define B4D_PRED_ABOVE3     4
void b4d_set_plane_rows(uint32_t plane_rows);   /* process-wide plane height for B4D_PRED_MEAN3D (0 = one plane) */
void b4d_set_predictor(int predictor);

/* Entropy coder (process-wide, stored per block):
 *   B4D_ENTROPY_HUFFMAN : per-block canonical Huffman + zero-run symbols
 *   B4D_ENTROPY_RICE    : adaptive Golomb-Rice, 128-sample groups with sparse-group and zero-run modes,
 *                         no code table; a few % larger, faster to write.
 *   B4D_ENTROPY_ANS     : rANS with per-block context-conditioned frequency tables; smallest output,
 *                         5-20% below Huffman when residuals are sparse (high q, denoised input). */
#define B4D_ENTROPY_HUFFMAN 0
#define B4D_ENTROPY_RICE    1
#define B4D_ENTROPY_ANS     2   /* interleaved rANS, 3 local-activity contexts: best ratio at any q */
void b4d_set_entropy(int coder);

/* Seeded subtractive dither (process-wide, stored per block; format v8). The quantiser's bin edges move by a pseudo-random
 * fraction of a step at every pixel and the decoder removes the same fraction, so the error stays within q/2 sigma and no
 * longer depends on the signal: mean bias goes to ~0 at low ADU, variance rises by ~q^2/12. Mode 2 dithers the stabilised
 * value; mode 1 the prediction residual, inside the closed loop. on: 0 off, 1 every pixel, 2 only the pixels of the noise
 * list (b4d_params.pixmap; without a list in the block nothing is dithered). */
void b4d_set_dither(int on);

/* Per-context parameters (preferred for multi-threaded / multi-context use).
 * Point cparams->codec_params at one of these; when it is NULL the process-wide
 * settings above are used. Decoding never needs it (blocks are self-describing). */
typedef struct {
  uint32_t magic;          /* must be B4D_PARAMS_MAGIC */
  float    conversion, offset, read_noise;
  uint32_t row_width;
  int      predictor;      /* B4D_PRED_* */
  int      entropy;        /* B4D_ENTROPY_* (0 huffman, 1 rice, 2 ans) */
  uint32_t plane_rows;     /* rows per plane for B4D_PRED_MEAN3D (0 = the block is one plane); ignored by the other predictors */
  int      dither;         /* seeded subtractive dither (format v8): 0 off, 1 every pixel, 2 the listed pixels only; modes 1 and 2 */
  const struct b4d_pixmap *pixmap;   /* per-pixel calibration (below) or NULL; mode 1 with B4D_PRED_ABOVE3, or mode 2 with B4D_PRED_ABOVE */
  uint32_t map_y0, map_x0;           /* where the block's first pixel sits on the map; plane_rows gives the plane height */
  const struct b4d_vst *vst;         /* exact VST (format v9) for this camera, or NULL for the Anscombe curve; modes 1 and 2 */
} b4d_params;
#define B4D_PARAMS_MAGIC 0x42344450u   /* "B4DP" */

/* Per-pixel calibration of one camera, shared read-only by any number of encoders. The camera model stays global: a short
 * list of pixels (tier[y * width + x] != 0) that are noisier than the model get a coarser step, so q keeps meaning the same
 * fraction of each pixel's own noise. Noise in T units is sqrt(1 + extra / max(T^2 / 4, C)), C = rn^2 + 3/8, extra =
 * extra_var[tier - 1] in e^2, evaluated from the prediction, so the step shrinks back to q as the signal rises. A listed
 * pixel whose step reaches skip * q is left out of the prediction of the row below, so its noise does not spread (skip = 0:
 * never). delta (optional, 1/256 T) is a listed pixel's offset from the sensor's level, removed before prediction and added
 * back on decoding. All of this travels in the block (format v7), so decoding needs no calibration and no other row of the
 * sensor is touched: unlisted pixels cost exactly what they cost without a map.
 * flat_a, flat_b (optional, dense): encoder-only flat fielding, T -= a; T += (T - T(0)) * b / 2^15. The stream is ordinary and
 * decodes to the corrected image; it replaces the raw values, so it is never applied unless given.
 * The encoder needs to know where its rows sit, so a block must be a whole chunk (one Blosc2 block per chunk). */
typedef struct b4d_pixmap b4d_pixmap;
#define B4D_PIX_TIERS 8
b4d_pixmap *b4d_pixmap_new(uint32_t height, uint32_t width, const uint8_t *tier, const int16_t *delta, const int16_t *flat_a,
                           const int16_t *flat_b, const float *extra_var, int ntiers, float skip);
void b4d_pixmap_free(b4d_pixmap *m);

/* Exact variance-stabilising transform (format v9). The generalised Anscombe curve 2 sqrt(e + rn^2 + 3/8) is one sigma per
 * unit only approximately near the offset (up to ~8 % off for real cameras, ADU rounding and clipping included), so q means
 * a different fraction of the noise in the dark than in the light. b4d_vst_new fits a curve that is one sigma per unit for
 * the camera model at every signal (the increments of Anscombe's times a gain g at nk knots; above lam_max Anscombe again).
 * The description (8 + 4 nk bytes) travels in every block, and the decoder rebuilds the same table from it, so decoding
 * needs nothing else. Shared read-only by any number of encoders; free with b4d_vst_free. max_err (optional): the largest
 * |residual| left at the signals it was fitted at.
 * objective: B4D_VST_STD, std(T(X) | lam) = 1 at every signal (the classic exact VST); or B4D_VST_VARIANCE, a step of q adds
 * q^2/12 of the pixel's own variance at every signal (what a noise measurement on decoded data sees). They differ near the
 * offset, where T bends across a dark pixel's spread. */
typedef struct b4d_vst b4d_vst;
#define B4D_VST_MAXK 64
#define B4D_VST_STD      0
#define B4D_VST_VARIANCE 1
b4d_vst *b4d_vst_new(float conversion, float offset, float read_noise, int objective, double *max_err);
b4d_vst *b4d_vst_from_knots(float conversion, float offset, float read_noise, int nk, float lam_max, const float *g);
int b4d_vst_knots(const b4d_vst *v, float *lam_max, float *g);                 /* nk; g needs B4D_VST_MAXK floats */
int b4d_vst_table(const b4d_vst *v, double *table65536);                       /* T(x), x = 0..65535 */
/* std(T(X) | lam) of the camera model for each of nl signals (electrons): through v, or with v NULL the Anscombe curve */
int b4d_vst_std(const b4d_vst *v, float conversion, float offset, float read_noise, const double *lams, int nl, double *std_out);
void b4d_vst_free(b4d_vst *v);

/* Build the compcode_meta byte. mode: 0 lossless, 1, 2. q: sigma multiples. */
uint8_t b4d_meta(int mode, float q);

/* Optional teardown: free the calling thread's scratch buffer / all caches. */
void b4d_free_thread_scratch(void);
void b4d_trim_thread_scratch(size_t keep_bytes);   /* free the calling thread's scratch only if it exceeds keep_bytes */
void b4d_cleanup(void);

/* Encode a block payload (header + streams) directly from int16 residuals, exactly as b4d_encoder
 * does after its prediction/quantisation stage. Used by the GPU path (residuals computed on the
 * device, entropy coding on the host). Returns payload size, 0 if incompressible, <0 on error. */
int b4d_encode_residual_block(const int16_t *residuals, uint32_t nelem, uint8_t mode, uint8_t typesize,
                              uint8_t qi, float conversion, float offset, float read_noise,
                              uint32_t row_width, int predictor, int entropy, uint8_t *out, int32_t out_cap);
/* same, with the plane height the residuals were predicted with (B4D_PRED_MEAN3D; 0 for the other predictors) */
int b4d_encode_residual_block2(const int16_t *residuals, uint32_t nelem, uint8_t mode, uint8_t typesize,
                               uint8_t qi, float conversion, float offset, float read_noise,
                               uint32_t row_width, uint32_t plane_rows, int predictor, int entropy, uint8_t *out, int32_t out_cap);

/* Native zarr v3 sharded block writer (b4d_zarr.c). Writes the shards covering a block of planes with a
 * pool of nthreads threads (chunk compression, shard assembly with a crc32c index, file IO with optional
 * flush). codec: B4D_ZARR_CODEC_*; for BLOBS the chunks are given pre-encoded (one per inner chunk of the
 * block, C order over shards then chunks) and only assembly/IO happen. Returns 0 or a negative error:
 * -2 chunks don't divide shards, -3 block origin not on shard boundaries, -4 block does not cover whole
 * shards (except at the array edge), -5 compression failed, -6 file IO failed. */
#define B4D_ZARR_CODEC_B4D   0
#define B4D_ZARR_CODEC_RAW   1
#define B4D_ZARR_CODEC_BLOSC 2
#define B4D_ZARR_CODEC_BLOBS 3
typedef struct { double t_compress, t_io; int groups_mask; long long bytes_written; int direct_used, direct_fallback;
                 int blosc1_fallback;   /* BLOSC codec: chunks stored uncompressed because the Blosc2->Blosc1 rewrite declined */
               } b4d_zarr_stats;
/* threaded box-mean downsampler: dst = ceil(n/f) per axis; factors 1 or 2 per axis (fast paths) or any integer */
int b4d_zarr_downsample(const void *src, int typesize, int nz, int ny, int nx, int fz, int fy, int fx, void *dst, int nthreads);
/* release cached per-worker arenas and aligned IO buffers (e.g. when a writer closes) */
void b4d_zarr_free_cache(void);
/* Entropy-code ntiles tiles of precomputed int16 residuals (nelem each, rows of cols, planes of plane_rows; the GPU
 * path's residual export) with nthreads pool threads, and frame each as the B4DP + Blosc2 chunk the C encoder would
 * produce. On return *blob_out (malloc'd; free with b4d_zarr_free_buffer) holds the blobs back to back, offsets/lengths
 * [ntiles] index them, status[ntiles] = 0 coded, 1 = left to the caller (all-zero / all-equal / incompressible tile:
 * the caller runs the full encoder on the samples so the bytes match the C path exactly). Returns 0 or -1. */
int b4d_zarr_encode_residual_tiles(const int16_t *res, int ntiles, uint32_t nelem, int mode, int qi, float conv, float offset, float rn,
                                   uint32_t cols, uint32_t plane_rows, int predictor, int entropy, uint8_t meta, int nthreads,
                                   uint8_t **blob_out, size_t *blob_len, int64_t *offsets, int64_t *lengths, int8_t *status);
void b4d_zarr_free_buffer(void *p);

int b4d_zarr_write_block(const void *block, int typesize, int nz, int ny, int nx,
                         int64_t z0, int64_t y0, int64_t x0, const int64_t ashape[3],
                         int sz, int sy, int sx, int cz, int cy, int cx,
                         int codec, const b4d_params *params, uint8_t meta, const char *cname, int clevel, int shuffle,
                         const uint8_t *const *blobs, const size_t *blob_sizes,
                         const char *root, const char *key_sep, int sync, int nthreads, long long *bytes_written, b4d_zarr_stats *stats,
                         void *ds_out, int fz, int fy, int fx, int direct, const char *key_prefix);
/* key_prefix: chunk-key components for leading axes of a higher-rank array (e.g. "0" for channel 0 of a
 * (1, z, y, x) array -> keys c/0/<z>/<y>/<x>); NULL or "" for a 3-D array. */
/* direct: 1 = assemble each shard into a sector-aligned buffer and write it unbuffered + write-through (Windows
 * FILE_FLAG_NO_BUFFERING|WRITE_THROUGH, Linux O_DIRECT), falling back to buffered writes where unavailable.
 * ds_out (optional, uint16 only): level-1 box mean of the block (factors 1 or 2 per axis) computed from the gathered
 * bands during compression, written to a buffer of ceil(n/f) per axis (clipped to the array); NULL to skip. */

/* The original B3D pipeline on one 2-D image (height rows of width pixels; a 3-D chunk is rows = ny*nz, like the
 * original filter). mode 0/1/2 (0 = lossless; q <= 0 forces 0), tile 24 in the original. out: malloc'd bitstream,
 * exactly the bytes the original filter stores for the chunk; free with b4d_orig_free. vst_table: 65536 floats of
 * (VST / q) per pixel value, or NULL for the IEEE table (b4d_orig_vst_table); the original GPU build's fast-math
 * table can be registered process-wide so every encoder call with these parameters picks it up. */
int b4d_orig_encode(const uint16_t *img, uint32_t width, uint32_t height, int mode, float q, float conversion, float offset,
                    float read_noise, uint32_t tile, const float *vst_table, uint8_t **out, size_t *out_len);
int b4d_orig_decode(const uint8_t *payload, size_t len, uint32_t width, uint32_t height, int mode, float q, float conversion,
                    float offset, float read_noise, uint32_t tile, uint16_t *out);
int b4d_orig_vst_table(float *tab65536, float conversion, float offset, float read_noise, float q);
int b4d_orig_register_vst_table(float conversion, float offset, float read_noise, float q, const float *tab65536); /* copy; NULL clears */
const float *b4d_orig_find_vst_table(float conversion, float offset, float read_noise, float q);
void b4d_orig_free(void *p);
#define B4D_ORIG_TILE 24

/* Raw encoder/decoder callbacks (exposed for testing). */
int b4d_encoder(const uint8_t *input, int32_t input_len, uint8_t *output,
                int32_t output_len, uint8_t meta, blosc2_cparams *cparams,
                const void *chunk);
int b4d_decoder(const uint8_t *input, int32_t input_len, uint8_t *output,
                int32_t output_len, uint8_t meta, blosc2_dparams *dparams,
                const void *chunk);

#ifdef __cplusplus
}
#endif
#endif /* B4D_BLOSC2_H */

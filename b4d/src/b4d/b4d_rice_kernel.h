/*
 * b4d_rice_kernel.h — B4D (mode 2 / mode 0, 'above' or 'mean3d' predictor, Rice coder) tile encoder as a CUDA kernel;
 * b4d_resid_tiles exports the int16 residuals of a batch of tiles for a host entropy coder (ANS).
 *
 * One thread block per tile (zarr chunk); 256 threads. Thread t owns stream position t (and t+256, ...) of
 * the B4DK_NS = 8 interleaved Rice streams (format v10 / v11): group g = 8*pos + s. All planning, zero-run detection, prefix sums and
 * bit packing are block-local. The output of each tile is the complete zarr chunk blob (B4DP container +
 * Blosc2 chunk header + block + B4D block header + 8 streams), byte-identical to the C codec, compacted into
 * a shared output buffer with an atomic bump allocator so only compressed bytes leave the device.
 *
 * The same source compiles with NVRTC (CuPy RawModule) and, with B4D_EMU defined, as plain C where the
 * kernel phases are run sequentially per thread index (used by the test suite on machines without a GPU).
 *
 * Persistent mode (sm_limit > 0): only thread blocks resident on SMs with id < sm_limit work; they take tiles from a
 * global counter, so a launch uses at most sm_limit SMs (the scaling benchmark's compute axis). Output is unchanged.
 *
 * Limits: typesize 1 or 2 (uint8 tiles are widened by the gather), tile rows*cols <= B4DK_MAXE (compile-time, default 262144 = 64x64x64), cols <= 1024.
 * Tiles that would not compress (block payload >= raw size), tiles whose bytes are all one repeated value,
 * and all-zero tiles are flagged for the host (status != 0): the host emits those chunks itself (rare).
 */
#ifndef B4DK_MAXE
#define B4DK_MAXE 262144
#endif
#define B4DK_G 128
#define B4DK_MAXG (B4DK_MAXE / B4DK_G)        /* groups per tile */
#define B4DK_NS 8                             /* Rice streams (v10 / v11; RICE_NS in b4d_blosc2.c) */
#define B4DK_T (B4DK_MAXG / B4DK_NS)          /* positions per stream */
#define B4DK_THREADS 256
#define B4DK_KMAX 13
#define B4DK_UMAX 31
#define B4DK_POSBITS 7
#define B4DK_ZRUNMAX 8
#define B4DK_HDR 44

#ifdef B4D_EMU
  #include <stdint.h>
  #include <string.h>
  #define __global__
  #define __device__
  #define __shared__ static
  #define __restrict__
  typedef struct { unsigned x; } b4dk_dim3;
  static b4dk_dim3 threadIdx, blockIdx;
  static inline unsigned atomicOr(unsigned *p, unsigned v) { unsigned o = *p; *p |= v; return o; }
  static inline unsigned atomicAnd(unsigned *p, unsigned v) { unsigned o = *p; *p &= v; return o; }
  static inline unsigned atomicAdd(unsigned *p, unsigned v) { unsigned o = *p; *p += v; return o; }
  static inline unsigned long long atomicAdd64(unsigned long long *p, unsigned long long v) { unsigned long long o = *p; *p += v; return o; }
  static inline int b4dk_clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
  static inline unsigned b4dk_bswap(unsigned x) { return __builtin_bswap32(x); }
  #define B4DK_SYNC() /* phases are looped per thread in emulation */
#else
  typedef unsigned char uint8_t; typedef unsigned short uint16_t; typedef unsigned int uint32_t; typedef int int32_t;
  typedef unsigned long long uint64_t; typedef short int16_t; typedef signed char int8_t;
  __device__ __forceinline__ unsigned long long atomicAdd64(unsigned long long *p, unsigned long long v) { return atomicAdd(p, v); }
  __device__ __forceinline__ int b4dk_clz(unsigned x) { return __clz((int)x); }
  __device__ __forceinline__ unsigned b4dk_bswap(unsigned x) { return __byte_perm(x, 0, 0x0123); }
  #define B4DK_SYNC() __syncthreads()
#endif

/* ---- per-tile parameters (one struct for the whole launch) */
typedef struct {
  int bz, by, bx;                 /* uploaded block dims; tiles beyond (bz, by, bx) read as zero (edge padding) */
  int ty, tx;                     /* tile grid (tz is implied by ntiles) */
  int cz, cy;                     /* tile depth and height (rows = cz*cy) */
  int rows, cols;                 /* tile rows (cz*cy) and columns (cx) */
  int nelem;                      /* rows*cols */
  int ngroups;                    /* ceil(nelem / 128) */
  int T;                          /* positions per stream = ceil(ngroups / B4DK_NS) */
  int mode, qi;                   /* header fields */
  int pred;                       /* 1 = above, 2 = mean3d (format v4) */
  float conv, offset, rn;
  unsigned long long out_cap;     /* capacity of the compact output buffer */
  int scratch_words;              /* words per stream in scratch (per tile B4DK_NS streams) */
  int ntiles;
  int ts;                         /* element size of the source array: 1 (uint8) or 2 (uint16); header fields and raw size */
  int vst_bytes;                  /* format v9: size of the exact-VST description that follows the 44-byte header (0: none) */
  const uint8_t *vst;             /* its bytes (device memory; host memory in the emulation) */
} b4dk_params;

/* ---- shared state of one tile */
typedef struct {
  int8_t  k[B4DK_MAXG];           /* 0..13 rice, 14 sparse, 15 zero, -1 nonexistent */
  int32_t vbits[B4DK_MAXG];       /* bits of the group's value fields */
  int32_t hbits[B4DK_MAXG];       /* header bits (0 for a zero group inside a run) */
  int32_t hval[B4DK_MAXG];
  int16_t nnz[B4DK_MAXG];
  int32_t sbits[B4DK_NS][B4DK_T];       /* per stream: group total bits -> exclusive prefix (in place) */
  int16_t rstart[B4DK_NS][B4DK_T];      /* zero-run start position per stream (-1) */
  int16_t rend[B4DK_NS][B4DK_T];
  int32_t tmp[B4DK_NS][B4DK_T];
  unsigned flags;                 /* bit0: any nonzero; bit1: not all-equal */
  unsigned stream_bits[B4DK_NS];
  unsigned total_payload;         /* block payload bytes */
  unsigned long long out_off;
  int status;                     /* 0 ok, 1 incompressible, 2 all zero, 3 repeated byte */
  int dith; unsigned seed;        /* v8 subtractive dither: the header carries FLAG_DITHER and the seed (4 more bytes) */
} b4dk_smem;

__device__ static inline int b4dk_floor_log2(unsigned u) { return 31 - b4dk_clz(u); }

/* tile view: element i (row-major within the tile) read from the block; outside the block = 0 */
typedef struct { const uint16_t *base; int oz, oy, ox; int bz, by, bx; int cy, cols; long long sy, sz; int xs, ys;
                 const short *res; } b4dk_tile;   /* res: this tile's precomputed residuals (b4d_resid_tiles) or NULL */
/* xs/ys: log2(cols) / log2(cy) when both are powers of two (shift/mask addressing), else -1 (divide) */
__device__ static inline unsigned b4dk_ld(const b4dk_tile *t, int i) {
  int x, y, z;
  if (t->xs >= 0) { x = i & (t->cols - 1); int r = i >> t->xs; y = r & (t->cy - 1); z = r >> t->ys; }
  else { x = i % t->cols; int r = i / t->cols; y = r % t->cy; z = r / t->cy; }
  int gz = t->oz + z, gy = t->oy + y, gx = t->ox + x;
  if (gz >= t->bz || gy >= t->by || gx >= t->bx) return 0u;
  return (unsigned)t->base[(long long)gz * t->sz + (long long)gy * t->sy + gx];
}
__device__ static inline int b4dk_log2_exact(int v) { int s = 0; while ((1 << s) < v) s++; return (1 << s) == v ? s : -1; }
/* quantised value of tile element i (LUT in mode 2, raw sample otherwise) */
__device__ static inline unsigned b4dk_vq(const b4dk_tile *t, const uint16_t *lut, int i, int mode) {
  unsigned x = b4dk_ld(t, i); return mode == 2 ? (unsigned)lut[x] : x;
}
/* prediction of element i. pred 1 ('above'): the row above, left in the first row. pred 2 ('mean3d'): rounded mean
   of the existing taps among the above row's (x-1, x, x+1) and the previous plane's 3x3 patch, planes of cy rows;
   the block's first row uses left. Identical arithmetic to mean3d_row_u16 in b4d_blosc2.c for a full tile. */
__device__ static inline unsigned b4dk_pred_at(const b4dk_tile *tile, const uint16_t *lut, int i, int cols, int cy, int mode, int pred) {
  if (pred == 2) {
    int plane = cy * cols; int z = i / plane; int rem = i - z * plane; int y = rem / cols; int x = rem - y * cols;
    if (z == 0 && y == 0) return x ? b4dk_vq(tile, lut, i - 1, mode) : 0u;
    unsigned s = 0, c = 0;
    for (int d = -1; d <= 1; d++) {
      int xx = x + d; if (xx < 0 || xx >= cols) continue;
      if (y > 0) { s += b4dk_vq(tile, lut, i - cols + d, mode); c++; }
      if (z > 0) {
        int b = i - plane + d;
        if (y > 0) { s += b4dk_vq(tile, lut, b - cols, mode); c++; }
        s += b4dk_vq(tile, lut, b, mode); c++;
        if (y + 1 < cy) { s += b4dk_vq(tile, lut, b + cols, mode); c++; }
      }
    }
    return (s + c / 2u) / c;
  }
  if (i < cols) return (i == 0) ? 0u : b4dk_vq(tile, lut, i - 1, mode);     /* first row: left */
  return b4dk_vq(tile, lut, i - cols, mode);                                  /* above */
}
/* 16-bit residual (mod 2^16) and its zigzag code for element i */
__device__ static inline unsigned b4dk_r_at(const b4dk_tile *tile, const uint16_t *lut, int i, int cols, int cy, int mode, int pred) {
  return (b4dk_vq(tile, lut, i, mode) - b4dk_pred_at(tile, lut, i, cols, cy, mode, pred)) & 0xFFFFu;
}
__device__ static inline unsigned b4dk_u_at(const b4dk_tile *tile, const uint16_t *lut, int i, int cols, int cy, int mode, int pred) {
  unsigned r = tile->res ? ((unsigned)(unsigned short)tile->res[i]) : b4dk_r_at(tile, lut, i, cols, cy, mode, pred);
  int sgn = (int)r - (int)((r & 0x8000u) << 1);
  return (unsigned)((sgn << 1) ^ (sgn >> 15)) & 0xFFFFu;
}
/* tile view of tile (tile0 + tile_i) of the block, contiguous (gathered) or in place */
__device__ static inline void b4dk_tile_setup(b4dk_tile *tv, const uint16_t *block, int tile_i, int tile0, int contig,
                                              int bz, int by, int bx, int ty, int tx, int cz, int cy, int cols, int nelem) {
  if (contig) {
    tv->base = block + (size_t)tile_i * (size_t)nelem; tv->oz = tv->oy = tv->ox = 0; tv->bz = cz; tv->by = cy; tv->bx = cols;
    tv->cy = cy; tv->cols = cols; tv->sy = cols; tv->sz = (long long)cy * cols;
  } else {
    int g = tile0 + tile_i; int k = g % tx; int j = (g / tx) % ty; int i = g / (tx * ty);
    tv->base = block; tv->oz = i * cz; tv->oy = j * cy; tv->ox = k * cols; tv->bz = bz; tv->by = by; tv->bx = bx; tv->cy = cy; tv->cols = cols;
    tv->sy = bx; tv->sz = (long long)by * bx;
  }
  int xs = b4dk_log2_exact(cols), ys = b4dk_log2_exact(cy); tv->xs = (xs >= 0 && ys >= 0) ? xs : -1; tv->ys = ys;
  tv->res = NULL;
}

/* ---- phase 0: zero this tile's scratch (B4DK_NS streams) and the shared flags */
__device__ static void b4dk_phase_zero(int tid, const b4dk_params *P, b4dk_smem *S, unsigned *scratch) {
  if (tid == 0) { S->flags = 0; S->status = 0; for (int s = 0; s < B4DK_NS; s++) S->stream_bits[s] = 0; }
  unsigned n = (unsigned)B4DK_NS * (unsigned)P->scratch_words;
  for (unsigned i = (unsigned)tid; i < n; i += B4DK_THREADS) scratch[i] = 0u;
}

/* ---- phase 1: per group plan (k, header, value bits), per-tile flags */
__device__ static void b4dk_phase_plan(int tid, const b4dk_params *P, const b4dk_tile *tile, const uint16_t *lut, b4dk_smem *S) {
  for (int g = tid; g < P->T * B4DK_NS; g += B4DK_THREADS) {
    if (g >= P->ngroups) { S->k[g] = -1; S->vbits[g] = 0; S->hbits[g] = 0; S->hval[g] = 0; S->nnz[g] = 0; continue; }
    int i0 = g * B4DK_G, glen = P->nelem - i0; if (glen > B4DK_G) glen = B4DK_G;
    unsigned sum = 0, nnz = 0, umax = 0; unsigned sbits = 4 + B4DK_POSBITS;
    unsigned any = 0, notsame = 0; unsigned x0 = b4dk_ld(tile, 0);
    uint16_t ug[B4DK_G];
    for (int j = 0; j < glen; j++) {
      unsigned x = b4dk_ld(tile, i0 + j); any |= x; notsame |= (x != x0);
      unsigned u = b4dk_u_at(tile, lut, i0 + j, P->cols, P->cy, P->mode, P->pred); ug[j] = (uint16_t)u;
      sum += u; nnz += (u != 0); umax = u > umax ? u : umax;
      if (u) sbits += B4DK_POSBITS + 2 * b4dk_floor_log2(u) + 1;
    }
    if (any) atomicOr(&S->flags, 1u);
    if (notsame) atomicOr(&S->flags, 2u);
    S->nnz[g] = (int16_t)nnz;
    if (sum == 0) { S->k[g] = 15; S->vbits[g] = 0; S->hbits[g] = 7; S->hval[g] = (15 << 3); continue; }  /* run headers fixed later */
    unsigned mean = sum / (unsigned)glen; int k0 = mean ? b4dk_floor_log2(mean) : 0; if (k0 > B4DK_KMAX) k0 = B4DK_KMAX;
    int klo = k0 > 0 ? k0 - 1 : 0, khi = k0 < B4DK_KMAX ? k0 + 1 : B4DK_KMAX;
    unsigned best = 0xFFFFFFFFu; int bestk = k0;
    for (int kk = klo; kk <= khi; kk++) {
      unsigned bits = 4;
      if ((umax >> kk) < B4DK_UMAX) {
        for (int j = 0; j < glen; j++) bits += (unsigned)ug[j] >> kk;
        bits += (unsigned)glen * (unsigned)(kk + 1);
      } else {
        for (int j = 0; j < glen; j++) { unsigned q = (unsigned)ug[j] >> kk; bits += (q >= B4DK_UMAX) ? (B4DK_UMAX + 16) : (q + 1 + kk); }
      }
      if (bits < best) { best = bits; bestk = kk; }
    }
    if (nnz < B4DK_G && sbits < best) { S->k[g] = 14; S->vbits[g] = (int)(sbits - 4 - B4DK_POSBITS); S->hbits[g] = 4 + B4DK_POSBITS; S->hval[g] = (14 << B4DK_POSBITS) | (int)nnz; }
    else { S->k[g] = (int8_t)bestk; S->vbits[g] = (int)(best - 4); S->hbits[g] = 4; S->hval[g] = bestk; }
  }
}

/* ---- phase 2: zero runs per stream (start/end via scans), header bits for run heads, group totals */
__device__ static void b4dk_phase_runs_init(int tid, const b4dk_params *P, b4dk_smem *S) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) {
      int g = B4DK_NS * pos + s; int z = (S->k[g] == 15);
      int zprev = pos > 0 ? (S->k[g - B4DK_NS] == 15) : 0;
      int znext = pos + 1 < P->T ? (S->k[g + B4DK_NS] == 15) : 0;
      S->rstart[s][pos] = (int16_t)((z && !zprev) ? pos : -1);
      S->rend[s][pos] = (int16_t)((z && !znext) ? pos : 0x7FFF);
    }
}
/* one Hillis-Steele step of prefix-max on rstart and suffix-min on rend (called with stride doubling) */
__device__ static void b4dk_phase_runs_step_read(int tid, const b4dk_params *P, b4dk_smem *S, int stride) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) {
      int a = S->rstart[s][pos]; if (pos >= stride) { int b = S->rstart[s][pos - stride]; a = b > a ? b : a; }
      int e = S->rend[s][pos]; if (pos + stride < P->T) { int f = S->rend[s][pos + stride]; e = f < e ? f : e; }
      S->tmp[s][pos] = (a & 0xFFFF) | (e << 16);
    }
}
__device__ static void b4dk_phase_runs_step_write(int tid, const b4dk_params *P, b4dk_smem *S) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) { int v = S->tmp[s][pos]; S->rstart[s][pos] = (int16_t)(v & 0xFFFF); S->rend[s][pos] = (int16_t)(v >> 16); }
}
__device__ static void b4dk_phase_totals(int tid, const b4dk_params *P, b4dk_smem *S) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) {
      int g = B4DK_NS * pos + s;
      if (S->k[g] == 15) {
        int start = S->rstart[s][pos], end = S->rend[s][pos];
        int pin = pos - start, runlen = end - start + 1;
        if (pin % B4DK_ZRUNMAX == 0) { int zc = runlen - pin; if (zc > B4DK_ZRUNMAX) zc = B4DK_ZRUNMAX; S->hbits[g] = 7; S->hval[g] = (15 << 3) | (zc - 1); }
        else { S->hbits[g] = 0; S->hval[g] = 0; }
      }
      S->sbits[s][pos] = (S->k[g] < 0) ? 0 : S->hbits[g] + S->vbits[g];
    }
}
/* exclusive prefix sum per stream over T positions: Hillis-Steele inclusive steps, then shift */
__device__ static void b4dk_phase_scan_read(int tid, const b4dk_params *P, b4dk_smem *S, int stride) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) { int v = S->sbits[s][pos]; if (pos >= stride) v += S->sbits[s][pos - stride]; S->tmp[s][pos] = v; }
}
__device__ static void b4dk_phase_scan_write(int tid, const b4dk_params *P, b4dk_smem *S) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS) for (int s = 0; s < B4DK_NS; s++) S->sbits[s][pos] = S->tmp[s][pos];
}
__device__ static void b4dk_phase_scan_finish(int tid, const b4dk_params *P, b4dk_smem *S) {
  /* inclusive -> exclusive: tmp holds inclusive; sbits[pos] = inclusive[pos-1] */
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) { S->tmp[s][pos] = pos > 0 ? S->sbits[s][pos - 1] : 0; }
}
__device__ static void b4dk_phase_scan_commit(int tid, const b4dk_params *P, b4dk_smem *S) {
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) {
      if (pos == P->T - 1) {
        S->stream_bits[s] = (unsigned)S->sbits[s][pos];                       /* inclusive sum at the last position = total */
        if (S->stream_bits[s] > (unsigned)P->scratch_words * 32u - 64u) S->status = 1;   /* would overflow the stream scratch */
      }
      S->sbits[s][pos] = S->tmp[s][pos];                                      /* exclusive prefix */
    }
}

/* ---- bit emission into a stream word buffer: MSB-first, big-endian words stored as bswapped LE words */
__device__ static inline void b4dk_emit(unsigned *words, unsigned bitpos, unsigned nbits, unsigned val) {
  if (!nbits) return;
  unsigned w = bitpos >> 5, off = bitpos & 31u;
  if (off + nbits <= 32u) { atomicOr(&words[w], b4dk_bswap(val << (32u - off - nbits))); }
  else {
    unsigned over = off + nbits - 32u;                              /* bits that spill into the next word */
    atomicOr(&words[w], b4dk_bswap(val >> over));
    atomicOr(&words[w + 1], b4dk_bswap((val & ((1u << over) - 1u)) << (32u - over)));
  }
}

/* ---- bit writer for one group's fields: the group's bit range is contiguous, so its interior words belong to this
   thread alone and are stored whole; only its first and last words, shared with the neighbouring groups, use atomicOr
   (on the zeroed scratch). Same bits as emitting each field with b4dk_emit. */
typedef struct { unsigned *words; unsigned w; unsigned long long acc; unsigned nacc; int shared_first; } b4dk_bw;
__device__ static inline void b4dk_bw_init(b4dk_bw *b, unsigned *words, unsigned bitpos) {
  b->words = words; b->w = bitpos >> 5; b->nacc = bitpos & 31u; b->acc = 0ull; b->shared_first = (b->nacc != 0);
}
__device__ static inline void b4dk_bw_put(b4dk_bw *b, unsigned nbits, unsigned val) {   /* nbits <= 32 */
  if (!nbits) return;
  b->acc = (b->acc << nbits) | (unsigned long long)(nbits < 32u ? (val & ((1u << nbits) - 1u)) : val); b->nacc += nbits;
  if (b->nacc >= 32u) {
    unsigned word = (unsigned)(b->acc >> (b->nacc - 32u));
    if (b->shared_first) { atomicOr(&b->words[b->w], b4dk_bswap(word)); b->shared_first = 0; }
    else b->words[b->w] = b4dk_bswap(word);
    b->w++; b->nacc -= 32u; b->acc &= (b->nacc ? ((1ull << b->nacc) - 1ull) : 0ull);
  }
}
__device__ static inline void b4dk_bw_end(b4dk_bw *b) {
  if (b->nacc) atomicOr(&b->words[b->w], b4dk_bswap((unsigned)(b->acc << (32u - b->nacc))));
}

/* ---- phase 3: pack every group's fields at its stream offset */
__device__ static void b4dk_phase_pack(int tid, const b4dk_params *P, const b4dk_tile *tile, const uint16_t *lut, b4dk_smem *S, unsigned *scratch) {
  if (S->status) return;
  for (int pos = tid; pos < P->T; pos += B4DK_THREADS)
    for (int s = 0; s < B4DK_NS; s++) {
      int g = B4DK_NS * pos + s; int k = S->k[g]; if (k < 0) continue;
      b4dk_bw b; b4dk_bw_init(&b, scratch + (size_t)s * (size_t)P->scratch_words, (unsigned)S->sbits[s][pos]);
      if (S->hbits[g]) b4dk_bw_put(&b, (unsigned)S->hbits[g], (unsigned)S->hval[g]);
      if (k != 15) {
        int i0 = g * B4DK_G, glen = P->nelem - i0; if (glen > B4DK_G) glen = B4DK_G;
        if (k == 14) {
          for (int j = 0; j < glen; j++) {
            unsigned u = b4dk_u_at(tile, lut, i0 + j, P->cols, P->cy, P->mode, P->pred); if (!u) continue;
            b4dk_bw_put(&b, B4DK_POSBITS, (unsigned)j);
            unsigned nb = (unsigned)b4dk_floor_log2(u) + 1u; b4dk_bw_put(&b, 2u * nb - 1u, u);
          }
        } else {
          for (int j = 0; j < glen; j++) {
            unsigned u = b4dk_u_at(tile, lut, i0 + j, P->cols, P->cy, P->mode, P->pred); unsigned q = u >> k;
            if (q >= B4DK_UMAX) { b4dk_bw_put(&b, B4DK_UMAX, 0x7FFFFFFFu); b4dk_bw_put(&b, 16u, u); }
            else { b4dk_bw_put(&b, q + 1u, ((1u << q) - 1u) << 1); if (k) b4dk_bw_put(&b, (unsigned)k, u & ((1u << k) - 1u)); }
          }
        }
      }
      b4dk_bw_end(&b);
    }
}

/* ---- phase 4 (thread 0): sizes, status, output offset (bump allocation) */
__device__ static void b4dk_phase_size(int tid, const b4dk_params *P, b4dk_smem *S, unsigned long long *out_used) {
  if (tid) return;
  unsigned sb[B4DK_NS]; unsigned payload = B4DK_HDR + (unsigned)P->vst_bytes + (S->dith ? 4u : 0u) + 4u * (B4DK_NS - 4);   /* + sizes of streams 3..6 */
  for (int s = 0; s < B4DK_NS; s++) { sb[s] = (S->stream_bits[s] + 7u) >> 3; payload += sb[s]; if (sb[s] > (unsigned)P->scratch_words * 4u) S->status = 1; }
  S->total_payload = payload;
  unsigned raw = (unsigned)P->nelem * (unsigned)P->ts;
  if (!(S->flags & 1u)) S->status = 2;                                  /* all zero: zero-run chunk (host) */
  else if (!(S->flags & 2u)) { unsigned x0 = 0; (void)x0; S->status = 3; }  /* all equal: host decides repeated-byte form */
  else if (payload + 40u >= raw) S->status = 1;                        /* Blosc2 would store the chunk raw: host C path */
  if (S->status) { S->out_off = ~0ull; return; }
  unsigned blob = 39u + 32u + 4u + 4u + payload;                      /* B4DP(7+24+8) + blosc2 hdr + bstart + csize + payload */
  unsigned long long off = atomicAdd64(out_used, (unsigned long long)blob);
  S->out_off = off;
  if (off + blob > P->out_cap) { S->status = 4; }                     /* output buffer full: host retries the batch */
}

__device__ static inline void b4dk_put_u32(uint8_t *p, unsigned v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
__device__ static inline void b4dk_put_f32(uint8_t *p, float f) { unsigned v; memcpy(&v, &f, 4); b4dk_put_u32(p, v); }

/* ---- phase 5: write the blob (headers by thread 0, stream bytes by all threads) */
__device__ static void b4dk_phase_write(int tid, const b4dk_params *P, b4dk_smem *S, const unsigned *scratch, uint8_t *out, int rows, int cols) {
  if (S->status) return;
  uint8_t *o = out + S->out_off;
  unsigned sb[B4DK_NS]; for (int s = 0; s < B4DK_NS; s++) sb[s] = (S->stream_bits[s] + 7u) >> 3;
  unsigned payload = S->total_payload, nbytes = (unsigned)P->nelem * (unsigned)P->ts;
  if (tid == 0) {
    /* B4DP container: magic, version 1, dtype 1 (uint16), ndim 3, shape (cz, cy, cx) as 3 x u64, npieces=1, piece size */
    o[0] = 'B'; o[1] = '4'; o[2] = 'D'; o[3] = 'P'; o[4] = 1; o[5] = (uint8_t)(P->ts == 1 ? 0 : 1); o[6] = 3;
    for (int i = 7; i < 31; i++) o[i] = 0;
    b4dk_put_u32(o + 7, (unsigned)P->cz); b4dk_put_u32(o + 15, (unsigned)P->cy); b4dk_put_u32(o + 23, (unsigned)cols);   /* 3 x u64 LE */
    unsigned csize = 32u + 4u + 4u + payload;
    b4dk_put_u32(o + 31, 1u); b4dk_put_u32(o + 35, csize);
    uint8_t *c = o + 39;
    /* Blosc2 header (32 bytes) */
    c[0] = 5; c[1] = 3; c[2] = 0xD5; c[3] = (uint8_t)P->ts;
    b4dk_put_u32(c + 4, nbytes); b4dk_put_u32(c + 8, nbytes); b4dk_put_u32(c + 12, csize);
    for (int i = 16; i < 32; i++) c[i] = 0;
    c[22] = 200; c[23] = (uint8_t)((P->mode & 3) | (P->qi << 2)); c[31] = 0;
    b4dk_put_u32(c + 32, 32u + 4u);                                   /* block start */
    b4dk_put_u32(c + 36, payload);                                    /* block csize */
    uint8_t *h = c + 40;                                              /* B4D block header (44 bytes) */
    h[0] = (uint8_t)(P->vst_bytes ? 11 : 10);                        /* v10 / v11: 8 Rice streams (carry every earlier option) */ h[1] = (uint8_t)(P->mode & 3); h[2] = (uint8_t)P->ts; h[3] = (uint8_t)P->qi;
    b4dk_put_f32(h + 4, P->conv); b4dk_put_f32(h + 8, P->offset); b4dk_put_f32(h + 12, P->rn);
    b4dk_put_u32(h + 16, (unsigned)cols); b4dk_put_u32(h + 20, (unsigned)P->nelem); b4dk_put_u32(h + 24, (unsigned)P->ngroups);
    b4dk_put_u32(h + 28, sb[0]); b4dk_put_u32(h + 32, sb[1]); b4dk_put_u32(h + 36, sb[2]);
    h[40] = (uint8_t)((P->pred >= 2 ? 8u : 1u) | 2u | (S->dith ? 32u : 0u));   /* predictor flag | FLAG_RICE (3 = above3: mean3d, no planes) | FLAG_DITHER */
    h[41] = h[42] = h[43] = 0;
    if (P->pred == 2) { h[41] = (uint8_t)(P->cy & 0xFF); h[42] = (uint8_t)((P->cy >> 8) & 0xFF); h[43] = (uint8_t)((P->cy >> 16) & 0xFF); }
    for (int i = 0; i < P->vst_bytes; i++) h[B4DK_HDR + i] = P->vst[i];   /* v9: the exact VST's description */
    if (S->dith) b4dk_put_u32(h + B4DK_HDR + P->vst_bytes, S->seed);  /* the seed ends the header */
    uint8_t *xs = h + B4DK_HDR + P->vst_bytes + (S->dith ? 4 : 0);    /* v10: the sizes of streams 3..6 open the payload */
    for (int s = 3; s < B4DK_NS - 1; s++) b4dk_put_u32(xs + 4 * (s - 3), sb[s]);
  }
  /* streams: bytes of the (bswapped) word buffers, concatenated at byte granularity */
  uint8_t *dst = o + 39 + 40 + B4DK_HDR + P->vst_bytes + (S->dith ? 4 : 0) + 4 * (B4DK_NS - 4);
  unsigned base = 0;
  for (int s = 0; s < B4DK_NS; s++) {
    const uint8_t *src = (const uint8_t *)(scratch + (size_t)s * (size_t)P->scratch_words);
    for (unsigned i = tid; i < sb[s]; i += B4DK_THREADS) dst[base + i] = src[i];
    base += sb[s];
  }
  (void)rows;
}

/* ---- the kernel: tiles (ntiles, rows, cols) uint16; lut 65536 x uint16; scratch per tile B4DK_NS*scratch_words words
   (zeroed by the host before launch); out compact buffer; out_used bump counter; per tile: offsets, lengths, status */
#ifndef B4D_EMU
/* gather tiles [tile0, tile0+ntiles) of the block into a contiguous (ntiles, cz*cy, cols) buffer; elements beyond
   the block (edge tiles) are zero. One thread per output element. */
__device__ static inline unsigned b4dk_smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
/* persistent mode: the next work item for this thread block (all threads get the same value), or -1 when done */
__device__ static inline long long b4dk_next(unsigned long long *work, long long n) {
  __shared__ long long item;
  __syncthreads();
  if (threadIdx.x == 0) item = (long long)atomicAdd(work, 1ull);
  __syncthreads();
  return item < n ? item : -1;
}
/* persistent mode: does this thread block work? Blocks on SMs >= sm_limit leave, except the last block of the grid to
   start (ticket in work[1]), which always works: with another kernel on the device (two pipeline slots) every block of
   a launch can land on SMs >= sm_limit, and the launch then ended with its tiles never encoded. */
__device__ static inline bool b4dk_admit(int sm_limit, unsigned long long *work) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    unsigned long long tk = atomicAdd(work + 1, 1ull);
    ok = (b4dk_smid() < (unsigned)sm_limit) || (tk == (unsigned long long)gridDim.x - 1);
  }
  __syncthreads();
  return ok != 0;
}

template <typename T>
__device__ static inline void b4dk_gather_el(const T *__restrict__ block, uint16_t *__restrict__ tiles, long long idx, long long nelem,
                                             int bz, int by, int bx, int ty, int tx, int tile0, int cz, int cy, int cols) {
  int t = (int)(idx / nelem); int i = (int)(idx - (long long)t * nelem);
  int x = i % cols; int r = i / cols; int y = r % cy; int z = r / cy;
  int g = tile0 + t; int k = g % tx; int j = (g / tx) % ty; int ii = g / (tx * ty);
  int gz = ii * cz + z, gy = j * cy + y, gx = k * cols + x;
  tiles[idx] = (gz < bz && gy < by && gx < bx) ? (uint16_t)block[((long long)gz * by + gy) * bx + gx] : (uint16_t)0;
}
template <typename T>
__device__ static inline void b4dk_gather(const T *__restrict__ block, uint16_t *__restrict__ tiles, int bz, int by, int bx, int ty, int tx,
                                          int tile0, int cz, int cy, int cols, int ntiles, int sm_limit, unsigned long long *work) {
  long long nelem = (long long)cz * cy * cols, total = nelem * ntiles;
  if (sm_limit <= 0) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) b4dk_gather_el(block, tiles, idx, nelem, bz, by, bx, ty, tx, tile0, cz, cy, cols);
    return;
  }
  if (!b4dk_admit(sm_limit, work)) return;
  const long long span = (long long)blockDim.x * 64, items = (total + span - 1) / span;
  for (long long w; (w = b4dk_next(work, items)) >= 0;)
    for (long long idx = w * span + threadIdx.x; idx < (w + 1) * span && idx < total; idx += blockDim.x)
      b4dk_gather_el(block, tiles, idx, nelem, bz, by, bx, ty, tx, tile0, cz, cy, cols);
}
extern "C" __global__ void b4d_gather_tiles(const uint16_t *__restrict__ block, uint16_t *__restrict__ tiles,
                                            int bz, int by, int bx, int ty, int tx, int tile0, int cz, int cy, int cols, int ntiles,
                                            int sm_limit, unsigned long long *work) {
  b4dk_gather(block, tiles, bz, by, bx, ty, tx, tile0, cz, cy, cols, ntiles, sm_limit, work);
}
/* the same from a uint8 block: tiles are widened to uint16, as the C encoder widens 8-bit input */
extern "C" __global__ void b4d_gather_tiles8(const uint8_t *__restrict__ block, uint16_t *__restrict__ tiles,
                                             int bz, int by, int bx, int ty, int tx, int tile0, int cz, int cy, int cols, int ntiles,
                                             int sm_limit, unsigned long long *work) {
  b4dk_gather(block, tiles, bz, by, bx, ty, tx, tile0, cz, cy, cols, ntiles, sm_limit, work);
}

#ifndef B4D_EMU
extern "C" __global__ void b4d_smem_bytes(unsigned *out) { if (threadIdx.x == 0) *out = (unsigned)sizeof(b4dk_smem); }

/* int16 residuals (mod 2^16, as b4d_encoder stores them) of tiles [tile0, tile0+ntiles): res[tile_i * nelem + i].
   One thread per element; the host entropy coder (b4d_encode_residual_block2) turns them into block payloads. */
extern "C" __global__ void b4d_resid_tiles(const uint16_t *__restrict__ block, const uint16_t *__restrict__ lut,
                                           int bz, int by, int bx, int ty, int tx, int tile0, int contig,
                                           int cz, int cy, int cols, int mode, int pred, int ntiles, short *__restrict__ res) {
  int nelem = cz * cy * cols;
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x, total = (long long)ntiles * nelem;
  if (gid >= total) return;
  int tile_i = (int)(gid / nelem), i = (int)(gid - (long long)tile_i * nelem);
  b4dk_tile tv; b4dk_tile_setup(&tv, block, tile_i, tile0, contig, bz, by, bx, ty, tx, cz, cy, cols, nelem);
  res[gid] = (short)b4dk_r_at(&tv, lut, i, cols, cy, mode, pred);
}

/* ---- v8 subtractive dither, as b4d_blosc2.c: the tile's seed is an order-free sum of mixed 64-bit words of its input
   bytes (dither_seed8), element i's dither byte is byte i % 8 of mix64(s0 + (i / 8 + 1) golden) (dither_fill_at). */
__device__ static inline uint64_t b4dk_mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
}
__device__ static inline uint64_t b4dk_dither_s0(unsigned seed) { return (uint64_t)seed * 0xD1342543DE82EF95ULL + 1ULL; }
__device__ static inline unsigned b4dk_dbyte(uint64_t s0, int i) {
  return (unsigned)(b4dk_mix64(s0 + (uint64_t)(i / 8 + 1) * 0x9E3779B97F4A7C15ULL) >> (8 * (i & 7))) & 0xFFu;
}
__device__ static inline int32_t b4dk_doff(unsigned r, int32_t st) { return (int32_t)((r * (unsigned)st) >> 8); }
/* seeds[t] of contiguous tiles [0, ntiles) (gathered, uint16 elements; ts = 1: the source bytes are their low bytes) */
__device__ static void b4dk_seed_tile(const uint16_t *__restrict__ tiles, int nelem, int ts, int t, unsigned *__restrict__ seeds) {
  __shared__ unsigned long long acc;
  if (threadIdx.x == 0) acc = 0ULL;
  __syncthreads();
  const uint16_t *e = tiles + (size_t)t * (size_t)nelem;
  const long long nbytes = (long long)nelem * ts, nw = (nbytes + 7) / 8;
  const int per = 8 / ts;                                          /* elements per word */
  unsigned long long h = 0ULL;
  for (long long k = threadIdx.x; k < nw; k += blockDim.x) {
    uint64_t w = 0; long long i0 = k * per;
    for (int b = 0; b < per && i0 + b < nelem; b++) w |= (uint64_t)(ts == 2 ? e[i0 + b] : (e[i0 + b] & 0xFFu)) << (8 * ts * b);
    h += b4dk_mix64(w ^ ((uint64_t)k * 0x9E3779B97F4A7C15ULL));
  }
  atomicAdd(&acc, h);
  __syncthreads();
  if (threadIdx.x == 0) { uint64_t z = b4dk_mix64(acc ^ ((uint64_t)nw * 0xD6E8FEB86659FD93ULL)); seeds[t] = (unsigned)(z ^ (z >> 32)); }
  __syncthreads();
}
extern "C" __global__ void b4d_seed8_tiles(const uint16_t *__restrict__ tiles, int nelem, int ts, int ntiles, unsigned *__restrict__ seeds,
                                           int sm_limit, unsigned long long *work) {
  if (sm_limit <= 0) { if ((int)blockIdx.x < ntiles) b4dk_seed_tile(tiles, nelem, ts, (int)blockIdx.x, seeds); return; }
  if (!b4dk_admit(sm_limit, work)) return;
  for (long long t; (t = b4dk_next(work, ntiles)) >= 0;) b4dk_seed_tile(tiles, nelem, ts, (int)t, seeds);
}
/* mode 2 'above' with dither: residuals of contiguous tiles, sample (lutd[x] + r) >> 8, as the C encoder. One thread per element. */
extern "C" __global__ void b4d_resid_d_tiles(const uint16_t *__restrict__ tiles, const int32_t *__restrict__ lutd, const unsigned *__restrict__ seeds,
                                             int nelem, int cols, int ntiles, short *__restrict__ res) {
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x, total = (long long)ntiles * nelem;
  if (gid >= total) return;
  int t = (int)(gid / nelem), i = (int)(gid - (long long)t * nelem);
  const uint16_t *e = tiles + (size_t)t * (size_t)nelem; const uint64_t s0 = b4dk_dither_s0(seeds[t]);
  unsigned v = (unsigned)((lutd[e[i]] + (int32_t)b4dk_dbyte(s0, i)) >> 8) & 0xFFFFu, p = 0;
  int ip = i >= cols ? i - cols : i - 1;                           /* above; the first row: left (element 0: 0) */
  if (ip >= 0) p = (unsigned)((lutd[e[ip]] + (int32_t)b4dk_dbyte(s0, ip)) >> 8) & 0xFFFFu;
  res[gid] = (short)((v - p) & 0xFFFFu);
}

/* Mode 1 'above3' or 'above' (above != 0) residuals (closed loop, as mode1_encode in b4d_blosc2.c with the mean3d predictor
   and no planes, or the above predictor): the
   tile is rows = cz*cy rows of cols. One thread block per tile, one thread per column; rows run in order because each is
   predicted from the reconstructed row above (rounded mean of its 3 neighbours, 2 at the edges), and the first row from
   the left, sequentially. Same fixed point and float rounding as the C encoder, so the residuals are identical.
   lut32 = rnd(T(x) * 256); qfix = qi * 64. bad[tile] is set when a residual leaves int16: the host then encodes the
   tile on the CPU, which stores it raw. Shared memory: 2 * cols int32. */
__device__ static inline int32_t b4dk_m1_q(int32_t s, int32_t p, float invq) {
  return __float2int_rn(__fmul_rn(__int2float_rn(s - p), invq));
}
__device__ static inline int32_t b4dk_div_round(int32_t s, int32_t c) { return s >= 0 ? (s + c / 2) / c : -((-s + c / 2) / c); }
/* v8 dither (seeds != NULL, the tile's seed from b4d_seed8_tiles): the residual is floor((s - p + d) / q) with
   d = (r q) >> 8, r the element's dither byte, and the reconstruction p + qr q - d + q/2, as mode1_encode */
__device__ static inline int32_t b4dk_m1_qd(int32_t s, int32_t p, int32_t d, float invq) {
  return (int32_t)floorf(__fmul_rn(__int2float_rn(s - p + d), invq));
}
__device__ static void b4dk_m1_tile(const uint16_t *__restrict__ block, const int32_t *__restrict__ lut32,
                                    int bz, int by, int bx, int ty, int tx, int tile0, int contig,
                                    int cz, int cy, int cols, int qfix, int tile_i, short *__restrict__ res, int *__restrict__ bad,
                                    const unsigned *__restrict__ seeds, int above);
extern "C" __global__ void b4d_resid_m1_tiles(const uint16_t *__restrict__ block, const int32_t *__restrict__ lut32,
                                              int bz, int by, int bx, int ty, int tx, int tile0, int contig,
                                              int cz, int cy, int cols, int qfix, int ntiles, short *__restrict__ res,
                                              int *__restrict__ bad, int sm_limit, unsigned long long *work, const unsigned *__restrict__ seeds,
                                              int above) {
  if (sm_limit <= 0) { if ((int)blockIdx.x < ntiles) b4dk_m1_tile(block, lut32, bz, by, bx, ty, tx, tile0, contig, cz, cy, cols, qfix, (int)blockIdx.x, res, bad, seeds, above); return; }
  if (!b4dk_admit(sm_limit, work)) return;
  for (long long t; (t = b4dk_next(work, ntiles)) >= 0;) b4dk_m1_tile(block, lut32, bz, by, bx, ty, tx, tile0, contig, cz, cy, cols, qfix, (int)t, res, bad, seeds, above);
}
__device__ static void b4dk_m1_tile(const uint16_t *__restrict__ block, const int32_t *__restrict__ lut32,
                                    int bz, int by, int bx, int ty, int tx, int tile0, int contig,
                                    int cz, int cy, int cols, int qfix, int tile_i, short *__restrict__ res, int *__restrict__ bad,
                                    const unsigned *__restrict__ seeds, int above) {
  extern __shared__ int32_t b4dk_m1_rows[];
  int nelem = cz * cy * cols, rows = cz * cy, j = threadIdx.x;
  b4dk_tile tv; b4dk_tile_setup(&tv, block, tile_i, tile0, contig, bz, by, bx, ty, tx, cz, cy, cols, nelem);
  short *r = res + (size_t)tile_i * (size_t)nelem;
  const float invq = __fdiv_rn(1.0f, __int2float_rn(qfix));
  const int dith = seeds != 0; const uint64_t s0 = dith ? b4dk_dither_s0(seeds[tile_i]) : 0; const int32_t hq = qfix >> 1;
  int32_t *prev = b4dk_m1_rows, *cur = b4dk_m1_rows + cols;
  int32_t flag = 0;
  if (j == 0) {                                          /* first row: left prediction, serial */
    int32_t p = 0;
    for (int x = 0; x < cols; x++) {
      int32_t sv = lut32[b4dk_ld(&tv, x)], q;
      if (dith) { int32_t d = b4dk_doff(b4dk_dbyte(s0, x), qfix); q = b4dk_m1_qd(sv, p, d, invq); p += q * qfix - d + hq; }
      else { q = b4dk_m1_q(sv, p, invq); p += q * qfix; }
      flag |= (q + 32768) >> 16; r[x] = (short)q; prev[x] = p;
    }
  }
  __syncthreads();
  for (int y = 1; y < rows; y++) {
    if (j < cols) {
      int32_t pr;
      if (above) pr = prev[j];
      else if (cols >= 3) {
        if (j == 0) pr = b4dk_div_round(prev[0] + prev[1], 2);
        else if (j == cols - 1) pr = b4dk_div_round(prev[j - 1] + prev[j], 2);
        else pr = b4dk_div_round(prev[j - 1] + prev[j] + prev[j + 1], 3);
      } else {                                           /* 1 or 2 columns: the taps that exist */
        int32_t s = 0, c = 0;
        for (int d = -1; d <= 1; d++) { int xx = j + d; if (xx >= 0 && xx < cols) { s += prev[xx]; c++; } }
        pr = b4dk_div_round(s, c);
      }
      int i = y * cols + j;
      int32_t sv = lut32[b4dk_ld(&tv, i)], q;
      if (dith) { int32_t d = b4dk_doff(b4dk_dbyte(s0, i), qfix); q = b4dk_m1_qd(sv, pr, d, invq); cur[j] = pr + q * qfix - d + hq; }
      else if (above) {                                  /* round half away from zero, as mode1_encode's 'above' pass */
        float xf = __fmul_rn(__int2float_rn(sv - pr), invq); q = __float2int_rz(__fadd_rn(xf, xf >= 0.0f ? 0.5f : -0.5f)); cur[j] = pr + q * qfix;
      }
      else { q = b4dk_m1_q(sv, pr, invq); cur[j] = pr + q * qfix; }
      flag |= (q + 32768) >> 16; r[i] = (short)q;
    }
    __syncthreads();
    int32_t *t = prev; prev = cur; cur = t;
  }
  if (flag) atomicOr((unsigned *)&bad[tile_i], 1u);
  __syncthreads();                                       /* the shared rows are reused by the next tile (persistent mode) */
}
#endif

__device__ static void b4dk_rice_one_at(const b4dk_params *Pp, b4dk_smem &S, int tile_i, int tile0, const uint16_t *__restrict__ block,
                                     const uint16_t *__restrict__ lut, int contig, unsigned *__restrict__ scratch, uint8_t *__restrict__ out,
                                     unsigned long long *out_used, unsigned long long *__restrict__ offsets, unsigned *__restrict__ lengths,
                                     int *__restrict__ status, const short *__restrict__ res, int use_res, const unsigned *__restrict__ seeds) {
  const b4dk_params &P = *Pp;
  int tid = threadIdx.x;
  if (tid == 0) { S.dith = seeds != 0; S.seed = seeds ? seeds[tile_i] : 0u; }
  b4dk_tile tv; b4dk_tile_setup(&tv, block, tile_i, tile0, contig, P.bz, P.by, P.bx, P.ty, P.tx, P.cz, P.cy, P.cols, P.nelem);
  if (use_res) tv.res = res + (size_t)tile_i * (size_t)P.nelem;     /* residuals from b4d_resid_tiles: one load per element */
  const b4dk_tile *tile = &tv;
  unsigned *sc = scratch + (size_t)tile_i * (size_t)B4DK_NS * (size_t)P.scratch_words;
  b4dk_phase_zero(tid, &P, &S, sc); __syncthreads();
  b4dk_phase_plan(tid, &P, tile, lut, &S); __syncthreads();
  b4dk_phase_runs_init(tid, &P, &S); __syncthreads();
  for (int stride = 1; stride < P.T; stride <<= 1) { b4dk_phase_runs_step_read(tid, &P, &S, stride); __syncthreads(); b4dk_phase_runs_step_write(tid, &P, &S); __syncthreads(); }
  b4dk_phase_totals(tid, &P, &S); __syncthreads();
  for (int stride = 1; stride < P.T; stride <<= 1) { b4dk_phase_scan_read(tid, &P, &S, stride); __syncthreads(); b4dk_phase_scan_write(tid, &P, &S); __syncthreads(); }
  b4dk_phase_scan_finish(tid, &P, &S); __syncthreads();
  b4dk_phase_scan_commit(tid, &P, &S); __syncthreads();
  b4dk_phase_pack(tid, &P, tile, lut, &S, sc); __syncthreads();
  b4dk_phase_size(tid, &P, &S, out_used); __syncthreads();
  b4dk_phase_write(tid, &P, &S, sc, out, P.rows, P.cols); __syncthreads();
  if (tid == 0) { offsets[tile_i] = S.out_off; lengths[tile_i] = S.status ? 0u : 39u + 40u + S.total_payload; status[tile_i] = S.status; }
  __syncthreads();
}

extern "C" __global__ void b4d_rice_tiles(const uint16_t *__restrict__ block, const uint16_t *__restrict__ lut,
                                           int bz, int by, int bx, int ty, int tx, int tile0, int contig,
                                           int cz, int cy, int cols, int mode, int qi, int pred, float conv, float offset, float rn,
                                           unsigned long long out_cap, int scratch_words, int ntiles,
                                           unsigned *__restrict__ scratch, uint8_t *__restrict__ out, unsigned long long *out_used,
                                           unsigned long long *__restrict__ offsets, unsigned *__restrict__ lengths, int *__restrict__ status,
                                           const short *__restrict__ res, int use_res, int ts, int sm_limit, unsigned long long *work,
                                           const unsigned *__restrict__ seeds, const uint8_t *__restrict__ vst, int vst_bytes) {
#ifdef B4DK_DYNSMEM
  /* dynamic shared memory: the 64^3 tile class needs ~55 KB, above the 48 KB static limit; the host sizes the launch
     with b4d_smem_bytes() and opts in with cudaFuncSetAttribute(MaxDynamicSharedMemorySize). Same layout, same output. */
  extern __shared__ __align__(16) unsigned char b4dk_smem_raw[];
  b4dk_smem &S = *reinterpret_cast<b4dk_smem *>(b4dk_smem_raw);
#else
  __shared__ b4dk_smem S;
#endif
  b4dk_params P;                                   /* scalar arguments (CuPy cannot pass a struct by value) */
  P.bz = bz; P.by = by; P.bx = bx; P.ty = ty; P.tx = tx;
  P.cz = cz; P.cy = cy; P.rows = cz * cy; P.cols = cols; P.nelem = P.rows * cols; P.ngroups = (P.nelem + 127) / 128; P.T = (P.ngroups + B4DK_NS - 1) / B4DK_NS;
  P.mode = mode; P.qi = qi; P.pred = pred; P.conv = conv; P.offset = offset; P.rn = rn; P.out_cap = out_cap; P.scratch_words = scratch_words; P.ntiles = ntiles;
  P.ts = ts; P.vst = vst; P.vst_bytes = vst_bytes;
  if (sm_limit <= 0) {
    if ((int)blockIdx.x < ntiles) b4dk_rice_one_at(&P, S, (int)blockIdx.x, tile0, block, lut, contig, scratch, out, out_used, offsets, lengths, status, res, use_res, seeds);
    return;
  }
  if (!b4dk_admit(sm_limit, work)) return;
  for (long long t; (t = b4dk_next(work, ntiles)) >= 0;)
    b4dk_rice_one_at(&P, S, (int)t, tile0, block, lut, contig, scratch, out, out_used, offsets, lengths, status, res, use_res, seeds);
}
#else
/* sequential emulation of one tile: each phase is run for every thread index before the next */
static void b4d_rice_tile_emu(const b4dk_tile *tile, const uint16_t *lut, const b4dk_params *P, unsigned *sc, uint8_t *out,
                              unsigned long long *out_used, unsigned long long *offset, unsigned *length, int *status) {
  static b4dk_smem S; memset(&S, 0, sizeof S);
  #define FOR_T for (int tid = 0; tid < B4DK_THREADS; tid++)
  FOR_T b4dk_phase_zero(tid, P, &S, sc);
  FOR_T b4dk_phase_plan(tid, P, tile, lut, &S);
  FOR_T b4dk_phase_runs_init(tid, P, &S);
  for (int stride = 1; stride < P->T; stride <<= 1) { FOR_T b4dk_phase_runs_step_read(tid, P, &S, stride); FOR_T b4dk_phase_runs_step_write(tid, P, &S); }
  FOR_T b4dk_phase_totals(tid, P, &S);
  for (int stride = 1; stride < P->T; stride <<= 1) { FOR_T b4dk_phase_scan_read(tid, P, &S, stride); FOR_T b4dk_phase_scan_write(tid, P, &S); }
  FOR_T b4dk_phase_scan_finish(tid, P, &S);
  FOR_T b4dk_phase_scan_commit(tid, P, &S);
  FOR_T b4dk_phase_pack(tid, P, tile, lut, &S, sc);
  FOR_T b4dk_phase_size(tid, P, &S, out_used);
  FOR_T b4dk_phase_write(tid, P, &S, sc, out, P->rows, P->cols);
  *offset = S.out_off; *length = S.status ? 0u : 39u + 40u + S.total_payload; *status = S.status;
  #undef FOR_T
}
/* batch entry point for ctypes: same arguments as the CUDA kernel */
void b4d_rice_tiles_emu(const uint16_t *block, const uint16_t *lut, const b4dk_params *P, int tile0, unsigned *scratch, uint8_t *out,
                        unsigned long long *out_used, unsigned long long *offsets, unsigned *lengths, int *status) {
  for (int t = 0; t < P->ntiles; t++) {
    int g = tile0 + t; int k = g % P->tx; int j = (g / P->tx) % P->ty; int i = g / (P->tx * P->ty);
    b4dk_tile tv; tv.base = block; tv.oz = i * P->cz; tv.oy = j * P->cy; tv.ox = k * P->cols; tv.bz = P->bz; tv.by = P->by; tv.bx = P->bx;
    tv.cy = P->cy; tv.cols = P->cols; tv.sy = P->bx; tv.sz = (long long)P->by * P->bx;
    int xs = b4dk_log2_exact(P->cols), ys = b4dk_log2_exact(P->cy); tv.xs = (xs >= 0 && ys >= 0) ? xs : -1; tv.ys = ys;
    unsigned *sc = scratch + (size_t)t * (size_t)B4DK_NS * (size_t)P->scratch_words;
    b4d_rice_tile_emu(&tv, lut, P, sc, out, out_used, &offsets[t], &lengths[t], &status[t]);
  }
}
#endif

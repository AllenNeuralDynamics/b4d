/*
 * b4d_blosc2.c — CPU implementation of the B³D algorithm as a Blosc2 codec.
 * MIT licence. See b4d_blosc2.h for an overview.
 *
 * v2 performance revision:
 *  - no libm calls in per-pixel loops (SSE cvt rounding, inlined sqrt-free inverse)
 *  - forward Anscombe/quantiser as per-meta cached 64K lookup tables
 *  - mode 1 closed loop in int32 fixed point (1/256 units) instead of double
 *  - row-peeled, branch-free interior loops (auto-vectorisable)
 *  - int16 residuals (mod 2^16 for lossless / mode 2), thread-local scratch,
 *    no malloc per block
 *  - single symbolisation pass with per-stream exact size computation,
 *    4 interleaved Huffman bitstreams, 64-bit bit I/O, 11-bit decode table
 */
#include "b4d_blosc2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <emmintrin.h>
#define B4D_HAVE_SSE2 1
#endif
#ifdef __AVX2__
#include <immintrin.h>
#define B4D_HAVE_AVX2 1
#if defined(_MSC_VER) && !defined(__clang__)
static inline uint64_t b4d_bswap64(uint64_t x) { return _byteswap_uint64(x); }
#else
static inline uint64_t b4d_bswap64(uint64_t x) { return __builtin_bswap64(x); }
#endif
static inline unsigned b4d_lzcnt64(uint64_t x) { return (unsigned)_lzcnt_u64(x); }   /* 64 for x == 0 (every AVX2 CPU has LZCNT) */
static inline uint32_t b4d_hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return (uint32_t)_mm_cvtsi128_si32(s);
}
static inline uint32_t b4d_hmax_epu32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i m  = _mm_max_epu32(lo, hi);
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1,0,3,2)));
  m = _mm_max_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2,3,0,1)));
  return (uint32_t)_mm_cvtsi128_si32(m);
}
#endif

/* ------------------------------------------------------------------------ */
/* Compiler portability: GCC/Clang builtins, MSVC intrinsics, plain-C fallback */
/* ------------------------------------------------------------------------ */
#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #define B4D_TLS __declspec(thread)
  static __inline int b4d_clz64(uint64_t x) { unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i; }
  static __inline int b4d_clz32(uint32_t x) { unsigned long i; _BitScanReverse(&i, x); return 31 - (int)i; }
  static __inline uint64_t b4d_mulhi64(uint64_t a, uint64_t b) { return __umulh(a, b); }
  #define b4d_atomic_load_ptr(pp)        ((void *)InterlockedCompareExchangePointer((void *volatile *)(pp), NULL, NULL))
  #define b4d_atomic_exchange_ptr(pp, v) ((void *)InterlockedExchangePointer((void *volatile *)(pp), (void *)(v)))
  static __inline int b4d_atomic_cas_ptr(void *volatile *pp, void **expected, void *desired) {
    void *seen = InterlockedCompareExchangePointer(pp, desired, *expected);
    if (seen == *expected) return 1; *expected = seen; return 0; }
  #define inline __inline
#elif defined(B4D_PORTABLE) || !(defined(__GNUC__) || defined(__clang__))
  /* plain C: no atomics (single-configuration or externally synchronised use only) */
  #define B4D_TLS _Thread_local
  static inline int b4d_clz64(uint64_t x) { int n = 0; while (!(x & (1ull << 63))) { x <<= 1; n++; } return n; }
  static inline int b4d_clz32(uint32_t x) { int n = 0; while (!(x & (1u << 31))) { x <<= 1; n++; } return n; }
  static inline uint64_t b4d_mulhi64(uint64_t a, uint64_t b) {
    uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32); }
  #define b4d_atomic_load_ptr(pp)        ((void *)*(pp))
  static inline void *b4d_atomic_exchange_ptr_impl(void **pp, void *v) { void *o = *pp; *pp = v; return o; }
  #define b4d_atomic_exchange_ptr(pp, v) b4d_atomic_exchange_ptr_impl((void **)(pp), (void *)(v))
  static inline int b4d_atomic_cas_ptr(void *volatile *pp, void **expected, void *desired) {
    if (*pp == *expected) { *pp = desired; return 1; } *expected = (void *)*pp; return 0; }
#else
  #define B4D_TLS _Thread_local
  static inline int b4d_clz64(uint64_t x) { return __builtin_clzll(x); }
  static inline int b4d_clz32(uint32_t x) { return __builtin_clz(x); }
  static inline uint64_t b4d_mulhi64(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }
  #define b4d_atomic_load_ptr(pp)        ((void *)__atomic_load_n((pp), __ATOMIC_ACQUIRE))
  #define b4d_atomic_exchange_ptr(pp, v) ((void *)__atomic_exchange_n((pp), (v), __ATOMIC_ACQ_REL))
  static inline int b4d_atomic_cas_ptr(void *volatile *pp, void **expected, void *desired) {
    return __atomic_compare_exchange_n((void **)pp, expected, desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE); }
#endif

/* ceil(2^(shift+63) / freq) without 128-bit division (used only when building rANS tables) */
static uint64_t b4d_rcp64(uint32_t freq, uint32_t shift) {
  uint64_t hi = (shift + 63 >= 64) ? (1ull << (shift + 63 - 64)) : 0, lo = (shift + 63 >= 64) ? 0 : (1ull << (shift + 63));
  uint64_t add = (uint64_t)freq - 1; lo += add; if (lo < add) hi++;
  uint64_t q = 0, r = 0;
  for (int i = 127; i >= 0; i--) {
    r = (r << 1) | ((i >= 64) ? ((hi >> (i - 64)) & 1) : ((lo >> i) & 1));
    if (r >= freq) { r -= freq; if (i < 64) q |= (1ull << i); }
  }
  return q;
}


/* ------------------------------------------------------------------------ */
/* Process-wide settings                                                    */
/* ------------------------------------------------------------------------ */

static float    g_conv   = 1.0f;
static float    g_offset = 0.0f;
static float    g_rn     = 0.0f;
static uint32_t g_roww   = 0;
static int      g_pred   = B4D_PRED_LEFT_ABOVE;
static int      g_coder  = B4D_ENTROPY_HUFFMAN;
static uint32_t g_ph     = 0;
static int      g_dither = 0;

#define FLAG_PRED_ABOVE  1u
#define FLAG_RICE        2u
#define FLAG_ANS         4u
#define FLAG_PRED_MEAN3D 8u          /* v4: predictor = mean3d (FLAG_PRED_ABOVE clear) */
#define FLAG_PRED_ORIGINAL 16u       /* v5: the original B3D pipeline (b4d_original.c); payload = the original filter's chunk bytes */
#define FLAG_DITHER      32u         /* v6: mode 2 with seeded subtractive dither; a u32 seed ends the header */
#define FLAG_PIXLIST     64u         /* v7: mode 1 with a per-pixel noise list; a u32 size and the list end the header */
#define FLAG_DLISTED     128u        /* v8: with FLAG_DITHER and FLAG_PIXLIST, only the listed pixels are dithered */
/* v8: FLAG_DITHER in mode 1 too (closed loop: the residual is quantised with the offset, the reconstruction takes it back out),
   FLAG_PIXLIST in mode 2 (above predictor), and the dither seed is dither_seed8 */
#define FLAG_PRED_MASK   (FLAG_PRED_ABOVE | FLAG_PRED_MEAN3D | FLAG_PRED_ORIGINAL)
static inline int pred_of_flags(uint8_t flags) { return (flags & FLAG_PRED_ORIGINAL) ? B4D_PRED_ORIGINAL : (flags & FLAG_PRED_MEAN3D) ? B4D_PRED_MEAN3D : (flags & FLAG_PRED_ABOVE) ? B4D_PRED_ABOVE : B4D_PRED_LEFT_ABOVE; }
static inline uint8_t flags_of_pred(int pred) { return pred == B4D_PRED_ORIGINAL ? FLAG_PRED_ORIGINAL : pred == B4D_PRED_MEAN3D ? FLAG_PRED_MEAN3D : pred == B4D_PRED_ABOVE ? FLAG_PRED_ABOVE : 0; }
static inline int norm_pred(int p) { return p == B4D_PRED_ORIGINAL ? B4D_PRED_ORIGINAL : (p == B4D_PRED_MEAN3D || p == B4D_PRED_ABOVE3) ? B4D_PRED_MEAN3D : p ? B4D_PRED_ABOVE : B4D_PRED_LEFT_ABOVE; }
static inline uint8_t version_of_pred(int pred) { return pred == B4D_PRED_ORIGINAL ? 5 : pred == B4D_PRED_MEAN3D ? 4 : B4D_VERSION; }   /* legacy predictors keep writing v3, byte for byte */

#define FIX_SHIFT 8                 /* mode 1 fixed point: 1/256 of a DN */
#define FIX_SCALE (1 << FIX_SHIFT)

static inline int32_t rnd_f(float x) {
#if defined(B4D_HAVE_SSE2)
  return _mm_cvtss_si32(_mm_set_ss(x));
#else
  return (int32_t)lrintf(x);
#endif
}
static inline int32_t rnd_d(double x) {
#if defined(B4D_HAVE_SSE2)
  return _mm_cvtsd_si32(_mm_set_sd(x));
#else
  return (int32_t)lrint(x);
#endif
}

/* ------------------------------------------------------------------------ */
/* Header (44 bytes) + nibble-packed code lengths (256 bytes)               */
/* ------------------------------------------------------------------------ */

#define HUFF_NSYM   512      /* 0..254 literal, 255 escape(16 raw bits), 256..511 zero-run 1..256 */
#define HUFF_MAXLEN 15
#define HUFF_TBITS  11
#define SYM_ESC     255
#define SYM_RUN0    256
#define MAX_RUN     256
#ifdef _MSC_VER
#define B4D_INLINE __forceinline
#else
#define B4D_INLINE inline __attribute__((always_inline))
#endif
#define NSTREAM     4        /* Huffman / rANS streams, and Rice streams before v10 */
#define RICE_NS     8        /* v10 / v11 Rice streams: the sizes of streams 0..2 are in the header, those of 3..6 are 4 u32s at the
                                payload start, stream 7 is the remainder (8 streams decode ~6.5 % faster per core than 4) */
#define RICE_XS     (RICE_NS - NSTREAM)
#define V_RICE8     10       /* first version with RICE_NS Rice streams; 10 = no exact VST, 11 = with the exact VST description */
#define V_HAS_VST(v) ((v) == 9 || (v) == 11)
#define HDR_LENS    (HUFF_NSYM / 2)

typedef struct {
  uint8_t  version, mode, typesize, qi;
  float    conv, offset, rn;
  uint32_t roww, nelem, nsym;
  uint32_t ssize[NSTREAM - 1];   /* byte size of streams 0..2 (3 = remainder) */
  uint8_t  flags;                /* FLAG_PRED_ABOVE | FLAG_RICE | FLAG_ANS | FLAG_PRED_MEAN3D */
  uint32_t ph;                   /* v4: rows per plane (mean3d); 0 = one plane */
  uint32_t seed;                 /* v6: dither seed (FLAG_DITHER) */
  uint32_t pix_bytes;            /* v7: size of the noise list (FLAG_PIXLIST) */
  const uint8_t *pix;            /* v7: the noise list */
  uint32_t vst_bytes;            /* v9: size of the exact-VST description (0: the Anscombe curve) */
  const uint8_t *vst;            /* v9: u8 nk, 3 x u8 0, f32 lam_max, f32 g[nk] (b4d_vst; the camera is in the header) */
} b4d_hdr;

#define HDR_BYTES 44
#define HDR_FIXED_OF(h) (HDR_BYTES + (((h)->flags & (FLAG_RICE | FLAG_ANS | FLAG_PRED_ORIGINAL)) ? 0 : HDR_LENS))
#define HDR_PIX_AT(h)   (HDR_FIXED_OF(h) + (size_t)(h)->vst_bytes)       /* v9: the VST description sits before the noise list */
#define HDR_TOTAL_OF(h) (HDR_PIX_AT(h) + (((h)->flags & FLAG_DITHER) ? 4 : 0) + (((h)->flags & FLAG_PIXLIST) ? 4 + (size_t)(h)->pix_bytes : 0))
#define RICE_G     128              /* residuals per Rice group */
#define RICE_KBITS 4               /* k in 0..13 Rice, 14 = sparse group (v2), 15 = all-zero group */
#define RICE_KMAX  13
#define RICE_KMAX_V1 14            /* v1 streams: k = 14 was a plain Rice parameter */
#define RICE_SPARSE 14
#define RICE_POSBITS 7             /* log2(RICE_G) */
#define RICE_ZRUNBITS 3            /* v2: k=15 is followed by (n-1) in 3 bits: n consecutive zero groups (same stream) */
#define RICE_ZRUNMAX (1 << RICE_ZRUNBITS)
#define RICE_UMAX  31              /* unary limit -> escape with 16 raw bits */

static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put_f32(uint8_t *p, float v)    { memcpy(p, &v, 4); }
static uint32_t get_u32(const uint8_t *p)   { uint32_t v; memcpy(&v, p, 4); return v; }
static float    get_f32(const uint8_t *p)   { float v; memcpy(&v, p, 4); return v; }

static void hdr_write(uint8_t *p, const b4d_hdr *h) {
  /* Rice writes v10 (v11 with the exact VST), which carry every earlier option and RICE_NS streams; others v9 with the VST */
  p[0] = (h->flags & FLAG_RICE) ? (h->vst_bytes ? 11 : V_RICE8) : h->vst_bytes ? 9 : h->version;
  p[1] = h->mode; p[2] = h->typesize; p[3] = h->qi;
  put_f32(p + 4, h->conv); put_f32(p + 8, h->offset); put_f32(p + 12, h->rn);
  put_u32(p + 16, h->roww); put_u32(p + 20, h->nelem); put_u32(p + 24, h->nsym);
  for (int k = 0; k < NSTREAM - 1; k++) put_u32(p + 28 + 4 * k, h->ssize[k]);
  p[40] = h->flags; p[41] = p[42] = p[43] = 0;
  if (h->version >= 4) { p[41] = (uint8_t)(h->ph & 0xFF); p[42] = (uint8_t)((h->ph >> 8) & 0xFF); p[43] = (uint8_t)((h->ph >> 16) & 0xFF); }
  if (h->flags & FLAG_DITHER) put_u32(p + HDR_TOTAL_OF(h) - 4, h->seed);
  if (h->vst_bytes) memcpy(p + HDR_FIXED_OF(h), h->vst, h->vst_bytes);
  if (h->flags & FLAG_PIXLIST) { put_u32(p + HDR_PIX_AT(h), h->pix_bytes); memcpy(p + HDR_PIX_AT(h) + 4, h->pix, h->pix_bytes); }
}
static void hdr_read(const uint8_t *p, b4d_hdr *h) {
  h->version = p[0]; h->mode = p[1]; h->typesize = p[2]; h->qi = p[3];
  h->conv = get_f32(p + 4); h->offset = get_f32(p + 8); h->rn = get_f32(p + 12);
  h->roww = get_u32(p + 16); h->nelem = get_u32(p + 20); h->nsym = get_u32(p + 24);
  for (int k = 0; k < NSTREAM - 1; k++) h->ssize[k] = get_u32(p + 28 + 4 * k);
  h->flags = p[40];
  h->ph = (h->version >= 4) ? ((uint32_t)p[41] | ((uint32_t)p[42] << 8) | ((uint32_t)p[43] << 16)) : 0;
  h->pix_bytes = 0; h->pix = NULL;          /* FLAG_PIXLIST: filled by the decoder once it knows the input is long enough */
  h->vst_bytes = 0; h->vst = NULL;          /* v9: likewise */
}

/* ------------------------------------------------------------------------ */
/* Anscombe VST (generalised, with read noise); used only to build tables   */
/* ------------------------------------------------------------------------ */

static inline double vst_anscombe(double x, float conv, float offset, float rn) {
  /* Generalised Anscombe: T(e) = 2*sqrt(e + rn^2 + 3/8), e = (x-offset)/conv electrons,
     Var(T) ~= 1 so one unit is one sigma. Below the baseline (e < 0) the noise is
     pure read noise, so T continues linearly with slope T'(0) = 1/sqrt(rn^2+3/8):
     monotonic, invertible, and one unit is still one sigma. */
  double e = (x - offset) / conv, c = (double)rn * rn + 0.375;
  if (e >= 0) return 2.0 * sqrt(e + c);
  return 2.0 * sqrt(c) + e / sqrt(c);
}

/* Exact VST (format v9). The Anscombe curve is one sigma per unit only approximately: near the offset its std is off by up
   to ~8 % (the 3/8 cannot cancel both orders of the expansion, and the camera rounds to ADU and clips at 0). The exact
   curve is a table T[0..65535] whose increments are Anscombe's times a gain g, piecewise linear in w = sqrt(lam + c)
   (c = rn^2 + 3/8, lam = the signal in electrons whose mean sits at the increment) on nk knots spread evenly in w over
   [0, lam_max]; above lam_max T is Anscombe plus a constant. b4d_vst_new chooses g so that std(T(X) | lam) = 1 for the
   ADU-rounded Poisson + Gaussian model at every knot. Only (nk, lam_max, g) travel in each block (8 + 4 nk bytes), and the
   table is rebuilt from them with +, -, *, / and sqrt (vst_build), the same everywhere. While a block is coded, tl_vt
   points at its table (NULL: Anscombe) and every curve evaluation below goes through it. */
struct b4d_vst { float conv, offset, rn, lam; int32_t nk; float g[B4D_VST_MAXK]; };   /* zero-filled beyond nk: a cache key */
#define VST_SER_BYTES(nk) (8u + 4u * (uint32_t)(nk))
static B4D_TLS const double *tl_vt = NULL;
static B4D_TLS const uint8_t *tl_vser = NULL;
static B4D_TLS uint32_t tl_vser_len = 0;

/* Where the increment of x sits among the knots: e = the signal (electrons, negative below the offset) whose mean is at its
   middle, y = T_anscombe(e) / 2 (sqrt(e + c), linear below 0), knots at y_j = y0 + (y1 - y0) (j / (nk - 1))^2 from
   y0 = y(-4 rn) to y1 = y(lam_max): dense near the offset, where Anscombe is furthest from one sigma per unit and the
   noise spans only a few ADU, sparse above, where the curve is already right. Below y0 the first knot's gain holds. */
static void vst_build(double *T, const struct b4d_vst *k) {
  const double c = (double)k->rn * k->rn + 0.375, w0 = sqrt(c), y0 = w0 - 2.0 * (double)k->rn / w0, y1 = sqrt((double)k->lam + c);
  const int nk = k->nk; double shift = 0.0; int joined = 0;
  T[0] = vst_anscombe(0, k->conv, k->offset, k->rn);
  for (int x = 0; x < 65535; x++) {
    if (!joined) {
      const double e = ((double)x + 0.5 - k->offset) / k->conv;
      if (e < (double)k->lam) {
        const double y = e >= 0 ? sqrt(e + c) : w0 + e / (2.0 * w0);
        const double pos = y <= y0 ? 0.0 : sqrt((y - y0) / (y1 - y0)) * (nk - 1);
        int j = (int)pos; if (j > nk - 2) j = nk - 2;
        const double fr = pos - j, gg = (double)k->g[j] + fr * ((double)k->g[j + 1] - (double)k->g[j]);
        const double d = vst_anscombe(x + 1, k->conv, k->offset, k->rn) - vst_anscombe(x, k->conv, k->offset, k->rn);
        T[x + 1] = T[x] + d * gg;
        continue;
      }
      joined = 1; shift = T[x] - vst_anscombe(x, k->conv, k->offset, k->rn);
    }
    T[x + 1] = vst_anscombe(x + 1, k->conv, k->offset, k->rn) + shift;
  }
}
/* T between integers: linear; outside [0, 65535]: the end slopes */
static inline double vt_fwd(const double *t, double x) {
  if (x <= 0) return t[0] + x * (t[1] - t[0]);
  if (x >= 65535) return t[65535] + (x - 65535) * (t[65535] - t[65534]);
  long i = (long)x; double fr = x - (double)i;
  return fr == 0 ? t[i] : t[i] + fr * (t[i + 1] - t[i]);
}
static inline double vt_inv(const double *t, double y) {
  if (y <= t[0]) return (y - t[0]) / (t[1] - t[0]);
  if (y >= t[65535]) return 65535 + (y - t[65535]) / (t[65535] - t[65534]);
  long lo = 0, hi = 65535;                       /* t[lo] <= y < t[hi] */
  while (hi - lo > 1) { long mid = (lo + hi) >> 1; if (t[mid] <= y) lo = mid; else hi = mid; }
  return (double)lo + (y - t[lo]) / (t[hi] - t[lo]);
}

static inline double vst_fwd(double x, float conv, float offset, float rn) {
  return tl_vt ? vt_fwd(tl_vt, x) : vst_anscombe(x, conv, offset, rn);
}
typedef struct { double y0, sq, c, conv, offset; const double *t; } vst_inv_t;
static inline void vst_inv_init(vst_inv_t *v, float conv, float offset, float rn) {
  v->c = (double)rn * rn + 0.375; v->sq = sqrt(v->c); v->y0 = 2.0 * v->sq; v->conv = conv; v->offset = offset; v->t = tl_vt;
}
static inline double vst_inv(const vst_inv_t *v, double y) {
  if (v->t) return vt_inv(v->t, y);
  double e = (y >= v->y0) ? (0.25 * y * y - v->c) : (y - v->y0) * v->sq;
  return v->offset + v->conv * e;
}
/* Subtractive dither leaves the error in T uniform on [-s/2, s/2) and independent of the signal, but the inverse is quadratic above
   the baseline (offset + conv (sq u + max(u, 0)^2 / 4), u = y - y0), so decoding y as is lands high by conv E[max(u+e, 0)^2 - max(u, 0)^2] / 4
   (conv s^2 / 48 well above the baseline, 0 well below). A dithered value is decoded less that expectation, so its mean is that of
   the input. y and s in T units. */
static inline double dither_curv(const vst_inv_t *v, double y, double s) {
  if (v->t) {  /* exact table: the mean of the inverse over the step, less its value (32-point midpoint rule) */
    if (s <= 0) return 0.0;
    double acc = 0.0;
    for (int i = 0; i < 32; i++) acc += vt_inv(v->t, y + s * ((i + 0.5) / 32.0 - 0.5));
    return acc / 32.0 - vt_inv(v->t, y);
  }
  const double a = 0.5 * s, u = y - v->y0;
  if (u + a <= 0 || a <= 0) return 0.0;
  const double hi = u + a, lo = u - a > 0 ? u - a : 0.0;
  return 0.25 * v->conv * ((hi * hi * hi - lo * lo * lo) / (6.0 * a) - (u > 0 ? u * u : 0.0));
}

/* ------------------------------------------------------------------------ */
/* Public setters                                                           */
/* ------------------------------------------------------------------------ */

/* vt: the exact-VST table the entries were built from (NULL: Anscombe), part of the key */
typedef struct { uint16_t tab[65536]; uint16_t smin, smax; int valid; float conv, offset, rn; uint8_t meta; const double *vt; } lut16_t;
typedef struct { int32_t  tab[65536]; float conv, offset, rn; uint8_t meta; const double *vt; } lut32_t;
typedef struct { int32_t  tab[65536]; int valid; float conv, offset, rn; uint8_t meta; const double *vt; } lutd_t;

/* small caches of tables keyed by (meta, camera) etc.: open addressing (cache_find / cache_insert below); entries are
   published atomically and never freed while the process codes (released by lut_clear / b4d_cleanup) */
#define LUT_SLOTS 64
static lut16_t *g_lut16[LUT_SLOTS];
static lut32_t *g_lut32[LUT_SLOTS];
static lutd_t  *g_lutd[LUT_SLOTS];
static void *g_pxtab[LUT_SLOTS];     /* px_tab: per-pixel noise steps, defined with the noise list */
#define VTAB_SLOTS 16
static void *g_vtab[VTAB_SLOTS];     /* vtab_t: exact-VST tables, one per description */
#define VTDEC_SLOTS 16
static void *g_vtdec[VTDEC_SLOTS];   /* vtdec_t: their mode-1 decode tables */
static void lut_clear(void);

static inline unsigned lut_slot(uint8_t meta, float conv, float offset, float rn) {
  uint32_t a, b, c; memcpy(&a, &conv, 4); memcpy(&b, &offset, 4); memcpy(&c, &rn, 4);
  return (meta * 2654435761u ^ a * 40503u ^ b * 2246822519u ^ c * 3266489917u ^ (uint32_t)((uintptr_t)tl_vt >> 6) * 668265263u) % LUT_SLOTS;
}

/* tables evicted from a slot are parked here (another thread may still read
   them) and released by lut_clear() / b4d_cleanup() */
typedef struct retired { void *p; struct retired *next; } retired_t;
static retired_t *g_retired = NULL;
static void lut_retire(void *p) {
  if (!p) return;
  retired_t *r = (retired_t *)malloc(sizeof *r);
  if (!r) return;
  r->p = p;
  r->next = (retired_t *)b4d_atomic_load_ptr(&g_retired);
  while (!b4d_atomic_cas_ptr((void *volatile *)&g_retired, (void **)&r->next, r)) {}
}
/* Open addressing: a table lives in the first free slot from its key's home slot; a lookup probes from home
   until it finds the key or an empty slot. Slots are filled once and emptied only by lut_clear, so a probe sequence has no
   holes. The direct-mapped cache this replaces evicted a table whenever two keys in use shared a slot: the levels of a
   multiscale writer (one camera model per level) then rebuilt their tables on every call and parked every evicted table
   until cleanup (Sony IMX811 with 4 levels: compression 3-4x slower, +150 GB resident in 60 s). Now only more keys in use
   than slots evict. */
typedef int (*cache_match_fn)(const void *entry, const void *key);
static void *cache_find(void **tab, unsigned n, unsigned home, cache_match_fn match, const void *key) {
  for (unsigned i = 0; i < n; i++) {
    void *e = b4d_atomic_load_ptr(&tab[(home + i) % n]);
    if (!e) return NULL;
    if (match(e, key)) return e;
  }
  return NULL;
}
static void *cache_insert(void **tab, unsigned n, unsigned home, cache_match_fn match, const void *key, void *mine) {
  for (unsigned i = 0; i < n; i++) {
    void *seen = NULL;
    if (b4d_atomic_cas_ptr((void *volatile *)&tab[(home + i) % n], &seen, mine)) return mine;
    if (match(seen, key)) { free(mine); return seen; }            /* another thread inserted the same table first */
  }
  lut_retire(b4d_atomic_exchange_ptr(&tab[home % n], mine));       /* every slot holds another key: evict the home one */
  return mine;
}
typedef struct { uint8_t meta; float conv, offset, rn; const double *vt; } lut_key;
#define LUT_MATCH_FN(name, type) \
  static int name(const void *e, const void *k) { const type *l = (const type *)e; const lut_key *q = (const lut_key *)k; \
    return l->meta == q->meta && l->conv == q->conv && l->offset == q->offset && l->rn == q->rn && l->vt == q->vt; }
LUT_MATCH_FN(lut16_match, lut16_t)
LUT_MATCH_FN(lut32_match, lut32_t)
LUT_MATCH_FN(lutd_match, lutd_t)

static void lut_clear(void) {
  for (int m = 0; m < LUT_SLOTS; m++) {
    free(b4d_atomic_exchange_ptr(&g_lut16[m], NULL));
    free(b4d_atomic_exchange_ptr(&g_lut32[m], NULL));
    free(b4d_atomic_exchange_ptr(&g_lutd[m], NULL));
    free(b4d_atomic_exchange_ptr(&g_pxtab[m], NULL));
  }
  for (int m = 0; m < VTAB_SLOTS; m++) free(b4d_atomic_exchange_ptr(&g_vtab[m], NULL));
  for (int m = 0; m < VTDEC_SLOTS; m++) free(b4d_atomic_exchange_ptr(&g_vtdec[m], NULL));
  retired_t *r = (retired_t *)b4d_atomic_exchange_ptr(&g_retired, NULL);
  while (r) { retired_t *n = r->next; free(r->p); free(r); r = n; }
}

/* Not thread-safe against concurrent compression (call before starting). */
void b4d_set_camera(float conversion, float offset, float read_noise) {
  if (conversion <= 0) conversion = 1.0f;
  g_conv = conversion; g_offset = offset; g_rn = read_noise;     /* LUT cache is keyed by camera */
}
void b4d_set_row_width(uint32_t row_width) { g_roww = row_width; }
void b4d_set_predictor(int predictor) { g_pred = norm_pred(predictor); if (predictor == B4D_PRED_ABOVE3) g_ph = 0; }
void b4d_set_plane_rows(uint32_t plane_rows) { g_ph = plane_rows; }
void b4d_set_dither(int on)            { g_dither = on != 0; }
void b4d_set_entropy(int coder)       { g_coder = (coder == B4D_ENTROPY_ANS) ? B4D_ENTROPY_ANS : coder ? B4D_ENTROPY_RICE : B4D_ENTROPY_HUFFMAN; }

uint8_t b4d_meta(int mode, float q) {
  int qi = (int)lroundf(q * 4.0f);
  if (qi < 0) qi = 0;
  if (qi > 63) qi = 63;
  if (mode < 0 || mode > 2) mode = 0;
  if (qi == 0) mode = 0;
  return (uint8_t)((qi << 2) | mode);
}

static lut16_t *lut16_get(uint8_t meta, float conv, float offset, float rn) {
  unsigned slot = lut_slot(meta, conv, offset, rn);
  const lut_key key = {meta, conv, offset, rn, tl_vt};
  lut16_t *l = (lut16_t *)cache_find((void **)g_lut16, LUT_SLOTS, slot, lut16_match, &key);
  if (l) return l;
  l = (lut16_t *)malloc(sizeof *l);
  if (!l) return NULL;
  double q = (meta >> 2) * 0.25;
  long smin = rnd_d(vst_fwd(0, conv, offset, rn) / q);            /* T is monotonic: min at x = 0 */
  long smax = smin; int valid = 1;
  for (int x = 0; x < 65536; x++) {
    long sv = rnd_d(vst_fwd(x, conv, offset, rn) / q) - smin;
    if (sv < 0) sv = 0;
    if (sv > 65535) valid = 0, sv = 65535;
    l->tab[x] = (uint16_t)sv;
    if (sv + smin > smax) smax = sv + smin;
  }
  l->smin = 0; l->smax = (uint16_t)(smax - smin); l->valid = valid;
  l->meta = meta; l->conv = conv; l->offset = offset; l->rn = rn; l->vt = tl_vt;
  return (lut16_t *)cache_insert((void **)g_lut16, LUT_SLOTS, slot, lut16_match, &key, l);
}
static lut32_t *lut32_get(uint8_t meta, float conv, float offset, float rn) {
  unsigned slot = lut_slot(meta, conv, offset, rn);
  const lut_key key = {meta, conv, offset, rn, tl_vt};
  lut32_t *l = (lut32_t *)cache_find((void **)g_lut32, LUT_SLOTS, slot, lut32_match, &key);
  if (l) return l;
  l = (lut32_t *)malloc(sizeof *l);
  if (!l) return NULL;
  for (int x = 0; x < 65536; x++) l->tab[x] = rnd_d(vst_fwd(x, conv, offset, rn) * FIX_SCALE);
  l->meta = meta; l->conv = conv; l->offset = offset; l->rn = rn; l->vt = tl_vt;
  return (lut32_t *)cache_insert((void **)g_lut32, LUT_SLOTS, slot, lut32_match, &key, l);
}

/* Dithered mode 2 (v6): T/q in fixed point, DITHER_SCALE units per step, relative to x = 0. A pixel's code is
   (tab[x] + r) >> DITHER_SHIFT with r in [0, DITHER_SCALE); the decoder takes the centre of the bin that r selected. */
#define DITHER_SHIFT 8
#define DITHER_SCALE (1 << DITHER_SHIFT)
static inline int32_t dither_base(double q, float conv, float offset, float rn) { return rnd_d(vst_fwd(0, conv, offset, rn) / q * DITHER_SCALE); }
static lutd_t *lutd_get(uint8_t meta, float conv, float offset, float rn) {
  unsigned slot = lut_slot(meta, conv, offset, rn);
  const lut_key key = {meta, conv, offset, rn, tl_vt};
  lutd_t *l = (lutd_t *)cache_find((void **)g_lutd, LUT_SLOTS, slot, lutd_match, &key);
  if (l) return l;
  l = (lutd_t *)malloc(sizeof *l);
  if (!l) return NULL;
  double q = (meta >> 2) * 0.25;
  int32_t base = dither_base(q, conv, offset, rn);
  for (int x = 0; x < 65536; x++) { int32_t v = rnd_d(vst_fwd(x, conv, offset, rn) / q * DITHER_SCALE) - base; l->tab[x] = v < 0 ? 0 : v; }
  l->valid = ((l->tab[65535] + DITHER_SCALE) >> DITHER_SHIFT) <= 65535;
  l->meta = meta; l->conv = conv; l->offset = offset; l->rn = rn; l->vt = tl_vt;
  return (lutd_t *)cache_insert((void **)g_lutd, LUT_SLOTS, slot, lutd_match, &key, l);
}

/* ---- exact VST (v9): tables by description, their decode tables, the optimiser ---- */
typedef struct { struct b4d_vst k; uint32_t ser_len; uint8_t ser[VST_SER_BYTES(B4D_VST_MAXK)]; double t[65536]; } vtab_t;
static void vst_serialise(const struct b4d_vst *k, uint8_t *p) {
  p[0] = (uint8_t)k->nk; p[1] = p[2] = p[3] = 0; put_f32(p + 4, k->lam);
  for (int j = 0; j < k->nk; j++) put_f32(p + 8 + 4 * j, k->g[j]);
}
/* a description is usable when it builds a strictly increasing table: nk in 2..MAXK, lam_max > 0, every g positive and finite */
static int vst_valid(const struct b4d_vst *k) {
  if (k->nk < 2 || k->nk > B4D_VST_MAXK || !(k->lam > 0) || !(k->lam < 1e9f) || !(k->conv > 0) || !(k->rn >= 0)) return 0;
  for (int j = 0; j < k->nk; j++) if (!(k->g[j] > 1e-3f) || !(k->g[j] < 1e3f)) return 0;
  return 1;
}
static int vtab_match(const void *e, const void *k) { return !memcmp(&((const vtab_t *)e)->k, k, sizeof(struct b4d_vst)); }
static const vtab_t *vtab_get(const struct b4d_vst *k) {
  uint32_t hsh = 2166136261u; const uint8_t *kb = (const uint8_t *)k;
  for (size_t i = 0; i < sizeof *k; i++) hsh = (hsh ^ kb[i]) * 16777619u;
  const unsigned slot = hsh % VTAB_SLOTS;
  vtab_t *v = (vtab_t *)cache_find((void **)g_vtab, VTAB_SLOTS, slot, vtab_match, k);
  if (v) return v;
  if (!vst_valid(k)) return NULL;
  v = (vtab_t *)malloc(sizeof *v);
  if (!v) return NULL;
  v->k = *k; v->ser_len = VST_SER_BYTES(k->nk); vst_serialise(k, v->ser);
  vst_build(v->t, k);
  return (const vtab_t *)cache_insert((void **)g_vtab, VTAB_SLOTS, slot, vtab_match, k, v);
}
/* the key of a description: camera + (nk, lam, g), the unused knots zero */
static void vst_key(struct b4d_vst *k, float conv, float offset, float rn, int nk, float lam, const float *g) {
  memset(k, 0, sizeof *k);
  k->conv = conv; k->offset = offset; k->rn = rn; k->nk = nk; k->lam = lam;
  for (int j = 0; j < nk && j < B4D_VST_MAXK; j++) k->g[j] = g[j];
}

/* Mode-1 decode through the exact table, once per (table, ofs, step cs, vmax): x[a - a0] = rnd(inv(T) - (mean of inv over the
   step - inv(T))), T = (a + ofs) / FIX_SCALE, for every fixed-point a that decodes inside [0, vmax]; the mean from a running
   integral of inv (trapezoid on the 1/FIX_SCALE grid). One lookup per pixel afterwards. */
typedef struct { const double *t; double ofs, cs, vmax; int32_t a0; uint32_t n; uint16_t x[]; } vtdec_t;
typedef struct { const double *t; double ofs, cs, vmax; } vtdec_key;
static int vtdec_match(const void *e, const void *k) {
  const vtdec_t *d = (const vtdec_t *)e; const vtdec_key *q = (const vtdec_key *)k;
  return d->t == q->t && d->ofs == q->ofs && d->cs == q->cs && d->vmax == q->vmax;
}
static const vtdec_t *vtdec_get(const double *t, double ofs, double cs, double vmax) {
  const unsigned slot = ((unsigned)((uintptr_t)t >> 6) * 2654435761u + (unsigned)lround(ofs * 4 + 8) * 31u + (unsigned)lround(cs * 64) * 7u +
                         (vmax > 255 ? 1u : 0u)) % VTDEC_SLOTS;
  const vtdec_key key = {t, ofs, cs, vmax};
  vtdec_t *d = (vtdec_t *)cache_find((void **)g_vtdec, VTDEC_SLOTS, slot, vtdec_match, &key);
  if (d) return d;
  const double S = FIX_SCALE;
  const int32_t h = (int32_t)lround(cs * S / 2);                       /* half a step, in fixed-point units */
  const int32_t a0 = (int32_t)floor(vt_fwd(t, -2.0) * S) - 2 * h - 4, a1 = (int32_t)ceil(vt_fwd(t, vmax + 2.0) * S) + 2 * h + 4;
  const uint32_t n = (uint32_t)(a1 - a0 + 1), m = n + 2 * (uint32_t)h;
  d = (vtdec_t *)malloc(sizeof *d + (size_t)n * sizeof(uint16_t) + 4);   /* +4: vtdec_apply's 32-bit gathers read x[n] */
  double *inv = (double *)malloc((size_t)m * sizeof(double)), *I = (double *)malloc((size_t)m * sizeof(double));
  if (!d || !inv || !I) { free(d); free(inv); free(I); return NULL; }
  long lo = 0;
  for (uint32_t j = 0; j < m; j++) {                                  /* the inverse along the grid: one walk over the table */
    const double y = ((double)(a0 - h + (int32_t)j) + ofs) / S;
    while (lo < 65534 && t[lo + 1] <= y) lo++;
    inv[j] = y < t[0] ? (y - t[0]) / (t[1] - t[0]) : (double)lo + (y - t[lo]) / (t[lo + 1] - t[lo]);
    I[j] = j ? I[j - 1] + 0.5 * (inv[j] + inv[j - 1]) : 0.0;
  }
  for (uint32_t i = 0; i < n; i++) {
    const uint32_t j = i + (uint32_t)h;
    const double mean = h > 0 ? (I[j + h] - I[j - h]) / (2.0 * h) : inv[j];
    const double v = (double)rnd_d(2.0 * inv[j] - mean);
    d->x[i] = (uint16_t)(v < 0 ? 0 : (v > vmax ? vmax : v));
  }
  free(inv); free(I);
  d->t = t; d->ofs = ofs; d->cs = cs; d->vmax = vmax; d->a0 = a0; d->n = n;
  return (const vtdec_t *)cache_insert((void **)g_vtdec, VTDEC_SLOTS, slot, vtdec_match, &key, d);
}

/* P(x | lam) for the camera: x = clip(round(offset + conv (P + G)), 0, 65535), P ~ Poisson(lam), G ~ N(0, rn^2) (electrons).
   Returns the first x and the count; p is malloc'd. */
static double *vst_xdist(double lam, float conv, float offset, float rn, int *x0_out, int *n_out) {
  const double rr = rn > 1e-3f ? (double)rn : 1e-3, sq = sqrt(lam), span = 9.0 * rr * conv + 3.0;
  const long kmin = lam > 0 ? (long)fmax(0.0, floor(lam - 12 * sq - 40)) : 0, kmax = lam > 0 ? (long)ceil(lam + 12 * sq + 40) : 0;
  long xlo = (long)floor(offset + conv * (double)kmin - span), xhi = (long)ceil(offset + conv * (double)kmax + span);
  if (xlo < 0) xlo = 0;
  if (xhi > 65535) xhi = 65535;
  if (xhi < xlo) xhi = xlo;
  const int n = (int)(xhi - xlo + 1);
  double *p = (double *)calloc((size_t)n, sizeof(double));
  if (!p) return NULL;
  const double r2 = 1.0 / (rr * sqrt(2.0)), lg = lam > 0 ? log(lam) : 0.0;
  for (long k = kmin; k <= kmax; k++) {
    const double pk = lam > 0 ? exp((double)k * lg - lam - lgamma((double)k + 1)) : (k == 0 ? 1.0 : 0.0);
    if (pk < 1e-16) continue;
    long a = (long)floor(offset + conv * (double)k - span), b = (long)ceil(offset + conv * (double)k + span);
    if (a < xlo) a = xlo;
    if (b > xhi) b = xhi;
    for (long x = a; x <= b; x++) {
      const double lo = x == 0 ? -INFINITY : ((double)x - 0.5 - offset) / conv, hi = x == 65535 ? INFINITY : ((double)x + 0.5 - offset) / conv;
      const double plo = x == 0 ? 0.0 : 0.5 * erfc(-(lo - (double)k) * r2), phi = x == 65535 ? 1.0 : 0.5 * erfc(-(hi - (double)k) * r2);
      p[x - xlo] += pk * (phi - plo);
    }
  }
  double s = 0; for (int i = 0; i < n; i++) s += p[i];
  if (s > 0) for (int i = 0; i < n; i++) p[i] /= s;
  *x0_out = (int)xlo; *n_out = n;
  return p;
}
static double vst_std_of(const double *T, const double *p, int x0, int n) {
  double m = 0; for (int i = 0; i < n; i++) m += p[i] * T[x0 + i];
  double v = 0; for (int i = 0; i < n; i++) { const double d = T[x0 + i] - m; v += p[i] * d * d; }
  return sqrt(v);
}
/* The variance a step of one T unit adds, E[(dx/dT)^2] / 12 (dx/dT at x: the mean of 1/d^2 over its two increments), over
   the variance of x itself (varx), x 12: 1 means a step of q adds q^2/12 of every pixel's own variance, at every signal. */
static double vst_added_of(const double *T, const double *p, int x0, int n, double varx) {
  double a = 0;
  for (int i = 0; i < n; i++) {
    const int x = x0 + i;
    const double dl = x > 0 ? T[x] - T[x - 1] : T[1] - T[0], dr = x < 65535 ? T[x + 1] - T[x] : dl;
    a += p[i] * 0.5 * (1.0 / (dl * dl) + 1.0 / (dr * dr));
  }
  return a / varx;
}
static double vst_varx(const double *p, int x0, int n) {
  double m = 0, v = 0;
  for (int i = 0; i < n; i++) m += p[i] * (x0 + i);
  for (int i = 0; i < n; i++) { const double d = (x0 + i) - m; v += p[i] * d * d; }
  return v;
}

b4d_vst *b4d_vst_from_knots(float conv, float offset, float rn, int nk, float lam_max, const float *g) {
  if (nk < 2 || nk > B4D_VST_MAXK || !g) return NULL;
  b4d_vst *v = (b4d_vst *)malloc(sizeof *v);
  if (!v) return NULL;
  vst_key(v, conv > 0 ? conv : 1.0f, offset, rn, nk, lam_max, g);
  if (!vst_valid(v)) { free(v); return NULL; }
  return v;
}
void b4d_vst_free(b4d_vst *v) { free(v); }
int b4d_vst_knots(const b4d_vst *v, float *lam_max, float *g) {
  if (!v) return 0;
  if (lam_max) *lam_max = v->lam;
  if (g) for (int j = 0; j < v->nk; j++) g[j] = v->g[j];
  return v->nk;
}
int b4d_vst_table(const b4d_vst *v, double *T) {
  if (!v || !T) return -1;
  if (!vst_valid(v)) return -1;
  vst_build(T, v);
  return 0;
}
/* std(T(X) | lam) under the camera model, for the exact table (v) or, with v NULL, the Anscombe curve of that camera */
int b4d_vst_std(const b4d_vst *v, float conv, float offset, float rn, const double *lams, int nl, double *out) {
  double *T = (double *)malloc(65536 * sizeof(double));
  if (!T) return -1;
  if (v) { conv = v->conv; offset = v->offset; rn = v->rn; vst_build(T, v); }
  else for (int x = 0; x < 65536; x++) T[x] = vst_anscombe(x, conv, offset, rn);
  for (int i = 0; i < nl; i++) {
    int x0, n; double *p = vst_xdist(lams[i], conv, offset, rn, &x0, &n);
    if (!p) { free(T); return -1; }
    out[i] = vst_std_of(T, p, x0, n); free(p);
  }
  free(T);
  return 0;
}
/* The optimiser: g at nk knots so that std(T(X) | lam_j) = 1 at every knot (g[nk - 1] = 1 joins Anscombe), by repeated
   least squares (Gauss-Newton, a light smoothness penalty). nk: about one knot per unit of T over [0, lam_max] (16..MAXK); lam_max = max(400, 40 rn^2) e-,
   within the 16-bit range. Encoder-side only: what it chooses travels in the blocks, so it need not be reproducible. */
#define VST_NK 48                                          /* knots chosen by b4d_vst_new: 200 bytes per block */
b4d_vst *b4d_vst_new(float conv, float offset, float rn, int objective, double *max_err) {
  if (objective != B4D_VST_STD && objective != B4D_VST_VARIANCE) return NULL;
  if (!(conv > 0) || !(rn >= 0)) return NULL;
  double lam = fmax(400.0, 40.0 * (double)rn * rn);
  if (offset + conv * lam > 50000.0) lam = (50000.0 - offset) / conv;
  if (lam < 4.0) return NULL;
  const int nk = VST_NK;
  const int M = 3 * nk, U = nk - 1; const double RIDGE = 0.02, EPS = 1e-4;
  b4d_vst *v = (b4d_vst *)malloc(sizeof *v);
  double *T = (double *)malloc(65536 * sizeof(double));
  double **P = (double **)calloc((size_t)M, sizeof(double *)); int *x0 = (int *)calloc((size_t)M, sizeof(int)), *np = (int *)calloc((size_t)M, sizeof(int));
  if (!v || !T || !P || !x0 || !np) { free(v); free(T); free(P); free(x0); free(np); return NULL; }
  float g1[B4D_VST_MAXK]; for (int j = 0; j < nk; j++) g1[j] = 1.0f;
  vst_key(v, conv, offset, rn, nk, (float)lam, g1);
  /* fitted at M signals: 0, then geometric from 0.02 e- to lam_max */
  int ok = 1; double vx[3 * VST_NK];
  for (int i = 0; i < M && ok; i++) {
    const double l = i == 0 ? 0.0 : 0.02 * pow((double)v->lam / 0.02, (double)(i - 1) / (M - 2));
    P[i] = vst_xdist(l, conv, offset, rn, &x0[i], &np[i]); ok = P[i] != NULL;
    if (ok) vx[i] = vst_varx(P[i], x0[i], np[i]);
  }
  /* the residual at signal i: std(T) - 1 (B4D_VST_STD), or the added variance in units of q^2/12 of the pixel's own - 1 */
  #define VST_RES(i) (objective == B4D_VST_VARIANCE ? vst_added_of(T, P[i], x0[i], np[i], vx[i]) - 1.0 \
                                                    : vst_std_of(T, P[i], x0[i], np[i]) - 1.0)
  /* damped Gauss-Newton on th = log g[0..nk-2]: residuals std - 1 at those signals, plus a small penalty on the second
     difference of th, so that knots closer together than the noise can resolve neither zig-zag nor pile the stretch into one place */
  double *J = (double *)malloc((size_t)M * U * sizeof(double)), *res = (double *)malloc((size_t)M * sizeof(double));
  double *A = (double *)malloc((size_t)U * U * sizeof(double)), *bv = (double *)malloc((size_t)U * sizeof(double));
  ok = ok && J && res && A && bv;
  for (int it = 0; it < 12 && ok; it++) {
    vst_build(T, v);
    double e = 0.0;
    for (int i = 0; i < M; i++) { res[i] = VST_RES(i); e = fmax(e, fabs(res[i])); }
    if (e < 2e-4) break;
    for (int j = 0; j < U; j++) {                           /* Jacobian by forward differences */
      const float gj = v->g[j]; v->g[j] = (float)(gj * exp(EPS)); const double de = log((double)v->g[j] / gj);
      vst_build(T, v);
      for (int i = 0; i < M; i++) J[(size_t)i * U + j] = (VST_RES(i) - res[i]) / de;
      v->g[j] = gj;
    }
    for (int a = 0; a < U; a++) {                            /* normal equations (J'J + R'R) d = -(J'r + R'R th) */
      double sb = 0; for (int i = 0; i < M; i++) sb += J[(size_t)i * U + a] * res[i];
      bv[a] = -sb;
      for (int b = 0; b <= a; b++) { double sa = 0; for (int i = 0; i < M; i++) sa += J[(size_t)i * U + a] * J[(size_t)i * U + b]; A[a * U + b] = A[b * U + a] = sa; }
    }
    double th[B4D_VST_MAXK]; for (int j = 0; j < nk; j++) th[j] = log((double)v->g[j]);   /* th[nk - 1] = 0 (g = 1) */
    for (int j = 0; j < nk - 1; j++) {                       /* rows of the penalty: th[j] - th[j+1] (th[nk-1] = 0 fixed): where no
                                                                signal decides, the gain stays level (a uniform stretch) */
      const int ix[2] = { j, j + 1 }; const double cf[2] = { 1, -1 };
      const double val = th[j] - th[j + 1];
      for (int q = 0; q < 2; q++) {
        if (ix[q] >= U) continue;
        bv[ix[q]] -= RIDGE * RIDGE * cf[q] * val;
        for (int r = 0; r < 2; r++) if (ix[r] < U) A[ix[q] * U + ix[r]] += RIDGE * RIDGE * cf[q] * cf[r];
      }
    }
    for (int a = 0; a < U; a++) A[a * U + a] += 1e-9;
    for (int c = 0; c < U; c++) {                            /* Gaussian elimination (symmetric positive definite: no pivoting) */
      const double pv = A[c * U + c];
      for (int r = c + 1; r < U; r++) {
        const double f = A[r * U + c] / pv;
        for (int k = c; k < U; k++) A[r * U + k] -= f * A[c * U + k];
        bv[r] -= f * bv[c];
      }
    }
    for (int r = U - 1; r >= 0; r--) { double sm = bv[r]; for (int k = r + 1; k < U; k++) sm -= A[r * U + k] * bv[k]; bv[r] = sm / A[r * U + r]; }
    for (int j = 0; j < U; j++) {
      const double d = bv[j] > 0.3 ? 0.3 : (bv[j] < -0.3 ? -0.3 : bv[j]);
      const float gj = (float)(v->g[j] * exp(d));
      v->g[j] = gj < 0.05f ? 0.05f : (gj > 20.0f ? 20.0f : gj);
    }
  }
  free(J); free(res); free(A); free(bv);
  if (ok && max_err) {                                     /* at the signals it was fitted at */
    vst_build(T, v); double e = 0.0;
    for (int i = 0; i < M; i++) e = fmax(e, fabs(VST_RES(i)));
    *max_err = e;
  }
  #undef VST_RES
  for (int i = 0; i < M; i++) free(P[i]);
  free(P); free(x0); free(np); free(T);
  if (!ok || !vst_valid(v)) { free(v); return NULL; }
  return v;
}
/* The seed is a hash of the block's input: repeatable, independent of threads and order, and different between blocks, so
   the error does not repeat from chunk to chunk and averages away across frames. */
static uint32_t dither_seed(const uint8_t *p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)n;
  for (; n >= 8; p += 8, n -= 8) { uint64_t w; memcpy(&w, p, 8); h = (h ^ w) * 0xFF51AFD7ED558CCDULL; h ^= h >> 29; }
  for (; n; p++, n--) { h = (h ^ *p) * 0xFF51AFD7ED558CCDULL; h ^= h >> 29; }
  return (uint32_t)(h ^ (h >> 32));
}
/* splitmix64: eight 8-bit dither values per draw; the stream is part of the v6 format. Reference form of the stream
   (state s = dither_s0(seed), w = 0); the encoder and decoder use the batched dither_fill_at below. */
typedef struct { uint64_t s, w; } dither_rng;
static inline uint32_t dither_next(dither_rng *d, uint32_t i) {
  if (!(i & 7)) {
    uint64_t z = (d->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; d->w = z ^ (z >> 31);
  }
  uint32_t r = (uint32_t)(d->w & (DITHER_SCALE - 1)); d->w >>= DITHER_SHIFT;
  return r;
}
static inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
}
/* The same stream as dither_next, all n values at once: draw k is mix64(s0 + (k + 1) * golden), its bytes in order. */
static inline uint64_t dither_s0(uint32_t seed) { return (uint64_t)seed * 0xD1342543DE82EF95ULL + 1; }
static inline uint64_t dither_word(uint64_t s0, uint32_t k) { return mix64(s0 + (uint64_t)(k + 1) * 0x9E3779B97F4A7C15ULL); }
/* elements [i0, i0 + cnt) of the stream: element i is byte i % 8 (little-endian) of draw i / 8 */
#ifdef B4D_HAVE_AVX2
/* z * c mod 2^64 in each 64-bit lane (AVX2 has no 64-bit multiply): lo(z) lo(c) + (hi(z) lo(c) + lo(z) hi(c)) << 32 */
static inline __m256i mul64c(__m256i z, __m256i clo, __m256i chi) {
  __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(z, 32), clo), _mm256_mul_epu32(z, chi));
  return _mm256_add_epi64(_mm256_mul_epu32(z, clo), _mm256_slli_epi64(cross, 32));
}
static inline __m256i mix64x4(__m256i z) {               /* mix64 in each lane */
  const __m256i c1l = _mm256_set1_epi64x((long long)0xBF58476D1CE4E5B9ULL), c1h = _mm256_set1_epi64x((long long)(0xBF58476D1CE4E5B9ULL >> 32));
  const __m256i c2l = _mm256_set1_epi64x((long long)0x94D049BB133111EBULL), c2h = _mm256_set1_epi64x((long long)(0x94D049BB133111EBULL >> 32));
  z = mul64c(_mm256_xor_si256(z, _mm256_srli_epi64(z, 30)), c1l, c1h);
  z = mul64c(_mm256_xor_si256(z, _mm256_srli_epi64(z, 27)), c2l, c2h);
  return _mm256_xor_si256(z, _mm256_srli_epi64(z, 31));
}
#endif
static void dither_fill_at(uint8_t *out, uint32_t i0, uint32_t cnt, uint64_t s0) {
  uint32_t i = i0, e = i0 + cnt;
  if (i & 7) { uint64_t z = dither_word(s0, i / 8); for (; (i & 7) && i < e; i++) *out++ = (uint8_t)(z >> (8 * (i & 7))); }
#ifdef B4D_HAVE_AVX2
  if (i + 32 <= e) {                                     /* 4 draws per vector: s0 + (k + 1) golden for k = i/8 .. i/8 + 3 */
    const uint64_t G = 0x9E3779B97F4A7C15ULL, k = i / 8;
    __m256i a = _mm256_set_epi64x((long long)(s0 + (k + 4) * G), (long long)(s0 + (k + 3) * G), (long long)(s0 + (k + 2) * G), (long long)(s0 + (k + 1) * G));
    const __m256i g4 = _mm256_set1_epi64x((long long)(4 * G));
    for (; i + 32 <= e; i += 32, out += 32, a = _mm256_add_epi64(a, g4)) _mm256_storeu_si256((__m256i *)out, mix64x4(a));
  }
#endif
  for (; i + 8 <= e; i += 8, out += 8) { uint64_t z = dither_word(s0, i / 8); memcpy(out, &z, 8); }
  if (i < e) { uint64_t z = dither_word(s0, i / 8); memcpy(out, &z, e - i); }
}
static void dither_fill(uint8_t *out, uint32_t n, uint32_t seed) { dither_fill_at(out, 0, n, dither_s0(seed)); }
static inline uint8_t dither_byte(uint64_t s0, uint32_t i) { return (uint8_t)(dither_word(s0, i / 8) >> (8 * (i & 7))); }
/* v8 seed: a sum of independently mixed 64-bit words (and the length), so it does not depend on the order the words are
   visited in and a GPU reduces it in parallel. Repeatable, and different between blocks, as dither_seed. */
static uint32_t dither_seed8(const uint8_t *p, size_t n) {
  uint64_t h = 0, k = 0;
#ifdef B4D_HAVE_AVX2
  if (n >= 32) {                                         /* 4 words per vector; the sum does not depend on the order */
    const uint64_t G = 0x9E3779B97F4A7C15ULL;
    __m256i kg = _mm256_set_epi64x((long long)(3 * G), (long long)(2 * G), (long long)G, 0), hv = _mm256_setzero_si256();
    const __m256i g4 = _mm256_set1_epi64x((long long)(4 * G));
    for (; n >= 32; p += 32, n -= 32, k += 4, kg = _mm256_add_epi64(kg, g4))
      hv = _mm256_add_epi64(hv, mix64x4(_mm256_xor_si256(_mm256_loadu_si256((const __m256i *)p), kg)));
    uint64_t l[4]; _mm256_storeu_si256((__m256i *)l, hv); h = l[0] + l[1] + l[2] + l[3];
  }
#endif
  for (; n >= 8; p += 8, n -= 8, k++) { uint64_t w; memcpy(&w, p, 8); h += mix64(w ^ (k * 0x9E3779B97F4A7C15ULL)); }
  if (n) { uint64_t w = 0; memcpy(&w, p, n); h += mix64(w ^ (k * 0x9E3779B97F4A7C15ULL)); k++; }
  h = mix64(h ^ (k * 0xD6E8FEB86659FD93ULL));
  return (uint32_t)(h ^ (h >> 32));
}
/* the dither offset of one value, in units of a step st (FIX units): floor(r * st / DITHER_SCALE), in [0, st) */
static inline int32_t dither_off(uint8_t r, int32_t st) { return (int32_t)(((uint32_t)r * (uint32_t)st) >> DITHER_SHIFT); }

/* ------------------------------------------------------------------------ */
/* Thread-local scratch arena                                               */
/* ------------------------------------------------------------------------ */

static B4D_TLS uint8_t *t_buf = NULL;
static B4D_TLS size_t   t_cap = 0;

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
static pthread_key_t  t_key; static pthread_once_t t_once = PTHREAD_ONCE_INIT;
static void scratch_dtor(void *p) { free(p); }
static void scratch_key_init(void) { pthread_key_create(&t_key, scratch_dtor); }
static void scratch_track(void *p) { pthread_once(&t_once, scratch_key_init); pthread_setspecific(t_key, p); }
#else
static void scratch_track(void *p) { (void)p; }   /* freed by b4d_free_thread_scratch() or leaked at exit */
#endif

static uint8_t *scratch(size_t bytes) {
  if (t_cap < bytes) {
    free(t_buf);
    t_buf = (uint8_t *)malloc(bytes + 64);
    t_cap = t_buf ? bytes : 0;
    scratch_track(t_buf);
  }
  return t_buf;
}

/* Optional: release this thread's scratch and the LUT cache (e.g. at exit). */
void b4d_free_thread_scratch(void) { scratch_track(NULL); free(t_buf); t_buf = NULL; t_cap = 0; }
void b4d_trim_thread_scratch(size_t keep_bytes) { if (t_cap > keep_bytes) b4d_free_thread_scratch(); }
void b4d_cleanup(void) { b4d_free_thread_scratch(); lut_clear(); }
#define ALIGN64(x) (((x) + 63) & ~(size_t)63)

/* ------------------------------------------------------------------------ */
/* Prediction / residual kernels (row peeled; interior loops vectorise)     */
/* residuals are computed mod 2^16 -> exact bijection for 16-bit samples    */
/* ------------------------------------------------------------------------ */

/* ---- mean3d predictor: prediction of one row (elements [base, base+len) of a block of n elements, rows of w,
   planes of ph rows) from already-known samples: the above row's (x-1, x, x+1) and the previous plane's 3x3 patch.
   pred = round(sum / count) over the taps that exist (count 1..12); rows are independent of each other, so the
   encoder and the decoder both vectorise. The same routine serves the uint16 sample domain (modes 0/2) and the
   int32 fixed-point domain of mode 1 (sums in int64, symmetric rounding). base == 0 (first row of the block) has no
   taps: the caller uses the left predictor there, exactly as the other predictors do. */
#ifdef B4D_HAVE_AVX2
static uint32_t mean3_row_avx2(const int32_t *up, int32_t *pr, uint32_t len);
#define M3_AVX2(T) if (sizeof(T) == 4) x0 = mean3_row_avx2((const int32_t *)up, (int32_t *)pred, len);   /* int32 rows (mode 1) */
#else
#define M3_AVX2(T)
#endif
#define MEAN3D_ROW_FN(NAME, T)                                                                                   \
static void NAME(const T *v, uint32_t n, uint32_t w, uint32_t ph, uint32_t base, uint32_t len, T *pred) {        \
  uint32_t plane = w * ph, z = base / plane, y = (base - z * plane) / w;                                         \
  const T *up = (y > 0) ? v + base - w : NULL;                                     /* above row, same plane */     \
  const T *p0 = (z > 0) ? v + base - plane : NULL;                                 /* previous plane, same row */  \
  const T *pm = (p0 && y > 0) ? p0 - w : NULL;                                                                    \
  const T *pp = (p0 && y + 1 < ph && base - plane + w + len <= n) ? p0 + w : NULL;  /* exists and complete */      \
  uint32_t uplen = up ? len : 0;                                                   /* above row covers len (rows before the last are complete) */ \
  if (uplen && base - w + len > n) uplen = n - (base - w);                                                        \
  if (up && uplen >= len && !p0 && len >= 3) {                                     /* fast path: above row only */ \
    uint32_t x0 = 1;                                                                                              \
    M3_AVX2(T)                                                                                                    \
    for (uint32_t x = x0; x + 1 < len; x++) {                                                                     \
      int32_t s = (int32_t)up[x - 1] + (int32_t)up[x] + (int32_t)up[x + 1];                                       \
      pred[x] = (T)(s >= 0 ? (s + 1) / 3 : -((-s + 1) / 3));                                                     \
    }                                                                                                             \
    { int32_t s = (int32_t)up[0] + (int32_t)up[1]; pred[0] = (T)(s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2)); }        \
    { uint32_t x = len - 1; int32_t s = (int32_t)up[x - 1] + (int32_t)up[x]; pred[x] = (T)(s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2)); } \
    return;                                                                                                       \
  }                                                                                                               \
  if (up && uplen >= len && p0 && pm && pp && len >= 3) {                          /* fast path: all taps exist */ \
    /* interior columns 1..len-2: 12 taps */                                                                     \
    for (uint32_t x = 1; x + 1 < len; x++) {                                                                      \
      int64_t s = (int64_t)up[x - 1] + up[x] + up[x + 1] + pm[x - 1] + pm[x] + pm[x + 1]                           \
                + p0[x - 1] + p0[x] + p0[x + 1] + pp[x - 1] + pp[x] + pp[x + 1];                                  \
      pred[x] = (T)(s >= 0 ? (s + 6) / 12 : -((-s + 6) / 12));                                                   \
    }                                                                                                             \
    { int64_t s = (int64_t)up[0] + up[1] + pm[0] + pm[1] + p0[0] + p0[1] + pp[0] + pp[1];                          \
      pred[0] = (T)(s >= 0 ? (s + 4) / 8 : -((-s + 4) / 8)); }                                                   \
    { uint32_t x = len - 1; int64_t s = (int64_t)up[x - 1] + up[x] + pm[x - 1] + pm[x] + p0[x - 1] + p0[x] + pp[x - 1] + pp[x]; \
      pred[x] = (T)(s >= 0 ? (s + 4) / 8 : -((-s + 4) / 8)); }                                                   \
    return;                                                                                                       \
  }                                                                                                               \
  for (uint32_t x = 0; x < len; x++) {                                             /* general: count the taps */   \
    int64_t s = 0; int c = 0;                                                                                     \
    for (int d = -1; d <= 1; d++) {                                                                               \
      int64_t xx = (int64_t)x + d; if (xx < 0 || xx >= (int64_t)w) continue;                                      \
      if (up && (uint32_t)xx < uplen) { s += up[xx]; c++; }                                                       \
      if (pm) { s += pm[xx]; c++; }                                                                               \
      if (p0 && (base - plane + (uint32_t)xx) < n) { s += p0[xx]; c++; }                                          \
      if (pp) { s += pp[xx]; c++; }                                                                               \
    }                                                                                                             \
    if (c == 0) { pred[x] = x ? v[base + x - 1] : (T)0; continue; }                /* no taps: left */            \
    pred[x] = (T)(s >= 0 ? (s + c / 2) / c : -((-s + c / 2) / c));                                                \
  }                                                                                                               \
}
MEAN3D_ROW_FN(mean3d_row_u16, uint16_t)
MEAN3D_ROW_FN(mean3d_row_i32, int32_t)

static inline uint32_t plane_rows_of(uint32_t ph, uint32_t n, uint32_t w) { return (ph == 0 || ph > (n + w - 1) / w) ? (n + w - 1) / w : ph; }

static void residual_u16(const uint16_t *v, int16_t *r, uint32_t n, uint32_t w, int pred_id, uint32_t ph) {
  if (n == 0) return;
  if (w == 0 || w > n) w = n;
  const int above_only = pred_id == B4D_PRED_ABOVE;
  r[0] = (int16_t)v[0];
  for (uint32_t j = 1; j < w; j++) r[j] = (int16_t)(uint16_t)(v[j] - v[j - 1]);
  if (pred_id == B4D_PRED_MEAN3D) {
    ph = plane_rows_of(ph, n, w);
    uint16_t predrow[4096]; uint16_t *pr = w <= 4096 ? predrow : (uint16_t *)malloc(2 * (size_t)w);
    for (uint32_t base = w; base < n; base += w) {
      uint32_t len = (n - base < w) ? n - base : w;
      mean3d_row_u16(v, n, w, ph, base, len, pr);
      const uint16_t *cur = v + base; int16_t *out = r + base;
      for (uint32_t j = 0; j < len; j++) out[j] = (int16_t)(uint16_t)(cur[j] - pr[j]);
    }
    if (pr != predrow) free(pr);
    return;
  }
  for (uint32_t base = w; base < n; base += w) {
    uint32_t len = (n - base < w) ? n - base : w;
    const uint16_t *cur = v + base, *up = v + base - w; int16_t *out = r + base;
    if (above_only) {
      for (uint32_t j = 0; j < len; j++) out[j] = (int16_t)(uint16_t)(cur[j] - up[j]);
    } else {
      out[0] = (int16_t)(uint16_t)(cur[0] - up[0]);
      for (uint32_t j = 1; j < len; j++)
        out[j] = (int16_t)(uint16_t)(cur[j] - (uint16_t)(((uint32_t)cur[j - 1] + up[j]) >> 1));
    }
  }
}

/* in place: r holds residuals on entry, reconstructed samples on exit */
static void reconstruct_u16(uint16_t *r, uint32_t n, uint32_t w, int pred_id, uint32_t ph) {
  if (n == 0) return;
  if (w == 0 || w > n) w = n;
  const int above_only = pred_id == B4D_PRED_ABOVE;
  for (uint32_t j = 1; j < w; j++) r[j] = (uint16_t)(r[j] + r[j - 1]);
  if (pred_id == B4D_PRED_MEAN3D) {
    ph = plane_rows_of(ph, n, w);
    uint16_t predrow[4096]; uint16_t *pr = w <= 4096 ? predrow : (uint16_t *)malloc(2 * (size_t)w);
    for (uint32_t base = w; base < n; base += w) {
      uint32_t len = (n - base < w) ? n - base : w;
      mean3d_row_u16(r, n, w, ph, base, len, pr);                        /* rows before base are already reconstructed */
      uint16_t *cur = r + base;
      for (uint32_t j = 0; j < len; j++) cur[j] = (uint16_t)(cur[j] + pr[j]);
    }
    if (pr != predrow) free(pr);
    return;
  }
  for (uint32_t base = w; base < n; base += w) {
    uint32_t len = (n - base < w) ? n - base : w;
    uint16_t *cur = r + base; const uint16_t *up = r + base - w;
    if (above_only) {
      for (uint32_t j = 0; j < len; j++) cur[j] = (uint16_t)(cur[j] + up[j]);
    } else {
      cur[0] = (uint16_t)(cur[0] + up[0]);
      for (uint32_t j = 1; j < len; j++)
        cur[j] = (uint16_t)(cur[j] + (uint16_t)(((uint32_t)cur[j - 1] + up[j]) >> 1));
    }
  }
}

/* Mode 1 streamed into the Rice stage (left_above, above, or above3 = mean3d on a single plane; no noise list): the closed
   loop runs a row at a time as the coder asks for it and keeps only the row above's reconstruction, so the block's hot set
   is the input tile + LUT + a few rows instead of tile + samples + reconstruction + residuals (4.5x the 16-bit tile). The
   arithmetic is mode1_encode's, element by element, so the bitstream is identical; bad != 0 where a residual left int16. */
typedef struct {
  const uint8_t *in8; const uint16_t *in16; const int32_t *lut; uint32_t n, w; int32_t qfix; int pred, dith; uint64_t ds0;
  int32_t *srow, *fprev, *fcur, *prow; int16_t *rrow; uint8_t *drow; uint32_t row; int started; int32_t bad;
} m1_src;

#ifdef B4D_HAVE_AVX2
/* AVX2 body of m1_row's above3 / above rows: the scalar loops' float and integer operations lane by lane (int -> float
   rounds to nearest, cvtps rounds by MXCSR as rnd_f, floor then truncate as floorf, x +- 1/2 truncated), so the residuals
   and reconstruction are identical. kind: 0 rnd_f, 1 floorf with dither, 2 round half away from zero. Returns the count
   done (a multiple of 8); the caller finishes the row with the scalar loops. */
static uint32_t m1_quant_avx2(const int32_t *sc, const int32_t *pr, const uint8_t *dc, int32_t *fc, int16_t *rc,
                              uint32_t len, int32_t qfix, float invq, int kind, int32_t *bad) {
  const __m256 vinv = _mm256_set1_ps(invq), half = _mm256_set1_ps(0.5f), sgn = _mm256_set1_ps(-0.0f);
  const __m256i vq = _mm256_set1_epi32(qfix), vhq = _mm256_set1_epi32(qfix >> 1), v32k = _mm256_set1_epi32(32768),
                lo16 = _mm256_set1_epi32(0xFFFF);
  __m256i vbad = _mm256_setzero_si256();
  uint32_t j = 0;
  for (; j + 8 <= len; j += 8) {
    __m256i p = _mm256_loadu_si256((const __m256i *)(pr + j)), d = p;
    __m256i t = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(sc + j)), p), qr;
    if (kind == 1) {
      d = _mm256_srli_epi32(_mm256_mullo_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(dc + j))), vq), DITHER_SHIFT);
      qr = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(t, d)), vinv)));
    } else {
      __m256 x = _mm256_mul_ps(_mm256_cvtepi32_ps(t), vinv);
      if (kind == 0) qr = _mm256_cvtps_epi32(x);
      else {                                            /* x >= 0 ? +1/2 : -1/2; -0.0 takes -1/2, which truncates to 0 as +1/2 does */
        __m256 h = _mm256_or_ps(_mm256_and_ps(x, sgn), half);
        qr = _mm256_cvttps_epi32(_mm256_add_ps(x, h));
      }
    }
    vbad = _mm256_or_si256(vbad, _mm256_srai_epi32(_mm256_add_epi32(qr, v32k), 16));
    __m256i r16 = _mm256_packus_epi32(_mm256_and_si256(qr, lo16), _mm256_and_si256(qr, lo16));
    _mm_storeu_si128((__m128i *)(rc + j), _mm256_castsi256_si128(_mm256_permute4x64_epi64(r16, 0x08)));
    __m256i f = _mm256_add_epi32(p, _mm256_mullo_epi32(qr, vq));
    if (kind == 1) f = _mm256_add_epi32(_mm256_sub_epi32(f, d), vhq);
    _mm256_storeu_si256((__m256i *)(fc + j), f);
  }
  __m128i b = _mm_or_si128(_mm256_castsi256_si128(vbad), _mm256_extracti128_si256(vbad, 1));
  b = _mm_or_si128(b, _mm_shuffle_epi32(b, 0x4E)); b = _mm_or_si128(b, _mm_shuffle_epi32(b, 0xB1));
  *bad |= _mm_cvtsi128_si32(b);
  return j;
}

/* pr[x] = rounded mean of up[x-1..x+1] for x in [1, len - 1) as mean3d_row_i32: sign(s) ((|s| + 1) / 3), the division by
   3 exact as (a 0xAAAAAAAB) >> 33. Returns the next x left to the scalar loop. */
static uint32_t mean3_row_avx2(const int32_t *up, int32_t *pr, uint32_t len) {
  const __m256i one = _mm256_set1_epi32(1), m = _mm256_set1_epi32((int32_t)0xAAAAAAABu);
  uint32_t x = 1;
  for (; x + 8 < len; x += 8) {
    __m256i s = _mm256_add_epi32(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(up + x - 1)),
                                                  _mm256_loadu_si256((const __m256i *)(up + x))),
                                 _mm256_loadu_si256((const __m256i *)(up + x + 1)));
    __m256i a = _mm256_add_epi32(_mm256_abs_epi32(s), one);
    __m256i ev = _mm256_srli_epi64(_mm256_mul_epu32(a, m), 33);
    __m256i od = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), m), 33), 32);
    _mm256_storeu_si256((__m256i *)(pr + x), _mm256_sign_epi32(_mm256_blend_epi32(ev, od, 0xAA), s));
  }
  return x;
}
#endif

/* residuals of row `row` (rows in order) into dst; the reconstruction becomes the next row's above row */
static void m1_row(m1_src *m, uint32_t row, int16_t *dst) {
  const uint32_t w = m->w, base = row * w, len = (m->n - base < w) ? m->n - base : w;
  const int32_t qfix = m->qfix, hq = qfix >> 1; const float invq = 1.0f / (float)qfix;
  int32_t *sc = m->srow, *fc = m->fcur, *pr = m->prow; const int32_t *up = m->fprev; int16_t *rc = dst; int32_t bad = 0;
  if (m->in16) {
    const uint16_t *in = m->in16 + base; uint32_t j = 0;
#ifdef B4D_HAVE_AVX2
    for (; j + 8 <= len; j += 8)
      _mm256_storeu_si256((__m256i *)(sc + j), _mm256_i32gather_epi32((const int *)m->lut, _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(in + j))), 4));
#endif
    for (; j < len; j++) sc[j] = m->lut[in[j]];
  }
  else { const uint8_t *in = m->in8 + base; for (uint32_t j = 0; j < len; j++) sc[j] = m->lut[in[j]]; }
  const uint8_t *dc = NULL;
  if (m->dith) { dither_fill_at(m->drow, base, len, m->ds0); dc = m->drow; }
  #define M1S(j, p) do { int32_t qr, pp_ = (p);                                                                         \
      if (dc) { int32_t dd = dither_off(dc[j], qfix); qr = (int32_t)floorf((float)(sc[j] - pp_ + dd) * invq);           \
                bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pp_ + qr * qfix - dd + hq; }                    \
      else { qr = rnd_f((float)(sc[j] - pp_) * invq); bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pp_ + qr * qfix; } \
    } while (0)
  if (row == 0) {
    M1S(0, 0);
    for (uint32_t j = 1; j < len; j++) M1S(j, fc[j - 1]);
  } else if (m->pred == B4D_PRED_MEAN3D || m->pred == B4D_PRED_ABOVE) {
    if (m->pred == B4D_PRED_MEAN3D) {                    /* mean3d_row_i32 on one plane: rounded mean of up[x-1..x+1] within the row */
      if (len >= 3) {
        uint32_t x0 = 1;
#ifdef B4D_HAVE_AVX2
        x0 = mean3_row_avx2(up, pr, len);
#endif
        for (uint32_t x = x0; x + 1 < len; x++) { int32_t s = up[x - 1] + up[x] + up[x + 1]; pr[x] = s >= 0 ? (s + 1) / 3 : -((-s + 1) / 3); }
        { int32_t s = up[0] + up[1]; pr[0] = s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2); }
        { uint32_t x = len - 1; int32_t s = up[x - 1] + up[x]; pr[x] = s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2); }
      } else if (len == 2) { int32_t s = up[0] + up[1]; pr[0] = pr[1] = s >= 0 ? (s + 1) / 2 : -((-s + 1) / 2); }
      else pr[0] = up[0];
    } else pr = (int32_t *)up;
    /* pass 1: quantised residual into fc; pass 2: overflow check, int16 store, reconstruction (as mode1_encode) */
    uint32_t j0 = 0;
#ifdef B4D_HAVE_AVX2
    j0 = m1_quant_avx2(sc, pr, dc, fc, rc, len, qfix, invq, dc ? 1 : m->pred == B4D_PRED_MEAN3D ? 0 : 2, &bad);
#endif
    if (dc) {
      for (uint32_t j = j0; j < len; j++) fc[j] = (int32_t)floorf((float)(sc[j] - pr[j] + dither_off(dc[j], qfix)) * invq);
      for (uint32_t j = j0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix - dither_off(dc[j], qfix) + hq; }
    } else if (m->pred == B4D_PRED_MEAN3D) {
      for (uint32_t j = j0; j < len; j++) fc[j] = rnd_f((float)(sc[j] - pr[j]) * invq);
      for (uint32_t j = j0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix; }
    } else {
      for (uint32_t j = j0; j < len; j++) { float x = (float)(sc[j] - pr[j]) * invq; fc[j] = (int32_t)(x + (x >= 0.0f ? 0.5f : -0.5f)); }
      for (uint32_t j = j0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix; }
    }
  } else {
    M1S(0, up[0]);
    for (uint32_t j = 1; j < len; j++) M1S(j, (fc[j - 1] + up[j]) >> 1);
  }
  #undef M1S
  m->bad |= bad; m->row = row; m->started = 1;
  int32_t *t = m->fprev; m->fprev = m->fcur; m->fcur = t;
}

/* Residual source for the entropy stage: either stored residuals (r) or, for the open-loop modes 0/2, the
   samples themselves (raw or through the stabilising LUT) from which a group's residuals are recomputed on the
   fly. Recomputing keeps the hot set at the input tile + LUT instead of tile + samples + residuals (3x), so
   larger chunks stay in L2; the arithmetic is residual_u16's, element by element, so the bitstream is identical.
   Mode 2 with dither (lutd): the sample is (lutd[x] + d) >> DITHER_SHIFT, d the element's dither byte, generated a row
   at a time into drow (as the stored path's whole-block stream, element for element). */
typedef struct {
  const int16_t *r; const uint8_t *in8; const uint16_t *in16; const uint16_t *lut; uint32_t n, w; int above;
  const int32_t *lutd; uint64_t ds0; uint8_t *drow; m1_src *m1;
  uint16_t *vcur, *vprev; uint32_t row; int cached;     /* 2-row cache of stabilised samples (one LUT lookup per element) */
} resid_src;

static inline uint16_t rs_v(const resid_src *s, uint32_t i) {
  uint32_t x = s->in16 ? s->in16[i] : s->in8[i];
  if (s->lutd) return (uint16_t)((s->lutd[x] + (int32_t)dither_byte(s->ds0, i)) >> DITHER_SHIFT);
  return s->lut ? s->lut[x] : (uint16_t)x;
}

static void rs_row(const resid_src *s, uint32_t row, uint16_t *dst) {
  uint32_t i0 = row * s->w, cnt = (s->n - i0 < s->w) ? s->n - i0 : s->w;
  if (s->lutd && s->drow) {
    const int32_t *t = s->lutd; const uint8_t *d = s->drow; dither_fill_at(s->drow, i0, cnt, s->ds0);
    if (s->in16) { const uint16_t *in = s->in16 + i0; for (uint32_t j = 0; j < cnt; j++) dst[j] = (uint16_t)((t[in[j]] + (int32_t)d[j]) >> DITHER_SHIFT); }
    else { const uint8_t *in = s->in8 + i0; for (uint32_t j = 0; j < cnt; j++) dst[j] = (uint16_t)((t[in[j]] + (int32_t)d[j]) >> DITHER_SHIFT); }
  }
  else if (s->in16 && s->lut) { const uint16_t *in = s->in16 + i0, *t = s->lut; for (uint32_t j = 0; j < cnt; j++) dst[j] = t[in[j]]; }
  else for (uint32_t j = 0; j < cnt; j++) dst[j] = rs_v(s, i0 + j);
}

/* residuals of elements [i0, i0 + len) into tmp (or a pointer into r when residuals are stored). Groups are
   visited in order, so the row cache advances one row at a time; a jump rebuilds both rows. */
static const int16_t *group_resid(resid_src *s, uint32_t i0, uint32_t len, int16_t *tmp) {
  if (s->r) return s->r + i0;
  if (s->m1) {                                                           /* mode 1: rows inside the group straight into tmp */
    m1_src *m = s->m1; const uint32_t w = m->w; uint32_t i = i0, k = 0;
    while (k < len) {
      uint32_t row = i / w, j = i - row * w, rlen = (m->n - row * w < w) ? m->n - row * w : w, span = rlen - j;
      if (span > len - k) span = len - k;
      if (j == 0 && span == rlen) m1_row(m, row, tmp + k);
      else {
        if (!m->started || row != m->row) m1_row(m, row, m->rrow);
        memcpy(tmp + k, m->rrow + j, 2 * (size_t)span);
      }
      i += span; k += span;
    }
    return tmp;
  }
  const uint32_t w = s->w;
  if (!s->vcur) {                                                        /* no cache (very wide rows): per element */
    for (uint32_t k = 0; k < len; k++) {
      uint32_t i = i0 + k; uint16_t cur = rs_v(s, i), pred;
      if (i < w) pred = i ? rs_v(s, i - 1) : 0;
      else if (s->above || (i % w) == 0) pred = rs_v(s, i - w);
      else pred = (uint16_t)(((uint32_t)rs_v(s, i - 1) + rs_v(s, i - w)) >> 1);
      tmp[k] = (int16_t)(uint16_t)(cur - pred);
    }
    return tmp;
  }
  uint32_t i = i0, k = 0;
  while (k < len) {
    uint32_t row = i / w, j = i - row * w, span = w - j; if (span > len - k) span = len - k;
    if (!s->cached || row != s->row) {
      if (s->cached && row == s->row + 1) { uint16_t *t = s->vprev; s->vprev = s->vcur; s->vcur = t; }
      else if (row > 0) rs_row(s, row - 1, s->vprev);
      rs_row(s, row, s->vcur); s->row = row; s->cached = 1;
    }
    const uint16_t *cur = s->vcur + j, *up = s->vprev + j; int16_t *out = tmp + k;
    if (row == 0) {                                                      /* first row: left prediction, r[0] = v[0] */
      for (uint32_t q = 0; q < span; q++) out[q] = (int16_t)(uint16_t)(cur[q] - ((j + q) ? cur[(int32_t)q - 1] : 0));
    } else if (s->above) {
      for (uint32_t q = 0; q < span; q++) out[q] = (int16_t)(uint16_t)(cur[q] - up[q]);
    } else {
      for (uint32_t q = 0; q < span; q++) {
        uint16_t pred = (j + q) ? (uint16_t)(((uint32_t)cur[(int32_t)q - 1] + up[q]) >> 1) : up[0];
        out[q] = (int16_t)(uint16_t)(cur[q] - pred);
      }
    }
    i += span; k += span;
  }
  return tmp;
}

/* Mode 1: closed loop in fixed point. s = stabilised sample (FIX units),
   f = reconstructed (FIX units), qfix = q in FIX units. Returns 0 if any
   residual overflowed int16 (caller stores block raw). */
/* s == NULL: the stabilised samples are looked up a row at a time from lut[in8 or in16] into srow (w int32) instead of
   a whole block of them, so the block's working set stays in the per-core cache when every core is coding (the
   output is the same). */
static int mode1_encode(const int32_t *s, int32_t *f, int16_t *r, uint32_t n,
                        uint32_t w, int32_t qfix, int pred_id, uint32_t ph, const uint8_t *dr,
                        const uint8_t *in8, const uint16_t *in16, const int32_t *lut, int32_t *srow) {
  if (w == 0 || w > n) w = n;
  uint32_t so = 0;                                     /* s[i] is sp[i - so] */
  const int32_t *sp = s;
  #define S_ROW(b, l) do { if (!s) { so = (b); sp = srow;                                                            \
      if (in16) for (uint32_t j_ = 0; j_ < (l); j_++) srow[j_] = lut[in16[(b) + j_]];                               \
      else for (uint32_t j_ = 0; j_ < (l); j_++) srow[j_] = lut[in8[(b) + j_]]; } } while (0)
  const int above_only = pred_id == B4D_PRED_ABOVE;
  const float invq = 1.0f / (float)qfix;
  const int32_t hq = qfix >> 1;
  int32_t bad = 0;
  /* dr (v8 dither, one value per element, or NULL): the residual is floor((s - p + d) / q) with d = dither_off(dr), and the
     reconstruction p + qr q - d + q/2, so the error is uniform on [-q/2, q/2) whatever the signal */
  #define M1_STEP(i, p) do { int32_t qr, pp_ = (p);                                                                     \
      if (dr) { int32_t dd = dither_off(dr[i], qfix); qr = (int32_t)floorf((float)(sp[(i) - so] - pp_ + dd) * invq);  \
                bad |= (qr + 32768) >> 16; r[i] = (int16_t)qr; f[i] = pp_ + qr * qfix - dd + hq; }                    \
      else { qr = rnd_f((float)(sp[(i) - so] - pp_) * invq); bad |= (qr + 32768) >> 16; r[i] = (int16_t)qr; f[i] = pp_ + qr * qfix; } \
    } while (0)
  S_ROW(0, w);
  M1_STEP(0, 0);
  for (uint32_t j = 1; j < w; j++) M1_STEP(j, f[j - 1]);
  if (pred_id == B4D_PRED_MEAN3D) {
    ph = plane_rows_of(ph, n, w);
    int32_t predrow[4096]; int32_t *pr = w <= 4096 ? predrow : (int32_t *)malloc(4 * (size_t)w);
    for (uint32_t base = w; base < n; base += w) {
      uint32_t len = (n - base < w) ? n - base : w;
      mean3d_row_i32(f, n, w, ph, base, len, pr);                        /* from reconstructed rows/planes */
      S_ROW(base, len);
      const int32_t *sc = sp + (base - so); int32_t *fc = f + base; int16_t *rc = r + base;
      /* pass 1: quantised residual into fc; pass 2: overflow check, int16 store, reconstruction (as M1_STEP, but the row has no serial dependency) */
      if (dr) {
        const uint8_t *dc = dr + base;
        for (uint32_t j = 0; j < len; j++) fc[j] = (int32_t)floorf((float)(sc[j] - pr[j] + dither_off(dc[j], qfix)) * invq);
        for (uint32_t j = 0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix - dither_off(dc[j], qfix) + hq; }
      } else {
        for (uint32_t j = 0; j < len; j++) fc[j] = rnd_f((float)(sc[j] - pr[j]) * invq);
        for (uint32_t j = 0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix; }
      }
    }
    if (pr != predrow) free(pr);
    return bad == 0;
  }
  for (uint32_t base = w; base < n; base += w) {
    uint32_t len = (n - base < w) ? n - base : w;
    S_ROW(base, len);
    if (above_only) {
      const int32_t *sc = sp + (base - so), *up = f + base - w; int32_t *fc = f + base; int16_t *rc = r + base;
      if (dr) {
        const uint8_t *dc = dr + base;
        for (uint32_t j = 0; j < len; j++) fc[j] = (int32_t)floorf((float)(sc[j] - up[j] + dither_off(dc[j], qfix)) * invq);
        for (uint32_t j = 0; j < len; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = up[j] + qr * qfix - dither_off(dc[j], qfix) + hq; }
        continue;
      }
      /* pass 1: quantised residual into fc (int32, vectorises: sub, cvt, mul, round, cvt) */
      for (uint32_t j = 0; j < len; j++) {
        float x = (float)(sc[j] - up[j]) * invq;
        fc[j] = (int32_t)(x + (x >= 0.0f ? 0.5f : -0.5f));          /* round half away from zero */
      }
      /* pass 2: overflow check, int16 store, reconstruction */
      for (uint32_t j = 0; j < len; j++) {
        int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = up[j] + qr * qfix;
      }
    } else {
      M1_STEP(base, f[base - w]);
      for (uint32_t j = 1; j < len; j++) {
        uint32_t i = base + j; int32_t p = (f[i - 1] + f[i - w]) >> 1;
        M1_STEP(i, p);
      }
    }
  }
  #undef M1_STEP
  #undef S_ROW
  return bad == 0;
}
static void mode1_decode(const int16_t *r, int32_t *f, uint32_t n, uint32_t w, int32_t qfix, int pred_id, uint32_t ph, const uint8_t *dr) {
  if (w == 0 || w > n) w = n;
  const int above_only = pred_id == B4D_PRED_ABOVE;
  const int32_t hq = dr ? qfix >> 1 : 0;
  #define DOFF(i) (dr ? dither_off(dr[i], qfix) : 0)
  f[0] = r[0] * qfix - DOFF(0) + hq;
  for (uint32_t j = 1; j < w; j++) f[j] = f[j - 1] + r[j] * qfix - DOFF(j) + hq;
  if (pred_id == B4D_PRED_MEAN3D) {
    ph = plane_rows_of(ph, n, w);
    int32_t predrow[4096]; int32_t *pr = w <= 4096 ? predrow : (int32_t *)malloc(4 * (size_t)w);
    for (uint32_t base = w; base < n; base += w) {
      uint32_t len = (n - base < w) ? n - base : w;
      mean3d_row_i32(f, n, w, ph, base, len, pr);
      if (dr) for (uint32_t j = 0; j < len; j++) f[base + j] = pr[j] + r[base + j] * qfix - dither_off(dr[base + j], qfix) + hq;
      else for (uint32_t j = 0; j < len; j++) f[base + j] = pr[j] + r[base + j] * qfix;
    }
    if (pr != predrow) free(pr);
    return;
  }
  for (uint32_t base = w; base < n; base += w) {
    uint32_t len = (n - base < w) ? n - base : w;
    if (above_only) {
      const int32_t *up = f + base - w; int32_t *fc = f + base; const int16_t *rc = r + base;
      if (dr) { const uint8_t *dc = dr + base; for (uint32_t j = 0; j < len; j++) fc[j] = up[j] + rc[j] * qfix - dither_off(dc[j], qfix) + hq; }
      else for (uint32_t j = 0; j < len; j++) fc[j] = up[j] + rc[j] * qfix;
    } else {
      f[base] = f[base - w] + r[base] * qfix - DOFF(base) + hq;
      for (uint32_t j = 1; j < len; j++) {
        uint32_t i = base + j; f[i] = ((f[i - 1] + f[i - w]) >> 1) + r[i] * qfix - DOFF(i) + hq;
      }
    }
  }
  #undef DOFF
}

/* ------------------------------------------------------------------------ */
/* Per-pixel noise list (v7: mode 1, above3)                                */
/*                                                                          */
/* The row passes above stay as they are, vectorised over every pixel; the  */
/* listed pixels are then redone one by one, so the cost follows the length */
/* of the list and not the width of the row.                                */
/*                                                                          */
/* In the block: u32 plane_rows, u32 entries, u32 offsets, f32 skip,        */
/* u8 tiers + 3 reserved, f32 extra[tiers], then the entries in raster      */
/* order within a plane, one byte each (gap << 3 | tier; gap 31 is followed */
/* by a varint of gap - 31), then the offsets: varint gap in entry index    */
/* and i16 delta.                                                           */
/* ------------------------------------------------------------------------ */

#define PX_BUCKETS 512               /* one per T unit; above the last the extra noise no longer moves the step */
#define PX_HDR(tiers) (20 + 4 * (size_t)(tiers))
#define PX_MAX_RAW_SHARE 400         /* the list stays under 1/400 of the block's raw bytes, about half of what it saves at q = 1 */

struct b4d_pixmap {
  uint32_t h, w, nent; int ntiers; float extra[B4D_PIX_TIERS], skip;
  uint32_t *row_start; uint16_t *col; uint8_t *tier; int16_t *delta;   /* listed pixels, by sensor row */
  int16_t *flat_a, *flat_b;                                            /* dense, h * w, or NULL */
};

void b4d_pixmap_free(b4d_pixmap *m) {
  if (!m) return;
  free(m->row_start); free(m->col); free(m->tier); free(m->delta); free(m->flat_a); free(m->flat_b); free(m);
}

b4d_pixmap *b4d_pixmap_new(uint32_t height, uint32_t width, const uint8_t *tier, const int16_t *delta, const int16_t *flat_a,
                           const int16_t *flat_b, const float *extra_var, int ntiers, float skip) {
  if (!height || !width || width > 65535 || ntiers < 1 || ntiers > B4D_PIX_TIERS) return NULL;
  b4d_pixmap *m = (b4d_pixmap *)calloc(1, sizeof *m);
  if (!m) return NULL;
  m->h = height; m->w = width; m->ntiers = ntiers; m->skip = skip;
  for (int t = 0; t < ntiers; t++) m->extra[t] = extra_var[t];
  const size_t np = (size_t)height * width;
  uint32_t count = 0;
  if (tier) for (size_t i = 0; i < np; i++) count += tier[i] != 0;
  m->nent = count;
  m->row_start = (uint32_t *)malloc(4 * ((size_t)height + 1));
  m->col = (uint16_t *)malloc(2 * (size_t)count + 2); m->tier = (uint8_t *)malloc((size_t)count + 1); m->delta = (int16_t *)calloc((size_t)count + 1, 2);
  if (flat_a) m->flat_a = (int16_t *)malloc(2 * np);
  if (flat_b) m->flat_b = (int16_t *)malloc(2 * np);
  if (!m->row_start || !m->col || !m->tier || !m->delta || (flat_a && !m->flat_a) || (flat_b && !m->flat_b)) { b4d_pixmap_free(m); return NULL; }
  if (flat_a) memcpy(m->flat_a, flat_a, 2 * np);
  if (flat_b) memcpy(m->flat_b, flat_b, 2 * np);
  uint32_t e = 0;
  for (uint32_t y = 0; y < height; y++) {
    m->row_start[y] = e;
    if (tier) for (uint32_t x = 0; x < width; x++) {
      size_t i = (size_t)y * width + x;
      if (!tier[i]) continue;
      if ((int)tier[i] > ntiers) { b4d_pixmap_free(m); return NULL; }
      m->col[e] = (uint16_t)x; m->tier[e] = (uint8_t)(tier[i] - 1); m->delta[e] = delta ? delta[i] : 0; e++;
    }
  }
  m->row_start[height] = e;
  return m;
}

typedef struct {
  int32_t step[B4D_PIX_TIERS][PX_BUCKETS]; float inv[B4D_PIX_TIERS][PX_BUCKETS];
  int32_t skip_step; int ntiers; float rn, skip, extra[B4D_PIX_TIERS]; uint8_t qi;
} px_tab;

/* step per tier and T unit. The decoder builds the same table from the values in the block, and the two must agree bit for
   bit, as the VST tables already must (IEEE doubles, no fast-math). */
typedef struct { uint8_t qi; float rn, skip; int ntiers; const float *extra; } pxtab_key;
static int pxtab_match(const void *e, const void *k) {
  const px_tab *l = (const px_tab *)e; const pxtab_key *q = (const pxtab_key *)k;
  return l->qi == q->qi && l->rn == q->rn && l->skip == q->skip && l->ntiers == q->ntiers && !memcmp(l->extra, q->extra, 4 * (size_t)q->ntiers);
}
static const px_tab *pxtab_get(uint8_t qi, float rn, int ntiers, const float *extra, float skip) {
  uint32_t a; memcpy(&a, &extra[ntiers - 1], 4);
  unsigned slot = lut_slot(qi, skip, (float)ntiers, rn) ^ (a * 2654435761u >> 26);
  slot %= LUT_SLOTS;
  const pxtab_key key = {qi, rn, skip, ntiers, extra};
  px_tab *l = (px_tab *)cache_find((void **)g_pxtab, LUT_SLOTS, slot, pxtab_match, &key);
  if (l) return l;
  l = (px_tab *)malloc(sizeof *l);
  if (!l) return NULL;
  memset(l->extra, 0, sizeof l->extra);
  const double c = (double)rn * rn + 0.375, qfix = (double)qi * (FIX_SCALE / 4);
  for (int t = 0; t < ntiers; t++) {
    for (int b = 0; b < PX_BUCKETS; b++) {
      double T = b + 0.5, u2 = 0.25 * T * T;
      if (u2 < c) u2 = c;
      int32_t st = rnd_d(qfix * sqrt(1.0 + (double)extra[t] / u2));
      l->step[t][b] = st; l->inv[t][b] = 1.0f / (float)st;
    }
    l->extra[t] = extra[t];
  }
  l->skip_step = skip > 0 ? rnd_d(qfix * (double)skip) : INT32_MAX;
  l->ntiers = ntiers; l->rn = rn; l->skip = skip; l->qi = qi;
  return (const px_tab *)cache_insert((void **)g_pxtab, LUT_SLOTS, slot, pxtab_match, &key, l);
}
static inline int px_bucket(int32_t p) { int b = p >> FIX_SHIFT; return b < 0 ? 0 : b >= PX_BUCKETS ? PX_BUCKETS - 1 : b; }

typedef struct {
  uint32_t plane_rows, nent;
  uint32_t *row_start; uint16_t *col; uint8_t *tier; int16_t *delta; int32_t *saved;
  const px_tab *tab;
} px_block;

static size_t px_mem_bytes(uint32_t plane_rows, uint32_t nent, uint32_t w) {
  return ALIGN64(4 * ((size_t)plane_rows + 1)) + ALIGN64(2 * (size_t)nent + 2) + ALIGN64((size_t)nent + 1) + ALIGN64(2 * (size_t)nent + 2) + 4 * (size_t)w;
}
static void px_mem_set(px_block *B, uint8_t *mem) {
  B->row_start = (uint32_t *)mem; mem += ALIGN64(4 * ((size_t)B->plane_rows + 1));
  B->col = (uint16_t *)mem; mem += ALIGN64(2 * (size_t)B->nent + 2);
  B->tier = mem; mem += ALIGN64((size_t)B->nent + 1);
  B->delta = (int16_t *)mem; mem += ALIGN64(2 * (size_t)B->nent + 2);
  B->saved = (int32_t *)mem;
}
static inline uint32_t px_lower(const uint16_t *col, uint32_t lo, uint32_t hi, uint32_t x) {
  while (lo < hi) { uint32_t mid = lo + (hi - lo) / 2; if (col[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}
/* how many listed pixels fall in plane_rows rows of w pixels at (y0, x0) */
static uint32_t px_count(const b4d_pixmap *m, uint32_t y0, uint32_t x0, uint32_t plane_rows, uint32_t w) {
  uint32_t c = 0;
  for (uint32_t y = y0; y < y0 + plane_rows && y < m->h; y++) {
    uint32_t lo = m->row_start[y], hi = m->row_start[y + 1];
    c += px_lower(m->col, lo, hi, x0 + w) - px_lower(m->col, lo, hi, x0);
  }
  return c;
}
static inline size_t px_put_varint(uint8_t *p, uint32_t v) {
  size_t k = 0;
  while (v >= 128) { p[k++] = (uint8_t)(v | 128); v >>= 7; }
  p[k++] = (uint8_t)v;
  return k;
}
static inline int px_get_varint(const uint8_t **pp, const uint8_t *end, uint32_t *out) {
  const uint8_t *p = *pp; uint32_t v = 0; int shift = 0;
  while (p < end && shift < 35) {
    uint8_t c = *p++; v |= (uint32_t)(c & 127) << shift; shift += 7;
    if (!(c & 128)) { *pp = p; *out = v; return 0; }
  }
  return -1;
}
/* worst case: a 5-byte gap per entry (+ 1 tier byte), a 5-byte index and 2-byte delta per offset */
static size_t px_payload_cap(uint32_t nent, int ntiers) { return PX_HDR(ntiers) + 13 * (size_t)nent; }

/* fill B (arrays set, plane_rows and nent known) from the map and serialise it; returns the payload size */
static size_t px_build(const b4d_pixmap *m, uint32_t y0, uint32_t x0, uint32_t w, px_block *B, uint8_t *out) {
  uint8_t *p = out + PX_HDR(m->ntiers); uint32_t e = 0, noff = 0; int64_t prev = -1;
  for (uint32_t y = 0; y < B->plane_rows; y++) {
    B->row_start[y] = e;
    uint32_t Y = y0 + y;
    if (Y >= m->h) continue;
    uint32_t lo = m->row_start[Y], hi = m->row_start[Y + 1], a = px_lower(m->col, lo, hi, x0), b = px_lower(m->col, lo, hi, x0 + w);
    for (uint32_t k = a; k < b; k++, e++) {
      uint32_t c = (uint32_t)m->col[k] - x0; int64_t idx = (int64_t)y * w + c; uint32_t gap = (uint32_t)(idx - prev - 1);
      prev = idx;
      B->col[e] = (uint16_t)c; B->tier[e] = m->tier[k]; B->delta[e] = m->delta[k]; noff += m->delta[k] != 0;
      if (gap < 31) *p++ = (uint8_t)(gap << 3 | m->tier[k]);
      else { *p++ = (uint8_t)(31u << 3 | m->tier[k]); p += px_put_varint(p, gap - 31); }
    }
  }
  B->row_start[B->plane_rows] = e;
  uint32_t pe = 0;
  for (uint32_t k = 0; k < e; k++) if (B->delta[k]) {
    p += px_put_varint(p, k - pe); pe = k;
    uint16_t d = (uint16_t)B->delta[k]; memcpy(p, &d, 2); p += 2;
  }
  put_u32(out, B->plane_rows); put_u32(out + 4, e); put_u32(out + 8, noff); put_f32(out + 12, m->skip);
  out[16] = (uint8_t)m->ntiers; out[17] = out[18] = out[19] = 0;
  for (int t = 0; t < m->ntiers; t++) put_f32(out + 20 + 4 * t, m->extra[t]);
  return (size_t)(p - out);
}

/* header of a list inside a block: plane_rows and entries, validated against the block */
static int px_peek(const uint8_t *p, size_t len, uint32_t n, uint32_t w, uint32_t *plane_rows, uint32_t *nent) {
  if (len < 20 || w < 3 || w > 65535 || n % w) return -1;
  const uint32_t ph = get_u32(p), ne = get_u32(p + 4);
  const int ntiers = p[16];
  if (ph == 0 || ph > n / w || ntiers < 1 || ntiers > B4D_PIX_TIERS || len < PX_HDR(ntiers) || ne > len - PX_HDR(ntiers) || (uint64_t)ne > (uint64_t)ph * w) return -1;
  *plane_rows = ph; *nent = ne;
  return 0;
}
static int px_parse(const uint8_t *p, size_t len, uint32_t w, uint8_t qi, float rn, px_block *B) {
  const uint32_t noff = get_u32(p + 8); const float skip = get_f32(p + 12); const int ntiers = p[16];
  float extra[B4D_PIX_TIERS];
  /* the step must stay far from overflow whatever the block says: at most 64 x q, at T(0) */
  const double c = (double)rn * rn + 0.375;
  for (int t = 0; t < ntiers; t++) { extra[t] = get_f32(p + 20 + 4 * t); if (!(extra[t] >= 0.0f) || (double)extra[t] > 4095.0 * c) return -1; }
  if (!(skip >= 0.0f)) return -1;
  const uint8_t *q = p + PX_HDR(ntiers), *end = p + len;
  uint64_t idx = 0; uint32_t y_prev = 0;
  const uint64_t limit = (uint64_t)B->plane_rows * w;
  B->row_start[0] = 0;
  for (uint32_t e = 0; e < B->nent; e++) {
    if (q >= end) return -1;
    uint8_t b = *q++; uint32_t gap = b >> 3, t = b & 7;
    if (gap == 31) { uint32_t g2; if (px_get_varint(&q, end, &g2)) return -1; gap = 31 + g2; }
    idx += gap;
    if (idx >= limit || (int)t >= ntiers) return -1;
    uint32_t y = (uint32_t)(idx / w);
    while (y_prev < y) B->row_start[++y_prev] = e;
    B->col[e] = (uint16_t)(idx - (uint64_t)y * w); B->tier[e] = (uint8_t)t; B->delta[e] = 0;
    idx++;
  }
  while (y_prev < B->plane_rows) B->row_start[++y_prev] = B->nent;
  if (noff > B->nent) return -1;
  uint32_t k = 0;
  for (uint32_t o = 0; o < noff; o++) {
    uint32_t g; if (px_get_varint(&q, end, &g) || end - q < 2) return -1;
    k += g; if (k >= B->nent) return -1;
    uint16_t d; memcpy(&d, q, 2); q += 2; B->delta[k] = (int16_t)d;
  }
  B->tab = pxtab_get(qi, rn, ntiers, extra, skip);
  return B->tab ? 0 : -1;
}

/* as mode1_encode for above3 (the first row uses the left predictor and the common step); then the listed pixels of each row
   are redone with their own step, from the same prediction. f is only the predictor's state here, so a pixel past the skip
   limit simply stores its prediction. */
/* dr (v8, or NULL): dither values; dall: also dither the pixels that are not listed (else only the listed ones) */
static int mode1_encode_px(const int32_t *s, int32_t *f, int16_t *r, uint32_t n, uint32_t w, int32_t qfix, const px_block *B,
                           const uint8_t *dr, int dall) {
  const float invq = 1.0f / (float)qfix; const uint32_t ph = n / w; const px_tab *T = B->tab; const int32_t skip = T->skip_step;
  const uint8_t *dc0 = dall ? dr : NULL; const int32_t hq = qfix >> 1;
  int32_t bad = 0, predrow[4096]; int32_t *pr = w <= 4096 ? predrow : (int32_t *)malloc(4 * (size_t)w);
  if (!pr) return 0;
  for (uint32_t j = 0; j < w; j++) {
    const int32_t p = j ? f[j - 1] : 0; int32_t qr;
    if (dc0) { int32_t dd = dither_off(dc0[j], qfix); qr = (int32_t)floorf((float)(s[j] - p + dd) * invq); f[j] = p + qr * qfix - dd + hq; }
    else { qr = rnd_f((float)(s[j] - p) * invq); f[j] = p + qr * qfix; }
    bad |= (qr + 32768) >> 16; r[j] = (int16_t)qr;
  }
  uint32_t y = B->plane_rows > 1 ? 1 : 0;
  for (uint32_t base = w; base < n; base += w) {
    mean3d_row_i32(f, n, w, ph, base, w, pr);
    const int32_t *sc = s + base; int32_t *fc = f + base; int16_t *rc = r + base;
    if (dc0) {
      const uint8_t *dc = dc0 + base;
      for (uint32_t j = 0; j < w; j++) fc[j] = (int32_t)floorf((float)(sc[j] - pr[j] + dither_off(dc[j], qfix)) * invq);
      for (uint32_t j = 0; j < w; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix - dither_off(dc[j], qfix) + hq; }
    } else {
      for (uint32_t j = 0; j < w; j++) fc[j] = rnd_f((float)(sc[j] - pr[j]) * invq);
      for (uint32_t j = 0; j < w; j++) { int32_t qr = fc[j]; bad |= (qr + 32768) >> 16; rc[j] = (int16_t)qr; fc[j] = pr[j] + qr * qfix; }
    }
    const uint32_t lo = B->row_start[y], hi = B->row_start[y + 1];
    for (uint32_t e = lo; e < hi; e++) {
      const uint32_t c = B->col[e]; const int32_t p = pr[c]; const int b = px_bucket(p), t = B->tier[e], st = T->step[t][b];
      int32_t qr, v;
      if (dr) { int32_t dd = dither_off(dr[base + c], st); qr = (int32_t)floorf((float)(sc[c] - B->delta[e] - p + dd) * T->inv[t][b]); v = p + qr * st - dd + (st >> 1); }
      else { qr = rnd_f((float)(sc[c] - B->delta[e] - p) * T->inv[t][b]); v = p + qr * st; }
      bad |= (qr + 32768) >> 16; rc[c] = (int16_t)qr; fc[c] = st >= skip ? p : v;
    }
    if (++y == B->plane_rows) y = 0;
  }
  if (pr != predrow) free(pr);
  return bad == 0;
}
/* f must end up holding the real values, and the row below must be predicted from the same state as the encoder's: each listed
   pixel's real value waits in saved until that prediction is done. */
static int mode1_decode_px(const int16_t *r, int32_t *f, uint32_t n, uint32_t w, int32_t qfix, const px_block *B,
                           const uint8_t *dr, int dall, int32_t *lst /* with dr: each listed occurrence's step, raster order */) {
  const uint32_t ph = n / w; const px_tab *T = B->tab; const int32_t skip = T->skip_step; int32_t *saved = B->saved;
  const uint8_t *dc0 = dall ? dr : NULL; const int32_t hq = dc0 ? qfix >> 1 : 0;
  int32_t predrow[4096]; int32_t *pr = w <= 4096 ? predrow : (int32_t *)malloc(4 * (size_t)w);
  if (!pr) return -1;
  for (uint32_t j = 0; j < w; j++) f[j] = (j ? f[j - 1] : 0) + r[j] * qfix - (dc0 ? dither_off(dc0[j], qfix) : 0) + hq;
  uint32_t y = B->plane_rows > 1 ? 1 : 0, plo = 0, phi = 0, pbase = 0;
  for (uint32_t base = w; base < n; base += w) {
    mean3d_row_i32(f, n, w, ph, base, w, pr);
    for (uint32_t e = plo; e < phi; e++) f[pbase + B->col[e]] = saved[e - plo] + B->delta[e];
    int32_t *fc = f + base; const int16_t *rc = r + base;
    if (dc0) { const uint8_t *dc = dc0 + base; for (uint32_t j = 0; j < w; j++) fc[j] = pr[j] + rc[j] * qfix - dither_off(dc[j], qfix) + hq; }
    else for (uint32_t j = 0; j < w; j++) fc[j] = pr[j] + rc[j] * qfix;
    const uint32_t lo = B->row_start[y], hi = B->row_start[y + 1];
    for (uint32_t e = lo; e < hi; e++) {
      const uint32_t c = B->col[e]; const int32_t p = pr[c]; const int32_t st = T->step[B->tier[e]][px_bucket(p)];
      int32_t v = (int32_t)((uint32_t)p + (uint32_t)(int32_t)rc[c] * (uint32_t)st);
      if (dr) { v = v - dither_off(dr[base + c], st) + (st >> 1); *lst++ = st; }
      saved[e - lo] = v; fc[c] = st >= skip ? p : v;
    }
    plo = lo; phi = hi; pbase = base;
    if (++y == B->plane_rows) y = 0;
  }
  for (uint32_t e = plo; e < phi; e++) f[pbase + B->col[e]] = saved[e - plo] + B->delta[e];
  if (pr != predrow) free(pr);
  return 0;
}

/* ---- mode 2 with the noise list (v8, above predictor). Codes are round(T/q) - smin, or with dither (all pixels)
   (T/q * DITHER_SCALE - base + r) >> DITHER_SHIFT. A listed pixel is quantised against its prediction with its own step, closed
   loop as in mode 1 (dithered if dr); its residual is that quantised value, and its code becomes the common code nearest to what
   it decodes to (or its prediction past the skip limit), which is what the pixels after it are predicted from. */
typedef struct { int dith; int32_t qfix, qi, smin, base; } m2code;
static inline int64_t floordiv64(int64_t a, int64_t b) { int64_t q = a / b; return (a % b && ((a < 0) != (b < 0))) ? q - 1 : q; }
static inline int32_t m2_T(const m2code *m, int32_t c) {           /* code -> T in FIX units (bin centre) */
  if (!m->dith) return (c + m->smin) * m->qfix;
  return (int32_t)floordiv64(((int64_t)c * DITHER_SCALE + m->base + DITHER_SCALE / 2) * m->qi, 4);
}
static inline uint16_t m2_code(const m2code *m, int32_t t) {       /* T in FIX units -> nearest code */
  int64_t c = !m->dith ? floordiv64((int64_t)t + m->qfix / 2, m->qfix) - m->smin
                       : floordiv64((int64_t)t * 4 - (int64_t)m->base * m->qi, (int64_t)DITHER_SCALE * m->qi);
  return (uint16_t)(c < 0 ? 0 : c > 65535 ? 65535 : c);
}
/* one listed pixel: residual and decoded value (T, FIX units, offset removed) from prediction code p */
static inline int32_t m2_px_quant(const px_block *B, uint32_t e, const m2code *m, int32_t p, int32_t t, const uint8_t *dr, uint32_t i, int32_t *rec, int32_t *st_out) {
  const px_tab *T = B->tab; const int32_t P = m2_T(m, p); const int b = px_bucket(P), tr = B->tier[e], st = T->step[tr][b];
  int32_t qr;
  if (dr) { int32_t dd = dither_off(dr[i], st); qr = (int32_t)floorf((float)(t - P + dd) * T->inv[tr][b]); *rec = P + qr * st - dd + (st >> 1); }
  else { qr = rnd_f((float)(t - P) * T->inv[tr][b]); *rec = P + qr * st; }
  *st_out = st;
  return qr;
}
static inline int32_t m2_px_rec(const px_block *B, uint32_t e, const m2code *m, int32_t p, int32_t qr, const uint8_t *dr, uint32_t i, int32_t *st_out) {
  const px_tab *T = B->tab; const int32_t P = m2_T(m, p); const int32_t st = T->step[B->tier[e]][px_bucket(P)];
  *st_out = st;
  int32_t v = (int32_t)((uint32_t)P + (uint32_t)qr * (uint32_t)st);
  return dr ? v - dither_off(dr[i], st) + (st >> 1) : v;
}
/* encoder: pass 0 settles the codes of the listed pixels in raster order (v in place); pass 1, after residual_u16, writes their
   residuals into r (every prediction is final by then, so the values are the same). t32: T in FIX units per input value. */
static int mode2_encode_px(uint16_t *v, int16_t *r, const uint8_t *in8, const uint16_t *in16, const int32_t *t32, uint32_t n, uint32_t w,
                           const px_block *B, const m2code *m, const uint8_t *dr, int pass) {
  const int32_t skip = B->tab->skip_step; int32_t bad = 0; uint32_t y = 0;
  for (uint32_t base = 0; base < n; base += w) {
    const uint32_t lo = B->row_start[y], hi = B->row_start[y + 1];
    for (uint32_t e = lo; e < hi; e++) {
      const uint32_t c = B->col[e], i = base + c;
      const int32_t p = base ? v[i - w] : (c ? v[i - 1] : 0);
      const int32_t t = t32[in16 ? in16[i] : in8[i]] - B->delta[e];
      int32_t rec, st; const int32_t qr = m2_px_quant(B, e, m, p, t, dr, i, &rec, &st);
      bad |= (qr + 32768) >> 16;
      if (pass == 0) v[i] = st >= skip ? (uint16_t)p : m2_code(m, rec);
      else r[i] = (int16_t)qr;
    }
    if (++y == B->plane_rows) y = 0;
  }
  return bad == 0;
}
/* decoder, above predictor: u holds the residuals on entry and the codes on exit; trec gets each listed occurrence's decoded value
   (T, FIX units, offset restored) and its step when dithered (else 0), in raster order */
static int mode2_decode_px(uint16_t *u, uint32_t n, uint32_t w, const px_block *B, const m2code *m, const uint8_t *dr, int32_t *trec) {
  const int32_t skip = B->tab->skip_step; uint32_t y = 0, k = 0; int16_t held[4096]; int16_t *hb = w <= 4096 ? held : (int16_t *)malloc(2 * (size_t)w);
  if (!hb) return -1;
  for (uint32_t base = 0; base < n; base += w) {
    const uint32_t lo = B->row_start[y], hi = B->row_start[y + 1];
    uint16_t *cur = u + base;
    if (base == 0) {                                     /* first row: left prediction, element by element */
      uint32_t e = lo;
      for (uint32_t j = 0; j < w; j++) {
        const uint16_t p = j ? cur[j - 1] : 0;
        if (e < hi && B->col[e] == j) {
          int32_t st; const int32_t rec = m2_px_rec(B, e, m, p, (int16_t)cur[j], dr, j, &st);
          trec[2 * k] = rec + B->delta[e]; trec[2 * k + 1] = dr ? st : 0; k++; cur[j] = st >= skip ? p : m2_code(m, rec); e++;
        } else cur[j] = (uint16_t)(cur[j] + p);
      }
    } else {
      const uint16_t *up = cur - w;
      for (uint32_t e = lo; e < hi; e++) hb[e - lo] = (int16_t)cur[B->col[e]];
      for (uint32_t j = 0; j < w; j++) cur[j] = (uint16_t)(cur[j] + up[j]);
      for (uint32_t e = lo; e < hi; e++) {
        const uint32_t c = B->col[e]; const uint16_t p = up[c];
        int32_t st; const int32_t rec = m2_px_rec(B, e, m, p, hb[e - lo], dr, base + c, &st);
        trec[2 * k] = rec + B->delta[e]; trec[2 * k + 1] = dr ? st : 0; k++; cur[c] = st >= skip ? p : m2_code(m, rec);
      }
    }
    if (++y == B->plane_rows) y = 0;
  }
  if (hb != held) free(hb);
  return 0;
}

/* encoder-only flat fielding: the block's rows against the dense maps */
static void px_flatten(int32_t *sv, uint32_t n, uint32_t w, const b4d_pixmap *m, uint32_t y0, uint32_t x0, uint32_t plane_rows, int32_t t0) {
  if (x0 >= m->w) return;
  const uint32_t len = m->w - x0 < w ? m->w - x0 : w;
  uint32_t y = 0;
  for (uint32_t base = 0; base < n; base += w) {
    const uint32_t Y = y0 + y;
    if (Y < m->h) {
      int32_t *s = sv + base; const size_t at = (size_t)Y * m->w + x0;
      if (m->flat_b) { const int16_t *g = m->flat_b + at; for (uint32_t j = 0; j < len; j++) s[j] += ((s[j] - t0) * g[j]) >> 15; }
      if (m->flat_a) { const int16_t *a = m->flat_a + at; for (uint32_t j = 0; j < len; j++) s[j] -= a[j]; }
    }
    if (++y == plane_rows) y = 0;
  }
}

/* ------------------------------------------------------------------------ */
/* 64-bit bit I/O (MSB first)                                               */
/* ------------------------------------------------------------------------ */

typedef struct { uint8_t *p; uint64_t acc; int n; } bitw;
typedef struct { const uint8_t *p, *end; uint64_t acc; int n; } bitr;

static inline void bw_init(bitw *b, uint8_t *p) { b->p = p; b->acc = 0; b->n = 0; }
static inline void bw_put(bitw *b, uint32_t bits, int len) {   /* len <= 31 */
  b->acc = (b->acc << len) | bits; b->n += len;
  if (b->n >= 32) {
    uint32_t v = (uint32_t)(b->acc >> (b->n - 32));
    b->p[0] = (uint8_t)(v >> 24); b->p[1] = (uint8_t)(v >> 16);
    b->p[2] = (uint8_t)(v >> 8);  b->p[3] = (uint8_t)v; b->p += 4; b->n -= 32;
  }
}
static inline void bw_flush(bitw *b) {
  while (b->n >= 8) { *b->p++ = (uint8_t)(b->acc >> (b->n - 8)); b->n -= 8; }
  if (b->n > 0) { *b->p++ = (uint8_t)(b->acc << (8 - b->n)); b->n = 0; }
}
static inline void br_init(bitr *b, const uint8_t *p, size_t len) { b->p = p; b->end = p + len; b->acc = 0; b->n = 0; }
static inline void br_refill(bitr *b) {           /* guarantee >= 57 bits */
  /* 8 bytes at once away from the end: the bits below the n counted ones are the stream's own next bits (the byte loop
     leaves zeros there), so they are only ever ORed with themselves and every value read is the same */
  if (b->n <= 56 && b->end - b->p >= 8) {
    uint64_t w; memcpy(&w, b->p, 8);
#if defined(_MSC_VER) && !defined(__clang__)
    w = _byteswap_uint64(w);
#else
    w = __builtin_bswap64(w);
#endif
    b->acc |= w >> b->n; const int nb = (64 - b->n) >> 3; b->p += nb; b->n += 8 * nb;
    return;
  }
  while (b->n <= 56) {
    uint64_t byte = (b->p < b->end) ? *b->p++ : 0;
    b->acc |= byte << (56 - b->n); b->n += 8;
  }
}
static inline uint32_t br_peek(const bitr *b, int k) { return (uint32_t)(b->acc >> (64 - k)); }
static inline void br_skip(bitr *b, int k) { b->acc <<= k; b->n -= k; }
static inline uint32_t br_bits(bitr *b, int k) { uint32_t v = br_peek(b, k); br_skip(b, k); return v; }
/* number of leading zero bits, capped at 16 (caller refilled) */
static inline int br_zeros(const bitr *b) {
  int z = b->acc ? b4d_clz64(b->acc) : 64;
  return z > 16 ? 16 : z;
}
/* number of leading one bits, capped at RICE_UMAX (caller refilled: >=57 bits) */
static inline int br_unary(bitr *b) {
  uint64_t inv = ~b->acc;
  int q = inv ? b4d_clz64(inv) : 64;
  if (q > RICE_UMAX) q = RICE_UMAX;
  return q;
}

/* ------------------------------------------------------------------------ */
/* Canonical Huffman (length-limited) + table-driven decoder                */
/* ------------------------------------------------------------------------ */

static int huff_lengths(const uint32_t *freq, uint8_t *len) {
  enum { MAXN = 2 * HUFF_NSYM };
  uint64_t f[MAXN]; int parent[MAXN]; uint8_t alive[MAXN]; int map[HUFF_NSYM];
  int n = 0;
  for (int s = 0; s < HUFF_NSYM; s++) { map[s] = -1; len[s] = 0; }
  for (int s = 0; s < HUFF_NSYM; s++) if (freq[s]) { map[s] = n; f[n] = freq[s]; parent[n] = -1; alive[n] = 1; n++; }
  if (n == 0) return 0;
  if (n == 1) { for (int s = 0; s < HUFF_NSYM; s++) if (map[s] >= 0) len[s] = 1; return 1; }
  int total = n;
  for (int left = n; left > 1; left--) {
    int a = -1, b = -1;
    for (int i = 0; i < total; i++) if (alive[i]) {
      if (a < 0 || f[i] < f[a]) { b = a; a = i; } else if (b < 0 || f[i] < f[b]) b = i;
    }
    alive[a] = alive[b] = 0;
    f[total] = f[a] + f[b]; parent[a] = parent[b] = total; parent[total] = -1; alive[total] = 1; total++;
  }
  int maxlen = 0;
  for (int s = 0; s < HUFF_NSYM; s++) if (map[s] >= 0) {
    int d = 0; for (int i = map[s]; parent[i] >= 0; i = parent[i]) d++;
    len[s] = (uint8_t)d; if (d > maxlen) maxlen = d;
  }
  return maxlen;
}
static void huff_build_lengths(uint32_t *freq, uint8_t *len) {
  while (huff_lengths(freq, len) > HUFF_MAXLEN)
    for (int s = 0; s < HUFF_NSYM; s++) if (freq[s]) freq[s] = (freq[s] >> 1) | 1;
}
static void huff_codes(const uint8_t *len, uint16_t *code) {
  uint32_t count[HUFF_MAXLEN + 2] = {0}, next[HUFF_MAXLEN + 2]; uint32_t c = 0;
  for (int s = 0; s < HUFF_NSYM; s++) count[len[s]]++;
  count[0] = 0;
  for (int l = 1; l <= HUFF_MAXLEN; l++) { c = (c + count[l - 1]) << 1; next[l] = c; }
  for (int s = 0; s < HUFF_NSYM; s++) if (len[s]) code[s] = (uint16_t)next[len[s]]++;
}

typedef struct {
  uint16_t tsym[1 << HUFF_TBITS]; uint8_t tlen[1 << HUFF_TBITS];     /* fast table */
  uint32_t count[HUFF_MAXLEN + 1], first[HUFF_MAXLEN + 1], index[HUFF_MAXLEN + 1];
  uint16_t sym[HUFF_NSYM];
} huff_dec;

/* returns 0 if the length table is not a valid (Kraft <= 1) prefix code */
static int huff_dec_init(huff_dec *d, const uint8_t *len) {
  memset(d->count, 0, sizeof d->count); memset(d->tlen, 0, sizeof d->tlen);
  uint64_t kraft = 0;
  for (int s = 0; s < HUFF_NSYM; s++) if (len[s]) { d->count[len[s]]++; kraft += 1ull << (HUFF_MAXLEN - len[s]); }
  if (kraft > (1ull << HUFF_MAXLEN)) return 0;
  uint32_t code = 0, idx = 0;
  for (int l = 1; l <= HUFF_MAXLEN; l++) {
    d->first[l] = code; d->index[l] = idx; code = (code + d->count[l]) << 1; idx += d->count[l];
  }
  uint32_t pos[HUFF_MAXLEN + 1]; memcpy(pos, d->index, sizeof pos);
  uint16_t codes[HUFF_NSYM]; huff_codes(len, codes);
  for (int s = 0; s < HUFF_NSYM; s++) if (len[s]) {
    d->sym[pos[len[s]]++] = (uint16_t)s;
    if (len[s] <= HUFF_TBITS) {
      uint32_t start = (uint32_t)codes[s] << (HUFF_TBITS - len[s]), cnt = 1u << (HUFF_TBITS - len[s]);
      for (uint32_t k = 0; k < cnt; k++) { d->tsym[start + k] = (uint16_t)s; d->tlen[start + k] = len[s]; }
    }
  }
  return 1;
}
static inline int huff_decode_sym(const huff_dec *d, bitr *b) {   /* caller refilled */
  uint32_t bits = br_peek(b, HUFF_MAXLEN);
  uint32_t t = bits >> (HUFF_MAXLEN - HUFF_TBITS);
  int l = d->tlen[t];
  if (l) { br_skip(b, l); return d->tsym[t]; }
  for (l = HUFF_TBITS + 1; l <= HUFF_MAXLEN; l++) {
    uint32_t code = bits >> (HUFF_MAXLEN - l);
    if (code - d->first[l] < d->count[l]) { br_skip(b, l); return d->sym[d->index[l] + (code - d->first[l])]; }
  }
  return -1;
}

/* ------------------------------------------------------------------------ */
/* Symbolisation: int16 residuals -> uint16 symbol stream (+ raw16 escapes) */
/* ------------------------------------------------------------------------ */

static inline uint32_t zigzag16(int16_t v) { int32_t x = v; return (uint32_t)(uint16_t)(((uint32_t)x << 1) ^ (uint32_t)(x >> 15)); }
static inline int16_t unzigzag16(uint32_t u) { return (int16_t)(uint16_t)((u >> 1) ^ (0u - (u & 1))); }

/* Emits symbol stream into sym[]; escapes are followed by their 16-bit payload
   as an extra uint16 entry (not counted as a symbol). Fills total freq and
   per-stream (symbol index mod 4) bit totals. Returns symbol count. */
static uint32_t symbolise(const int16_t *r, uint32_t n, uint16_t *sym, uint32_t *nsym_entries,
                          uint32_t *freq, uint32_t esc_per_stream[NSTREAM]) {
  uint32_t nsym = 0, k = 0, i = 0;
  memset(freq, 0, HUFF_NSYM * sizeof *freq);
  memset(esc_per_stream, 0, NSTREAM * sizeof *esc_per_stream);
  while (i < n) {
    if (r[i] == 0) {
      uint32_t run = 1, lim = (n - i < MAX_RUN) ? n - i : MAX_RUN;
      while (run < lim && r[i + run] == 0) run++;
      i += run;
      uint32_t s = SYM_RUN0 + run - 1; sym[k++] = (uint16_t)s; freq[s]++; nsym++;
    } else {
      uint32_t u = zigzag16(r[i++]);
      if (u < SYM_ESC) { sym[k++] = (uint16_t)u; freq[u]++; }
      else { esc_per_stream[nsym & (NSTREAM - 1)]++; freq[SYM_ESC]++; sym[k++] = SYM_ESC; sym[k++] = (uint16_t)u; }
      nsym++;
    }
  }
  *nsym_entries = k;
  return nsym;
}

/* ------------------------------------------------------------------------ */
/* Adaptive Golomb-Rice (groups of RICE_G, 4-bit k per group)               */
/* ------------------------------------------------------------------------ */

static inline uint32_t rice_val_bits(uint32_t u, int k) {
  uint32_t q = u >> k;
  return (q >= RICE_UMAX) ? (RICE_UMAX + 16) : (q + 1 + (uint32_t)k);
}

/* Elias-gamma cost of u >= 1: 2*floor(log2 u) + 1 bits */
static inline uint32_t gamma_bits(uint32_t u) { return 2u * (31u - (uint32_t)b4d_clz32(u)) + 1u; }

/* Choose the coding for a group and return its exact bit cost (incl. k field).
   k_out: 0..13 Rice parameter, RICE_SPARSE (positions + gamma values), 15 all-zero. */
static uint32_t rice_group_plan(const int16_t *r, uint32_t len, uint8_t *k_out, uint32_t *u /* [RICE_G] zigzagged out */) {
  /* Bit-identical to the straightforward version (same k choice, same cost), restructured for speed:
     the three candidate costs are branch-free sums when no value can escape, and the sparse-coding cost is
     only evaluated when its lower bound could beat the best Rice cost. This planner is ~2/3 of encode time.
     AVX2 path vectorises both the zigzag+stats first pass and the three-sum fast path. */
  uint32_t sum = 0, nnz = 0, umax = 0;
#ifdef B4D_HAVE_AVX2
  {
    __m256i vsum = _mm256_setzero_si256();
    __m256i vnnz = _mm256_setzero_si256();
    __m256i vumax = _mm256_setzero_si256();
    __m256i vzero = _mm256_setzero_si256();
    uint32_t i = 0, n16 = len & ~15u;
    for (; i < n16; i += 16) {
      /* load 16 int16_t, apply zigzag (z = (x<<1)^(x>>15)), widen to 2x 8 uint32_t */
      __m256i raw  = _mm256_loadu_si256((const __m256i *)(r + i));
      __m256i z16  = _mm256_xor_si256(_mm256_slli_epi16(raw, 1), _mm256_srai_epi16(raw, 15));
      __m256i zlo  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(z16));
      __m256i zhi  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z16, 1));
      _mm256_storeu_si256((__m256i *)(u + i),     zlo);
      _mm256_storeu_si256((__m256i *)(u + i + 8), zhi);
      vsum  = _mm256_add_epi32(vsum,  _mm256_add_epi32(zlo, zhi));
      /* count nonzero: cmpgt(z,0) gives -1 for nonzero; sub(-1) = add 1 */
      vnnz  = _mm256_sub_epi32(vnnz, _mm256_cmpgt_epi32(zlo, vzero));
      vnnz  = _mm256_sub_epi32(vnnz, _mm256_cmpgt_epi32(zhi, vzero));
      vumax = _mm256_max_epu32(vumax, _mm256_max_epu32(zlo, zhi));
    }
    for (; i < len; i++) {
      uint32_t z = zigzag16(r[i]); u[i] = z; sum += z; nnz += (z != 0); umax = z > umax ? z : umax;
    }
    sum  += b4d_hsum_epi32(vsum);
    nnz  += b4d_hsum_epi32(vnnz);
    { uint32_t vm = b4d_hmax_epu32(vumax); if (vm > umax) umax = vm; }
  }
#else
  for (uint32_t i = 0; i < len; i++) {
    uint32_t z = zigzag16(r[i]); u[i] = z; sum += z; nnz += (z != 0); umax = z > umax ? z : umax;
  }
#endif
  if (sum == 0) { *k_out = 15; return RICE_KBITS; }
  uint32_t mean = sum / len; int k = 0;
  while (k < RICE_KMAX && (mean >> (k + 1))) k++;
  int k0 = k > 0 ? k - 1 : 0, k2 = k < RICE_KMAX ? k + 1 : RICE_KMAX;
  uint32_t best = 0xFFFFFFFFu; int bestk = k;
  if ((umax >> k0) < RICE_UMAX) {                       /* no escape possible for any candidate: vectorisable pass */
    const int s1 = k0 + 1 <= k2 ? k0 + 1 : k2, s2 = k2;
#ifdef B4D_HAVE_AVX2
    {
      __m256i vk0v = _mm256_set1_epi32(k0);
      __m256i vs1v = _mm256_set1_epi32(s1);
      __m256i vs2v = _mm256_set1_epi32(s2);
      __m256i va0  = _mm256_setzero_si256();
      __m256i va1  = _mm256_setzero_si256();
      __m256i va2  = _mm256_setzero_si256();
      uint32_t n8 = len & ~7u;
      for (uint32_t i = 0; i < n8; i += 8) {
        __m256i z = _mm256_loadu_si256((const __m256i *)(u + i));
        va0 = _mm256_add_epi32(va0, _mm256_srlv_epi32(z, vk0v));
        va1 = _mm256_add_epi32(va1, _mm256_srlv_epi32(z, vs1v));
        va2 = _mm256_add_epi32(va2, _mm256_srlv_epi32(z, vs2v));
      }
      uint32_t a0 = b4d_hsum_epi32(va0);
      uint32_t a1 = b4d_hsum_epi32(va1);
      uint32_t a2 = b4d_hsum_epi32(va2);
      for (uint32_t i = n8; i < len; i++) {
        uint32_t z = u[i]; a0 += z >> k0; a1 += z >> s1; a2 += z >> s2;
      }
      uint32_t cand[3] = { a0, a1, a2 }; int ks[3] = { k0, s1, s2 };
      for (int c = 0; c < 3; c++) {
        if (c > 0 && ks[c] == ks[c - 1]) continue;
        uint32_t bits = RICE_KBITS + cand[c] + len * (uint32_t)(ks[c] + 1);
        if (bits < best) { best = bits; bestk = ks[c]; }
      }
    }
#else
    uint32_t a0 = 0, a1 = 0, a2 = 0;
    for (uint32_t i = 0; i < len; i++) { uint32_t z = u[i]; a0 += z >> k0; a1 += z >> s1; a2 += z >> s2; }
    uint32_t cand[3] = { a0, a1, a2 }; int ks[3] = { k0, s1, s2 };
    for (int c = 0; c < 3; c++) {
      if (c > 0 && ks[c] == ks[c - 1]) continue;        /* k0 == k2 at the ends of the range */
      uint32_t bits = RICE_KBITS + cand[c] + len * (uint32_t)(ks[c] + 1);
      if (bits < best) { best = bits; bestk = ks[c]; }
    }
#endif
  } else {
    for (int kk = k0; kk <= k2; kk++) {
      uint32_t bits = RICE_KBITS;
      for (uint32_t i = 0; i < len; i++) bits += rice_val_bits(u[i], kk);
      if (bits < best) { best = bits; bestk = kk; }
    }
  }
  /* sparse group: 4 + 7 + nnz x (7 + gamma_bits >= 1) bits; skip the exact count when it cannot win */
  if (nnz < RICE_G && RICE_KBITS + RICE_POSBITS + nnz * (RICE_POSBITS + 1) < best) {
    uint32_t sparse_bits = RICE_KBITS + RICE_POSBITS;
    for (uint32_t i = 0; i < len; i++) if (u[i]) sparse_bits += RICE_POSBITS + gamma_bits(u[i]);
    if (sparse_bits < best) { *k_out = RICE_SPARSE; return sparse_bits; }
  }
  *k_out = (uint8_t)bestk;
  return best;
}

/* branchless writer for the Rice emit: keeps up to 63 bits in acc (MSB-first); bwx_flush stores 8 bytes every
   element and advances by the whole bytes, so there is no per-put flush branch. It overruns its own end by up
   to 7 bytes, so it writes into a per-stream scratch area that is copied into place afterwards. */
typedef struct { uint8_t *p; uint64_t acc; unsigned n; } bitwx;
static inline void bwx_init(bitwx *b, uint8_t *p) { b->p = p; b->acc = 0; b->n = 0; }
static inline void bwx_put(bitwx *b, uint32_t v, unsigned len) {   /* n + len <= 64 must hold */
  b->acc |= (uint64_t)v << (64 - b->n - len); b->n += len;
}
static inline void bwx_flush(bitwx *b) {
#if defined(_MSC_VER) && !defined(__clang__)
  uint64_t be = _byteswap_uint64(b->acc);
#else
  uint64_t be = __builtin_bswap64(b->acc);
#endif
  memcpy(b->p, &be, 8); unsigned nb = b->n >> 3; b->p += nb; b->acc <<= nb * 8; b->n &= 7;
}
static inline size_t bwx_finish(bitwx *b, const uint8_t *start) {  /* flush the tail; returns the byte length */
  bwx_flush(b);
  if (b->n) { b->p[0] = (uint8_t)(b->acc >> 56); b->p++; b->n = 0; }
  return (size_t)(b->p - start);
}
static inline void rice_putx(bitwx *b, uint32_t u, int k) {        /* one element: unary + remainder, one flush */
  uint32_t q = u >> k;
  if (q >= RICE_UMAX) { bwx_put(b, 0x7FFFFFFFu, RICE_UMAX); bwx_flush(b); bwx_put(b, u, 16); bwx_flush(b); return; }
  bwx_put(b, ((1u << q) - 1) << 1, q + 1);                          /* <= 31 bits */
  if (k) bwx_put(b, u & ((1u << k) - 1), (unsigned)k);              /* <= 13 bits: n <= 7 + 31 + 13 = 51 */
  bwx_flush(b);
}

/* one element without the flush (q < RICE_UMAX: no escape), for runs of elements whose codes fit the writer together */
static inline void rice_put_nf(bitwx *b, uint32_t u, int k) {
  uint32_t q = u >> k;
  bwx_put(b, ((1u << q) - 1) << 1, q + 1);
  if (k) bwx_put(b, u & ((1u << k) - 1), (unsigned)k);
}
/* a Rice group's elements: when the longest code (from the group's largest value) lets 4 or 2 codes share a flush
   (n <= 7 after one, so 7 + 4 len <= 64), they do; the bits are the same as one flush per element */
static inline void rice_put_group(bitwx *b, const uint32_t *u, uint32_t len, int k) {
  uint32_t umax = 0, i = 0;
  for (uint32_t j = 0; j < len; j++) umax = u[j] > umax ? u[j] : umax;
  const uint32_t qmax = umax >> k, lmax = qmax + 1 + (uint32_t)k;
#ifdef B4D_HAVE_AVX2
  if (qmax < RICE_UMAX && 7 + 4 * lmax <= 64 && len == RICE_G) {   /* codes and lengths 8 at a time, then a short serial pack */
    uint32_t cd[RICE_G], ln[RICE_G];
    const __m128i vk = _mm_cvtsi32_si128(k);
    const __m256i one = _mm256_set1_epi32(1), two = _mm256_set1_epi32(2), msk = _mm256_set1_epi32((int)((1u << k) - 1)),
                  k1 = _mm256_set1_epi32(k + 1);
    for (uint32_t j = 0; j < RICE_G; j += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i *)(u + j)), q = _mm256_srl_epi32(v, vk);
      __m256i c = _mm256_or_si256(_mm256_sll_epi32(_mm256_sub_epi32(_mm256_sllv_epi32(one, _mm256_add_epi32(q, one)), two), vk),
                                  _mm256_and_si256(v, msk));
      _mm256_storeu_si256((__m256i *)(cd + j), c); _mm256_storeu_si256((__m256i *)(ln + j), _mm256_add_epi32(q, k1));
    }
    uint64_t acc = b->acc; unsigned n = b->n; uint8_t *p = b->p;
    for (uint32_t j = 0; j < RICE_G; j += 4) {        /* 4 codes of <= 14 bits per flush, as the scalar path below */
      n += ln[j];     acc |= (uint64_t)cd[j] << (64 - n);
      n += ln[j + 1]; acc |= (uint64_t)cd[j + 1] << (64 - n);
      n += ln[j + 2]; acc |= (uint64_t)cd[j + 2] << (64 - n);
      n += ln[j + 3]; acc |= (uint64_t)cd[j + 3] << (64 - n);
      uint64_t be = b4d_bswap64(acc); memcpy(p, &be, 8); unsigned nb = n >> 3; p += nb; acc <<= nb * 8; n &= 7;
    }
    b->acc = acc; b->n = n; b->p = p;
    return;
  }
#endif
  if (qmax < RICE_UMAX && 7 + 4 * lmax <= 64)
    for (; i + 4 <= len; i += 4) { rice_put_nf(b, u[i], k); rice_put_nf(b, u[i + 1], k); rice_put_nf(b, u[i + 2], k); rice_put_nf(b, u[i + 3], k); bwx_flush(b); }
  else if (qmax < RICE_UMAX && 7 + 2 * lmax <= 64)
    for (; i + 2 <= len; i += 2) { rice_put_nf(b, u[i], k); rice_put_nf(b, u[i + 1], k); bwx_flush(b); }
  for (; i < len; i++) rice_putx(b, u[i], k);
}

static inline void rice_put(bitw *b, uint32_t u, int k) {
  uint32_t q = u >> k;
  if (q >= RICE_UMAX) { bw_put(b, 0x7FFFFFFFu, RICE_UMAX); bw_put(b, u, 16); }
  else { bw_put(b, ((1u << q) - 1) << 1, (int)q + 1); if (k) bw_put(b, u & ((1u << k) - 1), k); }
}

static inline void gamma_put(bitw *b, uint32_t u) {            /* u >= 1, u < 2^16 */
  int nb = 32 - b4d_clz32(u);
  if (nb > 1) bw_put(b, 0, nb - 1);
  bw_put(b, u, nb);
}

static inline void rice_zero_run_emit(bitw *b, uint32_t ngroups) {   /* 1..RICE_ZRUNMAX */
  bw_put(b, 15, RICE_KBITS); bw_put(b, ngroups - 1, RICE_ZRUNBITS);
}

static inline void rice_group_emit(bitw *b, const int16_t *r, uint32_t len, int k) {   /* k != 15 */
  bw_put(b, (uint32_t)k, RICE_KBITS);
  if (k == RICE_SPARSE) {
    uint32_t nnz = 0;
    for (uint32_t i = 0; i < len; i++) nnz += (r[i] != 0);
    bw_put(b, nnz, RICE_POSBITS);
    for (uint32_t i = 0; i < len; i++) if (r[i]) { bw_put(b, i, RICE_POSBITS); gamma_put(b, zigzag16(r[i])); }
    return;
  }
  for (uint32_t i = 0; i < len; i++) rice_put(b, zigzag16(r[i]), k);
}

/* version 1: k=14 is a Rice parameter and k=15 is one zero group;
   version 2: k=14 is a sparse group and k=15 starts a run of zero groups (count follows).
   *pending counts zero groups already announced for this stream. */
/* br_refill without a data-dependent branch, for at least 8 bytes before the end: the bit count after it (57..64) depends on
   how many bits were left, so the plain loop's exit is hard to predict once per element or two */
static inline void br_refill_fast(bitr *b) {
  uint64_t w; memcpy(&w, b->p, 8);
#if defined(_MSC_VER) && !defined(__clang__)
  w = _byteswap_uint64(w);
#else
  w = __builtin_bswap64(w);
#endif
  const int n = b->n, nb = (64 - n) >> 3;               /* n == 64: nothing to add (and w >> 64 is undefined) */
  b->acc |= (w >> (n & 63)) & (uint64_t)((int64_t)(n - 64) >> 63);
  b->p += nb; b->n = n + 8 * nb;
}
static inline uint32_t rice_elem(bitr *b, int k) {
  if (b->end - b->p >= 8) br_refill_fast(b); else br_refill(b);
  int q = br_unary(b); uint32_t u;
  if (q >= RICE_UMAX) { br_skip(b, RICE_UMAX); u = br_bits(b, 16); }
  else if (k) {                                         /* unary and remainder in one shift of the accumulator */
    u = ((uint32_t)q << k) | (uint32_t)((b->acc << (q + 1)) >> (64 - k)); br_skip(b, q + 1 + k);
  } else { br_skip(b, q + 1); u = (uint32_t)q; }
  return u;
}
/* a group after its k field: zero and sparse groups are read here (returns 1, or -1 if malformed); 0 = a Rice group */
static int rice_group_special(bitr *b, int16_t *r, uint32_t len, int version, uint32_t *pending, int k) {
  if (k == 15) {
    if (version >= 2) *pending = br_bits(b, RICE_ZRUNBITS);     /* remaining groups of this run */
    memset(r, 0, len * sizeof *r); return 1;
  }
  if (k == RICE_SPARSE && version >= 2) {
    memset(r, 0, len * sizeof *r);
    uint32_t nnz = br_bits(b, RICE_POSBITS);
    for (uint32_t j = 0; j < nnz; j++) {
      br_refill(b);
      uint32_t pos = br_bits(b, RICE_POSBITS);
      int z = br_zeros(b); br_skip(b, z);
      uint32_t u = br_bits(b, z + 1);
      if (pos >= len || u == 0) return -1;
      r[pos] = unzigzag16(u);
    }
    return 1;
  }
  return 0;
}
static int rice_group_read(bitr *b, int16_t *r, uint32_t len, int version, uint32_t *pending) {
  if (*pending) { (*pending)--; memset(r, 0, len * sizeof *r); return 0; }
  br_refill(b);
  int k = (int)br_bits(b, RICE_KBITS);
  int sp = rice_group_special(b, r, len, version, pending, k);
  if (sp) return sp < 0 ? -1 : 0;
  bitr l = *b;                                         /* a local copy stays in registers (r's stores could alias *b) */
  for (uint32_t i = 0; i < len; i++) r[i] = unzigzag16(rice_elem(&l, k));
  *b = l;
  return 0;
}
/* groups g .. g + NSTREAM - 1, full and g a multiple of NSTREAM (so one per stream): the values of NSTREAM rice_group_read
   calls, but the Rice groups are read element by element in step, so the streams' bit chains (each element's shifts depend
   on the one before) overlap instead of running one after another */
#ifdef B4D_HAVE_AVX2
/* table decode: for each k, the code that starts at the top RTB bits of the window: its value (unzigzagged, low 16 bits)
   and length (bits 16..23; 0 = longer than RTB bits: decoded with the arithmetic) */
#define RTB 10
static uint32_t g_rtab[RICE_KMAX + 1][1 << RTB];
static volatile int g_rtab_ok = 0;
static void rtab_build(void) {
  for (int k = 0; k <= RICE_KMAX; k++)
    for (uint32_t x = 0; x < (1u << RTB); x++) {
      int q = 0; while (q < RTB && ((x >> (RTB - 1 - q)) & 1)) q++;
      uint32_t e = 0;
      if (q + 1 + k <= RTB) {
        const uint32_t rem = k ? (x >> (RTB - 1 - q - k)) & ((1u << k) - 1) : 0, u = ((uint32_t)q << k) | rem;
        e = ((uint32_t)(q + 1 + k) << 16) | (uint16_t)unzigzag16(u);
      }
      g_rtab[k][x] = e;
    }
  g_rtab_ok = 1;                                       /* every thread that races here writes the same values */
}
#endif
/* NS = 4 (v3..v9 Rice) or RICE_NS (v10 / v11); the two wrappers below give the compiler a constant NS */
static B4D_INLINE int rice_groups_read_ns(bitr *br, int16_t *r, int version, uint32_t *pending, const int NS) {
  bitr *bs[RICE_NS]; int16_t *rs[RICE_NS]; int ks[RICE_NS], m = 0;
  for (int s = 0; s < NS; s++) {
    int16_t *rr = r + (size_t)s * RICE_G;
    if (pending[s]) { pending[s]--; memset(rr, 0, RICE_G * sizeof *rr); continue; }
    br_refill(&br[s]);
    int k = (int)br_bits(&br[s], RICE_KBITS);
    int sp = rice_group_special(&br[s], rr, RICE_G, version, &pending[s], k);
    if (sp < 0) return -1;
    if (!sp) { bs[m] = &br[s]; rs[m] = rr; ks[m] = k; m++; }
  }
#ifdef B4D_HAVE_AVX2
  /* position reader: each element loads the 8 bytes at its bit position (no accumulator to refill), when every stream has
     room for a whole group of the longest codes (47 bits) plus the load; short codes come from the table. The values are
     rice_elem's. Afterwards each stream's reader is rebuilt at its bit position. */
  if (m == NS) {
    const uint8_t *bp[RICE_NS]; uint64_t pos[RICE_NS]; int room = 1, kok = 1;
    for (int s = 0; s < NS; s++) {
      const int nb = (bs[s]->n + 7) >> 3; bp[s] = bs[s]->p - nb; pos[s] = (uint64_t)(8 * nb - bs[s]->n);
      room &= bs[s]->end - bs[s]->p >= (RICE_G * 47) / 8 + 16; kok &= ks[s] <= RICE_KMAX;
    }
    if (room && kok) {
      if (!g_rtab_ok) rtab_build();
      const uint32_t *tb[RICE_NS];
      for (int s = 0; s < NS; s++) tb[s] = g_rtab[ks[s]];
      #define RTAB(S) do { uint64_t w_; memcpy(&w_, bp[S] + (pos[S] >> 3), 8); w_ = b4d_bswap64(w_) << (pos[S] & 7);             \
        const uint32_t e_ = tb[S][w_ >> (64 - RTB)];                                                                          \
        if (e_ >> 16) { rs[S][i] = (int16_t)(uint16_t)e_; pos[S] += e_ >> 16; }                                               \
        else { const unsigned q_ = b4d_lzcnt64(~w_); uint32_t u_;                                                             \
          if (q_ >= RICE_UMAX) { u_ = (uint32_t)((w_ << RICE_UMAX) >> 48); pos[S] += RICE_UMAX + 16; }                       \
          else { u_ = (q_ << ks[S]) | (uint32_t)(((w_ << (q_ + 1)) >> 1) >> (63 - ks[S])); pos[S] += q_ + 1 + (unsigned)ks[S]; } \
          rs[S][i] = unzigzag16(u_); } } while (0)
      if (NS == 4) for (uint32_t i = 0; i < RICE_G; i++) { RTAB(0); RTAB(1); RTAB(2); RTAB(3); }
      else for (uint32_t i = 0; i < RICE_G; i++) { RTAB(0); RTAB(1); RTAB(2); RTAB(3); RTAB(4); RTAB(5); RTAB(6); RTAB(7); }
      #undef RTAB
      for (int s = 0; s < NS; s++) {
        bs[s]->p = bp[s] + (pos[s] >> 3); bs[s]->acc = 0; bs[s]->n = 0; br_refill(bs[s]); br_skip(bs[s], (int)(pos[s] & 7));
      }
      return 0;
    }
  }
#endif
  if (m == NS && NS == 4) {                            /* local copies stay in registers (r's stores could alias br) */
    bitr l0 = *bs[0], l1 = *bs[1], l2 = *bs[2], l3 = *bs[3];
    for (uint32_t i = 0; i < RICE_G; i++) {
      rs[0][i] = unzigzag16(rice_elem(&l0, ks[0])); rs[1][i] = unzigzag16(rice_elem(&l1, ks[1]));
      rs[2][i] = unzigzag16(rice_elem(&l2, ks[2])); rs[3][i] = unzigzag16(rice_elem(&l3, ks[3]));
    }
    *bs[0] = l0; *bs[1] = l1; *bs[2] = l2; *bs[3] = l3;
  } else if (m == NS) {
    bitr l[RICE_NS];
    for (int s = 0; s < NS; s++) l[s] = *bs[s];
    for (uint32_t i = 0; i < RICE_G; i++)
      for (int s = 0; s < NS; s++) rs[s][i] = unzigzag16(rice_elem(&l[s], ks[s]));
    for (int s = 0; s < NS; s++) *bs[s] = l[s];
  } else
    for (uint32_t i = 0; i < RICE_G; i++)
      for (int j = 0; j < m; j++) rs[j][i] = unzigzag16(rice_elem(bs[j], ks[j]));
  return 0;
}
static int rice_groups_read_all4(bitr *br, int16_t *r, int version, uint32_t *pending) { return rice_groups_read_ns(br, r, version, pending, 4); }
static int rice_groups_read_all8(bitr *br, int16_t *r, int version, uint32_t *pending) { return rice_groups_read_ns(br, r, version, pending, RICE_NS); }

/* ------------------------------------------------------------------------ */
/* rANS coder with local-activity contexts (format v3, FLAG_ANS)            */
/* ------------------------------------------------------------------------ */

#define ANS_SCALE_BITS 12
#define ANS_M          (1u << ANS_SCALE_BITS)
#define ANS_L          (1ull << 31)             /* 64-bit state in [L, 2^32 L), 32-bit renormalisation words */
#define ANS_NCTX       3
#define ANS_TABLE_MAX  (ANS_NCTX * (HUFF_NSYM / 8 + (HUFF_NSYM * 12 + 7) / 8))   /* worst-case serialized tables */

typedef struct {
  uint16_t start[HUFF_NSYM], freq[HUFF_NSYM]; uint8_t present;
  uint64_t x_max[HUFF_NSYM], rcp_freq[HUFF_NSYM]; uint32_t bias[HUFF_NSYM]; uint16_t cmpl_freq[HUFF_NSYM], rcp_shift[HUFF_NSYM];
} ans_ctx_enc;

/* precompute division-free encode descriptors for every symbol of a context (64-bit reciprocals) */
static void ans_ctx_enc_prepare(ans_ctx_enc *c) {
  for (int s = 0; s < HUFF_NSYM; s++) {
    uint32_t freq = c->freq[s], start = c->start[s];
    if (!freq) continue;
    c->x_max[s] = ((ANS_L >> ANS_SCALE_BITS) << 32) * freq;
    c->cmpl_freq[s] = (uint16_t)((1u << ANS_SCALE_BITS) - freq);
    if (freq < 2) { c->rcp_freq[s] = ~0ull; c->rcp_shift[s] = 0; c->bias[s] = start + (1u << ANS_SCALE_BITS) - 1; }
    else {
      uint32_t shift = 0; while (freq > (1u << shift)) shift++;
      c->rcp_freq[s] = b4d_rcp64(freq, shift);
      c->rcp_shift[s] = (uint16_t)(shift - 1); c->bias[s] = start;
    }
  }
}
typedef struct { uint16_t start[HUFF_NSYM], freq[HUFF_NSYM]; uint16_t *sym; uint8_t present; } ans_ctx_dec;

/* context of the residual at position i from already-known residuals at i-1 (same row) and i-w */
static inline int ans_context(const int16_t *r, uint32_t i, uint32_t w) {
  /* previous residual in memory order (causal; at a row start it is the previous row's last pixel) + the one above */
  uint32_t a = i > 0 ? (uint32_t)abs(r[i - 1]) : 0;
  if (w && i >= w) a += (uint32_t)abs(r[i - w]);
  return a == 0 ? 0 : (a <= 4 ? 1 : 2);
}

#define ANS_CTX_MIN_GAIN 0.02   /* use per-context tables only if they save at least 2% (estimated) */

/* estimated fractional saving of 3 context-conditioned tables over one merged table (Shannon entropy) */
static double ans_context_gain(const uint32_t *count) {
  double bits_ctx = 0, bits_one = 0, tot_c[ANS_NCTX] = {0, 0, 0}, tot = 0; uint32_t merged[HUFF_NSYM] = {0};
  for (int c = 0; c < ANS_NCTX; c++) for (int sy = 0; sy < HUFF_NSYM; sy++) { tot_c[c] += count[c * HUFF_NSYM + sy]; merged[sy] += count[c * HUFF_NSYM + sy]; }
  for (int c = 0; c < ANS_NCTX; c++) { tot += tot_c[c]; for (int sy = 0; sy < HUFF_NSYM; sy++) { uint32_t k = count[c * HUFF_NSYM + sy]; if (k) bits_ctx -= k * log2((double)k / tot_c[c]); } }
  for (int sy = 0; sy < HUFF_NSYM; sy++) if (merged[sy]) bits_one -= merged[sy] * log2((double)merged[sy] / tot);
  return bits_one > 0 ? (bits_one - bits_ctx) / bits_one : 0.0;
}

/* normalise counts to sum exactly ANS_M, every present symbol >= 1 (deterministic) */
static void ans_normalise(const uint32_t *count, uint16_t *freq) {
  uint64_t total = 0; int npresent = 0;
  for (int s = 0; s < HUFF_NSYM; s++) { total += count[s]; npresent += count[s] != 0; }
  memset(freq, 0, HUFF_NSYM * sizeof *freq);
  if (total == 0) return;
  uint32_t sum = 0; int maxs = 0;
  for (int s = 0; s < HUFF_NSYM; s++) if (count[s]) {
    uint32_t f = (uint32_t)(((uint64_t)count[s] * ANS_M) / total);
    if (f == 0) f = 1;
    freq[s] = (uint16_t)f; sum += f;
    if (count[s] > count[maxs] || !count[maxs]) maxs = s;
  }
  /* fix the rounding drift on the most frequent symbol (it always has slack when sum > M) */
  while (sum > ANS_M) {
    int s = maxs;
    if (freq[s] > 1) { freq[s]--; sum--; }
    else { for (s = 0; s < HUFF_NSYM && sum > ANS_M; s++) if (freq[s] > 1) { freq[s]--; sum--; } }
  }
  if (sum < ANS_M) freq[maxs] += (uint16_t)(ANS_M - sum);
}

static void ans_starts(uint16_t *start, const uint16_t *freq) {
  uint32_t c = 0;
  for (int s = 0; s < HUFF_NSYM; s++) { start[s] = (uint16_t)c; c += freq[s]; }
}

/* serialize: 64-byte presence bitmap + 12 bits (freq-1) per present symbol, MSB-first */
static uint32_t ans_tables_write(uint8_t *out, const ans_ctx_enc *ctx) {
  uint8_t *p = out;
  for (int c = 0; c < ANS_NCTX; c++) {
    if (!ctx[c].present) { *p++ = 0; continue; }          /* 0 = context unused */
    *p++ = 1;
    uint8_t *bm = p; memset(bm, 0, HUFF_NSYM / 8); p += HUFF_NSYM / 8;
    bitw bw; bw_init(&bw, p);
    for (int s = 0; s < HUFF_NSYM; s++) if (ctx[c].freq[s]) { bm[s >> 3] |= (uint8_t)(1u << (s & 7)); bw_put(&bw, ctx[c].freq[s] - 1u, 12); }
    bw_flush(&bw); p = bw.p;
  }
  return (uint32_t)(p - out);
}

static int ans_tables_read(const uint8_t *in, uint32_t avail, ans_ctx_dec *ctx, uint16_t *symbuf) {
  const uint8_t *p = in, *end = in + avail;
  for (int c = 0; c < ANS_NCTX; c++) {
    ctx[c].present = 0; ctx[c].sym = symbuf + c * ANS_M;
    memset(ctx[c].freq, 0, sizeof ctx[c].freq);
    if (p >= end) return -1;
    uint8_t flag = *p++;
    if (!flag) continue;
    if (p + HUFF_NSYM / 8 > end) return -1;
    const uint8_t *bm = p; p += HUFF_NSYM / 8;
    int npresent = 0; for (int s = 0; s < HUFF_NSYM; s++) npresent += (bm[s >> 3] >> (s & 7)) & 1;
    size_t nbytes = ((size_t)npresent * 12 + 7) / 8;
    if (p + nbytes > end) return -1;
    bitr br; br_init(&br, p, nbytes); uint32_t sum = 0;
    for (int s = 0; s < HUFF_NSYM; s++) if ((bm[s >> 3] >> (s & 7)) & 1) { br_refill(&br); ctx[c].freq[s] = (uint16_t)(br_bits(&br, 12) + 1); sum += ctx[c].freq[s]; }
    if (sum != ANS_M) return -1;
    ans_starts(ctx[c].start, ctx[c].freq);
    for (int s = 0; s < HUFF_NSYM; s++) for (uint32_t k = 0; k < ctx[c].freq[s]; k++) ctx[c].sym[ctx[c].start[s] + k] = (uint16_t)s;
    ctx[c].present = 1; p += nbytes;
  }
  return (int)(p - in);
}

/* ---- rANS primitives, 64-bit state / 32-bit words (Giesen's rans64). Encoding runs backwards. */
typedef struct { uint64_t x; uint8_t *p; } rans_enc;     /* p: current write pointer (decrementing, 4-byte units) */
typedef struct { uint64_t x; const uint8_t *p, *end; } rans_dec;

static inline void rans_put_u32(rans_enc *e, uint32_t v) { e->p -= 4; memcpy(e->p, &v, 4); }
static inline uint32_t rans_get_u32(rans_dec *d) { uint32_t v = 0; if (d->p + 4 <= d->end) { memcpy(&v, d->p, 4); d->p += 4; } else d->p = d->end; return v; }

static inline void rans_enc_put(rans_enc *e, uint32_t start, uint32_t freq, int scale_bits) {   /* generic (raw bits) */
  uint64_t x = e->x, x_max = ((ANS_L >> scale_bits) << 32) * freq;
  if (x >= x_max) { rans_put_u32(e, (uint32_t)x); x >>= 32; }
  e->x = ((x / freq) << scale_bits) + (x % freq) + start;
}
static inline void rans_enc_put_sym(rans_enc *e, const ans_ctx_enc *c, uint32_t s) {                 /* division-free */
  uint64_t x = e->x;
  if (x >= c->x_max[s]) { rans_put_u32(e, (uint32_t)x); x >>= 32; }
  uint64_t q = b4d_mulhi64(x, c->rcp_freq[s]) >> c->rcp_shift[s];
  e->x = x + c->bias[s] + q * c->cmpl_freq[s];
}
static inline void rans_enc_flush(rans_enc *e) { rans_put_u32(e, (uint32_t)(e->x >> 32)); rans_put_u32(e, (uint32_t)e->x); }
static inline int rans_dec_init(rans_dec *d, const uint8_t *p, size_t len) {
  if (len < 8) return -1;
  uint32_t lo, hi; memcpy(&lo, p, 4); memcpy(&hi, p + 4, 4);    /* flush wrote hi then lo going backwards */
  d->x = ((uint64_t)hi << 32) | lo; d->p = p + 8; d->end = p + len; return 0;
}
static inline uint32_t rans_dec_slot(const rans_dec *d, int scale_bits) { return (uint32_t)(d->x & ((1u << scale_bits) - 1)); }
static inline void rans_dec_advance(rans_dec *d, uint32_t start, uint32_t freq, int scale_bits) {
  uint64_t x = freq * (d->x >> scale_bits) + rans_dec_slot(d, scale_bits) - start;
  if (x < ANS_L) x = (x << 32) | rans_get_u32(d);
  d->x = x;
}

/* Symbols of the block in stream order with their contexts, as (sym, ctx, esc payload) records. */
typedef struct { uint16_t sym, ctx, esc; } ans_rec;

/* Build per-stream record lists from residuals. Returns record count (== number of symbols). */
static uint32_t ans_symbolise(const int16_t *r, uint32_t n, uint32_t w, ans_rec *rec, uint32_t *count /* [NCTX][NSYM] */) {
  uint32_t k = 0, i = 0;
  memset(count, 0, ANS_NCTX * HUFF_NSYM * sizeof *count);
  while (i < n) {
    int c = ans_context(r, i, w);
    if (r[i] == 0) {
      uint32_t run = 1, lim = (n - i < MAX_RUN) ? n - i : MAX_RUN;
      while (run < lim && r[i + run] == 0) run++;
      i += run; uint32_t sy = SYM_RUN0 + run - 1;
      rec[k].sym = (uint16_t)sy; rec[k].ctx = (uint16_t)c; rec[k].esc = 0; count[c * HUFF_NSYM + sy]++; k++;
    } else {
      uint32_t u = zigzag16(r[i++]);
      if (u < SYM_ESC) { rec[k].sym = (uint16_t)u; rec[k].ctx = (uint16_t)c; rec[k].esc = 0; count[c * HUFF_NSYM + u]++; }
      else { rec[k].sym = SYM_ESC; rec[k].ctx = (uint16_t)c; rec[k].esc = (uint16_t)u; count[c * HUFF_NSYM + SYM_ESC]++; }
      k++;
    }
  }
  return k;
}

/* Encode all streams. rec: nsym records in order; symbol k goes to stream k & 3, encoded in reverse.
   Each stream is written into its own scratch buffer (rans encodes backwards) and copied to out. */
static int ans_encode_streams(const ans_rec *rec, uint32_t nsym, const ans_ctx_enc *ctx,
                              uint8_t *scratch_buf, size_t scratch_cap, uint8_t *out, size_t out_cap, uint32_t sbytes[NSTREAM]) {
  size_t per = scratch_cap / NSTREAM;
  size_t total = 0;
  for (int st = 0; st < NSTREAM; st++) {
    uint8_t *buf = scratch_buf + (size_t)st * per, *endp = buf + per;
    rans_enc e; e.x = ANS_L; e.p = endp;
    /* symbols of this stream: indices st, st+4, ... ; encode from the last to the first */
    if (nsym > (uint32_t)st) {
      uint32_t last = ((nsym - 1 - st) / NSTREAM) * NSTREAM + st;
      const ans_rec *rk = &rec[last], *rend = &rec[st];
      for (;; rk -= NSTREAM) {
        const ans_ctx_enc *cx = &ctx[rk->ctx];
        if ((size_t)(e.p - buf) < 64) return -1;                            /* scratch exhausted */
        if (rk->sym == SYM_ESC) {                                             /* raw bits first (decoder reads them after the symbol) */
          rans_enc_put(&e, rk->esc & 0xFF, 1, 8);
          rans_enc_put(&e, rk->esc >> 8, 1, 8);
        }
        rans_enc_put_sym(&e, cx, rk->sym);
        if (rk == rend) break;
      }
    }
    rans_enc_flush(&e);
    sbytes[st] = (uint32_t)(endp - e.p);
    if (total + sbytes[st] > out_cap) return -2;                             /* incompressible */
    memcpy(out + total, e.p, sbytes[st]); total += sbytes[st];
  }
  return (int)total;
}

static int ans_decode_streams(bitr br_unused[NSTREAM], const uint8_t *streams[NSTREAM], const uint32_t sbytes[NSTREAM],
                              uint32_t nsym, const ans_ctx_dec *ctx, int16_t *r, uint32_t n, uint32_t w) {
  (void)br_unused;
  rans_dec d[NSTREAM];
  for (int st = 0; st < NSTREAM; st++) if (rans_dec_init(&d[st], streams[st], sbytes[st]) != 0) return -1;
  const int single = !ctx[1].present && !ctx[2].present;     /* one table: no per-symbol context work */
  if (!ctx[0].present) return -1;
  uint32_t i = 0;
  for (uint32_t k = 0; k < nsym; k++) {
    rans_dec *ds = &d[k & (NSTREAM - 1)];
    if (i >= n) return -1;
    const ans_ctx_dec *cx = single ? &ctx[0] : &ctx[ans_context(r, i, w)];
    if (!cx->present) return -1;
    uint32_t slot = rans_dec_slot(ds, ANS_SCALE_BITS); uint32_t sy = cx->sym[slot];
    rans_dec_advance(ds, cx->start[sy], cx->freq[sy], ANS_SCALE_BITS);
    if (sy >= SYM_RUN0) {
      uint32_t run = sy - SYM_RUN0 + 1;
      if (i + run > n) return -1;
      if (run < 8) { for (uint32_t k2 = 0; k2 < run; k2++) r[i + k2] = 0; } else memset(r + i, 0, run * sizeof *r);
      i += run;
    } else {
      uint32_t u = sy;
      if (sy == SYM_ESC) {
        uint32_t hi = rans_dec_slot(ds, 8); rans_dec_advance(ds, hi, 1, 8);
        uint32_t lo = rans_dec_slot(ds, 8); rans_dec_advance(ds, lo, 1, 8);
        u = (hi << 8) | lo;
      }
      r[i++] = unzigzag16(u);
    }
  }
  return (i == n) ? 0 : -1;
}

/* ------------------------------------------------------------------------ */
/* Encoder                                                                  */
/* ------------------------------------------------------------------------ */

/* Entropy-code int16 residuals into a block payload (header + streams) at out. Shared by b4d_encoder
   and b4d_encode_residual_block. Returns payload size, 0 if it does not fit (incompressible), <0 error. */
/* work-area size the entropy stage needs for n residuals with a given coder */
static size_t entropy_work_bytes(uint32_t n, int coder) {
  if (coder == B4D_ENTROPY_ANS) return ALIGN64((size_t)n * sizeof(ans_rec)) + 4 * ((size_t)n * 3 + 4096) + 64;
  if (coder == B4D_ENTROPY_RICE) return RICE_NS * (6 * ((size_t)n / RICE_NS + RICE_G) + 64) + 128;   /* per-stream worst case (escapes) */
  return 4 * (size_t)n + 64;
}

static int entropy_encode_block_src(b4d_hdr *hp, resid_src *rs, uint8_t *output, size_t output_len, int coder,
                                    uint8_t *work, size_t work_cap) {
  b4d_hdr h = *hp;
  if (work_cap < entropy_work_bytes(h.nelem, coder)) return -1;
  const uint32_t n = h.nelem, w = h.roww;
  const int rice = coder == B4D_ENTROPY_RICE, ans = coder == B4D_ENTROPY_ANS;
  const int16_t *r = rs->r;                                              /* Huffman/ANS need stored residuals */
  if (!rice && !r) return -1;
  h.flags = (uint8_t)((h.flags & (FLAG_PRED_MASK | FLAG_DITHER | FLAG_PIXLIST | FLAG_DLISTED)) | (rice ? FLAG_RICE : 0) | (ans ? FLAG_ANS : 0));
  const size_t hdr_total = HDR_TOTAL_OF(&h);
  if (output_len <= hdr_total) return 0;
  const uint32_t ngroups = (n + RICE_G - 1) / RICE_G;

  if (ans) {
    /* scratch: records [n] + stream buffers */
    size_t off_rec = 0, off_str = ALIGN64((size_t)n * sizeof(ans_rec)), str_cap = 4 * ((size_t)n * 3 + 4096);
    uint8_t *buf = work;
    ans_rec *rec = (ans_rec *)(buf + off_rec);
    uint32_t count[ANS_NCTX * HUFF_NSYM];
    h.nsym = ans_symbolise(r, n, w, rec, count);
    /* contexts cost decode speed (a per-symbol dependency on the previous residual); use them only
       when the estimated saving over a single table exceeds ANS_CTX_MIN_GAIN */
    if (ans_context_gain(count) < ANS_CTX_MIN_GAIN) {
      for (int c = 1; c < ANS_NCTX; c++) for (int sy = 0; sy < HUFF_NSYM; sy++) { count[sy] += count[c * HUFF_NSYM + sy]; count[c * HUFF_NSYM + sy] = 0; }
      for (uint32_t k = 0; k < h.nsym; k++) rec[k].ctx = 0;
    }
    ans_ctx_enc ctx[ANS_NCTX];
    for (int c = 0; c < ANS_NCTX; c++) {
      ans_normalise(count + c * HUFF_NSYM, ctx[c].freq); ans_starts(ctx[c].start, ctx[c].freq);
      ctx[c].present = 0; for (int sy = 0; sy < HUFF_NSYM; sy++) ctx[c].present |= ctx[c].freq[sy] != 0;
      ans_ctx_enc_prepare(&ctx[c]);
    }
    uint8_t tables[ANS_TABLE_MAX + 16];
    uint32_t tlen = ans_tables_write(tables, ctx);
    if (hdr_total + tlen >= output_len) return 0;
    memcpy(output + hdr_total, tables, tlen);
    uint32_t sbytes[NSTREAM];
    int total = ans_encode_streams(rec, h.nsym, ctx, buf + off_str, str_cap, output + hdr_total + tlen,
                                   output_len - hdr_total - tlen, sbytes);
    if (total == -2) return 0;
    if (total < 0) return -1;
    for (int k = 0; k < NSTREAM - 1; k++) h.ssize[k] = sbytes[k];   /* stream 3 = remainder; tables are self-delimiting */
    hdr_write(output, &h);
    *hp = h;
    return (int)(hdr_total + tlen + (size_t)total);
  }

  if (rice) {
    /* Single pass: each group is planned and emitted at once into per-stream scratch (worst-case sized, only the
       written bytes are touched), so the input is read once and nothing per block is stored; the stream sizes
       are the emitted lengths. Zero-run tokens are deferred until the run closes: a zero group writes nothing,
       so the token lands at the same place in the stream as the look-ahead version did. */
    h.nsym = ngroups;
    const size_t scap = 6 * ((size_t)n / RICE_NS + RICE_G) + 64;
    bitwx bx[RICE_NS]; uint8_t *ss[RICE_NS]; uint32_t zrun[RICE_NS] = {0};
    for (int k = 0; k < RICE_NS; k++) { ss[k] = work + (size_t)k * scap; bwx_init(&bx[k], ss[k]); }
    uint32_t u[RICE_G]; int16_t gtmp[RICE_G + 16];
    for (uint32_t g = 0; g < ngroups; g++) {
      uint32_t st = g & (RICE_NS - 1), i0 = g * RICE_G, glen = (n - i0 < RICE_G) ? n - i0 : RICE_G;
      uint8_t kk;
      rice_group_plan(group_resid(rs, i0, glen, gtmp), glen, &kk, u);
      bitwx *b = &bx[st];
      if (kk == 15) {
        if (++zrun[st] == RICE_ZRUNMAX) { bwx_put(b, 15, RICE_KBITS); bwx_put(b, RICE_ZRUNMAX - 1, RICE_ZRUNBITS); bwx_flush(b); zrun[st] = 0; }
        continue;
      }
      if (zrun[st]) { bwx_put(b, 15, RICE_KBITS); bwx_put(b, zrun[st] - 1, RICE_ZRUNBITS); bwx_flush(b); zrun[st] = 0; }
      if (kk == RICE_SPARSE) {
        uint32_t nnz = 0;
        for (uint32_t i = 0; i < glen; i++) nnz += (u[i] != 0);
        bwx_put(b, RICE_SPARSE, RICE_KBITS); bwx_put(b, nnz, RICE_POSBITS); bwx_flush(b);
        for (uint32_t i = 0; i < glen; i++) if (u[i]) {
          int nb = 32 - b4d_clz32(u[i]);
          bwx_put(b, i, RICE_POSBITS); bwx_flush(b);
          if (nb > 1) { bwx_put(b, 0, (unsigned)(nb - 1)); }           /* gamma: nb-1 zeros then u in nb bits (<= 31 total) */
          bwx_put(b, u[i], (unsigned)nb); bwx_flush(b);
        }
      } else {
        bwx_put(b, (uint32_t)kk, RICE_KBITS); bwx_flush(b);
        bitwx l = *b;                                    /* a local copy stays in registers (the byte stores could alias *b) */
        rice_put_group(&l, u, glen, kk);
        *b = l;
      }
    }
    uint64_t total = hdr_total + 4 * RICE_XS; uint32_t sbytes[RICE_NS];
    for (int k = 0; k < RICE_NS; k++) {
      if (zrun[k]) { bwx_put(&bx[k], 15, RICE_KBITS); bwx_put(&bx[k], zrun[k] - 1, RICE_ZRUNBITS); bwx_flush(&bx[k]); }
      sbytes[k] = (uint32_t)bwx_finish(&bx[k], ss[k]); total += sbytes[k];
    }
    if (total >= (uint64_t)output_len) return 0;                 /* incompressible */
    for (int k = 0; k < NSTREAM - 1; k++) h.ssize[k] = sbytes[k];
    hdr_write(output, &h);
    *hp = h;
    uint8_t *p = output + hdr_total;
    for (int k = 0; k < RICE_XS; k++) { put_u32(p, sbytes[NSTREAM - 1 + k]); p += 4; }   /* v10: sizes of streams 3..6 */
    for (int k = 0; k < RICE_NS; k++) { memcpy(p, ss[k], sbytes[k]); p += sbytes[k]; }
    return (int)total;
  }

  size_t off_sym = 0;
  uint8_t *buf = work;
  uint64_t bits[NSTREAM] = {0, 0, 0, 0};
  uint8_t len[HUFF_NSYM]; uint16_t code[HUFF_NSYM]; uint32_t nentries = 0;
  uint16_t *sym = (uint16_t *)(buf + off_sym);
  {
    uint32_t freq[HUFF_NSYM], esc[NSTREAM];
    h.nsym = symbolise(r, n, sym, &nentries, freq, esc);
    memset(code, 0, sizeof code);
    huff_build_lengths(freq, len);
    huff_codes(len, code);
    uint32_t k = 0, si = 0;
    while (k < nentries) {
      uint32_t sy = sym[k++]; bits[si & (NSTREAM - 1)] += len[sy];
      if (sy == SYM_ESC) { k++; bits[si & (NSTREAM - 1)] += 16; }
      si++;
    }
  }

  uint64_t total = hdr_total; uint32_t sbytes[NSTREAM];
  for (int k = 0; k < NSTREAM; k++) { sbytes[k] = (uint32_t)((bits[k] + 7) / 8); total += sbytes[k]; }
  if (total >= (uint64_t)output_len) return 0;                   /* incompressible */
  for (int k = 0; k < NSTREAM - 1; k++) h.ssize[k] = sbytes[k];

  hdr_write(output, &h);
  *hp = h;
  for (int sy = 0; sy < HUFF_NSYM; sy += 2) output[HDR_BYTES + sy / 2] = (uint8_t)(len[sy] | (len[sy + 1] << 4));

  bitw bw[NSTREAM]; uint8_t *p = output + hdr_total;
  for (int k = 0; k < NSTREAM; k++) { bw_init(&bw[k], p); p += sbytes[k]; }
  {
    uint32_t k = 0, si = 0;
    while (k < nentries) {
      uint32_t sy = sym[k++]; bitw *b = &bw[si & (NSTREAM - 1)];
      if (sy == SYM_ESC) bw_put(b, ((uint32_t)code[sy] << 16) | sym[k++], len[sy] + 16);
      else bw_put(b, code[sy], len[sy]);
      si++;
    }
  }
  for (int k = 0; k < NSTREAM; k++) bw_flush(&bw[k]);
  return (int)total;
}

static int entropy_encode_block(b4d_hdr *hp, const int16_t *r, uint8_t *output, size_t output_len, int coder,
                                uint8_t *work, size_t work_cap) {
  resid_src rs; memset(&rs, 0, sizeof rs); rs.r = r;
  return entropy_encode_block_src(hp, &rs, output, output_len, coder, work, work_cap);
}

/* B4D_PRED_ORIGINAL: header (v5, tile in ssize[0]) + the original filter's chunk bitstream. 8-bit input is widened to
   16 bits like the original filter does (u8tou16), and narrowed back with its clamp on decode. */
static int orig_encode_block(b4d_hdr *h, const uint8_t *input, uint8_t *output, int32_t output_len) {
  const uint32_t n = h->nelem, w = (h->roww == 0 || h->roww > n) ? n : h->roww;
  if (n % w) return 0;
  const uint16_t *img = (const uint16_t *)input; uint16_t *tmp = NULL;
  if (h->typesize == 1) { tmp = (uint16_t *)malloc((size_t)n * 2); if (!tmp) return -1; for (uint32_t i = 0; i < n; i++) tmp[i] = input[i]; img = tmp; }
  const float q = h->qi * 0.25f;
  const float *tab = (h->qi && h->mode) ? b4d_orig_find_vst_table(h->conv, h->offset, h->rn, q) : NULL;
  uint8_t *payload = NULL; size_t plen = 0;
  int rc = b4d_orig_encode(img, w, n / w, h->mode, q, h->conv, h->offset, h->rn, B4D_ORIG_TILE, tab, &payload, &plen);
  free(tmp);
  if (rc) return -1;
  if (plen + HDR_BYTES > (size_t)output_len) { b4d_orig_free(payload); return 0; }
  h->flags = FLAG_PRED_ORIGINAL; h->ssize[0] = B4D_ORIG_TILE; h->nsym = 0; h->ph = 0;
  hdr_write(output, h);
  memcpy(output + HDR_BYTES, payload, plen);
  b4d_orig_free(payload);
  return (int)(plen + HDR_BYTES);
}
static int orig_decode_block(const b4d_hdr *h, const uint8_t *payload, size_t plen, uint8_t *output, int32_t output_len) {
  const uint32_t n = h->nelem, w = (h->roww == 0 || h->roww > n) ? n : h->roww;
  if (n % w || (int64_t)n * h->typesize > output_len) return -1;
  uint32_t tile = h->ssize[0] ? h->ssize[0] : B4D_ORIG_TILE;
  uint16_t *dst = (h->typesize == 2) ? (uint16_t *)output : (uint16_t *)malloc((size_t)n * 2);
  if (!dst) return -1;
  int rc = b4d_orig_decode(payload, plen, w, n / w, h->mode, h->qi * 0.25f, h->conv, h->offset, h->rn, tile, dst);
  if (h->typesize == 1) { if (rc == 0) for (uint32_t i = 0; i < n; i++) output[i] = (uint8_t)(dst[i] > 255 ? 255 : dst[i]); free(dst); }
  return rc ? -1 : (int)(n * h->typesize);
}

static int encoder_impl(const uint8_t *input, int32_t input_len, uint8_t *output,
                        int32_t output_len, uint8_t meta, blosc2_cparams *cparams) {
  int typesize = cparams ? cparams->typesize : 2;
  if (typesize != 1 && typesize != 2) return 0;

  /* parameters: per-context struct if supplied, else process-wide settings */
  float conv = g_conv, offset = g_offset, rn = g_rn; uint32_t roww = g_roww, ph = g_ph; int pred = g_pred, coder = g_coder, dither = g_dither;
  const b4d_params *pp = cparams ? (const b4d_params *)cparams->codec_params : NULL;
  if (pp && pp->magic == B4D_PARAMS_MAGIC) {
    conv = pp->conversion > 0 ? pp->conversion : 1.0f; offset = pp->offset; rn = pp->read_noise;
    roww = pp->row_width; pred = norm_pred(pp->predictor); ph = pp->predictor == B4D_PRED_ABOVE3 ? 0 : pp->plane_rows; dither = pp->dither;
    coder = (pp->entropy == B4D_ENTROPY_ANS) ? B4D_ENTROPY_ANS : pp->entropy ? B4D_ENTROPY_RICE : B4D_ENTROPY_HUFFMAN;
  }
  b4d_hdr h; memset(&h, 0, sizeof h);
  h.version = version_of_pred(pred);
  h.mode = meta & 3; h.qi = meta >> 2; h.typesize = (uint8_t)typesize;
  h.conv = conv; h.offset = offset; h.rn = rn; h.roww = roww; h.ph = (pred == B4D_PRED_MEAN3D) ? (ph & 0xFFFFFF) : 0;
  h.flags = flags_of_pred(pred);
  h.nelem = (uint32_t)(input_len / typesize);
  if (h.qi == 0) h.mode = 0;
  /* the noise list: mode 1 with above3, or mode 2 with above (v8) */
  const b4d_pixmap *pm = (pp && pp->magic == B4D_PARAMS_MAGIC &&
                          ((h.mode == 1 && pp->predictor == B4D_PRED_ABOVE3) || (h.mode == 2 && pred == B4D_PRED_ABOVE))) ? pp->pixmap : NULL;
  /* dither: 1 = every pixel, 2 = the listed pixels only (nothing to dither without a list); v8 in both modes */
  int dmode = (h.mode == 0 || pred == B4D_PRED_ORIGINAL) ? 0 : dither == 2 ? 2 : dither ? 1 : 0;
  if (dmode == 2 && !pm) dmode = 0;
  const uint32_t n = h.nelem, w = h.roww;
  const int above = (h.flags & FLAG_PRED_ABOVE) != 0;
  if (n == 0 || output_len <= HDR_BYTES) return 0;
  if (pred == B4D_PRED_ORIGINAL) return orig_encode_block(&h, input, output, output_len);
  if (tl_vt && h.mode) { h.vst = tl_vser; h.vst_bytes = tl_vser_len; }      /* v9: the exact VST's description */

  /* open-loop modes with the Rice coder: no sample/residual arrays; the entropy stage recomputes each
     group's residuals from the input (through the LUT for mode 2). Hot set = tile + LUT + Rice work.
     (mean3d keeps the stored-residual path below: its taps span three rows of two planes.) With dither on every pixel
     the dither stream is generated a row at a time too. */
  if (h.mode != 1 && coder == B4D_ENTROPY_RICE && pred != B4D_PRED_MEAN3D && dmode != 2 && !pm) {
    resid_src rs; memset(&rs, 0, sizeof rs);
    if (h.mode == 2 && dmode) {
      lutd_t *L = lutd_get(meta, conv, offset, rn);
      if (!L || !L->valid) return 0;
      rs.lutd = L->tab;
      h.flags |= FLAG_DITHER; h.seed = dither_seed8(input, (size_t)input_len); h.version = 8; rs.ds0 = dither_s0(h.seed);
    } else if (h.mode == 2) {
      lut16_t *L = lut16_get(meta, conv, offset, rn);
      if (!L || !L->valid) return 0;
      rs.lut = L->tab;
    }
    if (typesize == 1) rs.in8 = input; else rs.in16 = (const uint16_t *)input;
    rs.n = n; rs.w = (w == 0 || w > n) ? n : w; rs.above = above;
    size_t work_cap = entropy_work_bytes(n, coder), off_row = ALIGN64(work_cap);
    size_t rowb = rs.w <= 8192 ? ALIGN64(4 * (size_t)rs.w + 64) : 0;      /* 2-row sample cache (wider rows: per-element lookups) */
    size_t drb = (rowb && rs.lutd) ? ALIGN64((size_t)rs.w) : 0;          /* a row of dither */
    uint8_t *work = scratch(off_row + rowb + drb);
    if (!work) return -1;
    if (rowb) { rs.vcur = (uint16_t *)(work + off_row); rs.vprev = rs.vcur + rs.w; }
    if (drb) rs.drow = work + off_row + rowb;
    return entropy_encode_block_src(&h, &rs, output, (size_t)output_len, coder, work, work_cap);
  }

  /* mode 1 with the Rice coder, one plane (left_above, above, above3) and no noise list: the closed loop is streamed into
     the coder a row at a time (m1_src); scratch = Rice work + 4 int32 rows + a residual row + a dither row */
  if (h.mode == 1 && coder == B4D_ENTROPY_RICE && !pm && h.qi) {
    const uint32_t w1 = (w == 0 || w > n) ? n : w;
    if (pred != B4D_PRED_MEAN3D || plane_rows_of(h.ph, n, w1) == (n + w1 - 1) / w1) {
      lut32_t *L = lut32_get(meta, conv, offset, rn);
      if (!L) return -1;
      m1_src m; memset(&m, 0, sizeof m);
      m.in8 = typesize == 1 ? input : NULL; m.in16 = typesize == 2 ? (const uint16_t *)input : NULL; m.lut = L->tab;
      m.n = n; m.w = w1; m.qfix = (int32_t)h.qi * (FIX_SCALE / 4); m.pred = pred;
      if (dmode) {
        h.flags |= FLAG_DITHER; h.seed = dither_seed8(input, (size_t)input_len); h.version = 8;
        m.dith = 1; m.ds0 = dither_s0(h.seed);
      }
      size_t work_cap = entropy_work_bytes(n, coder), off_row = ALIGN64(work_cap), rb = ALIGN64(4 * (size_t)w1);
      uint8_t *work = scratch(off_row + 4 * rb + ALIGN64(2 * (size_t)w1) + ALIGN64((size_t)w1));
      if (!work) return -1;
      uint8_t *p = work + off_row;
      m.srow = (int32_t *)p; m.fprev = (int32_t *)(p + rb); m.fcur = (int32_t *)(p + 2 * rb); m.prow = (int32_t *)(p + 3 * rb);
      m.rrow = (int16_t *)(p + 4 * rb); m.drow = p + 4 * rb + ALIGN64(2 * (size_t)w1);
      resid_src rs; memset(&rs, 0, sizeof rs); rs.m1 = &m; rs.n = n; rs.w = w1;
      int rc = entropy_encode_block_src(&h, &rs, output, (size_t)output_len, coder, work, work_cap);
      return (rc > 0 && m.bad) ? 0 : rc;                         /* a residual outside int16: stored raw, as mode1_encode */
    }
  }

  /* scratch: samples u16 [n] | residuals i16 [n] | symbols u16 [2n] or rice k [n/G]
              | mode1 s32 [n] f32 [n] */
  size_t off_v = 0, off_r = ALIGN64(off_v + 2 * (size_t)n);
  size_t off_s = ALIGN64(off_r + 2 * (size_t)n), off_f = ALIGN64(off_s + 4 * (size_t)n);
  size_t off_work = (h.mode == 1) ? ALIGN64(off_f + 4 * (size_t)n) : off_s;
  size_t work_cap = entropy_work_bytes(n, coder);
  /* per-pixel map: count the block's listed pixels first, so their arrays and the serialised list can share the scratch */
  px_block B; memset(&B, 0, sizeof B);
  uint32_t plane = 0; size_t off_px = ALIGN64(off_work + work_cap), off_pay = off_px, px_cap = 0;
  if (pm && w >= 3 && n % w == 0) {
    const uint32_t rows = n / w;
    plane = (pp->plane_rows == 0 || pp->plane_rows > rows) ? rows : pp->plane_rows;
    const uint32_t nent = px_count(pm, pp->map_y0, pp->map_x0, plane, w);
    if (nent && PX_HDR(pm->ntiers) + (size_t)nent <= (size_t)input_len / PX_MAX_RAW_SHARE) {
      B.plane_rows = plane; B.nent = nent;
      off_pay = off_px + ALIGN64(px_mem_bytes(plane, nent, w)); px_cap = px_payload_cap(nent, pm->ntiers);
    }
  }
  const size_t off_dr = ALIGN64(off_pay + px_cap);
  uint8_t *buf = scratch(off_dr + (dmode ? (size_t)n : 0));
  if (!buf) return -1;
  uint16_t *v = (uint16_t *)(buf + off_v); int16_t *r = (int16_t *)(buf + off_r);
  if (B.nent) {
    uint8_t *pay = buf + off_pay;
    px_mem_set(&B, buf + off_px);
    size_t pix = px_build(pm, pp->map_y0, pp->map_x0, w, &B, pay);
    B.tab = pxtab_get(h.qi, rn, pm->ntiers, pm->extra, pm->skip);
    if (!B.tab) return -1;
    if (pix > (size_t)input_len / PX_MAX_RAW_SHARE) B.nent = 0;
    else { h.version = 7; h.flags |= FLAG_PIXLIST; h.pix_bytes = (uint32_t)pix; h.pix = pay; }
  }
  if (dmode == 2 && !B.nent) dmode = 0;
  const uint8_t *dr = NULL;
  if (dmode) {
    h.flags |= FLAG_DITHER | (dmode == 2 ? FLAG_DLISTED : 0); h.seed = dither_seed8(input, (size_t)input_len);
    dither_fill(buf + off_dr, n, h.seed); dr = buf + off_dr;
  }
  if (dmode || (B.nent && h.mode == 2)) h.version = 8;
  const int32_t qfix = (int32_t)h.qi * (FIX_SCALE / 4);

  /* ---- prediction / quantisation -> int16 residuals */
  if (h.mode == 0) {
    const uint16_t *src = (const uint16_t *)input;
    if (typesize == 1) { for (uint32_t i = 0; i < n; i++) v[i] = input[i]; src = v; }
    residual_u16(src, r, n, w, pred, h.ph);
  } else if (h.mode == 2) {
    const uint16_t *in16 = (const uint16_t *)input;
    if (dmode == 1) {
      lutd_t *L = lutd_get(meta, conv, offset, rn);
      if (!L || !L->valid) return 0;
      const int32_t *t = L->tab;
      if (typesize == 1) for (uint32_t i = 0; i < n; i++) v[i] = (uint16_t)((t[input[i]] + (int32_t)dr[i]) >> DITHER_SHIFT);
      else for (uint32_t i = 0; i < n; i++) v[i] = (uint16_t)((t[in16[i]] + (int32_t)dr[i]) >> DITHER_SHIFT);
    } else {
      lut16_t *L = lut16_get(meta, conv, offset, rn);
      if (!L || !L->valid) return 0;
      const uint16_t *t = L->tab;
      if (typesize == 1) for (uint32_t i = 0; i < n; i++) v[i] = t[input[i]];
      else for (uint32_t i = 0; i < n; i++) v[i] = t[in16[i]];
    }
    if (B.nent) {
      lut32_t *L32 = lut32_get(meta, conv, offset, rn);
      if (!L32) return -1;
      const double q = h.qi * 0.25;
      m2code mc = { dmode == 1, qfix, h.qi, rnd_d(vst_fwd(0, conv, offset, rn) / q), dither_base(q, conv, offset, rn) };
      const uint8_t *i8 = typesize == 1 ? input : NULL; const uint16_t *i16 = typesize == 2 ? in16 : NULL;
      if (!mode2_encode_px(v, r, i8, i16, L32->tab, n, w, &B, &mc, dr, 0)) return 0;
      residual_u16(v, r, n, w, pred, h.ph);
      mode2_encode_px(v, r, i8, i16, L32->tab, n, w, &B, &mc, dr, 1);
    } else residual_u16(v, r, n, w, pred, h.ph);
  } else {
    lut32_t *L = lut32_get(meta, conv, offset, rn);
    if (!L) return -1;
    int32_t *sv = (int32_t *)(buf + off_s), *f = (int32_t *)(buf + off_f);
    const int32_t *t = L->tab;
    const int flat = plane && (pm->flat_a || pm->flat_b);
    if (B.nent || flat) {                               /* the per-pixel paths take the whole block of samples */
      if (typesize == 1) for (uint32_t i = 0; i < n; i++) sv[i] = t[input[i]];
      else { const uint16_t *in16 = (const uint16_t *)input; for (uint32_t i = 0; i < n; i++) sv[i] = t[in16[i]]; }
      if (flat) px_flatten(sv, n, w, pm, pp->map_y0, pp->map_x0, plane, rnd_d(vst_fwd(offset, conv, offset, rn) * FIX_SCALE));
    }
    if (B.nent) { if (!mode1_encode_px(sv, f, r, n, w, qfix, &B, dr, dmode == 1)) return 0; }
    else if (!mode1_encode(flat ? sv : NULL, f, r, n, w, qfix, pred, h.ph, dr, typesize == 1 ? input : NULL,
                           typesize == 2 ? (const uint16_t *)input : NULL, t, sv)) return 0;
  }

  return entropy_encode_block(&h, r, output, (size_t)output_len, coder, buf + off_work, work_cap);
}

/* The exact VST (b4d_params.vst, modes 1 and 2) is in force for the whole block: tl_vt points at its table while it is coded. */
int b4d_encoder(const uint8_t *input, int32_t input_len, uint8_t *output,
                int32_t output_len, uint8_t meta, blosc2_cparams *cparams,
                const void *chunk) {
  (void)chunk;
  const b4d_params *pp = cparams ? (const b4d_params *)cparams->codec_params : NULL;
  const b4d_vst *vd = (pp && pp->magic == B4D_PARAMS_MAGIC && pp->vst && (meta & 3) && (meta >> 2)) ? pp->vst : NULL;
  const vtab_t *vt = NULL;
  if (vd) {
    const float conv = pp->conversion > 0 ? pp->conversion : 1.0f;
    if (vd->conv != conv || vd->offset != pp->offset || vd->rn != pp->read_noise) return -1;   /* built for another camera */
    if (!(vt = vtab_get(vd))) return -1;
  }
  const double *s_vt = tl_vt; const uint8_t *s_ser = tl_vser; const uint32_t s_len = tl_vser_len;
  tl_vt = vt ? vt->t : NULL; tl_vser = vt ? vt->ser : NULL; tl_vser_len = vt ? vt->ser_len : 0;
  const int rc = encoder_impl(input, input_len, output, output_len, meta, cparams);
  tl_vt = s_vt; tl_vser = s_ser; tl_vser_len = s_len;
  return rc;
}

int b4d_encode_residual_block2(const int16_t *residuals, uint32_t nelem, uint8_t mode, uint8_t typesize,
                               uint8_t qi, float conversion, float offset, float read_noise,
                               uint32_t row_width, uint32_t plane_rows, int predictor, int entropy, uint8_t *out, int32_t out_cap) {
  if (out_cap <= HDR_BYTES || nelem == 0) return 0;
  int pred = norm_pred(predictor);
  if (pred == B4D_PRED_ORIGINAL) return -1;          /* no residual interface: the original pipeline runs end to end in b4d_encoder */
  b4d_hdr h; memset(&h, 0, sizeof h);
  h.version = version_of_pred(pred); h.mode = mode; h.typesize = typesize; h.qi = qi;
  h.conv = conversion > 0 ? conversion : 1.0f; h.offset = offset; h.rn = read_noise; h.roww = row_width; h.nelem = nelem;
  h.ph = (pred == B4D_PRED_MEAN3D && predictor != B4D_PRED_ABOVE3) ? (plane_rows & 0xFFFFFF) : 0;
  h.flags = flags_of_pred(pred);
  size_t work_cap = entropy_work_bytes(nelem, entropy);
  uint8_t *work = scratch(work_cap);
  if (!work) return -1;
  return entropy_encode_block(&h, residuals, out, (size_t)out_cap, entropy, work, work_cap);
}
int b4d_encode_residual_block(const int16_t *residuals, uint32_t nelem, uint8_t mode, uint8_t typesize,
                              uint8_t qi, float conversion, float offset, float read_noise,
                              uint32_t row_width, int predictor, int entropy, uint8_t *out, int32_t out_cap) {
  return b4d_encode_residual_block2(residuals, nelem, mode, typesize, qi, conversion, offset, read_noise, row_width, 0, predictor, entropy, out, out_cap);
}
/* ------------------------------------------------------------------------ */
/* Decoder                                                                  */
/* ------------------------------------------------------------------------ */

static int decoder_impl(const uint8_t *input, int32_t input_len, uint8_t *output, int32_t output_len);
int b4d_decoder(const uint8_t *input, int32_t input_len, uint8_t *output,
                int32_t output_len, uint8_t meta, blosc2_dparams *dparams,
                const void *chunk) {
  (void)meta; (void)dparams; (void)chunk;
  const double *s_vt = tl_vt;
  tl_vt = NULL;                                        /* decoder_impl sets it from a v9 header */
  const int rc = decoder_impl(input, input_len, output, output_len);
  tl_vt = s_vt;
  return rc;
}
/* output[k] = d->x[clamp(a[k] - a0, 0, n - 1)] (the table decode of DECODE_FIX); the index in int32 with wrap then clamp, so
   every read is inside the table whatever a holds */
static void vtdec_apply(const vtdec_t *d, const int32_t *a, uint32_t n, uint8_t *out, int ts) {
  const int32_t a0 = d->a0, top = (int32_t)d->n - 1; const uint16_t *x = d->x; uint32_t k = 0;
  if (ts == 2) {
    uint16_t *o = (uint16_t *)out;
#ifdef B4D_HAVE_AVX2
    const __m256i va0 = _mm256_set1_epi32(a0), vtop = _mm256_set1_epi32(top), z = _mm256_setzero_si256(), lo16 = _mm256_set1_epi32(0xFFFF);
    for (; k + 16 <= n; k += 16) {
      __m256i i0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(a + k)), va0);
      __m256i i1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(a + k + 8)), va0);
      i0 = _mm256_min_epi32(_mm256_max_epi32(i0, z), vtop); i1 = _mm256_min_epi32(_mm256_max_epi32(i1, z), vtop);
      __m256i g0 = _mm256_and_si256(_mm256_i32gather_epi32((const int *)x, i0, 2), lo16);
      __m256i g1 = _mm256_and_si256(_mm256_i32gather_epi32((const int *)x, i1, 2), lo16);
      _mm256_storeu_si256((__m256i *)(o + k), _mm256_permute4x64_epi64(_mm256_packus_epi32(g0, g1), 0xD8));
    }
#endif
    for (; k < n; k++) { int32_t i = (int32_t)((uint32_t)a[k] - (uint32_t)a0); i = i < 0 ? 0 : (i > top ? top : i); o[k] = x[i]; }
  } else
    for (; k < n; k++) { int32_t i = (int32_t)((uint32_t)a[k] - (uint32_t)a0); i = i < 0 ? 0 : (i > top ? top : i); out[k] = (uint8_t)x[i]; }
}

static int decoder_impl(const uint8_t *input, int32_t input_len, uint8_t *output, int32_t output_len) {
  if (input_len < HDR_BYTES) return -1;
  b4d_hdr h; hdr_read(input, &h);
  if (h.version < 1 || h.version > B4D_VERSION_MAX || (h.typesize != 1 && h.typesize != 2) || h.mode > 2) return -1;
  if (h.version >= V_RICE8 && !(h.flags & FLAG_RICE)) return -1;        /* v10 / v11 are Rice only */
  if ((int64_t)h.nelem * h.typesize > output_len) return -1;
  if (V_HAS_VST(h.version)) {                          /* the exact VST: rebuild (or find) its table from the description */
    const size_t at = HDR_FIXED_OF(&h);
    if (h.mode == 0 || (size_t)input_len < at + 8) return -1;
    const int nk = input[at];
    if (nk < 2 || nk > B4D_VST_MAXK || (size_t)input_len < at + VST_SER_BYTES(nk)) return -1;
    float g[B4D_VST_MAXK]; for (int j = 0; j < nk; j++) g[j] = get_f32(input + at + 8 + 4 * j);
    struct b4d_vst k; vst_key(&k, h.conv, h.offset, h.rn, nk, get_f32(input + at + 4), g);
    const vtab_t *vt = vtab_get(&k);
    if (!vt) return -1;
    tl_vt = vt->t; h.vst = input + at; h.vst_bytes = VST_SER_BYTES(nk);
  }
  if (h.flags & FLAG_PIXLIST) {
    const size_t at = HDR_PIX_AT(&h);
    if ((size_t)input_len < at + 4) return -1;
    h.pix_bytes = get_u32(input + at); h.pix = input + at + 4;
    if ((size_t)h.pix_bytes > (size_t)input_len - at - 4) return -1;
  }
  const size_t hdr_total = HDR_TOTAL_OF(&h);
  if ((size_t)input_len < hdr_total) return -1;
  const uint32_t n = h.nelem, w = h.roww;
  const int pred = pred_of_flags(h.flags), rice = (h.flags & FLAG_RICE) != 0, ans = (h.flags & FLAG_ANS) != 0;
  if (ans && h.version < 3) return -1;
  if (pred == B4D_PRED_MEAN3D && h.version < 4) return -1;
  if (pred == B4D_PRED_ORIGINAL && h.version < 5) return -1;
  const int dither = (h.flags & FLAG_DITHER) != 0, dlisted = (h.flags & FLAG_DLISTED) != 0;
  if (dither && (h.version < 6 || h.mode == 0 || (h.mode == 1 && h.version < 8))) return -1;
  const int pixlist = (h.flags & FLAG_PIXLIST) != 0;
  if (pixlist && (h.mode == 1 ? (h.version < 7 || pred != B4D_PRED_MEAN3D || h.ph != 0)
                              : (h.mode != 2 || h.version < 8 || pred != B4D_PRED_ABOVE))) return -1;
  if (dlisted && (h.version < 8 || !dither || !pixlist)) return -1;
  const int dall = dither && !dlisted;                 /* every pixel is dithered (else only the listed ones, or none) */
  if (n == 0) return 0;
  if (pred == B4D_PRED_ORIGINAL) return orig_decode_block(&h, input + HDR_BYTES, (size_t)input_len - HDR_BYTES, output, output_len);

  /* mode-2 inverse LUT range (computed up front so scratch is allocated once) */
  const double vmax = (h.typesize == 1) ? 255.0 : 65535.0;
  const double q = h.qi * 0.25;
  long smin = 0; uint32_t span = 0;
  if (h.mode == 2 && q <= 0) return -1;
  if (h.mode == 2 && !dall) {
    smin = rnd_d(vst_fwd(0, h.conv, h.offset, h.rn) / q);
    long smax = rnd_d(vst_fwd(vmax, h.conv, h.offset, h.rn) / q);
    if (smax < smin) smax = smin;
    span = (uint32_t)(smax - smin) + 1;
  }
  size_t off_r = 0, off_f = ALIGN64(2 * (size_t)n), off_d = ALIGN64(off_f + 4 * (size_t)n);
  size_t off_inv = ALIGN64(off_d + sizeof(huff_dec) + (ans ? ANS_NCTX * ANS_M * sizeof(uint16_t) : 0));
  px_block B; memset(&B, 0, sizeof B);
  size_t off_px = ALIGN64(off_inv + 2 * (size_t)span), px_mem = 0, trec_n = 0;
  if (pixlist) {
    if (px_peek(h.pix, h.pix_bytes, n, w, &B.plane_rows, &B.nent)) return -1;
    px_mem = px_mem_bytes(B.plane_rows, B.nent, w);
    trec_n = 2 * (size_t)B.nent * ((n / w + B.plane_rows - 1) / B.plane_rows);   /* mode 2: (T, step) per occurrence; mode 1: steps */
  }
  const size_t off_trec = ALIGN64(off_px + px_mem), off_dr = ALIGN64(off_trec + 4 * trec_n);
  uint8_t *buf = scratch(off_dr + (dither ? (size_t)n : 0));
  if (!buf) return -1;
  int16_t *r = (int16_t *)(buf + off_r);
  if (pixlist) {
    px_mem_set(&B, buf + off_px);
    if (px_parse(h.pix, h.pix_bytes, w, h.qi, h.rn, &B)) return -1;
  }
  const uint8_t *dr = NULL;
  if (dither) { dither_fill(buf + off_dr, n, get_u32(input + hdr_total - 4)); dr = buf + off_dr; }

  /* ---- streams (ANS: self-delimiting frequency tables come first) */
  const uint8_t *p = input + hdr_total; size_t rem = (size_t)input_len - hdr_total;
  ans_ctx_dec actx[ANS_NCTX];
  if (ans) {
    int tlen = ans_tables_read(p, (uint32_t)rem, actx, (uint16_t *)(buf + off_d + sizeof(huff_dec)));
    if (tlen < 0) return -1;
    p += tlen; rem -= (size_t)tlen;
  }
  const int rice8 = rice && h.version >= V_RICE8, ns = rice8 ? RICE_NS : NSTREAM;
  bitr br[RICE_NS]; const uint8_t *streams[RICE_NS]; uint32_t ssz[RICE_NS], xs[RICE_XS];
  if (rice8) {                                         /* v10 / v11: the sizes of streams 3..6 open the payload */
    if (rem < 4 * (size_t)RICE_XS) return -1;
    for (int k = 0; k < RICE_XS; k++) xs[k] = get_u32(p + 4 * k);
    p += 4 * RICE_XS; rem -= 4 * RICE_XS;
  }
  for (int k = 0; k < ns; k++) {
    size_t sz = (k < NSTREAM - 1) ? h.ssize[k] : (k < ns - 1) ? xs[k - (NSTREAM - 1)] : rem;
    if (sz > rem) return -1;
    br_init(&br[k], p, sz); streams[k] = p; ssz[k] = (uint32_t)sz; p += sz; rem -= sz;
  }

  if (ans) {
    if (ans_decode_streams(br, streams, ssz, h.nsym, actx, r, n, w) != 0) return -1;
  } else if (rice) {
    const uint32_t ngroups = (n + RICE_G - 1) / RICE_G;
    if (h.nsym != ngroups) return -1;
    uint32_t pending[RICE_NS] = {0};
    uint32_t g = 0;
    if (rice8)
      for (; (g + RICE_NS) * RICE_G <= n; g += RICE_NS)        /* whole sets of one group per stream */
        { if (rice_groups_read_all8(br, r + (size_t)g * RICE_G, h.version, pending) != 0) return -1; }
    else
      for (; (g + NSTREAM) * RICE_G <= n; g += NSTREAM)
        { if (rice_groups_read_all4(br, r + (size_t)g * RICE_G, h.version, pending) != 0) return -1; }
    for (; g < ngroups; g++) {
      uint32_t i0 = g * RICE_G, glen = (n - i0 < RICE_G) ? n - i0 : RICE_G, s = g & (uint32_t)(ns - 1);
      if (rice_group_read(&br[s], r + i0, glen, h.version, &pending[s]) != 0) return -1;
    }
    for (int k = 0; k < ns; k++) if (br[k].p > br[k].end) return -1;
  } else {
    uint8_t len[HUFF_NSYM];
    for (int sy = 0; sy < HUFF_NSYM; sy += 2) { len[sy] = input[HDR_BYTES + sy / 2] & 0xF; len[sy + 1] = input[HDR_BYTES + sy / 2] >> 4; }
    huff_dec *d = (huff_dec *)(buf + off_d);
    if (!huff_dec_init(d, len)) return -1;
    uint32_t i = 0;
    for (uint32_t si = 0; si < h.nsym; si++) {
      bitr *b = &br[si & (NSTREAM - 1)];
      br_refill(b);
      int sy = huff_decode_sym(d, b);
      if (sy < 0) return -1;
      if (sy >= SYM_RUN0) {
        uint32_t run = (uint32_t)(sy - SYM_RUN0) + 1;
        if (i + run > n) return -1;
        memset(r + i, 0, run * sizeof *r); i += run;
      } else {
        if (i >= n) return -1;
        uint32_t u = (uint32_t)sy;
        if (sy == SYM_ESC) u = br_bits(b, 16);
        r[i++] = unzigzag16(u);
      }
    }
    if (i != n) return -1;
  }

  #define STORE(idx, val) do { if (h.typesize == 1) output[idx] = (uint8_t)(val); else ((uint16_t *)output)[idx] = (uint16_t)(val); } while (0)
  /* Decode of fixed-point T values a[k] (T = (a + ofs) * step, less the dither curvature for step cs when cs > 0): through a table
     over a's range when that range is small next to n (always, for camera data: a dark tile spans a few hundred values), else
     per pixel. The table holds exactly the per-pixel values. */
  #define DECODE_FIX(a, ofs, step, cs) do {                                                                              \
    const vtdec_t *vd_ = (vi.t && (step) == 1.0 / FIX_SCALE) ? vtdec_get(vi.t, (ofs), (cs), (double)vmax) : NULL;        \
    if (vd_) { vtdec_apply(vd_, a, n, output, h.typesize); break; }                                                    \
    int32_t lo_ = INT32_MAX, hi_ = INT32_MIN;                                                                           \
    for (uint32_t k = 0; k < n; k++) { lo_ = a[k] < lo_ ? a[k] : lo_; hi_ = a[k] > hi_ ? a[k] : hi_; }                  \
    const uint64_t span_ = n ? (uint64_t)((int64_t)hi_ - lo_ + 1) : 0;                                                   \
    uint16_t *tab_ = (span_ && span_ <= n / 2) ? (uint16_t *)malloc(2 * span_) : NULL;                                   \
    for (uint64_t s_ = 0; tab_ && s_ < span_; s_++) {                                                                    \
      const double t_ = ((double)(lo_ + (int64_t)s_) + (ofs)) * (step);                                                  \
      double x_ = (double)rnd_d(vst_inv(&vi, t_) - ((cs) > 0 ? dither_curv(&vi, t_, (cs)) : 0.0));                        \
      tab_[s_] = (uint16_t)(x_ < 0 ? 0 : (x_ > vmax ? vmax : x_));                                                       \
    }                                                                                                                    \
    if (tab_) {                                                                                                          \
      if (h.typesize == 2) { uint16_t *o_ = (uint16_t *)output; for (uint32_t k = 0; k < n; k++) o_[k] = tab_[a[k] - lo_]; } \
      else for (uint32_t k = 0; k < n; k++) output[k] = (uint8_t)tab_[a[k] - lo_];                                       \
      free(tab_);                                                                                                        \
    } else for (uint32_t k = 0; k < n; k++) {                                                                            \
      const double t_ = ((double)a[k] + (ofs)) * (step);                                                                 \
      double x_ = (double)rnd_d(vst_inv(&vi, t_) - ((cs) > 0 ? dither_curv(&vi, t_, (cs)) : 0.0));                        \
      x_ = x_ < 0 ? 0 : (x_ > vmax ? vmax : x_); STORE(k, (uint32_t)x_);                                                 \
    }                                                                                                                    \
  } while (0)

  if (h.mode == 0) {
    uint16_t *u = (uint16_t *)r;                 /* residuals are the mod-2^16 stream */
    reconstruct_u16(u, n, w, pred, h.ph);
    if (h.typesize == 2) memcpy(output, u, 2 * (size_t)n);
    else for (uint32_t k = 0; k < n; k++) output[k] = (uint8_t)u[k];
  } else if (h.mode == 2) {
    uint16_t *u = (uint16_t *)r;
    int32_t *trec = (int32_t *)(buf + off_trec);
    vst_inv_t vi; vst_inv_init(&vi, h.conv, h.offset, h.rn);
    if (pixlist) {
      m2code mc = { dall, (int32_t)h.qi * (FIX_SCALE / 4), h.qi, (int32_t)rnd_d(vst_fwd(0, h.conv, h.offset, h.rn) / q), dither_base(q, h.conv, h.offset, h.rn) };
      if (mode2_decode_px(u, n, w, &B, &mc, dr, trec)) return -1;
    } else reconstruct_u16(u, n, w, pred, h.ph);
    if (dall) {
      const int32_t base = dither_base(q, h.conv, h.offset, h.rn);
      /* fixed-point value of each pixel, in the int32 scratch after the residuals (r is n int32, u its first half) */
      int32_t *fx = (int32_t *)(buf + off_f);
      for (uint32_t k = 0; k < n; k++) fx[k] = ((int32_t)u[k] << DITHER_SHIFT) - (int32_t)dr[k] + base;
      /* the encoder's remainder is uniform on 0..DITHER_SCALE-1, so its mean is (DITHER_SCALE - 1) / 2 */
      DECODE_FIX(fx, (DITHER_SCALE - 1) * 0.5, q / DITHER_SCALE, h.version >= 8 ? q : 0.0);
    } else {
      uint16_t *inv = (uint16_t *)(buf + off_inv);
      for (uint32_t sidx = 0; sidx < span; sidx++) {
        double x = (double)rnd_d(vst_inv(&vi, (double)(smin + (long)sidx) * q));
        inv[sidx] = (uint16_t)(x < 0 ? 0 : (x > vmax ? vmax : x));
      }
      for (uint32_t k = 0; k < n; k++) {
        uint32_t sidx = u[k]; if (sidx >= span) sidx = span - 1;
        STORE(k, inv[sidx]);
      }
    }
    if (pixlist) {                                       /* listed pixels: their own decoded values, in the encoder's order */
      uint32_t y = 0, k = 0;
      for (uint32_t base = 0; base < n; base += w) {
        for (uint32_t e = B.row_start[y]; e < B.row_start[y + 1]; e++, k++) {
          const double t = (double)trec[2 * k] * (1.0 / FIX_SCALE);
          double x = (double)rnd_d(vst_inv(&vi, t) - dither_curv(&vi, t, (double)trec[2 * k + 1] * (1.0 / FIX_SCALE)));
          x = x < 0 ? 0 : (x > vmax ? vmax : x); STORE(base + B.col[e], (uint32_t)x);
        }
        if (++y == B.plane_rows) y = 0;
      }
    }
  } else {
    int32_t *f = (int32_t *)(buf + off_f);
    int32_t *lst = (int32_t *)(buf + off_trec);
    if (pixlist) { if (mode1_decode_px(r, f, n, w, (int32_t)h.qi * (FIX_SCALE / 4), &B, dr, dall, lst)) return -1; }
    else mode1_decode(r, f, n, w, (int32_t)h.qi * (FIX_SCALE / 4), pred, h.ph, dr);
    vst_inv_t vi; vst_inv_init(&vi, h.conv, h.offset, h.rn);
    /* dithered: s - f = ((s - p + d) mod step) - step/2 averages -1/2 (fixed-point units, even step), so f is read 1/2 lower.
       The closed loop leaves |s - f| <= step/2 spread evenly across the step (exactly uniform with dither), so either way f is decoded
       less the curvature of the inverse for a uniform error of one step; without it the mean is high by conv step^2 / 48 */
    DECODE_FIX(f, dall ? -0.5 : 0.0, 1.0 / FIX_SCALE, q);
    if (pixlist && dr) {                                 /* dithered listed pixels: with their own step (rows after the first) */
      uint32_t y = B.plane_rows > 1 ? 1 : 0, k = 0;
      for (uint32_t base = w; base < n; base += w) {
        for (uint32_t e = B.row_start[y]; e < B.row_start[y + 1]; e++, k++) {
          const uint32_t i = base + B.col[e]; const double t = ((double)f[i] - ((lst[k] & 1) ? 0.0 : 0.5)) * (1.0 / FIX_SCALE);
          double x = (double)rnd_d(vst_inv(&vi, t) - dither_curv(&vi, t, (double)lst[k] * (1.0 / FIX_SCALE)));
          x = x < 0 ? 0 : (x > vmax ? vmax : x); STORE(i, (uint32_t)x);
        }
        if (++y == B.plane_rows) y = 0;
      }
    }
  }
  #undef STORE
  return (int)(n * h.typesize);
}

/* ------------------------------------------------------------------------ */
/* Registration                                                             */
/* ------------------------------------------------------------------------ */

int b4d_register(void) {
  blosc2_codec codec;
  memset(&codec, 0, sizeof codec);
  codec.compcode = B4D_CODEC_ID;
  codec.compname = (char *)B4D_CODEC_NAME;
  codec.complib  = B4D_CODEC_ID;
  codec.version  = B4D_VERSION;
  codec.encoder  = b4d_encoder;
  codec.decoder  = b4d_decoder;
  return blosc2_register_codec(&codec);
}

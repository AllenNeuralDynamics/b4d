/* b4d_original.c — the original B3D codec (Balazs et al. 2017, github.com/balintbalazs/B3D) as a block option.
 *
 * Bit-for-bit reimplementation of the original HDF5 filter's per-chunk pipeline, validated against the demo file
 * shipped with the original repository and against its win64 GPU binary (see b4d/validation/test_original_b3d.py):
 *
 *   lossless (q = 0): predictor7 on 24x24 tiles (int16 wrap) -> symbolize -> run-length (zero runs <= 255) + Huffman
 *   mode 1 (q > 0)  : float VST  v = (2 sqrt(max(x-offset,0)/conv + rn^2) - 2 rn) / q, closed-loop predictor7 with
 *                     round-half-away-from-zero quantisation of the residual -> symbolize -> RL + Huffman
 *   mode 2 (q > 0)  : VST, round to uint16, predictor7 -> symbolize -> RL + Huffman
 *
 * The entropy stage is cudaCompress's encodeRLHuff: two Huffman tables (compacted symbols, zero counts) designed on
 * the host with std::priority_queue and std::sort of Visual Studio 2013; both are emulated here exactly (tie-breaking
 * decides which symbol gets which code), followed by canonical codes, 32-bit big-endian-within-word bit packing,
 * 16-bit offset increments every 128 symbols and the table serialisation of HuffmanEncodeTable::writeToBitStream.
 *
 * The original GPU build used --use_fast_math: its VST (approximate division and square root, no fused add) differs
 * from IEEE single precision in ~17% of the 65536 table entries by one ulp, which flips a quantiser rounding about
 * once per million pixels. Pass the GPU's table (b4d.original.gpu_vst_table, the same expression compiled with
 * fast math on an NVIDIA device) through vst_table for byte identity in the lossy modes; with NULL an IEEE table is
 * used. The inverse VST of the GPU build fuses its last multiply-add (fmaf below); validated against its decoder.
 */
#include "b4d_blosc2.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF   /* the float stages must round exactly like the reference: no fused multiply-add */

#define ORIG_ZERO_COUNT_MAX 255u
#define ORIG_CODING_BLOCK   128u
#define ORIG_ISORT_MAX      32

/* ------------------------------------------------------------------------------------------------ bit stream */
typedef struct { uint32_t *w; size_t cap; uint32_t bitsize; uint32_t pos; } obw;   /* pos = bit position */

static int obw_reserve(obw *b, uint32_t bits) {
  size_t need = ((size_t)bits + 31) / 32;
  if (need <= b->cap) return 0;
  size_t ncap = b->cap ? b->cap : 1024;
  while (ncap < need) ncap *= 2;
  uint32_t *nw = (uint32_t *)realloc(b->w, ncap * sizeof(uint32_t));
  if (!nw) return -1;
  memset(nw + b->cap, 0, (ncap - b->cap) * sizeof(uint32_t));
  b->w = nw; b->cap = ncap;
  return 0;
}
static int obw_bits(obw *b, uint32_t data, uint32_t n) {                      /* BitStream::writeBits, n <= 32 */
  uint32_t end = b->pos + n;
  if (end > b->bitsize) { b->bitsize = end; if (obw_reserve(b, end)) return -1; }
  while (n > 0) {
    uint32_t bi = b->pos & 31, cur = 32 - bi; if (cur > n) cur = n;
    uint32_t mask = (cur == 32) ? 0xFFFFFFFFu : ((1u << cur) - 1u);
    uint32_t value = (data >> (n - cur)) & mask;
    uint32_t off = 32 - cur - bi;
    uint32_t *wp = &b->w[b->pos >> 5];
    *wp = (*wp & ~(mask << off)) | (value << off);
    b->pos += cur; n -= cur;
  }
  return 0;
}
static int obw_align32(obw *b) {
  uint32_t r = b->pos & 31;
  if (r) b->pos += 32 - r;
  if (b->pos > b->bitsize) { b->bitsize = b->pos; if (obw_reserve(b, b->pos)) return -1; }
  return 0;
}
static int obw_u32(obw *b, const uint32_t *v, uint32_t count) {               /* writeAligned<uint> */
  if (obw_align32(b)) return -1;
  if (!count) return 0;
  uint32_t end = b->pos + 32 * count;
  if (end > b->bitsize) { b->bitsize = end; if (obw_reserve(b, end)) return -1; }
  memcpy(b->w + (b->pos >> 5), v, (size_t)count * 4);
  b->pos = end;
  return 0;
}
static int obw_u16(obw *b, const uint16_t *v, uint32_t count) {               /* writeAligned<ushort> */
  if (obw_align32(b)) return -1;
  if (!count) return 0;
  uint32_t end = b->pos + 16 * count;
  if (end > b->bitsize) { b->bitsize = end; if (obw_reserve(b, end + 32)) return -1; }
  memcpy((uint8_t *)b->w + (b->pos >> 3), v, (size_t)count * 2);
  b->pos = end;
  return obw_align32(b);
}

typedef struct { const uint32_t *w; uint32_t nwords; uint32_t pos; uint32_t bitsize; } obr;

static int obr_bits(obr *b, uint32_t n, uint32_t *out) {                      /* readBits, n <= 32 */
  uint32_t v = 0;
  if (b->pos + n > b->bitsize) return -1;
  while (n > 0) {
    uint32_t bi = b->pos & 31, cur = 32 - bi; if (cur > n) cur = n;
    uint32_t mask = (cur == 32) ? 0xFFFFFFFFu : ((1u << cur) - 1u);
    uint32_t off = 32 - cur - bi;
    v = (v << cur) | ((b->w[b->pos >> 5] >> off) & mask);
    b->pos += cur; n -= cur;
  }
  *out = v;
  return 0;
}
static int obr_align32(obr *b) { uint32_t r = b->pos & 31; if (r) b->pos += 32 - r; return b->pos <= b->bitsize ? 0 : -1; }
static int obr_u32(obr *b, uint32_t *out) {
  if (obr_align32(b) || b->pos + 32 > b->bitsize) return -1;
  *out = b->w[b->pos >> 5]; b->pos += 32; return 0;
}

/* ------------------------------------------------------------------------------------------------ Huffman table */
typedef struct onode { uint32_t sym, prob; int length; struct onode *left, *right; } onode;

/* std::priority_queue<onode*, vector, InvComparer>: pred(a, b) == (b->prob < a->prob); VS2013 heap routines */
static inline int pq_pred(const onode *a, const onode *b) { return b->prob < a->prob; }
static void pq_push_heap(onode **a, int hole, int top, onode *val) {
  int idx = (hole - 1) / 2;
  while (top < hole && pq_pred(a[idx], val)) { a[hole] = a[idx]; hole = idx; idx = (hole - 1) / 2; }
  a[hole] = val;
}
static void pq_adjust_heap(onode **a, int hole, int bottom, onode *val) {
  int top = hole, idx = 2 * hole + 2;
  while (idx < bottom) {
    if (pq_pred(a[idx], a[idx - 1])) --idx;
    a[hole] = a[idx]; hole = idx; idx = 2 * idx + 2;
  }
  if (idx == bottom) { a[hole] = a[bottom - 1]; hole = bottom - 1; }
  pq_push_heap(a, hole, top, val);
}
static void pq_make_heap(onode **a, int n) { for (int hole = n / 2; hole > 0;) { --hole; pq_adjust_heap(a, hole, n, a[hole]); } }
static onode *pq_pop(onode **a, int *n) {                                      /* returns the former top */
  onode *topv = a[0];
  if (*n > 1) { onode *val = a[*n - 1]; (*n)--; pq_adjust_heap(a, 0, *n, val); }
  else (*n)--;
  return topv;
}
static void pq_push(onode **a, int *n, onode *val) { a[*n] = val; (*n)++; pq_push_heap(a, *n - 1, 0, val); }

/* VS2013 std::sort on onode* by codeword length (ties keep the implementation's order, which the reference relies on) */
static inline int ls_pred(const onode *a, const onode *b) { return a->length < b->length; }
#define LS_SWAP(i, j) do { onode *t_ = a[i]; a[i] = a[j]; a[j] = t_; } while (0)
static void ls_med3(onode **a, int f, int m, int l) {
  if (ls_pred(a[m], a[f])) LS_SWAP(m, f);
  if (ls_pred(a[l], a[m])) { LS_SWAP(l, m); if (ls_pred(a[m], a[f])) LS_SWAP(m, f); }
}
static void ls_median(onode **a, int f, int m, int l) {
  if (40 < l - f) {
    int step = (l - f + 1) / 8;
    ls_med3(a, f, f + step, f + 2 * step); ls_med3(a, m - step, m, m + step);
    ls_med3(a, l - 2 * step, l - step, l); ls_med3(a, f + step, m, l - step);
  } else ls_med3(a, f, m, l);
}
static void ls_partition(onode **a, int first, int last, int *out_pfirst, int *out_plast) {
  int mid = first + (last - first) / 2;
  ls_median(a, first, mid, last - 1);
  int pfirst = mid, plast = pfirst + 1;
  while (first < pfirst && !ls_pred(a[pfirst - 1], a[pfirst]) && !ls_pred(a[pfirst], a[pfirst - 1])) --pfirst;
  while (plast < last && !ls_pred(a[plast], a[pfirst]) && !ls_pred(a[pfirst], a[plast])) ++plast;
  int gfirst = plast, glast = pfirst;
  for (;;) {
    for (; gfirst < last;) {
      if (ls_pred(a[pfirst], a[gfirst])) ++gfirst;
      else if (ls_pred(a[gfirst], a[pfirst])) break;
      else { LS_SWAP(plast, gfirst); ++plast; ++gfirst; }
    }
    for (; first < glast;) {
      if (ls_pred(a[glast - 1], a[pfirst])) --glast;
      else if (ls_pred(a[pfirst], a[glast - 1])) break;
      else { --pfirst; LS_SWAP(pfirst, glast - 1); --glast; }
    }
    if (glast == first && gfirst == last) { *out_pfirst = pfirst; *out_plast = plast; return; }
    if (glast == first) {                       /* no room at bottom, rotate pivot upward */
      if (plast != gfirst) LS_SWAP(pfirst, plast);
      ++plast; LS_SWAP(pfirst, gfirst); ++pfirst; ++gfirst;
    } else if (gfirst == last) {                /* no room at top, rotate pivot downward */
      --glast; --pfirst;
      if (glast != pfirst) LS_SWAP(glast, pfirst);
      --plast; LS_SWAP(pfirst, plast);
    } else { LS_SWAP(gfirst, glast - 1); ++gfirst; --glast; }
  }
}
static void ls_insertion(onode **a, int first, int last) {
  for (int nxt = first + 1; nxt < last; nxt++) {
    onode *val = a[nxt];
    if (ls_pred(val, a[first])) { memmove(a + first + 1, a + first, (size_t)(nxt - first) * sizeof *a); a[first] = val; }
    else { int hole = nxt; while (ls_pred(val, a[hole - 1])) { a[hole] = a[hole - 1]; hole--; } a[hole] = val; }
  }
}
static void ls_heap_adjust(onode **a, int hole, int bottom, onode *val) {
  int top = hole, idx = 2 * hole + 2;
  while (idx < bottom) { if (ls_pred(a[idx], a[idx - 1])) --idx; a[hole] = a[idx]; hole = idx; idx = 2 * idx + 2; }
  if (idx == bottom) { a[hole] = a[bottom - 1]; hole = bottom - 1; }
  idx = (hole - 1) / 2;
  while (top < hole && ls_pred(a[idx], val)) { a[hole] = a[idx]; hole = idx; idx = (hole - 1) / 2; }
  a[hole] = val;
}
static void ls_heapsort(onode **a, int first, int last) {
  onode **s = a + first; int n = last - first;
  for (int hole = n / 2; hole > 0;) { --hole; ls_heap_adjust(s, hole, n, s[hole]); }
  for (int end = n; end > 1; end--) { onode *val = s[end - 1]; s[end - 1] = s[0]; ls_heap_adjust(s, 0, end - 1, val); }
}
static void ls_sort_rec(onode **a, int first, int last, int ideal) {
  int count;
  while (ORIG_ISORT_MAX < (count = last - first) && 0 < ideal) {
    int pfirst, plast; ls_partition(a, first, last, &pfirst, &plast);
    ideal /= 2; ideal += ideal / 2;
    if (pfirst - first < last - plast) { ls_sort_rec(a, first, pfirst, ideal); first = plast; }
    else { ls_sort_rec(a, plast, last, ideal); last = pfirst; }
  }
  if (ORIG_ISORT_MAX < count) ls_heapsort(a, first, last);
  else if (1 < count) ls_insertion(a, first, last);
}

typedef struct {
  uint32_t symbol_max;          /* reduce(max) over the stream (0 for an empty stream) */
  uint32_t nleaves;             /* distinct symbols actually used (== table symbols) */
  uint32_t maxlen;              /* longest codeword (0 if <= 1 distinct symbol) */
  uint32_t *count_per_len;      /* [maxlen] */
  uint32_t *symbols;            /* [nleaves] in table order (sorted by length, VS2013 order within a length) */
  uint32_t *code, *len;         /* [symbol_max + 1] encode table */
} otable;

static void otable_free(otable *t) { free(t->count_per_len); free(t->symbols); free(t->code); free(t->len); memset(t, 0, sizeof *t); }

/* HuffmanEncodeTable::design + build for one stream */
static int otable_design(otable *t, const uint16_t *stream, uint32_t count) {
  memset(t, 0, sizeof *t);
  if (count == 0) return 0;                           /* empty table: writes 0 lengths, 0 symbols, symbolBits 0 */
  uint32_t smax = 0;
  for (uint32_t i = 0; i < count; i++) if (stream[i] > smax) smax = stream[i];
  t->symbol_max = smax;
  uint32_t nbins = smax + 1;
  uint32_t *hist = (uint32_t *)calloc(nbins, sizeof(uint32_t));
  if (!hist) return -1;
  for (uint32_t i = 0; i < count; i++) hist[stream[i]]++;
  /* cudaCompress pads each symbol stream to a multiple of 8 elements with 0xFFFF before its GPU histogram; the
     kernel reads whole 16-byte words and counts a value only when it lies inside the bin range [0, symbolMax], so
     the padding is counted exactly when the largest symbol is 0xFFFF itself (full-range 16-bit residuals): its
     count is then inflated by 0..7, which changes the tree and therefore the codes. */
  if (smax == 0xFFFFu) hist[smax] += (8u - (count & 7u)) & 7u;
  uint32_t k = 0;
  for (uint32_t s = 0; s < nbins; s++) if (hist[s]) k++;
  onode *nodes = (onode *)calloc(2 * (size_t)k, sizeof(onode));
  onode **leaves = (onode **)malloc((size_t)k * sizeof(onode *));
  onode **heap = (onode **)malloc((size_t)k * sizeof(onode *));
  if (!nodes || !leaves || !heap) { free(hist); free(nodes); free(leaves); free(heap); return -1; }
  uint32_t next = 0;
  for (uint32_t s = 0; s < nbins; s++) if (hist[s]) { nodes[next].sym = s; nodes[next].prob = hist[s]; nodes[next].length = -1; leaves[next] = &nodes[next]; next++; }
  free(hist);
  int hn = (int)k;
  memcpy(heap, leaves, (size_t)k * sizeof(onode *));
  pq_make_heap(heap, hn);
  while (hn > 1) {
    onode *l = pq_pop(heap, &hn);
    onode *r = pq_pop(heap, &hn);
    onode *n = &nodes[next++];
    n->sym = 0xFFFFFFFFu; n->prob = l->prob + r->prob; n->length = -1; n->left = l; n->right = r;
    pq_push(heap, &hn, n);
  }
  onode *root = heap[0];
  /* assignCodewordLength: depth-first, left before right */
  uint32_t maxlen = 0;
  {
    size_t cap = 2 * (size_t)k + 2, sp = 0;
    onode **st = (onode **)malloc(cap * sizeof(onode *)); int *sl = (int *)malloc(cap * sizeof(int));
    if (!st || !sl) { free(st); free(sl); free(nodes); free(leaves); free(heap); return -1; }
    st[sp] = root; sl[sp] = 0; sp++;
    while (sp) {
      sp--; onode *n = st[sp]; int L = sl[sp];
      n->length = L; if ((uint32_t)L > maxlen) maxlen = (uint32_t)L;
      if (n->right) { st[sp] = n->right; sl[sp] = L + 1; sp++; }
      if (n->left) { st[sp] = n->left; sl[sp] = L + 1; sp++; }
    }
    free(st); free(sl);
  }
  ls_sort_rec(leaves, 0, (int)k, (int)k);
  t->nleaves = k; t->maxlen = maxlen;
  t->count_per_len = (uint32_t *)calloc(maxlen ? maxlen : 1, sizeof(uint32_t));
  t->symbols = (uint32_t *)malloc((size_t)k * sizeof(uint32_t));
  t->code = (uint32_t *)calloc(nbins, sizeof(uint32_t));
  t->len = (uint32_t *)malloc((size_t)nbins * sizeof(uint32_t));
  if (!t->count_per_len || !t->symbols || !t->code || !t->len) { free(nodes); free(leaves); free(heap); otable_free(t); return -1; }
  memset(t->len, 0xFF, (size_t)nbins * sizeof(uint32_t));
  for (uint32_t i = 0; i < k; i++) {
    const onode *n = leaves[i];
    if (n->length > 0) t->count_per_len[n->length - 1]++;
    t->symbols[i] = n->sym;
  }
  /* canonical codewords by table index */
  uint32_t code = 0, prevlen = maxlen ? 1 : 0, idx = 0;
  if (maxlen == 0) { t->code[t->symbols[0]] = 0; t->len[t->symbols[0]] = 0; }
  else {
    for (uint32_t L = 1; L <= maxlen; L++) {
      for (uint32_t c = 0; c < t->count_per_len[L - 1]; c++) {
        if (idx == 0) { code = 0; prevlen = L; }
        else { code = code + 1; code <<= (L - prevlen); prevlen = L; }
        t->code[t->symbols[idx]] = code; t->len[t->symbols[idx]] = L;
        idx++;
      }
    }
  }
  free(nodes); free(leaves); free(heap);
  return 0;
}

static uint32_t required_bits(uint32_t v) { uint32_t b = 0; while (v) { v >>= 1; b++; } return b; }

static int otable_write(obw *b, const otable *t) {                            /* HuffmanEncodeTable::writeToBitStream */
  if (obw_bits(b, t->maxlen, 5)) return -1;
  for (uint32_t i = 0; i < t->maxlen; i++) if (obw_bits(b, t->count_per_len[i], 16)) return -1;
  if (obw_bits(b, t->nleaves, 16)) return -1;
  uint32_t symbits = required_bits(t->symbol_max);
  if (obw_bits(b, symbits & 15u, 4)) return -1;           /* 16 is written as 0 (the original's documented quirk) */
  for (uint32_t i = 0; i < t->nleaves; i++) if (obw_bits(b, t->symbols[i], symbits)) return -1;
  return 0;
}

/* one Huffman-coded stream: table, [u32 bitsize][codewords][u16 offset increments] */
static int ostream_write(obw *b, const uint16_t *stream, uint32_t count) {
  otable t;
  if (otable_design(&t, stream, count)) return -1;
  int rc = -1;
  uint32_t noff = (count + ORIG_CODING_BLOCK - 1) / ORIG_CODING_BLOCK;
  uint16_t *inc = (uint16_t *)malloc(((size_t)noff + 1) * sizeof(uint16_t));
  if (!inc) goto done;
  if (otable_write(b, &t)) goto done;
  if (obw_align32(b)) goto done;
  uint32_t sizepos = b->pos >> 5, zero = 0;
  if (obw_u32(b, &zero, 1)) goto done;
  uint32_t start = b->pos, prev = 0;
  for (uint32_t i = 0; i < count; i++) {
    uint32_t s = stream[i];
    if ((i % ORIG_CODING_BLOCK) == 0) { uint32_t off = b->pos - start; inc[i / ORIG_CODING_BLOCK] = (uint16_t)(off - prev); prev = off; }
    if (t.maxlen && obw_bits(b, t.code[s], t.len[s])) goto done;
  }
  b->w[sizepos] = b->pos - start;
  if (obw_align32(b)) goto done;
  if (obw_u16(b, inc, noff)) goto done;
  rc = 0;
done:
  free(inc); otable_free(&t);
  return rc;
}

/* ------------------------------------------------------------------------------------------------ pixel stages */
static inline int16_t pred7_i16(const int16_t *in, uint32_t w, uint32_t x, uint32_t y, uint32_t tile) {
  uint32_t u = x % tile, v = y % tile;
  int32_t p;
  if (u == 0) p = (v > 0) ? in[(y - 1) * w + x] : 0;
  else if (v == 0) p = in[y * w + x - 1];
  else p = ((int32_t)in[y * w + x - 1] + (int32_t)in[(y - 1) * w + x]) >> 1;
  return (int16_t)((int32_t)in[y * w + x] - p);
}
static void predictor7(const int16_t *in, int16_t *out, uint32_t w, uint32_t h, uint32_t tile) {
  for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) out[y * w + x] = pred7_i16(in, w, x, y, tile);
}
static void unpredictor7(const int16_t *in, int16_t *out, uint32_t w, uint32_t h, uint32_t tile) {
  for (uint32_t y = 0; y < h; y++) {
    uint32_t v = y % tile;
    for (uint32_t x = 0; x < w; x++) {
      uint32_t u = x % tile; int32_t p;
      if (u == 0) p = (v > 0) ? out[(y - 1) * w + x] : 0;
      else if (v == 0) p = out[y * w + x - 1];
      else p = ((int32_t)out[y * w + x - 1] + (int32_t)out[(y - 1) * w + x]) >> 1;
      out[y * w + x] = (int16_t)((int32_t)in[y * w + x] + p);
    }
  }
}
static inline int orig_round(float eps) {                                     /* int Q = eps + (eps < 0 ? -0.5 : 0.5) */
  double d = (double)eps + (eps < 0.0f ? -0.5 : 0.5);
  return (int)d;
}
static void predictor7_wnll(const float *in, float *buf, int16_t *out, uint32_t w, uint32_t h, uint32_t tile) {
  for (uint32_t y = 0; y < h; y++) {
    uint32_t v = y % tile;
    for (uint32_t x = 0; x < w; x++) {
      uint32_t u = x % tile; float pred;
      if (u == 0) pred = (v > 0) ? buf[(y - 1) * w + x] : 0.0f;
      else if (v == 0) pred = buf[y * w + x - 1];
      else pred = (buf[y * w + x - 1] + buf[(y - 1) * w + x]) / 2.0f;
      float eps = (u == 0 && v == 0) ? in[y * w + x] : in[y * w + x] - pred;
      int Q = orig_round(eps);
      out[y * w + x] = (int16_t)Q;
      buf[y * w + x] = (u == 0 && v == 0) ? (float)Q : (float)Q + pred;
    }
  }
}
static void unpredictor7_wnll(const int16_t *in, float *out, uint32_t w, uint32_t h, uint32_t tile) {
  for (uint32_t y = 0; y < h; y++) {
    uint32_t v = y % tile;
    for (uint32_t x = 0; x < w; x++) {
      uint32_t u = x % tile; float q = (float)in[y * w + x];
      if (u == 0) out[y * w + x] = (v > 0) ? q + out[(y - 1) * w + x] : q;
      else if (v == 0) out[y * w + x] = q + out[y * w + x - 1];
      else out[y * w + x] = q + (out[y * w + x - 1] + out[(y - 1) * w + x]) / 2.0f;
    }
  }
}
static inline uint16_t orig_f2u(float f) {                                    /* out = in + (in < 0 ? -1 : 1) * 0.5 (double), stored to uint16 */
  double d = (double)f + (f < 0.0f ? -0.5 : 0.5);
  long long iv = (long long)d;
  return (uint16_t)iv;
}

int b4d_orig_vst_table(float *tab, float conversion, float offset, float read_noise, float q) {
  if (!tab || !(q > 0.0f) || !(conversion > 0.0f)) return -1;
  const float invq = 1.0f / q, s2 = read_noise * read_noise, two_s = 2.0f * read_noise;
  for (uint32_t x = 0; x < 65536; x++) {
    float d = (float)x - offset; if (d < 0.0f) d = 0.0f;
    float t = d / conversion + s2;
    float v = 2.0f * sqrtf(t) - two_s;
    tab[x] = v * invq;
  }
  return 0;
}

/* ------------------------------------------------------------------------------------------------ encoder */
int b4d_orig_encode(const uint16_t *img, uint32_t width, uint32_t height, int mode, float q, float conversion, float offset,
                    float read_noise, uint32_t tile, const float *vst_table, uint8_t **out, size_t *out_len) {
  if (!img || !out || !out_len || width == 0 || height == 0 || tile == 0) return -1;
  const size_t n = (size_t)width * height;
  if (n > 0x7FFFFFFFu) return -1;
  int16_t *res = (int16_t *)malloc(n * sizeof(int16_t));
  uint16_t *sym = (uint16_t *)malloc(n * sizeof(uint16_t));
  uint16_t *cs = (uint16_t *)malloc(n * sizeof(uint16_t));
  uint16_t *zc = (uint16_t *)malloc(n * sizeof(uint16_t));
  float *own_tab = NULL; float *fbuf = NULL; uint16_t *ubuf = NULL;
  int rc = -1;
  if (!res || !sym || !cs || !zc) goto done;
  if (!(q > 0.0f)) mode = 0;
  if (mode == 0) {
    predictor7((const int16_t *)img, res, width, height, tile);
  } else {
    const float *tab = vst_table ? vst_table : b4d_orig_find_vst_table(conversion, offset, read_noise, q);   /* explicit, registered (GPU-exact), else IEEE */
    if (!tab) { own_tab = (float *)malloc(65536 * sizeof(float)); if (!own_tab || b4d_orig_vst_table(own_tab, conversion, offset, read_noise, q)) goto done; tab = own_tab; }
    if (mode == 1) {
      float *fin = (float *)malloc(n * sizeof(float)); fbuf = (float *)calloc(n, sizeof(float));
      if (!fin || !fbuf) { free(fin); goto done; }
      for (size_t i = 0; i < n; i++) fin[i] = tab[img[i]];
      predictor7_wnll(fin, fbuf, res, width, height, tile);
      free(fin);
    } else {
      ubuf = (uint16_t *)malloc(n * sizeof(uint16_t)); if (!ubuf) goto done;
      for (size_t i = 0; i < n; i++) ubuf[i] = orig_f2u(tab[img[i]]);
      predictor7((const int16_t *)ubuf, res, width, height, tile);
    }
  }
  /* symbolize: 2|v| + (v >> 15)  (negatives map to odd = 2|v| - 1) */
  for (size_t i = 0; i < n; i++) { int32_t v = res[i]; sym[i] = (uint16_t)(2 * (v < 0 ? -v : v) + (v < 0 ? -1 : 0)); }
  /* run-length of zeros: (symbol, zeros before it), extra (0, 255) entries per 256 zeros, trailing zeros dropped */
  uint32_t nc = 0; size_t prev = (size_t)-1;
  for (size_t i = 0; i < n; i++) {
    if (!sym[i]) continue;
    size_t gap = i - prev - 1;
    uint32_t extra = (uint32_t)(gap / (ORIG_ZERO_COUNT_MAX + 1)), rem = (uint32_t)(gap % (ORIG_ZERO_COUNT_MAX + 1));
    for (uint32_t e = 0; e < extra; e++) { cs[nc] = 0; zc[nc] = (uint16_t)ORIG_ZERO_COUNT_MAX; nc++; }
    cs[nc] = sym[i]; zc[nc] = (uint16_t)rem; nc++;
    prev = i;
  }
  obw b; memset(&b, 0, sizeof b);
  if (obw_u32(&b, &nc, 1)) goto done;
  if (ostream_write(&b, cs, nc)) { free(b.w); goto done; }
  if (ostream_write(&b, zc, nc)) { free(b.w); goto done; }
  *out = (uint8_t *)b.w; *out_len = (size_t)((b.bitsize + 31) / 32) * 4;
  rc = 0;
done:
  free(res); free(sym); free(cs); free(zc); free(own_tab); free(fbuf); free(ubuf);
  return rc;
}

/* ------------------------------------------------------------------------------------------------ decoder */
typedef struct { int first[32], cmin[32], cmax[32]; uint32_t nsym; uint16_t *syms; uint32_t maxlen; } odtable;

static int odtable_read(obr *b, odtable *t) {
  uint32_t nlen, cnt, symbits;
  memset(t, 0, sizeof *t);
  if (obr_bits(b, 5, &nlen) || nlen > 32) return -1;
  uint32_t counts[32]; uint32_t total = 0;
  for (uint32_t i = 0; i < nlen; i++) { if (obr_bits(b, 16, &cnt)) return -1; counts[i] = cnt; total += cnt; }
  if (obr_bits(b, 16, &t->nsym) || obr_bits(b, 4, &symbits)) return -1;
  if (symbits == 0 && t->nsym > 1) symbits = 16;
  if (nlen == 0 && t->nsym > 0) total++;               /* single codeword of length 0; an empty stream has no codeword at all */
  if (total != t->nsym) return -1;
  t->syms = (uint16_t *)malloc(((size_t)t->nsym + 1) * sizeof(uint16_t));
  if (!t->syms) return -1;
  for (uint32_t i = 0; i < t->nsym; i++) { uint32_t s; if (obr_bits(b, symbits, &s)) return -1; t->syms[i] = (uint16_t)s; }
  t->maxlen = nlen;
  for (int i = 0; i < 32; i++) t->first[i] = t->cmin[i] = t->cmax[i] = -1;
  if (nlen) {                                          /* HuffmanDecodeTable::build */
    int code = 0, entry = 0, prevlen = 0;
    for (uint32_t L = 1; L <= nlen; L++) {
      if (!counts[L - 1]) continue;
      if (prevlen == 0) code = 0; else { code += 1; code <<= (int)(L - prevlen); }
      t->first[L - 1] = entry; t->cmin[L - 1] = code;
      code += (int)counts[L - 1] - 1; entry += (int)counts[L - 1];
      t->cmax[L - 1] = code; prevlen = (int)L;
    }
  }
  return 0;
}
static int ostream_read(obr *b, uint16_t *out, uint32_t count) {
  odtable t; uint32_t bitsize;
  if (odtable_read(b, &t)) { free(t.syms); return -1; }
  if (obr_u32(b, &bitsize)) { free(t.syms); return -1; }
  uint32_t start = b->pos;
  if (t.nsym == 0) { if (count) { free(t.syms); return -1; } }
  else if (t.maxlen == 0) { for (uint32_t i = 0; i < count; i++) out[i] = t.syms[0]; }
  else {
    for (uint32_t i = 0; i < count; i++) {
      /* peek up to 32 bits (zero beyond the end) */
      uint32_t wi = b->pos >> 5, bi = b->pos & 31;
      uint64_t win = ((uint64_t)(wi < b->nwords ? b->w[wi] : 0) << 32) | (uint64_t)(wi + 1 < b->nwords ? b->w[wi + 1] : 0);
      uint32_t peek = (uint32_t)((win >> (32 - bi)) & 0xFFFFFFFFu);
      uint32_t L = 1;
      for (;; L++) {
        if (L > t.maxlen) { free(t.syms); return -1; }
        int code = (int)(peek >> (32 - L));
        if (code <= t.cmax[L - 1]) {
          uint32_t idx = (uint32_t)(t.first[L - 1] + code - t.cmin[L - 1]);
          if (idx >= t.nsym) { free(t.syms); return -1; }
          out[i] = t.syms[idx]; break;
        }
      }
      b->pos += L;
      if (b->pos > b->bitsize) { free(t.syms); return -1; }
    }
  }
  if (b->pos - start != bitsize) { free(t.syms); return -1; }
  free(t.syms);
  if (obr_align32(b)) return -1;
  uint32_t noff = (count + ORIG_CODING_BLOCK - 1) / ORIG_CODING_BLOCK;
  b->pos += 16 * noff;                                 /* skipAligned<ushort>(offsets) */
  return obr_align32(b);
}

int b4d_orig_decode(const uint8_t *payload, size_t len, uint32_t width, uint32_t height, int mode, float q, float conversion, float offset,
                    float read_noise, uint32_t tile, uint16_t *out) {
  if (!payload || !out || width == 0 || height == 0 || tile == 0 || (len & 3)) return -1;
  const size_t n = (size_t)width * height;
  obr b; b.w = (const uint32_t *)payload; b.nwords = (uint32_t)(len / 4); b.pos = 0; b.bitsize = (uint32_t)(len * 8);
  uint32_t nc;
  if (obr_u32(&b, &nc) || nc > n) return -1;
  uint16_t *cs = (uint16_t *)malloc(((size_t)nc + 1) * sizeof(uint16_t)), *zc = (uint16_t *)malloc(((size_t)nc + 1) * sizeof(uint16_t));
  int16_t *res = (int16_t *)calloc(n, sizeof(int16_t));
  int rc = -1;
  if (!cs || !zc || !res) goto done;
  if (ostream_read(&b, cs, nc) || ostream_read(&b, zc, nc)) goto done;
  {
    size_t idx = 0;
    for (uint32_t i = 0; i < nc; i++) {
      idx += zc[i];
      if (idx >= n) goto done;
      uint32_t s = cs[i]; int neg = (int)(s & 1u);
      res[idx] = (int16_t)((1 - 2 * neg) * (int)((s + (uint32_t)neg) / 2u));
      idx++;
    }
  }
  if (!(q > 0.0f)) mode = 0;
  if (mode == 0) {
    unpredictor7(res, (int16_t *)out, width, height, tile);
  } else {
    const float s2 = read_noise * read_noise, two_s = 2.0f * read_noise;
    if (mode == 2) {
      int16_t *u = (int16_t *)malloc(n * sizeof(int16_t)); if (!u) goto done;
      unpredictor7(res, u, width, height, tile);
      for (size_t i = 0; i < n; i++) {
        float D = (float)(uint16_t)u[i] * q;
        D = D + two_s;
        float v = (D >= two_s) ? fmaf((D * D) * 0.25f - s2, conversion, offset) : offset;   /* the GPU build fused the last multiply-add */
        out[i] = orig_f2u(v);
      }
      free(u);
    } else {
      float *f = (float *)calloc(n, sizeof(float)); if (!f) goto done;
      unpredictor7_wnll(res, f, width, height, tile);
      for (size_t i = 0; i < n; i++) {
        float D = f[i] * q;
        D = D + two_s;
        float v = (D >= two_s) ? fmaf((D * D) * 0.25f - s2, conversion, offset) : offset;
        out[i] = orig_f2u(v);
      }
      free(f);
    }
  }
  rc = 0;
done:
  free(cs); free(zc); free(res);
  return rc;
}

void b4d_orig_free(void *p) { free(p); }

/* process-wide registry of externally computed VST tables (the original GPU build's fast-math values) */
#define ORIG_TAB_SLOTS 8
typedef struct { int used; float conv, offset, rn, q; float *tab; } orig_tab_slot;
static orig_tab_slot g_orig_tabs[ORIG_TAB_SLOTS];
#if defined(_WIN32)
#include <windows.h>
static CRITICAL_SECTION g_orig_cs; static volatile long g_orig_cs_init = 0;
static void orig_lock(void) { if (!g_orig_cs_init) { static volatile long once = 0; if (InterlockedCompareExchange(&once, 1, 0) == 0) { InitializeCriticalSection(&g_orig_cs); g_orig_cs_init = 1; } else while (!g_orig_cs_init) Sleep(0); } EnterCriticalSection(&g_orig_cs); }
static void orig_unlock(void) { LeaveCriticalSection(&g_orig_cs); }
#else
#include <pthread.h>
static pthread_mutex_t g_orig_mx = PTHREAD_MUTEX_INITIALIZER;
static void orig_lock(void) { pthread_mutex_lock(&g_orig_mx); }
static void orig_unlock(void) { pthread_mutex_unlock(&g_orig_mx); }
#endif
int b4d_orig_register_vst_table(float conversion, float offset, float read_noise, float q, const float *tab) {
  orig_lock();
  int slot = -1;
  for (int i = 0; i < ORIG_TAB_SLOTS; i++) {
    orig_tab_slot *s = &g_orig_tabs[i];
    if (s->used && s->conv == conversion && s->offset == offset && s->rn == read_noise && s->q == q) { slot = i; break; }
  }
  if (!tab) { if (slot >= 0) { free(g_orig_tabs[slot].tab); memset(&g_orig_tabs[slot], 0, sizeof g_orig_tabs[slot]); } orig_unlock(); return 0; }
  if (slot < 0) for (int i = 0; i < ORIG_TAB_SLOTS; i++) if (!g_orig_tabs[i].used) { slot = i; break; }
  if (slot < 0) { orig_unlock(); return -1; }
  float *copy = (float *)malloc(65536 * sizeof(float));
  if (!copy) { orig_unlock(); return -1; }
  memcpy(copy, tab, 65536 * sizeof(float));
  free(g_orig_tabs[slot].tab);
  g_orig_tabs[slot].used = 1; g_orig_tabs[slot].conv = conversion; g_orig_tabs[slot].offset = offset; g_orig_tabs[slot].rn = read_noise; g_orig_tabs[slot].q = q; g_orig_tabs[slot].tab = copy;
  orig_unlock();
  return 0;
}
const float *b4d_orig_find_vst_table(float conversion, float offset, float read_noise, float q) {
  const float *r = NULL;
  orig_lock();
  for (int i = 0; i < ORIG_TAB_SLOTS; i++) {
    const orig_tab_slot *s = &g_orig_tabs[i];
    if (s->used && s->conv == conversion && s->offset == offset && s->rn == read_noise && s->q == q) { r = s->tab; break; }
  }
  orig_unlock();
  return r;   /* tables are never freed while registered: safe to use after unlock */
}

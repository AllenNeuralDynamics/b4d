/*
 * b4d_test.c — unit tests for the B³D Blosc2 codec.
 * Includes the implementation directly so internal kernels can be tested.
 */
#include "b4d_blosc2.c"
#include <stdio.h>
#include <inttypes.h>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } else g_pass++; } while (0)
#define SECTION(name) printf("[%s]\n", name)

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 11); }
static double rndf(void) { return (rnd() & 0xFFFFFF) / 16777216.0; }

/* ---------------------------------------------------------------- zigzag */
static void test_zigzag(void) {
  SECTION("zigzag16");
  int ok = 1;
  for (int v = -32768; v <= 32767; v++) {
    uint32_t u = zigzag16((int16_t)v);
    if (u > 65535 || unzigzag16(u) != (int16_t)v) { ok = 0; break; }
  }
  CHECK(ok, "zigzag16 not a bijection on int16");
  CHECK(zigzag16(0) == 0 && zigzag16(-1) == 1 && zigzag16(1) == 2 && zigzag16(-2) == 3, "zigzag ordering");
}

/* ---------------------------------------------------------------- bit I/O */
static void test_bitio(void) {
  SECTION("bit I/O");
  enum { N = 20000 };
  static uint32_t vals[N]; static int lens[N];
  uint8_t *buf = malloc(N * 4 + 16);
  bitw w; bw_init(&w, buf);
  uint64_t total = 0;
  for (int i = 0; i < N; i++) {
    lens[i] = 1 + (int)(rnd() % 31);
    vals[i] = rnd() & ((lens[i] == 32) ? 0xFFFFFFFFu : ((1u << lens[i]) - 1));
    bw_put(&w, vals[i], lens[i]); total += lens[i];
  }
  bw_flush(&w);
  CHECK((size_t)(w.p - buf) == (total + 7) / 8, "writer byte count exact (%zu vs %" PRIu64 ")", (size_t)(w.p - buf), (total + 7) / 8);
  bitr r; br_init(&r, buf, (size_t)(w.p - buf));
  int ok = 1;
  for (int i = 0; i < N && ok; i++) { br_refill(&r); if (br_bits(&r, lens[i]) != vals[i]) ok = 0; }
  CHECK(ok, "bit round trip");
  /* unary helper */
  bw_init(&w, buf); bw_put(&w, 0x7Fu << 1, 8); bw_put(&w, 0, 1); bw_put(&w, 0x7FFFFFFFu, 31); bw_put(&w, 0, 1); bw_flush(&w);
  br_init(&r, buf, (size_t)(w.p - buf)); br_refill(&r);
  CHECK(br_unary(&r) == 7, "unary 7"); br_skip(&r, 8);
  CHECK(br_unary(&r) == 0, "unary 0"); br_skip(&r, 1);
  CHECK(br_unary(&r) == RICE_UMAX, "unary capped at %d", RICE_UMAX);
  free(buf);
}

/* ---------------------------------------------------------------- Huffman */
static void huff_case(const char *name, uint32_t *freq) {
  uint8_t len[HUFF_NSYM]; uint16_t code[HUFF_NSYM] = {0};
  uint32_t f2[HUFF_NSYM]; memcpy(f2, freq, sizeof f2);
  huff_build_lengths(f2, len);
  huff_codes(len, code);
  int maxl = 0, nsym = 0; double kraft = 0;
  for (int s = 0; s < HUFF_NSYM; s++) {
    if (freq[s] && !len[s]) { CHECK(0, "%s: symbol %d has freq but no code", name, s); return; }
    if (!freq[s] && len[s]) { CHECK(0, "%s: symbol %d has code but no freq", name, s); return; }
    if (len[s]) { nsym++; if (len[s] > maxl) maxl = len[s]; kraft += ldexp(1.0, -len[s]); }
  }
  CHECK(maxl <= HUFF_MAXLEN, "%s: max length %d", name, maxl);
  CHECK(kraft <= 1.0 + 1e-12, "%s: Kraft sum %f", name, kraft);
  /* encode/decode a random stream drawn from freq */
  enum { M = 5000 };
  static uint16_t syms[M];
  uint32_t tot = 0; for (int s = 0; s < HUFF_NSYM; s++) tot += freq[s];
  for (int i = 0; i < M; i++) {
    uint32_t t = rnd() % tot, acc = 0; int s = 0;
    for (; s < HUFF_NSYM; s++) { acc += freq[s]; if (t < acc) break; }
    syms[i] = (uint16_t)s;
  }
  uint8_t *buf = malloc(M * 4 + 16); bitw w; bw_init(&w, buf);
  for (int i = 0; i < M; i++) bw_put(&w, code[syms[i]], len[syms[i]]);
  bw_flush(&w);
  huff_dec *d = malloc(sizeof *d); huff_dec_init(d, len);
  bitr r; br_init(&r, buf, (size_t)(w.p - buf)); int ok = 1;
  for (int i = 0; i < M && ok; i++) { br_refill(&r); if (huff_decode_sym(d, &r) != syms[i]) ok = 0; }
  CHECK(ok, "%s: decode mismatch (nsym=%d maxlen=%d)", name, nsym, maxl);
  free(buf); free(d);
}
static void test_huffman(void) {
  SECTION("Huffman");
  uint32_t f[HUFF_NSYM];
  memset(f, 0, sizeof f); f[7] = 100; huff_case("single symbol", f);
  memset(f, 0, sizeof f); f[0] = 50; f[1] = 50; huff_case("two symbols", f);
  for (int s = 0; s < HUFF_NSYM; s++) f[s] = 1; huff_case("uniform 512", f);
  /* Laplacian-ish: forces long codes -> exercises length limiting */
  memset(f, 0, sizeof f); for (int s = 0; s < 60; s++) f[s] = (uint32_t)(1e9 * pow(0.5, s)) + 1; f[SYM_RUN0] = 1; f[SYM_RUN0 + 255] = 1;
  huff_case("geometric (length-limited)", f);
  /* Fibonacci frequencies produce a maximally deep tree */
  memset(f, 0, sizeof f); { uint32_t a = 1, b = 1; for (int s = 0; s < 40; s++) { f[s] = a; uint32_t c = a + b; a = b; b = c; } }
  huff_case("fibonacci", f);
  for (int s = 0; s < HUFF_NSYM; s++) f[s] = rnd() % 1000; f[3] = 0; f[400] = 0; huff_case("random", f);
}

/* ---------------------------------------------------------------- Rice */
static void test_rice(void) {
  SECTION("Rice");
  for (int trial = 0; trial < 300; trial++) {
    int16_t g[RICE_G], back[RICE_G];
    uint32_t len = (trial % 7 == 0) ? 1 + rnd() % RICE_G : RICE_G;
    int kind = trial % 7;
    for (uint32_t i = 0; i < len; i++) {
      int v;
      if (kind == 0) v = 0;
      else if (kind == 1) v = (int)(rnd() % 3) - 1;
      else if (kind == 2) v = (int)(rnd() % 65536) - 32768;             /* full range -> escapes */
      else if (kind == 3) v = (int)(-log(rndf() + 1e-9) * 40) * ((rnd() & 1) ? 1 : -1);
      else if (kind == 4) v = (i == 5) ? 32767 : ((i == 6) ? -32768 : (int)(rnd() % 5) - 2); /* outliers among small */
      else if (kind == 5) v = (rnd() % 16 == 0) ? ((rnd() & 1) ? 1 : -1) : 0;                /* sparse +-1 (q=4 sigma regime) */
      else v = (i == 3) ? -20000 : (i == 17 ? 300 : 0);                                     /* sparse with large values */
      g[i] = (int16_t)v;
    }
    uint8_t k; uint32_t uz[RICE_G]; uint32_t planned = rice_group_plan(g, len, &k, uz);   /* (u out-array added in 466e191; test updated) */
    uint8_t buf[RICE_G * 8 + 16]; bitw w; bw_init(&w, buf);
    if (k == 15) rice_zero_run_emit(&w, 1); else rice_group_emit(&w, g, len, k);
    bw_flush(&w);
    size_t nb = (size_t)(w.p - buf);
    if (k == 15) planned += RICE_ZRUNBITS;                 /* run count is added by the encoder loop */
    if (nb != (planned + 7) / 8) { CHECK(0, "trial %d: planned %u bits (%u B) but wrote %zu B", trial, planned, (planned + 7) / 8, nb); continue; }
    bitr r; br_init(&r, buf, nb); uint32_t pend = 0;
    CHECK(rice_group_read(&r, back, len, B4D_VERSION, &pend) == 0, "trial %d read ok", trial);
    CHECK(memcmp(g, back, len * sizeof *g) == 0, "trial %d (kind %d, len %u, k %d) round trip", trial, kind, len, k);
    if (kind == 0) CHECK(k == 15 && planned == RICE_KBITS + RICE_ZRUNBITS, "all-zero group costs %d bits", RICE_KBITS + RICE_ZRUNBITS);
    if (kind == 5 && len == RICE_G) { uint32_t nnz = 0; for (uint32_t i = 0; i < len; i++) nnz += g[i] != 0;
      if (nnz && nnz <= 3) CHECK(k == RICE_SPARSE, "trial %d: %u non-zeros should pick sparse mode (got k=%d)", trial, nnz, k); }
  }
  /* v1 compatibility: a k=14 field in a version-1 stream is a plain Rice parameter */
  { int16_t g[RICE_G], back[RICE_G]; for (int i = 0; i < RICE_G; i++) g[i] = (int16_t)(20000 + i);
    uint8_t buf[RICE_G * 8 + 16]; bitw w; bw_init(&w, buf);
    bw_put(&w, 14, RICE_KBITS); for (int i = 0; i < RICE_G; i++) rice_put(&w, zigzag16(g[i]), 14); bw_flush(&w);
    bitr r; br_init(&r, buf, (size_t)(w.p - buf)); uint32_t pend = 0; rice_group_read(&r, back, RICE_G, 1, &pend);
    CHECK(memcmp(g, back, sizeof g) == 0, "v1 k=14 group decodes as Rice"); }
  CHECK(1, "rice done");
}

/* ---------------------------------------------------------------- rANS */
static void test_ans(void) {
  SECTION("rANS");
  /* normalisation: sums to M, every present symbol >= 1, deterministic */
  for (int trial = 0; trial < 200; trial++) {
    uint32_t count[HUFF_NSYM] = {0}; int kind = trial % 4;
    for (int s = 0; s < HUFF_NSYM; s++)
      count[s] = kind == 0 ? (s < 3 ? 1000 : 0) : kind == 1 ? (uint32_t)(1e6 * pow(0.7, s)) : kind == 2 ? (s == 200 ? 1 : (s == 0 ? 5000000 : 0)) : rnd() % 50;
    uint16_t freq[HUFF_NSYM]; ans_normalise(count, freq);
    uint32_t sum = 0; int ok = 1;
    for (int s = 0; s < HUFF_NSYM; s++) { sum += freq[s]; if ((count[s] != 0) != (freq[s] != 0)) ok = 0; }
    CHECK(sum == ANS_M && ok, "trial %d: normalised sum %u", trial, sum);
  }
  /* table serialise / parse */
  { ans_ctx_enc enc[ANS_NCTX]; ans_ctx_dec dec[ANS_NCTX]; uint16_t *symbuf = malloc(ANS_NCTX * ANS_M * 2);
    for (int c = 0; c < ANS_NCTX; c++) {
      uint32_t count[HUFF_NSYM] = {0}; if (c != 1) for (int s = 0; s < HUFF_NSYM; s += 1 + c) count[s] = rnd() % 100 + 1;
      ans_normalise(count, enc[c].freq); ans_starts(enc[c].start, enc[c].freq); enc[c].present = c != 1;
    }
    uint8_t buf[ANS_TABLE_MAX + 16]; uint32_t len = ans_tables_write(buf, enc);
    int rd = ans_tables_read(buf, len, dec, symbuf);
    CHECK(rd == (int)len, "tables parse length %d vs %u", rd, len);
    int ok = 1; for (int c = 0; c < ANS_NCTX; c++) { if (dec[c].present != enc[c].present) ok = 0; if (enc[c].present && memcmp(dec[c].freq, enc[c].freq, sizeof enc[c].freq)) ok = 0; }
    CHECK(ok, "tables round trip");
    CHECK(ans_tables_read(buf, len - 3, dec, symbuf) < 0, "truncated tables rejected");
    free(symbuf); }
  /* residual stream round trip through symbolise -> encode -> decode, all contexts, escapes, runs */
  for (int trial = 0; trial < 60; trial++) {
    uint32_t n = 1 + rnd() % 5000, w = (trial % 3 == 0) ? 0 : 1 + rnd() % 64;
    int16_t *r = malloc(2 * n), *back = malloc(2 * n);
    for (uint32_t i = 0; i < n; i++) {
      int kind = trial % 5; int v;
      if (kind == 0) v = 0; else if (kind == 1) v = (rnd() % 8 == 0) ? (int)(rnd() % 7) - 3 : 0;
      else if (kind == 2) v = (int)(rnd() % 65536) - 32768; else if (kind == 3) v = (int)(-log(rndf() + 1e-9) * 3) * ((rnd() & 1) ? 1 : -1);
      else v = (i % 97 == 0) ? 30000 : (int)(rnd() % 3) - 1;
      r[i] = (int16_t)v;
    }
    ans_rec *rec = malloc(n * sizeof *rec); uint32_t count[ANS_NCTX * HUFF_NSYM];
    uint32_t nsym = ans_symbolise(r, n, w, rec, count);
    ans_ctx_enc enc[ANS_NCTX];
    for (int c = 0; c < ANS_NCTX; c++) { ans_normalise(count + c * HUFF_NSYM, enc[c].freq); ans_starts(enc[c].start, enc[c].freq); enc[c].present = 0; for (int s = 0; s < HUFF_NSYM; s++) enc[c].present |= enc[c].freq[s] != 0; ans_ctx_enc_prepare(&enc[c]); }
    size_t scap = 4 * (n * 3 + 4096); uint8_t *scr = malloc(scap), *out = malloc(scap); uint32_t sb[NSTREAM];
    int total = ans_encode_streams(rec, nsym, enc, scr, scap, out, scap, sb);
    CHECK(total > 0, "trial %d encode ok (%d)", trial, total);
    if (total > 0) {
      uint8_t tb[ANS_TABLE_MAX + 16]; uint32_t tl = ans_tables_write(tb, enc);
      ans_ctx_dec dec[ANS_NCTX]; uint16_t *symbuf = malloc(ANS_NCTX * ANS_M * 2); ans_tables_read(tb, tl, dec, symbuf);
      const uint8_t *streams[NSTREAM]; const uint8_t *p = out; for (int k = 0; k < NSTREAM; k++) { streams[k] = p; p += sb[k]; }
      int rc = ans_decode_streams(NULL, streams, sb, nsym, dec, back, n, w);
      CHECK(rc == 0 && memcmp(r, back, 2 * n) == 0, "trial %d (n=%u w=%u) residual round trip", trial, n, w);
      free(symbuf);
    }
    free(r); free(back); free(rec); free(scr); free(out);
  }
}

/* ---------------------------------------------------------------- predictors */
static void test_predictors(void) {
  SECTION("residual/reconstruct");
  uint32_t ns[] = {1, 2, 3, 31, 32, 33, 100, 1000, 4097};
  uint32_t ws[] = {0, 1, 7, 32, 100, 5000};
  uint32_t phs[] = {0, 1, 2, 3, 8, 1000};
  for (size_t a = 0; a < sizeof ns / sizeof *ns; a++) for (size_t b = 0; b < sizeof ws / sizeof *ws; b++) for (int pred = 0; pred < 3; pred++)
  for (size_t c = 0; c < (pred == 2 ? sizeof phs / sizeof *phs : 1); c++) {
    uint32_t n = ns[a], w = ws[b], ph = phs[c];
    uint16_t *v = malloc(2 * n), *u = malloc(2 * n);
    for (uint32_t i = 0; i < n; i++) v[i] = (i % 17 == 0) ? 65535 : (i % 23 == 0) ? 0 : (uint16_t)rnd();
    residual_u16(v, (int16_t *)u, n, w, pred, ph);
    reconstruct_u16(u, n, w, pred, ph);
    CHECK(memcmp(v, u, 2 * n) == 0, "n=%u w=%u pred=%d ph=%u bijection", n, w, pred, ph);
    free(v); free(u);
  }
  /* above-only predictor should give exactly cur - up for rows >= 1 */
  { uint16_t v[8] = {1, 2, 3, 4, 10, 20, 30, 40}; int16_t r[8];
    residual_u16(v, r, 8, 4, 1, 0);
    CHECK(r[4] == 9 && r[5] == 18 && r[6] == 27 && r[7] == 36 && r[1] == 1, "above-only values"); }
  /* mean3d: 2 planes of 3 rows x 4 columns; check a few predictions by hand */
  { uint16_t v[24] = { 10, 20, 30, 40,   50, 60, 70, 80,   90, 100, 110, 120,      /* plane 0 */
                       11, 21, 31, 41,   51, 61, 71, 81,   91, 101, 111, 121 };    /* plane 1 */
    int16_t r[24]; residual_u16(v, r, 24, 4, 2, 3);
    /* plane 0 row 1 x=1: taps = above row (10,20,30) -> 20 ; residual 60-20 = 40 */
    CHECK(r[5] == 40, "mean3d in-plane row: got %d", r[5]);
    /* plane 0 row 0 is the left predictor: r[1] = 20-10 */
    CHECK(r[0] == 10 && r[1] == 10, "mean3d first row uses left");
    /* plane 1 row 1 x=1 (index 17): above row (11,21,31) + prev plane rows 0..2 cols 0..2 = (10,20,30,50,60,70,90,100,110)
       sum = 63 + 540 = 603, 12 taps -> (603+6)/12 = 50 ; value 61 -> residual 11 */
    CHECK(r[17] == 11, "mean3d full 12-tap: got %d", r[17]);
    /* plane 1 row 0 x=0 (index 12): no above row; prev plane rows 0..1 cols 0..1 = 10,20,50,60 -> (140+2)/4 = 35 ; 11-35 = -24 */
    CHECK(r[12] == -24, "mean3d prev-plane-only corner: got %d", r[12]);
    uint16_t u[24]; memcpy(u, r, sizeof u); reconstruct_u16(u, 24, 4, 2, 3);
    CHECK(memcmp(u, v, sizeof v) == 0, "mean3d hand case round-trips"); }
}

/* ---------------------------------------------------------------- mode 1 closed loop */
static void test_mode1_loop(void) {
  SECTION("mode 1 closed loop");
  uint32_t n = 4000, w = 64;
  int32_t *s = malloc(4 * n), *f = malloc(4 * n), *f2 = malloc(4 * n); int16_t *r = malloc(2 * n);
  for (int above = 0; above < 3; above++) for (int qi = 1; qi <= 16; qi *= 2) {
    int32_t qfix = qi * (FIX_SCALE / 4);
    for (uint32_t i = 0; i < n; i++) s[i] = (int32_t)(rnd() % (700 * FIX_SCALE));   /* 0..700 sigma */
    CHECK(mode1_encode(s, f, r, n, w, qfix, above, 5), "encode ok (above=%d qi=%d)", above, qi);
    int32_t maxdev = 0;
    for (uint32_t i = 0; i < n; i++) { int32_t d = abs(f[i] - s[i]); if (d > maxdev) maxdev = d; }
    CHECK(maxdev <= (qfix + 1) / 2, "above=%d qi=%d: max |recon-s| = %d, bound %d", above, qi, maxdev, (qfix + 1) / 2);
    mode1_decode(r, f2, n, w, qfix, above, 5);
    CHECK(memcmp(f, f2, 4 * n) == 0, "above=%d qi=%d: decoder tracks encoder state", above, qi);
  }
  /* overflow detection: a jump of > 32767 steps */
  for (uint32_t i = 0; i < n; i++) s[i] = (i & 1) ? 0x7FFFFFF : 0;
  CHECK(!mode1_encode(s, f, r, n, w, 64, 0, 0), "int16 overflow detected");
  free(s); free(f); free(f2); free(r);
}

/* ---------------------------------------------------------------- full codec (direct) */
static void fill_image(uint16_t *img, uint32_t n, uint32_t w, int kind, float conv, float off) {
  for (uint32_t i = 0; i < n; i++) {
    double e;
    uint32_t x = w ? i % w : i, y = w ? i / w : 0;
    switch (kind) {
      case 0: e = 50; break;                                                    /* constant */
      case 1: e = 20 + 0.1 * x + 0.05 * y; break;                               /* gradient */
      case 2: e = 30 + 3000 * exp(-(pow((double)x - 40, 2) + pow((double)y - 10, 2)) / 30); break; /* spot */
      case 3: e = 5 + 60 * rndf(); break;                                       /* noisy */
      default: e = 1 + 4 * rndf(); break;                                       /* dim: read noise dominates */
    }
    e += sqrt(e) * (rndf() + rndf() + rndf() - 1.5) * 2;                        /* ~shot noise */
    e += 3.0 * (rndf() + rndf() + rndf() + rndf() - 2.0) * 1.7;                  /* ~3 e- read noise */
    double dn = off + conv * e; img[i] = (uint16_t)(dn < 0 ? 0 : dn > 65535 ? 65535 : dn);
  }
}

static void test_codec_direct(void) {
  SECTION("encoder/decoder (direct)");
  const float conv = 2.0f, off = 100.0f, rn = 3.0f;
  b4d_set_camera(conv, off, rn);
  uint32_t ns[] = {1, 2, 31, 32, 33, 640, 641, 4096, 60000};
  uint32_t w = 64;
  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
  for (int ts = 1; ts <= 2; ts++) for (size_t a = 0; a < sizeof ns / sizeof *ns; a++) for (int kind = 0; kind < 5; kind++)
  for (int flags = 0; flags < 9; flags++) for (int mode = 0; mode <= 2; mode++) for (int qi = 2; qi <= 8; qi *= 2) {
    if (mode == 0 && qi != 2) continue;
    uint32_t n = ns[a];
    /* flags 0..5: predictor (flags & 1) x entropy (flags >> 1); flags 6..8: mean3d x entropy, 5 rows per plane */
    if (flags < 6) { b4d_set_predictor(flags & 1); b4d_set_entropy(flags >> 1); b4d_set_plane_rows(0); }
    else { b4d_set_predictor(2); b4d_set_entropy(flags - 6); b4d_set_plane_rows(5); }
    b4d_set_row_width(w);
    cp.typesize = ts;
    uint16_t *img16 = malloc(2 * n); fill_image(img16, n, w, kind, conv, off);
    uint8_t *in = malloc(2 * n), *out = malloc(2 * n), *comp = malloc(2 * n + 1024);
    if (ts == 2) memcpy(in, img16, 2 * n);
    else for (uint32_t i = 0; i < n; i++) { uint32_t v = img16[i] >> 4; in[i] = (uint8_t)(v > 255 ? 255 : v); }
    if (ts == 1) b4d_set_camera(conv / 16, off / 16, rn); else b4d_set_camera(conv, off, rn);
    uint8_t meta = b4d_meta(mode, qi * 0.25f);
    int cs = b4d_encoder(in, (int32_t)(n * ts), comp, (int32_t)(2 * n + 1024), meta, &cp, NULL);
    if (cs == 0) { free(img16); free(in); free(out); free(comp); continue; }   /* incompressible: fine */
    CHECK(cs > 0, "ts=%d n=%u kind=%d flags=%d mode=%d: encoder error %d", ts, n, kind, flags, mode, cs);
    if (cs <= 0) { free(img16); free(in); free(out); free(comp); continue; }
    int ds = b4d_decoder(comp, cs, out, (int32_t)(n * ts), meta, NULL, NULL);
    CHECK(ds == (int)(n * ts), "ts=%d n=%u kind=%d flags=%d mode=%d: decoder returned %d", ts, n, kind, flags, mode, ds);
    if (ds != (int)(n * ts)) { free(img16); free(in); free(out); free(comp); continue; }
    if (mode == 0) {
      CHECK(memcmp(in, out, n * ts) == 0, "ts=%d n=%u kind=%d flags=%d lossless mismatch", ts, n, kind, flags);
    } else {
      double q = qi * 0.25, worst = 0; float c = (ts == 1) ? conv / 16 : conv, o = (ts == 1) ? off / 16 : off;
      for (uint32_t i = 0; i < n; i++) {
        double x = (ts == 2) ? ((uint16_t *)in)[i] : in[i], y = (ts == 2) ? ((uint16_t *)out)[i] : out[i];
        double e = (x - o) / c; if (e < 0) e = 0;
        double sig = c * sqrt(e + (double)rn * rn);
        double ratio = fabs(x - y) / (0.5 * q * sig + 1.0);       /* q/2 sigma + 1 DN rounding slack */
        if (ratio > worst) { worst = ratio; if (ratio > 1.05 && getenv("B4D_DEBUG")) printf("    i=%u x=%.0f y=%.0f sig=%.2f\n", i, x, y, sig); }
      }
      CHECK(worst <= 1.05, "ts=%d n=%u kind=%d flags=%d mode=%d q=%.2f: error exceeds q/2 sigma (x%.2f)", ts, n, kind, flags, mode, q, worst);
    }
    free(img16); free(in); free(out); free(comp);
  }
  b4d_set_camera(conv, off, rn); b4d_set_predictor(0); b4d_set_plane_rows(0);
}

/* ---------------------------------------------------------------- residual-level API */
static void test_residual_api(void) {
  SECTION("b4d_encode_residual_block == b4d_encoder");
  uint32_t n = 3000, w = 60; uint16_t *img = malloc(2 * n); fill_image(img, n, w, 2, 2.0f, 100.0f);
  uint8_t *a = malloc(2 * n + 1024), *b = malloc(2 * n + 1024); int16_t *r = malloc(2 * n);
  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS; cp.typesize = 2;
  b4d_set_camera(2.0f, 100.0f, 3.0f); b4d_set_row_width(w);
  for (int ent = 0; ent <= 2; ent++) for (int pred = 0; pred < 3; pred++) {
    b4d_set_predictor(pred); b4d_set_entropy(ent); b4d_set_plane_rows(pred == 2 ? 10 : 0);
    /* lossless: residuals are the mod-2^16 predictor residuals */
    residual_u16(img, r, n, w, pred, pred == 2 ? 10 : 0);
    int sa = b4d_encoder((uint8_t *)img, 2 * n, a, 2 * n + 1024, b4d_meta(0, 0), &cp, NULL);
    int sb = pred == 2 ? b4d_encode_residual_block2(r, n, 0, 2, 0, 2.0f, 100.0f, 3.0f, w, 10, pred, ent, b, 2 * n + 1024)
                       : b4d_encode_residual_block(r, n, 0, 2, 0, 2.0f, 100.0f, 3.0f, w, pred, ent, b, 2 * n + 1024);
    CHECK(sa > 0 && sa == sb && memcmp(a, b, sa) == 0, "ent=%d pred=%d: identical payload (%d vs %d)", ent, pred, sa, sb);
    if (pred == 2) CHECK(a[0] == 4 && (a[40] & 8), "mean3d writes format v4 with the mean3d flag (version %d flags %d)", a[0], a[40]);
    else CHECK(a[0] == 3, "legacy predictors keep format v3 (version %d)", a[0]);
  }
  b4d_set_predictor(0); b4d_set_plane_rows(0);
  free(img); free(a); free(b); free(r);
}

/* ---------------------------------------------------------------- robustness */
static void test_robustness(void) {
  SECTION("robustness");
  uint32_t n = 4096; uint16_t *img = malloc(2 * n); fill_image(img, n, 64, 2, 2.0f, 100.0f);
  uint8_t *comp = malloc(2 * n + 1024), *out = malloc(2 * n);
  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS; cp.typesize = 2;
  b4d_set_row_width(64);
  for (int flags = 0; flags < 6; flags++) for (int mode = 0; mode <= 2; mode++) {
    b4d_set_predictor(flags & 1); b4d_set_entropy(flags >> 1);
    uint8_t meta = b4d_meta(mode, 1.0f);
    int cs = b4d_encoder((uint8_t *)img, 2 * n, comp, 2 * n + 1024, meta, &cp, NULL);
    CHECK(cs > 0, "encode");
    /* truncations must not crash and must not claim success with too-short output */
    int crash_free = 1;
    for (int cut = 0; cut < cs; cut += (cs / 37) + 1) {
      int ds = b4d_decoder(comp, cut, out, 2 * n, meta, NULL, NULL);
      if (ds > 0 && ds != (int)(2 * n)) crash_free = 0;
    }
    CHECK(crash_free, "flags=%d mode=%d truncation handling", flags, mode);
    /* output buffer too small */
    CHECK(b4d_decoder(comp, cs, out, 2 * n - 2, meta, NULL, NULL) < 0, "small output rejected");
    /* corrupted header fields */
    uint8_t save = comp[0]; comp[0] = 99;  CHECK(b4d_decoder(comp, cs, out, 2 * n, meta, NULL, NULL) < 0, "bad version rejected"); comp[0] = save;
    save = comp[1]; comp[1] = 3;           CHECK(b4d_decoder(comp, cs, out, 2 * n, meta, NULL, NULL) < 0, "bad mode rejected"); comp[1] = save;
    /* random bit flips: no crash */
    for (int t = 0; t < 50; t++) {
      uint8_t *c2 = malloc(cs); memcpy(c2, comp, cs);
      for (int k = 0; k < 3; k++) c2[rnd() % cs] ^= (uint8_t)(1u << (rnd() & 7));
      b4d_decoder(c2, cs, out, 2 * n, meta, NULL, NULL);
      free(c2);
    }
    CHECK(1, "bit flips survived");
  }
  /* meta byte */
  CHECK(b4d_meta(1, 1.0f) == ((4 << 2) | 1), "meta mode1 q=1");
  CHECK(b4d_meta(2, 15.75f) == ((63 << 2) | 2), "meta q max");
  CHECK(b4d_meta(2, 100.0f) == ((63 << 2) | 2), "meta q clamps");
  CHECK(b4d_meta(1, 0.0f) == 0, "meta q=0 forces lossless");
  CHECK(b4d_meta(7, 1.0f) == (4 << 2), "meta bad mode -> lossless");
  free(img); free(comp); free(out);
}

/* ---------------------------------------------------------------- blosc2 integration */
static void test_blosc(void) {
  SECTION("blosc2 integration");
  CHECK(b4d_register() >= 0 || 1, "register");   /* may already be registered */
  const uint32_t W = 512, H = 384, N = W * H;
  uint16_t *img = malloc(2 * N); fill_image(img, N, W, 2, 2.0f, 100.0f);
  b4d_set_camera(2.0f, 100.0f, 3.0f); b4d_set_row_width(W);
  int32_t blocksizes[] = {0, 16 * W * 2, 100 * W * 2 + 6, 2 * N};   /* auto, aligned, misaligned rows, single */
  for (int flags = 0; flags < 6; flags++) for (int mode = 0; mode <= 2; mode++) for (size_t b = 0; b < 4; b++) for (int nt = 1; nt <= 2; nt++) {
    b4d_set_predictor(flags & 1); b4d_set_entropy(flags >> 1);
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS; cp.typesize = 2; cp.compcode = B4D_CODEC_ID;
    cp.compcode_meta = b4d_meta(mode, 1.0f); cp.nthreads = nt; cp.blocksize = blocksizes[b];
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
    blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS; dp.nthreads = nt;
    uint8_t *comp = malloc(2 * N + BLOSC2_MAX_OVERHEAD); uint16_t *out = malloc(2 * N);
    blosc2_context *cc = blosc2_create_cctx(cp), *dc = blosc2_create_dctx(dp);
    int cs = blosc2_compress_ctx(cc, img, 2 * N, comp, 2 * N + BLOSC2_MAX_OVERHEAD);
    int ds = cs > 0 ? blosc2_decompress_ctx(dc, comp, cs, out, 2 * N) : -1;
    CHECK(cs > 0 && ds == (int)(2 * N), "flags=%d mode=%d bs=%d nt=%d: cs=%d ds=%d", flags, mode, blocksizes[b], nt, cs, ds);
    if (ds == (int)(2 * N)) {
      if (mode == 0) CHECK(memcmp(img, out, 2 * N) == 0, "flags=%d bs=%d nt=%d lossless", flags, blocksizes[b], nt);
      else {
        double worst = 0;
        for (uint32_t i = 0; i < N; i++) {
          double e = (img[i] - 100.0) / 2; if (e < 0) e = 0; double sig = 2 * sqrt(e + 9);
          double r = fabs((double)img[i] - out[i]) / (0.5 * sig + 1.0); if (r > worst) worst = r;
        }
        CHECK(worst <= 1.05, "flags=%d mode=%d bs=%d: lossy error bound (x%.2f)", flags, mode, blocksizes[b], worst);
      }
      CHECK(cs < (int)(2 * N), "flags=%d mode=%d bs=%d: actually compressed (%d)", flags, mode, blocksizes[b], cs);
    }
    blosc2_free_ctx(cc); blosc2_free_ctx(dc); free(comp); free(out);
  }
  /* incompressible input still round-trips (codec returns 0 -> blosc memcpy) */
  {
    for (uint32_t i = 0; i < N; i++) img[i] = (uint16_t)rnd();
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS; cp.typesize = 2; cp.compcode = B4D_CODEC_ID; cp.compcode_meta = 0;
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
    blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS;
    uint8_t *comp = malloc(2 * N + BLOSC2_MAX_OVERHEAD); uint16_t *out = malloc(2 * N);
    blosc2_context *cc = blosc2_create_cctx(cp), *dc = blosc2_create_dctx(dp);
    int cs = blosc2_compress_ctx(cc, img, 2 * N, comp, 2 * N + BLOSC2_MAX_OVERHEAD);
    int ds = cs > 0 ? blosc2_decompress_ctx(dc, comp, cs, out, 2 * N) : -1;
    CHECK(ds == (int)(2 * N) && memcmp(img, out, 2 * N) == 0, "incompressible data round trip (cs=%d)", cs);
    blosc2_free_ctx(cc); blosc2_free_ctx(dc); free(comp); free(out);
  }
  free(img);
}

int main(void) {
  blosc2_init();
  test_zigzag();
  test_bitio();
  test_huffman();
  test_rice();
  test_ans();
  test_predictors();
  test_mode1_loop();
  test_codec_direct();
  test_residual_api();
  test_robustness();
  test_blosc();
  b4d_cleanup();
  blosc2_destroy();
  printf("\n%d checks passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}

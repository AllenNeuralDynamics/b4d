/*
 * b4d_example.c — round-trip test of the B³D Blosc2 codec on a synthetic
 * fluorescence-like image (smooth background + spots + Poisson noise).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#include "blosc2.h"
#include "b4d_blosc2.h"

#define W 1024
#define H 1024
#define N (W * H)

static const float CONV = 2.0f, OFFSET = 100.0f, RN = 3.0f;   /* camera model */

static double urand(void) { return (rand() + 1.0) / (RAND_MAX + 2.0); }
static long poisson(double lam) {          /* Knuth for small, normal approx for large */
  if (lam > 60) { double u = urand(), v = urand();
    return lround(lam + sqrt(lam) * sqrt(-2 * log(u)) * cos(6.283185307 * v)); }
  double L = exp(-lam), p = 1; long k = 0;
  do { k++; p *= urand(); } while (p > L);
  return k - 1;
}

static void make_image(uint16_t *img) {
  double *truth = malloc(N * sizeof *truth);
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
    truth[y * W + x] = 60.0 + 40.0 * x / W;                       /* electrons */
  for (int k = 0; k < 400; k++) {                                  /* blobs */
    int cx = rand() % W, cy = rand() % H; double amp = 200 + rand() % 1500;
    for (int dy = -8; dy <= 8; dy++) for (int dx = -8; dx <= 8; dx++) {
      int x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= W || y >= H) continue;
      truth[y * W + x] += amp * exp(-(dx * dx + dy * dy) / 8.0);
    }
  }
  for (int i = 0; i < N; i++) {
    double e = poisson(truth[i]) + RN * sqrt(-2 * log(urand())) * cos(6.283185307 * urand());
    double dn = OFFSET + CONV * e; img[i] = (uint16_t)(dn < 0 ? 0 : dn > 65535 ? 65535 : dn);
  }
  free(truth);
}

static double sigma_dn(double dn) {       /* camera noise model, in DN */
  double e = (dn - OFFSET) / CONV; if (e < 0) e = 0;
  return CONV * sqrt(e + RN * RN);
}

static int run(const char *label, const uint16_t *img, int mode, float q, int nthreads, int reps) {
  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
  cp.typesize = 2; cp.compcode = B4D_CODEC_ID; cp.compcode_meta = b4d_meta(mode, q);
  cp.nthreads = nthreads; cp.blocksize = 256 * W * 2;                    /* 256 rows / block */
  for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
  blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS; dp.nthreads = nthreads;

  size_t nbytes = N * sizeof(uint16_t);
  uint8_t *comp = malloc(nbytes + BLOSC2_MAX_OVERHEAD);
  uint16_t *out = malloc(nbytes);

  blosc2_context *cctx = blosc2_create_cctx(cp), *dctx = blosc2_create_dctx(dp);
  int csize = 0, dsize = 0; double tc = 1e9, td = 1e9;
  for (int rep = 0; rep < reps; rep++) {           /* best of N */
    double t0 = now();
    csize = blosc2_compress_ctx(cctx, img, (int32_t)nbytes, comp, (int32_t)nbytes + BLOSC2_MAX_OVERHEAD);
    double t = now() - t0; if (t < tc) tc = t;
    if (csize <= 0) { printf("%-22s compress failed (%d)\n", label, csize); return 1; }
    t0 = now();
    dsize = blosc2_decompress_ctx(dctx, comp, csize, out, (int32_t)nbytes);
    t = now() - t0; if (t < td) td = t;
    if (dsize != (int)nbytes) { printf("%-22s decompress failed (%d)\n", label, dsize); return 1; }
  }
  double sum_err_sig = 0, max_err_sig = 0; long mismatches = 0;
  for (int i = 0; i < N; i++) {
    double d = fabs((double)img[i] - out[i]); if (d) mismatches++;
    double r = d / sigma_dn(img[i]); sum_err_sig += r; if (r > max_err_sig) max_err_sig = r;
  }
  printf("%s %5.2fx %9.0f %9.0f %8.3f %6.2f %s\n",
         label, (double)nbytes / csize, nbytes / 1e6 / tc, nbytes / 1e6 / td,
         sum_err_sig / N, max_err_sig,
         mode == 0 ? (mismatches ? "LOSSLESS FAIL" : "bit-exact") : "");
  int fail = (mode == 0 && mismatches) || (mode && sum_err_sig / N > 1.0);
  blosc2_free_ctx(cctx); blosc2_free_ctx(dctx); free(comp); free(out);
  return fail;
}

int main(void) {
  srand(7);
  blosc2_init();
  if (b4d_register() < 0) { fprintf(stderr, "codec registration failed\n"); return 1; }
  b4d_set_camera(CONV, OFFSET, RN);
  b4d_set_row_width(W);

  uint16_t *img = malloc(N * sizeof *img);
  make_image(img);
  printf("Synthetic %dx%d uint16 image, camera: %.1f DN/e-, offset %.0f, read noise %.1f e-\n\n",
         W, H, CONV, OFFSET, RN);

  int fails = 0;
  int nthreads = getenv("B4D_THREADS") ? atoi(getenv("B4D_THREADS")) : 1;
  if (nthreads < 1) nthreads = 1;
  const char *pn[2] = {"left+above", "above"}, *cn[3] = {"huffman", "rice", "ans"};
  struct { const char *l; int mode; float q; } cases[] = {
    {"lossless", 0, 0}, {"mode1 q=1σ", 1, 1}, {"mode1 q=2σ", 1, 2}, {"mode2 q=1σ", 2, 1}, {"mode2 q=2σ", 2, 2} };
  printf("%d thread(s), best of 5  (set B4D_THREADS=N to change)\n", nthreads);
  printf("%-12s %-8s %-12s %6s %9s %9s %8s %6s\n", "predictor", "coder", "case", "ratio", "comp MB/s", "dec MB/s", "mean|e|/σ", "max");
  for (int p = 0; p < 2; p++) for (int c = 0; c < 3; c++) {
    b4d_set_predictor(p); b4d_set_entropy(c);
    for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
      char label[64]; snprintf(label, sizeof label, "%-12s %-8s %-12s", pn[p], cn[c], cases[k].l);
      fails += run(label, img, cases[k].mode, cases[k].q, nthreads, 5);
    }
  }
  b4d_set_predictor(B4D_PRED_LEFT_ABOVE); b4d_set_entropy(B4D_ENTROPY_HUFFMAN);

  free(img);
  blosc2_destroy();
  printf("\n%s\n", fails ? "SOME TESTS FAILED" : "all tests passed");
  return fails ? 1 : 0;
}

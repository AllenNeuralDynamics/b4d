/*
 * _b4d.c — minimal CPython extension exposing the B³D Blosc2 codec.
 *
 * Exposes two functions that work on contiguous buffers (the numpy-facing
 * logic lives in b4d/__init__.py):
 *
 *   _b4d.compress(buf, typesize, meta, row_width, predictor, entropy,
 *                 conv, offset, read_noise, nthreads, blocksize) -> bytes
 *   _b4d.decompress(buf, nthreads) -> bytearray
 *   _b4d.nbytes(buf) -> int           (uncompressed size of a chunk)
 *
 * Parameters are passed per call through Blosc2's cparams->codec_params, so
 * concurrent calls from several Python threads run fully in parallel (the GIL
 * is released around compression and decompression).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "blosc2.h"
#include "b4d_blosc2.h"

#if defined(_MSC_VER)
#define B4D_PY_TLS __declspec(thread)
#else
#define B4D_PY_TLS _Thread_local
#endif

/* Reusable per-thread compression buffer: avoids page-faulting a fresh
   input-sized allocation on every call (matters at multi-GB/s). */
static B4D_PY_TLS uint8_t *t_out = NULL;
static B4D_PY_TLS size_t   t_out_cap = 0;

static uint8_t *thread_out_buffer(size_t cap) {
  if (t_out_cap < cap) {
    free(t_out);
    t_out = (uint8_t *)malloc(cap);
    t_out_cap = t_out ? cap : 0;
  }
  return t_out;
}

/* Persistent per-thread Blosc2 compression context: creating a context per call makes Blosc2 spawn
   its worker threads on every call, which dominates short runs at high thread counts. The context is
   reused while typesize/meta/nthreads/blocksize are unchanged; codec params live in TLS so the
   pointer stored in the context stays valid. */
static B4D_PY_TLS blosc2_context *t_cctx = NULL;
static B4D_PY_TLS blosc2_cparams  t_cp;
static B4D_PY_TLS b4d_params      t_params;
static B4D_PY_TLS int             t_cctx_valid = 0;

/* Blosc2's tuner clamps cparams.blocksize to the chunk size and stores the result in the context, so a reused
   context would carry a smaller block size into later calls. Requesting min(blocksize, nbytes) makes the clamp a
   no-op and the cache key (which includes blocksize) recreates the context when it changes. blocksize 0 (auto)
   cannot be made idempotent: those calls use a fresh context. */
static int32_t effective_blocksize(int blocksize, int32_t nbytes, int typesize) {
  if (blocksize <= 0) return 0;
  int32_t bs = blocksize > nbytes ? nbytes : blocksize;
  if (bs > typesize) bs = bs / typesize * typesize;
  return bs;
}

static blosc2_context *thread_cctx(const blosc2_cparams *cp, const b4d_params *params) {
  if (t_cctx_valid && t_cctx && t_cp.typesize == cp->typesize && t_cp.compcode_meta == cp->compcode_meta &&
      t_cp.nthreads == cp->nthreads && t_cp.blocksize == cp->blocksize) {
    t_params = *params;                       /* same context, possibly different camera/options */
    return t_cctx;
  }
  if (t_cctx) blosc2_free_ctx(t_cctx);
  t_cp = *cp; t_params = *params; t_cp.codec_params = &t_params;
  t_cctx = blosc2_create_cctx(t_cp); t_cctx_valid = t_cctx != NULL;
  return t_cctx;
}

/* per-pixel calibration: a capsule from pixmap_new(), or None */
#define B4D_PIXMAP_CAPSULE "b4d.pixmap"
static void pixmap_capsule_free(PyObject *c) { b4d_pixmap_free((b4d_pixmap *)PyCapsule_GetPointer(c, B4D_PIXMAP_CAPSULE)); }
static int pixmap_of(PyObject *o, const b4d_pixmap **out) {
  *out = NULL;
  if (!o || o == Py_None) return 0;
  *out = (const b4d_pixmap *)PyCapsule_GetPointer(o, B4D_PIXMAP_CAPSULE);
  return *out ? 0 : -1;
}

/* exact VST (format v9): a capsule from vst_new() / vst_from_knots(), or None */
#define B4D_VST_CAPSULE "b4d.vst"
static void vst_capsule_free(PyObject *c) { b4d_vst_free((b4d_vst *)PyCapsule_GetPointer(c, B4D_VST_CAPSULE)); }
static int vst_of(PyObject *o, const b4d_vst **out) {
  *out = NULL;
  if (!o || o == Py_None) return 0;
  *out = (const b4d_vst *)PyCapsule_GetPointer(o, B4D_VST_CAPSULE);
  return *out ? 0 : -1;
}
/* vst_new(conv, offset, rn, objective=0) -> (capsule, max |residual|): the optimiser (b4d_vst_new; 0 std, 1 variance) */
static PyObject *py_vst_new(PyObject *self, PyObject *args) {
  (void)self; double conv, offset, rn, err = -1.0; b4d_vst *v; int objective = B4D_VST_STD;
  if (!PyArg_ParseTuple(args, "ddd|i", &conv, &offset, &rn, &objective)) return NULL;
  Py_BEGIN_ALLOW_THREADS
  v = b4d_vst_new((float)conv, (float)offset, (float)rn, objective, &err);
  Py_END_ALLOW_THREADS
  if (!v) { PyErr_SetString(PyExc_ValueError, "no exact VST for this camera model"); return NULL; }
  PyObject *c = PyCapsule_New(v, B4D_VST_CAPSULE, vst_capsule_free);
  if (!c) { b4d_vst_free(v); return NULL; }
  return Py_BuildValue("(Nd)", c, err);
}
/* vst_knots(capsule) -> (lam_max, [g]) */
static PyObject *py_vst_knots(PyObject *self, PyObject *arg) {
  (void)self; const b4d_vst *v; float lam, g[B4D_VST_MAXK];
  if (vst_of(arg, &v) || !v) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "not a vst capsule"); return NULL; }
  int nk = b4d_vst_knots(v, &lam, g);
  PyObject *l = PyList_New(nk);
  if (!l) return NULL;
  for (int j = 0; j < nk; j++) PyList_SET_ITEM(l, j, PyFloat_FromDouble(g[j]));
  return Py_BuildValue("(dN)", (double)lam, l);
}
/* vst_table(capsule) -> bytes(float64[65536]) */
static PyObject *py_vst_table(PyObject *self, PyObject *arg) {
  (void)self; const b4d_vst *v;
  if (vst_of(arg, &v) || !v) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "not a vst capsule"); return NULL; }
  PyObject *b = PyBytes_FromStringAndSize(NULL, 65536 * sizeof(double));
  if (!b) return NULL;
  if (b4d_vst_table(v, (double *)PyBytes_AS_STRING(b))) { Py_DECREF(b); PyErr_SetString(PyExc_ValueError, "invalid exact VST"); return NULL; }
  return b;
}
/* vst_std(capsule|None, conv, offset, rn, lams (float64 buffer)) -> bytes(float64[n]): std of T(X) | lam for the camera model */
static PyObject *py_vst_std(PyObject *self, PyObject *args) {
  (void)self; PyObject *vo; double conv, offset, rn; Py_buffer lv; const b4d_vst *v; int rc;
  if (!PyArg_ParseTuple(args, "Odddy*", &vo, &conv, &offset, &rn, &lv)) return NULL;
  if (vst_of(vo, &v)) { PyBuffer_Release(&lv); return NULL; }
  const int nl = (int)(lv.len / 8);
  PyObject *b = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)nl * 8);
  if (!b) { PyBuffer_Release(&lv); return NULL; }
  double *out = (double *)PyBytes_AS_STRING(b);
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_vst_std(v, (float)conv, (float)offset, (float)rn, (const double *)lv.buf, nl, out);
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&lv);
  if (rc) { Py_DECREF(b); return PyErr_NoMemory(); }
  return b;
}

/* pixmap_new(height, width, tier|None, delta|None, flat_a|None, flat_b|None, extra_var(list of float), skip) -> capsule.
   tier: uint8 (0 = not listed, k = tier k - 1); delta, flat_a, flat_b: int16; each height * width elements. */
static PyObject *py_pixmap_new(PyObject *self, PyObject *args) {
  (void)self;
  unsigned int h, w; PyObject *objs[4], *extra; double skip;
  if (!PyArg_ParseTuple(args, "IIOOOOOd", &h, &w, &objs[0], &objs[1], &objs[2], &objs[3], &extra, &skip)) return NULL;
  Py_buffer views[4]; const void *ptr[4] = {NULL, NULL, NULL, NULL}; int have[4] = {0, 0, 0, 0}; PyObject *out = NULL;
  float ex[B4D_PIX_TIERS]; Py_ssize_t ne = PySequence_Check(extra) ? PySequence_Size(extra) : -1;
  if (ne < 1 || ne > B4D_PIX_TIERS) { PyErr_SetString(PyExc_ValueError, "extra_var needs 1 to 8 values"); return NULL; }
  for (Py_ssize_t i = 0; i < ne; i++) {
    PyObject *it = PySequence_GetItem(extra, i); double v = it ? PyFloat_AsDouble(it) : -1.0; Py_XDECREF(it);
    if (PyErr_Occurred()) return NULL;
    ex[i] = (float)v;
  }
  for (int i = 0; i < 4; i++) {
    if (objs[i] == Py_None) continue;
    if (PyObject_GetBuffer(objs[i], &views[i], PyBUF_SIMPLE) != 0) goto done;
    have[i] = 1; ptr[i] = views[i].buf;
    if ((size_t)views[i].len != (size_t)h * w * (i == 0 ? 1 : 2)) { PyErr_SetString(PyExc_ValueError, "map size does not match height * width"); goto done; }
  }
  {
    b4d_pixmap *m = b4d_pixmap_new(h, w, (const uint8_t *)ptr[0], (const int16_t *)ptr[1], (const int16_t *)ptr[2], (const int16_t *)ptr[3], ex, (int)ne, (float)skip);
    if (!m) { PyErr_SetString(PyExc_ValueError, "invalid pixel map"); goto done; }
    out = PyCapsule_New(m, B4D_PIXMAP_CAPSULE, pixmap_capsule_free);
    if (!out) b4d_pixmap_free(m);
  }
done:
  for (int i = 0; i < 4; i++) if (have[i]) PyBuffer_Release(&views[i]);
  return out;
}

static PyObject *py_compress(PyObject *self, PyObject *args, PyObject *kwargs) {
  (void)self;
  static char *kw[] = {"buf", "typesize", "meta", "row_width", "predictor", "entropy",
                       "conv", "offset", "read_noise", "nthreads", "blocksize", "plane_rows", "dither", "pixmap", "map_y0", "map_x0", "vst", NULL};
  Py_buffer view; int typesize, meta, predictor, entropy, nthreads = 1, blocksize = 0, dither = 0;
  unsigned int row_width, plane_rows = 0, map_y0 = 0, map_x0 = 0; double conv, offset, read_noise; PyObject *pixobj = Py_None; const b4d_pixmap *pixmap;
  PyObject *vstobj = Py_None; const b4d_vst *vst;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*iiIiiddd|iiIiOIIO", kw, &view, &typesize, &meta, &row_width,
                                   &predictor, &entropy, &conv, &offset, &read_noise, &nthreads, &blocksize, &plane_rows, &dither, &pixobj, &map_y0, &map_x0, &vstobj))
    return NULL;
  if (pixmap_of(pixobj, &pixmap) || vst_of(vstobj, &vst)) { PyBuffer_Release(&view); return NULL; }
  if (!PyBuffer_IsContiguous(&view, 'C')) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "buffer must be C-contiguous"); return NULL; }
  if (typesize != 1 && typesize != 2) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "typesize must be 1 or 2"); return NULL; }
  if (view.len > BLOSC2_MAX_BUFFERSIZE) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "chunk too large for one Blosc2 chunk; split it"); return NULL; }
  if (view.len % typesize) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "buffer length not a multiple of typesize"); return NULL; }

  size_t cap = (size_t)view.len + BLOSC2_MAX_OVERHEAD;
  uint8_t *scratch = thread_out_buffer(cap);
  if (!scratch) { PyBuffer_Release(&view); return PyErr_NoMemory(); }

  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
  cp.typesize = typesize; cp.compcode = B4D_CODEC_ID; cp.compcode_meta = (uint8_t)meta;
  cp.nthreads = (int16_t)(nthreads < 1 ? 1 : nthreads); cp.blocksize = effective_blocksize(blocksize, (int32_t)view.len, typesize);
  for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;

  b4d_params params = { B4D_PARAMS_MAGIC, (float)conv, (float)offset, (float)read_noise, row_width, predictor, entropy, plane_rows, dither, pixmap, map_y0, map_x0, vst };

  int csize;
  Py_BEGIN_ALLOW_THREADS
  if (cp.blocksize == 0) {                                        /* auto block size: fresh context per call */
    b4d_params p2 = params; cp.codec_params = &p2;
    blosc2_context *ctx = blosc2_create_cctx(cp);
    csize = ctx ? blosc2_compress_ctx(ctx, view.buf, (int32_t)view.len, scratch, (int32_t)cap) : -1;
    if (ctx) blosc2_free_ctx(ctx);
  } else {
    blosc2_context *ctx = thread_cctx(&cp, &params);
    csize = ctx ? blosc2_compress_ctx(ctx, view.buf, (int32_t)view.len, scratch, (int32_t)cap) : -1;
  }
  Py_END_ALLOW_THREADS

  PyBuffer_Release(&view);
  if (csize <= 0) { PyErr_Format(PyExc_RuntimeError, "blosc2_compress_ctx failed (%d)", csize); return NULL; }
  return PyBytes_FromStringAndSize((const char *)scratch, csize);   /* exactly-sized copy of the compressed bytes */
}

static B4D_PY_TLS blosc2_context *t_dctx = NULL;
static B4D_PY_TLS int t_dctx_threads = 0;
static blosc2_context *thread_dctx(int nthreads) {
  if (t_dctx && t_dctx_threads == nthreads) return t_dctx;
  if (t_dctx) blosc2_free_ctx(t_dctx);
  blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS; dp.nthreads = (int16_t)nthreads;
  t_dctx = blosc2_create_dctx(dp); t_dctx_threads = nthreads;
  return t_dctx;
}

static PyObject *py_decompress(PyObject *self, PyObject *args, PyObject *kwargs) {
  (void)self;
  static char *kw[] = {"buf", "nthreads", NULL};
  Py_buffer view; int nthreads = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*|i", kw, &view, &nthreads)) return NULL;
  int32_t nbytes = 0, cbytes = 0, bsize = 0;
  if (blosc2_cbuffer_sizes(view.buf, &nbytes, &cbytes, &bsize) < 0 || cbytes > view.len) {
    PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "not a valid Blosc2 chunk"); return NULL;
  }
  PyObject *out = PyByteArray_FromStringAndSize(NULL, nbytes);
  if (!out) { PyBuffer_Release(&view); return NULL; }
  int dsize;
  Py_BEGIN_ALLOW_THREADS
  blosc2_context *ctx = thread_dctx(nthreads < 1 ? 1 : nthreads);
  dsize = ctx ? blosc2_decompress_ctx(ctx, view.buf, (int32_t)view.len, PyByteArray_AS_STRING(out), nbytes) : -1;
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&view);
  if (dsize != nbytes) { Py_DECREF(out); PyErr_Format(PyExc_RuntimeError, "blosc2_decompress_ctx failed (%d)", dsize); return NULL; }
  return out;
}

/* encode_residual_tiles(res_int16, ntiles, nelem, mode, qi, conv, offset, rn, cols, plane_rows, predictor, entropy, meta, nthreads)
   -> (blobs: bytes, offsets: bytes[int64 x ntiles], lengths: bytes[int64 x ntiles], status: bytes[int8 x ntiles])
   Host entropy stage of the GPU path: every tile's residuals coded and framed on the writer's thread pool (no GIL). */
static PyObject *py_encode_residual_tiles(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer view; int ntiles, mode, qi, predictor, entropy, meta, nthreads; unsigned int nelem, cols, plane_rows; double conv, offset, rn;
  if (!PyArg_ParseTuple(args, "y*iIiidddIIiiii", &view, &ntiles, &nelem, &mode, &qi, &conv, &offset, &rn, &cols, &plane_rows, &predictor, &entropy, &meta, &nthreads))
    return NULL;
  if (view.len < (Py_ssize_t)ntiles * (Py_ssize_t)nelem * 2) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "residual buffer too small"); return NULL; }
  int64_t *offs = (int64_t *)malloc((size_t)ntiles * sizeof(int64_t)), *lens = (int64_t *)malloc((size_t)ntiles * sizeof(int64_t));
  int8_t *st = (int8_t *)malloc((size_t)ntiles);
  uint8_t *blob = NULL; size_t blen = 0; int rc;
  if (!offs || !lens || !st) { free(offs); free(lens); free(st); PyBuffer_Release(&view); return PyErr_NoMemory(); }
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_zarr_encode_residual_tiles((const int16_t *)view.buf, ntiles, nelem, mode, qi, (float)conv, (float)offset, (float)rn, cols, plane_rows,
                                      predictor, entropy, (uint8_t)meta, nthreads, &blob, &blen, offs, lens, st);
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&view);
  if (rc != 0) { free(offs); free(lens); free(st); b4d_zarr_free_buffer(blob); PyErr_SetString(PyExc_RuntimeError, "b4d_zarr_encode_residual_tiles failed"); return NULL; }
  PyObject *pb = PyBytes_FromStringAndSize((const char *)blob, (Py_ssize_t)blen);
  PyObject *po = PyBytes_FromStringAndSize((const char *)offs, (Py_ssize_t)ntiles * 8);
  PyObject *pl = PyBytes_FromStringAndSize((const char *)lens, (Py_ssize_t)ntiles * 8);
  PyObject *ps = PyBytes_FromStringAndSize((const char *)st, (Py_ssize_t)ntiles);
  b4d_zarr_free_buffer(blob); free(offs); free(lens); free(st);
  if (!pb || !po || !pl || !ps) { Py_XDECREF(pb); Py_XDECREF(po); Py_XDECREF(pl); Py_XDECREF(ps); return NULL; }
  return Py_BuildValue("(NNNN)", pb, po, pl, ps);
}

/* compress_tiles(buf, nz, ny, nx, tz, ty, tx, typesize, meta, row_width_flag, predictor, entropy, conv, offset,
                  read_noise, nthreads, blocksize) -> list of bytes (one Blosc2 chunk per tile, tiles in C order)
   Cuts a contiguous (nz, ny, nx) array into (tz, ty, tx) tiles and compresses each as an independent Blosc2 chunk
   inside ONE GIL release: no per-tile Python work. row_width = tile width (2-D prediction within a tile). */
/* compress_tiles' tile and arena buffers, kept per thread between calls (a fresh multi-MB malloc is fresh pages: a page
   fault per 4 KiB on every call); released after a call that needed more than CT_KEEP */
#if defined(_MSC_VER)
#define CT_TLS __declspec(thread)
#else
#define CT_TLS _Thread_local
#endif
#define CT_KEEP ((size_t)256 << 20)
static CT_TLS uint8_t *ct_tile = NULL, *ct_arena = NULL; static CT_TLS size_t ct_tile_cap = 0, ct_arena_cap = 0;
static uint8_t *ct_get(uint8_t **p, size_t *cap, size_t need) {
  if (*cap < need) { free(*p); *p = (uint8_t *)malloc(need); *cap = *p ? need : 0; }
  return *p;
}
static void ct_trim(void) {
  if (ct_arena_cap > CT_KEEP) { free(ct_arena); ct_arena = NULL; ct_arena_cap = 0; }
  if (ct_tile_cap > CT_KEEP) { free(ct_tile); ct_tile = NULL; ct_tile_cap = 0; }
}

static PyObject *py_compress_tiles(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer view; int nz, ny, nx, tz, ty, tx, typesize, meta, predictor, entropy, nthreads = 1, blocksize = 0, dither = 0; double conv, offset, rn;
  PyObject *pixobj = Py_None; unsigned int map_y0 = 0, map_x0 = 0; const b4d_pixmap *pixmap; PyObject *vstobj = Py_None; const b4d_vst *vst;
  if (!PyArg_ParseTuple(args, "y*iiiiiiiiiiddd|iiiOIIO", &view, &nz, &ny, &nx, &tz, &ty, &tx, &typesize, &meta, &predictor, &entropy,
                        &conv, &offset, &rn, &nthreads, &blocksize, &dither, &pixobj, &map_y0, &map_x0, &vstobj)) return NULL;
  if (pixmap_of(pixobj, &pixmap) || vst_of(vstobj, &vst)) { PyBuffer_Release(&view); return NULL; }
  if (typesize != 1 && typesize != 2) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "typesize must be 1 or 2"); return NULL; }
  if ((size_t)view.len < (size_t)nz * ny * nx * typesize || tz <= 0 || ty <= 0 || tx <= 0) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "bad shape/tile"); return NULL; }
  int nbz = (nz + tz - 1) / tz, nby = (ny + ty - 1) / ty, nbx = (nx + tx - 1) / tx, ntiles = nbz * nby * nbx;
  /* outputs: per-tile compressed size and an arena that grows by tile */
  size_t tile_max = (size_t)tz * ty * tx * typesize + BLOSC2_MAX_OVERHEAD;
  uint8_t *tilebuf = ct_get(&ct_tile, &ct_tile_cap, tile_max);
  size_t arena_cap = (size_t)ntiles * (tile_max / 2 + 4096), arena_len = 0;
  if (arena_cap < ct_arena_cap) arena_cap = ct_arena_cap;
  uint8_t *arena = ct_get(&ct_arena, &ct_arena_cap, arena_cap);
  int32_t *sizes = (int32_t *)malloc((size_t)ntiles * sizeof(int32_t));
  size_t *offs = (size_t *)malloc((size_t)ntiles * sizeof(size_t));
  if (!tilebuf || !arena || !sizes || !offs) { free(sizes); free(offs); ct_trim(); PyBuffer_Release(&view); return PyErr_NoMemory(); }
  b4d_params params = { B4D_PARAMS_MAGIC, (float)conv, (float)offset, (float)rn, (uint32_t)tx, predictor, entropy, (uint32_t)ty, dither, pixmap, 0, 0, vst };
  blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
  cp.typesize = typesize; cp.compcode = B4D_CODEC_ID; cp.compcode_meta = (uint8_t)meta;
  cp.nthreads = (int16_t)(nthreads < 1 ? 1 : nthreads);
  for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
  int err = 0;
  Py_BEGIN_ALLOW_THREADS
  const uint8_t *src = (const uint8_t *)view.buf;
  int t = 0;
  for (int bz = 0; bz < nbz && !err; bz++) for (int by = 0; by < nby && !err; by++) for (int bx = 0; bx < nbx && !err; bx++, t++) {
    int z0 = bz * tz, y0 = by * ty, x0 = bx * tx;
    int dz = (nz - z0 < tz) ? nz - z0 : tz, dy = (ny - y0 < ty) ? ny - y0 : ty, dx = (nx - x0 < tx) ? nx - x0 : tx;
    /* gather the tile (rows are contiguous runs of dx*typesize bytes) */
    size_t rowb = (size_t)dx * typesize; uint8_t *dst = tilebuf; const uint8_t *tin = tilebuf;
    if (dx == nx && (dy == ny || dz == 1))                                           /* the tile is one contiguous run */
      tin = src + (((size_t)z0 * ny + y0) * nx) * typesize;
    else
    for (int z = 0; z < dz; z++) for (int y = 0; y < dy; y++) {
      memcpy(dst, src + (((size_t)(z0 + z) * ny + (y0 + y)) * nx + x0) * typesize, rowb); dst += rowb;
    }
    int32_t nbytes = (int32_t)((size_t)dz * dy * dx * typesize);
    if (arena_len + (size_t)nbytes + BLOSC2_MAX_OVERHEAD > arena_cap) {          /* grow arena */
      size_t ncap = arena_cap * 2 + (size_t)nbytes + BLOSC2_MAX_OVERHEAD; uint8_t *na = (uint8_t *)realloc(arena, ncap);
      if (!na) { err = -1; break; } arena = na; arena_cap = ncap; ct_arena = na; ct_arena_cap = ncap;
    }
    params.row_width = (uint32_t)dx; params.plane_rows = (uint32_t)dy;             /* 2-D / 3-D prediction within the tile */
    cp.blocksize = effective_blocksize(blocksize, nbytes, typesize);
    if (pixmap) { cp.blocksize = nbytes; params.map_y0 = map_y0 + (uint32_t)y0; params.map_x0 = map_x0 + (uint32_t)x0; }   /* one block: it must know where its rows sit */
    if (predictor == B4D_PRED_MEAN3D && cp.blocksize > 0) {                          /* whole planes per Blosc2 block */
      int32_t pb = (int32_t)dy * dx * typesize, planes = cp.blocksize / pb; if (planes < 1) planes = 1;
      cp.blocksize = planes * pb; if (cp.blocksize > nbytes) cp.blocksize = nbytes;
    }
    int csize;
    if (cp.blocksize == 0) {
      b4d_params p2 = params; cp.codec_params = &p2;
      blosc2_context *c1 = blosc2_create_cctx(cp);
      csize = c1 ? blosc2_compress_ctx(c1, tin, nbytes, arena + arena_len, (int32_t)(arena_cap - arena_len)) : -1;
      if (c1) blosc2_free_ctx(c1);
    } else {
      blosc2_context *ctx = thread_cctx(&cp, &params);
      csize = ctx ? blosc2_compress_ctx(ctx, tin, nbytes, arena + arena_len, (int32_t)(arena_cap - arena_len)) : -1;
    }
    if (csize <= 0) { err = -2; break; }
    offs[t] = arena_len; sizes[t] = csize; arena_len += (size_t)csize;
  }
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&view);
  PyObject *out = NULL;
  if (err) PyErr_Format(PyExc_RuntimeError, "compress_tiles failed (%d)", err);
  else {
    out = PyList_New(ntiles);
    for (int i = 0; i < ntiles && out; i++) {
      PyObject *b = PyBytes_FromStringAndSize((const char *)arena + offs[i], sizes[i]);
      if (!b) { Py_DECREF(out); out = NULL; break; }
      PyList_SET_ITEM(out, i, b);
    }
  }
  free(sizes); free(offs); ct_trim();                                            /* tile / arena kept for the next call */
  return out;
}

/* zarr_write_block(block_buf, typesize, nz, ny, nx, z0, y0, x0, ashape(3), shards(3), chunks(3), codec,
                    params(conv, offset, rn, predictor, entropy, meta), cname, clevel, shuffle, blobs|None,
                    root, key_sep, sync, nthreads) -> bytes written.  GIL released for the whole call. */
static PyObject *py_zarr_write_block(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer view; int typesize, nz, ny, nx, codec, predictor, entropy, meta, dither, clevel, shuffle, sync, nthreads;
  long long z0, y0, x0, as0, as1, as2; int sz, sy, sx, cz, cy, cx; double conv, offset, rn;
  const char *cname, *root, *key_sep; PyObject *blobs;
  PyObject *ds_obj = Py_None; int fz = 1, fy = 1, fx = 1, direct = 0; const char *key_prefix = "";
  PyObject *pixobj = Py_None; unsigned int map_y0 = 0, map_x0 = 0; const b4d_pixmap *pixmap; PyObject *vstobj = Py_None; const b4d_vst *vst;
  if (!PyArg_ParseTuple(args, "y*iiiiLLL(LLL)(iii)(iii)i(dddiiii)siiOssii|O(iii)isOIIO", &view, &typesize, &nz, &ny, &nx, &z0, &y0, &x0,
                        &as0, &as1, &as2, &sz, &sy, &sx, &cz, &cy, &cx, &codec, &conv, &offset, &rn, &predictor, &entropy, &meta, &dither,
                        &cname, &clevel, &shuffle, &blobs, &root, &key_sep, &sync, &nthreads, &ds_obj, &fz, &fy, &fx, &direct, &key_prefix, &pixobj, &map_y0, &map_x0, &vstobj))
    return NULL;
  if (pixmap_of(pixobj, &pixmap) || vst_of(vstobj, &vst)) { PyBuffer_Release(&view); return NULL; }
  Py_buffer dsv; int have_ds = 0;
  if (ds_obj != Py_None) {
    if (PyObject_GetBuffer(ds_obj, &dsv, PyBUF_WRITABLE) != 0) { PyBuffer_Release(&view); return NULL; }
    have_ds = 1;
  }
  if (codec != B4D_ZARR_CODEC_BLOBS && (size_t)view.len < (size_t)nz * ny * nx * typesize) {
    PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "block buffer smaller than nz*ny*nx*typesize"); return NULL;
  }
  /* pre-encoded blobs: keep Py_buffers alive for the duration */
  Py_ssize_t nb = 0; Py_buffer *bviews = NULL; const uint8_t **bptr = NULL; size_t *blen = NULL;
  Py_buffer packv, offv, lenv; int have_pack = 0;
  if (codec == B4D_ZARR_CODEC_BLOBS && PyTuple_Check(blobs) && PyTuple_Size(blobs) == 3) {
    /* packed form: (buffer, offsets int64[n], lengths int64[n]) in writer order; length < 0 = absent chunk */
    if (PyObject_GetBuffer(PyTuple_GetItem(blobs, 0), &packv, PyBUF_SIMPLE) != 0) { PyBuffer_Release(&view); return NULL; }
    if (PyObject_GetBuffer(PyTuple_GetItem(blobs, 1), &offv, PyBUF_SIMPLE) != 0) { PyBuffer_Release(&packv); PyBuffer_Release(&view); return NULL; }
    if (PyObject_GetBuffer(PyTuple_GetItem(blobs, 2), &lenv, PyBUF_SIMPLE) != 0) { PyBuffer_Release(&offv); PyBuffer_Release(&packv); PyBuffer_Release(&view); return NULL; }
    have_pack = 1; nb = (Py_ssize_t)(offv.len / 8);
    if (lenv.len / 8 != nb) { PyBuffer_Release(&lenv); PyBuffer_Release(&offv); PyBuffer_Release(&packv); PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "offsets/lengths size mismatch"); return NULL; }
    bptr = (const uint8_t **)calloc((size_t)nb, sizeof *bptr); blen = (size_t *)calloc((size_t)nb, sizeof *blen);
    const long long *po = (const long long *)offv.buf, *pl = (const long long *)lenv.buf;
    for (Py_ssize_t i = 0; i < nb; i++) {
      if (pl[i] < 0) { bptr[i] = NULL; blen[i] = 0; continue; }
      if (po[i] < 0 || (unsigned long long)(po[i] + pl[i]) > (unsigned long long)packv.len) { free(bptr); free(blen); PyBuffer_Release(&lenv); PyBuffer_Release(&offv); PyBuffer_Release(&packv); PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "blob offset/length outside the packed buffer"); return NULL; }
      bptr[i] = (const uint8_t *)packv.buf + po[i]; blen[i] = (size_t)pl[i];
    }
  } else if (codec == B4D_ZARR_CODEC_BLOBS) {
    if (!PySequence_Check(blobs)) { PyBuffer_Release(&view); PyErr_SetString(PyExc_TypeError, "blobs must be a sequence of bytes"); return NULL; }
    nb = PySequence_Size(blobs);
    bviews = (Py_buffer *)calloc((size_t)nb, sizeof *bviews); bptr = (const uint8_t **)calloc((size_t)nb, sizeof *bptr); blen = (size_t *)calloc((size_t)nb, sizeof *blen);
    for (Py_ssize_t i = 0; i < nb; i++) {
      PyObject *it = PySequence_GetItem(blobs, i);
      if (it == Py_None) { bptr[i] = NULL; blen[i] = 0; Py_DECREF(it); continue; }
      if (PyObject_GetBuffer(it, &bviews[i], PyBUF_SIMPLE) != 0) { Py_DECREF(it); for (Py_ssize_t k = 0; k < i; k++) if (bviews[k].obj) PyBuffer_Release(&bviews[k]); free(bviews); free(bptr); free(blen); PyBuffer_Release(&view); return NULL; }
      bptr[i] = (const uint8_t *)bviews[i].buf; blen[i] = (size_t)bviews[i].len; Py_DECREF(it);
    }
  }
  b4d_params params = { B4D_PARAMS_MAGIC, (float)conv, (float)offset, (float)rn, (uint32_t)cx, predictor, entropy, (uint32_t)cy, dither, pixmap, map_y0, map_x0, vst };
  int64_t ashape[3] = { as0, as1, as2 }; long long written = 0; int rc; b4d_zarr_stats st; memset(&st, 0, sizeof st);
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_zarr_write_block(view.buf, typesize, nz, ny, nx, z0, y0, x0, ashape, sz, sy, sx, cz, cy, cx, codec, &params, (uint8_t)meta,
                            cname, clevel, shuffle, bptr, blen, root, key_sep, sync, nthreads, &written, &st,
                            have_ds ? dsv.buf : NULL, fz, fy, fx, direct, key_prefix);
  Py_END_ALLOW_THREADS
  if (bviews) for (Py_ssize_t i = 0; i < nb; i++) if (bviews[i].obj) PyBuffer_Release(&bviews[i]);
  if (have_pack) { PyBuffer_Release(&lenv); PyBuffer_Release(&offv); PyBuffer_Release(&packv); }
  free(bviews); free(bptr); free(blen); PyBuffer_Release(&view);
  if (have_ds) PyBuffer_Release(&dsv);
  if (rc != 0) { PyErr_Format(PyExc_RuntimeError, "zarr_write_block failed (%d)", rc); return NULL; }
  return Py_BuildValue("{s:L,s:d,s:d,s:i,s:i,s:i,s:i}", "bytes", written, "t_compress", st.t_compress, "t_io", st.t_io, "groups_mask", st.groups_mask,
                       "direct_used", st.direct_used, "direct_fallback", st.direct_fallback, "blosc1_fallback", st.blosc1_fallback);
}

/* downsample(src_buf, dst_buf, typesize, nz, ny, nx, fz, fy, fx, nthreads): box mean into dst (ceil shapes) */
static PyObject *py_downsample(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer src, dst; int typesize, nz, ny, nx, fz, fy, fx, nthreads;
  if (!PyArg_ParseTuple(args, "y*w*iiiiiiii", &src, &dst, &typesize, &nz, &ny, &nx, &fz, &fy, &fx, &nthreads)) return NULL;
  size_t oz = (nz + fz - 1) / fz, oy = (ny + fy - 1) / fy, ox = (nx + fx - 1) / fx;
  if ((size_t)src.len < (size_t)nz * ny * nx * typesize || (size_t)dst.len < oz * oy * ox * typesize) {
    PyBuffer_Release(&src); PyBuffer_Release(&dst); PyErr_SetString(PyExc_ValueError, "buffer too small"); return NULL;
  }
  int rc;
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_zarr_downsample(src.buf, typesize, nz, ny, nx, fz, fy, fx, dst.buf, nthreads);
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&src); PyBuffer_Release(&dst);
  if (rc) { PyErr_SetString(PyExc_RuntimeError, "downsample failed"); return NULL; }
  Py_RETURN_NONE;
}

static PyObject *py_zarr_free_cache(PyObject *self, PyObject *args) { (void)self; (void)args; b4d_zarr_free_cache(); Py_RETURN_NONE; }


/* orig_encode(img_u16, width, height, mode, q, conv, offset, rn, tile, vst_table|None) -> bytes
   The original B3D filter's chunk bitstream (no B4DP/Blosc2 container). */
static PyObject *py_orig_encode(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer view, tview; int width, height, mode, tile; double q, conv, offset, rn; PyObject *tab = Py_None;
  if (!PyArg_ParseTuple(args, "y*iiiddddi|O", &view, &width, &height, &mode, &q, &conv, &offset, &rn, &tile, &tab)) return NULL;
  const float *tabp = NULL;
  tview.obj = NULL;
  if (tab != Py_None) {
    if (PyObject_GetBuffer(tab, &tview, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&view); return NULL; }
    if (tview.len != 65536 * (Py_ssize_t)sizeof(float)) { PyBuffer_Release(&tview); PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "vst_table must be 65536 float32"); return NULL; }
    tabp = (const float *)tview.buf;
  }
  if (width <= 0 || height <= 0 || view.len != (Py_ssize_t)width * height * 2) {
    if (tview.obj) PyBuffer_Release(&tview);
    PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "img must be width*height uint16"); return NULL;
  }
  uint8_t *out = NULL; size_t olen = 0; int rc;
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_orig_encode((const uint16_t *)view.buf, (uint32_t)width, (uint32_t)height, mode, (float)q, (float)conv, (float)offset, (float)rn,
                       (uint32_t)tile, tabp, &out, &olen);
  Py_END_ALLOW_THREADS
  if (tview.obj) PyBuffer_Release(&tview);
  PyBuffer_Release(&view);
  if (rc) { PyErr_SetString(PyExc_RuntimeError, "b4d_orig_encode failed"); return NULL; }
  PyObject *res = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)olen);
  b4d_orig_free(out);
  return res;
}

/* orig_decode(payload, width, height, mode, q, conv, offset, rn, tile) -> bytearray(uint16 width*height) */
static PyObject *py_orig_decode(PyObject *self, PyObject *args) {
  (void)self;
  Py_buffer view; int width, height, mode, tile; double q, conv, offset, rn;
  if (!PyArg_ParseTuple(args, "y*iiiddddi", &view, &width, &height, &mode, &q, &conv, &offset, &rn, &tile)) return NULL;
  if (width <= 0 || height <= 0) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "bad shape"); return NULL; }
  PyObject *out = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)width * height * 2);
  if (!out) { PyBuffer_Release(&view); return NULL; }
  int rc;
  Py_BEGIN_ALLOW_THREADS
  rc = b4d_orig_decode((const uint8_t *)view.buf, (size_t)view.len, (uint32_t)width, (uint32_t)height, mode, (float)q, (float)conv, (float)offset,
                       (float)rn, (uint32_t)tile, (uint16_t *)PyByteArray_AS_STRING(out));
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&view);
  if (rc) { Py_DECREF(out); PyErr_SetString(PyExc_ValueError, "b4d_orig_decode: corrupt or mismatched bitstream"); return NULL; }
  return out;
}

/* orig_vst_table(conv, offset, rn, q) -> bytes(float32[65536]) : the IEEE table the C encoder uses by default */
static PyObject *py_orig_vst_table(PyObject *self, PyObject *args) {
  (void)self;
  double conv, offset, rn, q;
  if (!PyArg_ParseTuple(args, "dddd", &conv, &offset, &rn, &q)) return NULL;
  PyObject *out = PyBytes_FromStringAndSize(NULL, 65536 * (Py_ssize_t)sizeof(float));
  if (!out) return NULL;
  if (b4d_orig_vst_table((float *)PyBytes_AS_STRING(out), (float)conv, (float)offset, (float)rn, (float)q)) { Py_DECREF(out); PyErr_SetString(PyExc_ValueError, "bad parameters"); return NULL; }
  return out;
}

/* orig_register_vst_table(conv, offset, rn, q, table|None) : process-wide override used by every encoder call with these parameters */
static PyObject *py_orig_register_vst_table(PyObject *self, PyObject *args) {
  (void)self;
  double conv, offset, rn, q; PyObject *tab;
  if (!PyArg_ParseTuple(args, "ddddO", &conv, &offset, &rn, &q, &tab)) return NULL;
  int rc;
  if (tab == Py_None) rc = b4d_orig_register_vst_table((float)conv, (float)offset, (float)rn, (float)q, NULL);
  else {
    Py_buffer tview;
    if (PyObject_GetBuffer(tab, &tview, PyBUF_SIMPLE) < 0) return NULL;
    if (tview.len != 65536 * (Py_ssize_t)sizeof(float)) { PyBuffer_Release(&tview); PyErr_SetString(PyExc_ValueError, "table must be 65536 float32"); return NULL; }
    rc = b4d_orig_register_vst_table((float)conv, (float)offset, (float)rn, (float)q, (const float *)tview.buf);
    PyBuffer_Release(&tview);
  }
  if (rc) { PyErr_SetString(PyExc_RuntimeError, "no free table slot"); return NULL; }
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  {"compress",   (PyCFunction)(void (*)(void))py_compress,   METH_VARARGS | METH_KEYWORDS, "compress(buf, ...) -> bytes"},
  {"pixmap_new", py_pixmap_new, METH_VARARGS, "pixmap_new(height, width, tier, delta, flat_a, flat_b, extra_var, skip) -> capsule"},
  {"vst_new", py_vst_new, METH_VARARGS, "vst_new(conv, offset, rn, objective=0) -> (capsule, max |residual|): exact VST (format v9); objective 0 std, 1 variance"},
  {"vst_knots", py_vst_knots, METH_O, "vst_knots(capsule) -> (lam_max, [g])"},
  {"vst_table", py_vst_table, METH_O, "vst_table(capsule) -> bytes(float64[65536])"},
  {"vst_std", py_vst_std, METH_VARARGS, "vst_std(capsule|None, conv, offset, rn, lams float64) -> bytes(float64[n])"},
  {"decompress", (PyCFunction)(void (*)(void))py_decompress, METH_VARARGS | METH_KEYWORDS, "decompress(buf, nthreads=1) -> bytearray"},
  {"zarr_write_block", py_zarr_write_block, METH_VARARGS, "native zarr v3 sharded block writer"},
  {"downsample", py_downsample, METH_VARARGS, "threaded box downsampling by integer factors"},
  {"zarr_free_cache", py_zarr_free_cache, METH_NOARGS, "free the native writer's cached arenas"},
  {"compress_tiles", py_compress_tiles, METH_VARARGS, "compress_tiles(buf, nz, ny, nx, tz, ty, tx, typesize, meta, predictor, entropy, conv, offset, rn[, nthreads, blocksize]) -> [bytes]"},
  {"encode_residual_tiles", py_encode_residual_tiles, METH_VARARGS, "encode_residual_tiles(res, ntiles, nelem, mode, qi, conv, offset, rn, cols, plane_rows, predictor, entropy, meta, nthreads) -> (blobs, offsets, lengths, status)"},
  {"orig_encode", py_orig_encode, METH_VARARGS, "orig_encode(img_u16, width, height, mode, q, conv, offset, rn, tile[, vst_table]) -> bytes"},
  {"orig_decode", py_orig_decode, METH_VARARGS, "orig_decode(payload, width, height, mode, q, conv, offset, rn, tile) -> bytearray"},
  {"orig_vst_table", py_orig_vst_table, METH_VARARGS, "orig_vst_table(conv, offset, rn, q) -> bytes(float32[65536])"},
  {"orig_register_vst_table", py_orig_register_vst_table, METH_VARARGS, "orig_register_vst_table(conv, offset, rn, q, table|None)"},
  {NULL, NULL, 0, NULL}
};

static void module_free(void *m) {
  (void)m; free(t_out); t_out = NULL; t_out_cap = 0;
  if (t_cctx) { blosc2_free_ctx(t_cctx); t_cctx = NULL; t_cctx_valid = 0; }
  if (t_dctx) { blosc2_free_ctx(t_dctx); t_dctx = NULL; }
  b4d_cleanup(); blosc2_destroy();
}

static struct PyModuleDef moddef = { PyModuleDef_HEAD_INIT, "b4d._b4d", "B4D codec bindings", -1, methods, NULL, NULL, NULL, module_free };

#if defined(__clang__)
  #define B4D_BUILT_WITH "clang " __clang_version__
#elif defined(_MSC_VER)
  #define B4D_STR2(x) #x
  #define B4D_STR(x) B4D_STR2(x)
  #define B4D_BUILT_WITH "msvc " B4D_STR(_MSC_VER)
#elif defined(__GNUC__)
  #define B4D_BUILT_WITH "gcc " __VERSION__
#else
  #define B4D_BUILT_WITH "unknown"
#endif

PyMODINIT_FUNC PyInit__b4d(void) {
  blosc2_init();
  if (b4d_register() < 0) { PyErr_SetString(PyExc_RuntimeError, "b4d_register failed"); return NULL; }
  PyObject *m = PyModule_Create(&moddef);
  if (m) PyModule_AddStringConstant(m, "COMPILER", B4D_BUILT_WITH);
  if (!m) return NULL;
  PyModule_AddIntConstant(m, "CODEC_ID", B4D_CODEC_ID);
  return m;
}

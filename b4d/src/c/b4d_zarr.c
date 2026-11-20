/*
 * b4d_zarr.c — native writer for zarr v3 sharded stores.
 *
 * One call writes a block of planes (nz x ny x nx, uint8/uint16, C-contiguous) into the shards covering
 * [z0, z0+nz) x [y0, y0+ny) x [x0, x0+nx) of a zarr v3 array whose outer chunks (= shards) are
 * (sz, sy, sx) and inner chunks (cz, cy, cx), using the "sharding_indexed" codec with index codecs
 * [bytes LE, crc32c] and index_location "end". The block must start on shard boundaries in every axis;
 * partial shards/chunks at the array edge are padded with the fill value (0) as zarr requires.
 *
 * Steps, all native: a pool of nthreads threads compresses every inner chunk of the block (gathered from
 * the strided block into a contiguous tile) with the selected codec; then shard files are assembled
 * (chunk blobs in C order + index) and written through a temporary name, optionally flushed (fsync /
 * FlushFileBuffers), and renamed into place. Shards are written concurrently by the pool.
 *
 * Codecs:  B4D_ZARR_CODEC_B4D   - B4D chunk = B4DP container + Blosc2 chunk, identical to b4d.compress()
 *          B4D_ZARR_CODEC_RAW   - zarr "bytes" codec (little-endian)
 *          B4D_ZARR_CODEC_BLOSC - zarr "blosc" codec (lz4/zstd/...; blosc1-compatible frames from a per-worker blosc2 context,
 *                                 so the compression runs on the pool like B4D; cname/clevel/shuffle)
 *          B4D_ZARR_CODEC_BLOBS - chunks pre-encoded by the caller (any numcodecs codec); only shard
 *                                 assembly + IO happen here
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "b4d_blosc2.h"
#include <blosc2.h>
#if defined(_MSC_VER) && !defined(__clang__)
  #define B4DZ_RESTRICT __restrict
#else
  #define B4DZ_RESTRICT __restrict__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <io.h>
  #include <direct.h>
  #include <fcntl.h>
  #define b4dz_mkdir(p) _mkdir(p)
  typedef HANDLE b4dz_thread;
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <fcntl.h>
  #define b4dz_mkdir(p) mkdir(p, 0775)
  typedef pthread_t b4dz_thread;
#endif

/* ------------------------------------------------------------------ atomics (counter for the task queue) */
#if defined(_MSC_VER) && !defined(__clang__)
static long b4dz_fetch_add(volatile long *p, long v) { return InterlockedExchangeAdd(p, v); }
#elif defined(__GNUC__) || defined(__clang__)
static long b4dz_fetch_add(volatile long *p, long v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
#else
static long b4dz_fetch_add(volatile long *p, long v) { long o = *p; *p += v; return o; }
#endif

/* ------------------------------------------------------------------ wall clock */
#if defined(_WIN32)
static double b4dz_now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
static double b4dz_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#endif

/* ------------------------------------------------------------------ crc32c (Castagnoli), table driven */
static uint32_t crc32c_table[256];
static int crc32c_ready = 0;
static void crc32c_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
    crc32c_table[i] = c;
  }
  crc32c_ready = 1;
}
static uint32_t crc32c(const uint8_t *p, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; i++) c = crc32c_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ portable threads */
typedef void (*b4dz_fn)(void *);
#if defined(_WIN32)
typedef struct { b4dz_fn fn; void *arg; } b4dz_tramp;
static DWORD WINAPI b4dz_tramp_run(LPVOID p) { b4dz_tramp *t = (b4dz_tramp *)p; t->fn(t->arg); free(t); return 0; }
/* Windows splits > 64 logical CPUs into processor groups and a new thread inherits its creator's group, so a
   plain pool would never leave the first 64 CPUs. Each worker is created suspended, assigned to a group
   round-robin (all CPUs of that group), then resumed. idx = worker index. */
static int b4dz_spawn(b4dz_thread *th, b4dz_fn fn, void *arg, int idx) {
  b4dz_tramp *t = (b4dz_tramp *)malloc(sizeof *t); if (!t) return -1; t->fn = fn; t->arg = arg;
  HANDLE h = CreateThread(NULL, 0, b4dz_tramp_run, t, CREATE_SUSPENDED, NULL);
  if (!h) { free(t); return -1; }
  WORD ngroups = GetActiveProcessorGroupCount();
  if (ngroups > 1) {
    GROUP_AFFINITY ga; memset(&ga, 0, sizeof ga);
    ga.Group = (WORD)(idx % ngroups);
    DWORD ncpu = GetActiveProcessorCount(ga.Group);
    ga.Mask = (ncpu >= 64) ? ~(KAFFINITY)0 : (((KAFFINITY)1 << ncpu) - 1);
    SetThreadGroupAffinity(h, &ga, NULL);                  /* best effort: on failure the thread keeps the parent's group */
  }
  ResumeThread(h);
  *th = h; return 0;
}

#else
typedef struct { b4dz_fn fn; void *arg; } b4dz_tramp;
static void *b4dz_tramp_run(void *p) { b4dz_tramp *t = (b4dz_tramp *)p; t->fn(t->arg); free(t); return NULL; }
static int b4dz_spawn(b4dz_thread *th, b4dz_fn fn, void *arg, int idx) {
  (void)idx;                                               /* Linux/macOS schedule across all CPUs by default */
  b4dz_tramp *t = (b4dz_tramp *)malloc(sizeof *t); if (!t) return -1; t->fn = fn; t->arg = arg;
  return pthread_create(th, NULL, b4dz_tramp_run, t);
}

#endif

/* ------------------------------------------------------------------ arena cache (reused across calls: fresh pages fault slowly) */
#define B4DZ_ARENA_CACHE 512
static uint8_t *g_arena_ptr[B4DZ_ARENA_CACHE]; static size_t g_arena_cap[B4DZ_ARENA_CACHE]; static int g_arena_n = 0;
#if defined(_WIN32)
static CRITICAL_SECTION g_arena_cs; static int g_arena_cs_init = 0;
static void arena_lock(void) { if (!g_arena_cs_init) { InitializeCriticalSection(&g_arena_cs); g_arena_cs_init = 1; } EnterCriticalSection(&g_arena_cs); }
static void arena_unlock(void) { LeaveCriticalSection(&g_arena_cs); }
#else
static pthread_mutex_t g_arena_mx = PTHREAD_MUTEX_INITIALIZER;
static void arena_lock(void) { pthread_mutex_lock(&g_arena_mx); }
static void arena_unlock(void) { pthread_mutex_unlock(&g_arena_mx); }
#endif
/* best fit per worker: the smallest cached arena of at least `hint` bytes, else the largest there is. Calls of very
   different sizes share the cache (multiscale: level 0 at 128 threads and the pyramid levels at 64 threads, running
   concurrently); taking the last-returned arena handed level 0 the pyramid's small arenas, which it then regrew with
   realloc + fresh page faults on every block (Sony IMX811 multiscale compress went from 0.75 to 3.3-4.3 s per block) */
static void arena_take(uint8_t **ptr, size_t *cap, int n, size_t hint) {
  arena_lock();
  for (int i = 0; i < n; i++) {
    if (g_arena_n <= 0) { ptr[i] = NULL; cap[i] = 0; continue; }
    int best = -1, big = 0;
    for (int k = 0; k < g_arena_n; k++) {
      if (g_arena_cap[k] > g_arena_cap[big]) big = k;
      if (g_arena_cap[k] >= hint && (best < 0 || g_arena_cap[k] < g_arena_cap[best])) best = k;
    }
    if (best < 0) best = big;
    ptr[i] = g_arena_ptr[best]; cap[i] = g_arena_cap[best];
    g_arena_n--; g_arena_ptr[best] = g_arena_ptr[g_arena_n]; g_arena_cap[best] = g_arena_cap[g_arena_n];
  }
  arena_unlock();
}
/* per-worker scratch slabs (band + tile + cbuf), cached across calls like the arenas: a fresh malloc per block
   re-faults 2-3 MB per worker per block (a fixed ~tens of ms per block at 128 workers) */
#define B4DZ_WBUF_CACHE 512
#define B4DZ_AL64(x) (((x) + 63) & ~(size_t)63)
static uint8_t *g_wbuf_ptr[B4DZ_WBUF_CACHE]; static size_t g_wbuf_cap[B4DZ_WBUF_CACHE]; static int g_wbuf_n = 0;
static uint8_t *wbuf_take(size_t need, size_t *cap) {
  uint8_t *p = NULL; size_t c = 0;
  arena_lock();
  if (g_wbuf_n > 0) { g_wbuf_n--; p = g_wbuf_ptr[g_wbuf_n]; c = g_wbuf_cap[g_wbuf_n]; }
  arena_unlock();
  if (p && c < need) { free(p); p = NULL; c = 0; }
  if (!p) { p = (uint8_t *)malloc(need); c = p ? need : 0; }
  *cap = c; return p;
}
static void wbuf_give(uint8_t *p, size_t cap) {
  if (!p) return;
  arena_lock();
  if (g_wbuf_n < B4DZ_WBUF_CACHE) { g_wbuf_ptr[g_wbuf_n] = p; g_wbuf_cap[g_wbuf_n] = cap; g_wbuf_n++; p = NULL; }
  arena_unlock();
  free(p);
}

static void arena_give(uint8_t **ptr, size_t *cap, int n) {
  arena_lock();
  for (int i = 0; i < n; i++) { if (ptr[i] && g_arena_n < B4DZ_ARENA_CACHE) { g_arena_ptr[g_arena_n] = ptr[i]; g_arena_cap[g_arena_n] = cap[i]; g_arena_n++; } else free(ptr[i]); }
  arena_unlock();
}

/* ------------------------------------------------------------------ aligned IO buffers (direct/unbuffered writes) */
#define B4DZ_SECTOR 4096                          /* multiple of every logical sector size in use (512, 4096) */
#define B4DZ_IOBUF_CACHE 256
static uint8_t *g_io_ptr[B4DZ_IOBUF_CACHE]; static size_t g_io_cap[B4DZ_IOBUF_CACHE]; static int g_io_n = 0;
static void *aligned_alloc_sector(size_t n) {
#if defined(_WIN32)
  return _aligned_malloc(n, B4DZ_SECTOR);
#else
  void *p = NULL; return posix_memalign(&p, B4DZ_SECTOR, n) == 0 ? p : NULL;
#endif
}
static void aligned_free_sector(void *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}
static uint8_t *iobuf_take(size_t need, size_t *cap_out) {   /* reuse a cached buffer >= need, else allocate */
  arena_lock();
  int best = -1;
  for (int i = 0; i < g_io_n; i++) if (g_io_cap[i] >= need && (best < 0 || g_io_cap[i] < g_io_cap[best])) best = i;
  uint8_t *p = NULL; size_t cap = 0;
  if (best >= 0) { p = g_io_ptr[best]; cap = g_io_cap[best]; g_io_n--; g_io_ptr[best] = g_io_ptr[g_io_n]; g_io_cap[best] = g_io_cap[g_io_n]; }
  arena_unlock();
  if (!p) { cap = (need + (size_t)B4DZ_SECTOR - 1) / B4DZ_SECTOR * B4DZ_SECTOR; cap += cap / 8; p = (uint8_t *)aligned_alloc_sector(cap); if (!p) return NULL; }
  *cap_out = cap; return p;
}
static void iobuf_give(uint8_t *p, size_t cap) {
  arena_lock();
  if (g_io_n < B4DZ_IOBUF_CACHE) { g_io_ptr[g_io_n] = p; g_io_cap[g_io_n] = cap; g_io_n++; p = NULL; }
  arena_unlock();
  if (p) aligned_free_sector(p);
}

/* write `len` bytes (buffer sector-aligned, capacity >= len rounded up to a sector) unbuffered + write-through,
   then set the exact file length. Returns 0 on success, 1 if direct IO is unavailable here (caller falls back),
   -1 on an IO error. */
static int write_file_direct(const char *path, uint8_t *buf, size_t len) {
  size_t padded = (len + (size_t)B4DZ_SECTOR - 1) / B4DZ_SECTOR * B4DZ_SECTOR;
  if (padded > len) memset(buf + len, 0, padded - len);
#if defined(_WIN32)
  HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, NULL);
  if (h == INVALID_HANDLE_VALUE) return 1;
  size_t done = 0;
  while (done < padded) {
    DWORD chunk = (DWORD)((padded - done > (1u << 30)) ? (1u << 30) : (padded - done)), got = 0;
    if (!WriteFile(h, buf + done, chunk, &got, NULL) || got != chunk) { CloseHandle(h); return -1; }
    done += got;
  }
  CloseHandle(h);
  if (padded != len) {                                   /* exact length: reopen buffered and set EOF */
    HANDLE h2 = CreateFileA(path, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h2 == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER li; li.QuadPart = (LONGLONG)len;
    if (!SetFilePointerEx(h2, li, NULL, FILE_BEGIN) || !SetEndOfFile(h2)) { CloseHandle(h2); return -1; }
    FlushFileBuffers(h2); CloseHandle(h2);
  }
  return 0;
#elif defined(O_DIRECT)
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) return 1;
  size_t done = 0;
  while (done < padded) {
    ssize_t got = write(fd, buf + done, padded - done);
    if (got <= 0) { close(fd); return -1; }
    done += (size_t)got;
  }
  if (padded != len && ftruncate(fd, (off_t)len) != 0) { close(fd); return -1; }
  fsync(fd); close(fd);
  return 0;
#else
  (void)path; (void)buf; (void)len; return 1;
#endif
}

void b4d_zarr_free_cache(void) {                           /* release cached arenas and IO buffers (e.g. at writer close) */
  arena_lock();
  for (int i = 0; i < g_arena_n; i++) free(g_arena_ptr[i]);
  g_arena_n = 0;
  for (int i = 0; i < g_wbuf_n; i++) free(g_wbuf_ptr[i]);
  g_wbuf_n = 0;
  for (int i = 0; i < g_io_n; i++) aligned_free_sector(g_io_ptr[i]);
  g_io_n = 0;
  arena_unlock();
}

/* ------------------------------------------------------------------ job description */
typedef struct {
  /* array geometry */
  const uint8_t *block; int typesize;
  int nz, ny, nx;                      /* block extent */
  int64_t z0, y0, x0;                  /* block origin in the array */
  int64_t ashape[3];                   /* full array shape (for edge clipping) */
  int sz, sy, sx;                      /* shard shape */
  int cz, cy, cx;                      /* inner chunk shape */
  /* codec */
  int codec; b4d_params params; uint8_t meta; int clevel; int shuffle; char cname[16];
  int blosc_typesize;
  /* pre-encoded blobs (BLOBS codec): one per inner chunk of the block in C order over (shards, chunks) */
  const uint8_t *const *blobs; const size_t *blob_sizes;
  /* output */
  const char *root; const char *key_sep; int sync; int nthreads; int direct;
  const char *key_prefix;              /* leading-axis chunk indices (e.g. "0" for channel 0), or "" */
  volatile long direct_used, direct_fallback, blosc1_fallback;
  /* work */
  int nsz, nsy, nsx;                   /* shards in the block */
  int ncz, ncy, ncx;                   /* chunks per shard */
  int group;                           /* x-tiles gathered per band (<= B4DZ_GROUP; band kept to ~1 MB) */
  size_t nshards, nchunks, ngroups;    /* totals; groups = compression tasks */
  uint8_t **out; size_t *out_len;      /* per chunk (pointers resolved after compression) */
  int *out_worker; size_t *out_off;    /* per chunk: arena of the worker that produced it + offset */
  uint8_t **arena; size_t *arena_len, *arena_cap;   /* per worker */
  volatile long next_worker;
  volatile long groups_mask;           /* bit g set when a worker ran in processor group g (Windows) */
  double t_compress, t_io;
  volatile long next_chunk, next_shard;
  volatile long error;
  /* fused level-1 downsampling (optional): box mean of the gathered band written into ds_out, whose
     shape is ceil(nz/fz) x ceil(ny/fy) x ceil(nx/fx) for the block (clipped to the array) */
  uint8_t *ds_out; int fz, fy, fx; int ds_oy, ds_ox;
  /* stats */
  long long *shard_bytes;              /* per shard, summed by the caller thread */
} b4dz_job;

static int64_t b4dz_min64(int64_t a, int64_t b) { return a < b ? a : b; }

#define B4DZ_GROUP 16        /* max x-tiles gathered together as one band (long gather runs); the band is capped at
                                B4DZ_BAND_BYTES so it stays cache-resident between the gather and the tile cuts */
#define B4DZ_BAND_BYTES ((size_t)1 << 20)

/* chunk index -> (shard, chunk-in-shard) decomposition helpers */
static void chunk_origin(const b4dz_job *j, size_t si, size_t ki, int *bz, int *by, int *bx) {
  int szi = (int)(si / ((size_t)j->nsy * j->nsx)), syi = (int)((si / j->nsx) % j->nsy), sxi = (int)(si % j->nsx);
  int kzi = (int)(ki / ((size_t)j->ncy * j->ncx)), kyi = (int)((ki / j->ncx) % j->ncy), kxi = (int)(ki % j->ncx);
  *bz = szi * j->sz + kzi * j->cz; *by = syi * j->sy + kyi * j->cy; *bx = sxi * j->sx + kxi * j->cx;
}

/* append `n` bytes to worker w's arena; returns the offset (arena may move: pointers are resolved at the end) */
static int arena_put(b4dz_job *j, int w, const uint8_t *a, size_t na, const uint8_t *b, size_t nb, size_t *off_out) {
  size_t need = j->arena_len[w] + na + nb;
  if (need > j->arena_cap[w]) {
    size_t ncap = j->arena_cap[w] ? j->arena_cap[w] : (size_t)16 << 20;
    while (ncap < need) ncap *= 2;
    uint8_t *na_ = (uint8_t *)realloc(j->arena[w], ncap); if (!na_) return -1;
    j->arena[w] = na_; j->arena_cap[w] = ncap;
  }
  *off_out = j->arena_len[w];
  if (na) memcpy(j->arena[w] + j->arena_len[w], a, na);
  if (nb) memcpy(j->arena[w] + j->arena_len[w] + na, b, nb);
  j->arena_len[w] = need;
  return 0;
}

/* ---- zarr "blosc" codec (lz4, zstd, ...) on the pool: Blosc1-format chunks from a per-worker Blosc2 context ----------
   c-blosc2's only Blosc1-format entry point, blosc1_compress(), serialises every call behind one global mutex, so the
   pool would run it one chunk at a time. blosc2_compress_ctx() is lock-free per context but always writes the 32-byte
   Blosc2 header and run-length-encodes constant streams (cbytes 0 / negative), which c-blosc1 readers (numcodecs and
   zarr-python, tensorstore, neuroglancer) reject. blosc1_from_blosc2() rewrites the chunk into the Blosc1 layout:
   16-byte header, block starts shifted, compressed streams copied verbatim, run streams re-emitted as a hand-built LZ4
   block or zstd RLE frame (stored raw for other codecs). Anything unexpected falls back to a Blosc1 memcpyed chunk
   (header + raw tile, always valid), counted in stats->blosc1_fallback. */
static int32_t b1_rd32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static void b1_wr32(uint8_t *p, int32_t v) { memcpy(p, &v, 4); }

/* LZ4 block decoding to n copies of v: one literal, a match at offset 1, then the 5 literals the block format requires */
static int32_t b1_run_lz4(uint8_t *o, size_t cap, uint8_t v, int32_t n) {
  if (n < 16) return -1;
  int32_t ml = n - 6 - 4, k = 0;                              /* match length less the 4-byte minimum */
  size_t need = 4 + (ml >= 15 ? 1 + (size_t)(ml - 15) / 255 : 0) + 6;
  if (need > cap) return -1;
  o[k++] = (uint8_t)((1 << 4) | (ml < 15 ? ml : 15)); o[k++] = v; o[k++] = 1; o[k++] = 0;
  if (ml >= 15) { int32_t r = ml - 15; while (r >= 255) { o[k++] = 255; r -= 255; } o[k++] = (uint8_t)r; }
  o[k++] = 5 << 4; for (int i = 0; i < 5; i++) o[k++] = v;
  return k;
}
/* zstd frame decoding to n copies of v: single-segment header + RLE blocks of at most 128 KiB */
static int32_t b1_run_zstd(uint8_t *o, size_t cap, uint8_t v, int32_t n) {
  if (n < 1) return -1;
  int32_t k = 0, rem = n;
  if (cap < 13 + ((size_t)n / 131072 + 1) * 4) return -1;
  o[k++] = 0x28; o[k++] = 0xB5; o[k++] = 0x2F; o[k++] = 0xFD;
  if (n <= 255) { o[k++] = 0x20; o[k++] = (uint8_t)n; }
  else if (n <= 65535 + 256) { o[k++] = 0x60; uint32_t m = (uint32_t)(n - 256); o[k++] = (uint8_t)m; o[k++] = (uint8_t)(m >> 8); }
  else { o[k++] = 0xA0; b1_wr32(o + k, n); k += 4; }
  do {
    int32_t bs = rem < 131072 ? rem : 131072; rem -= bs;
    uint32_t h = (uint32_t)(rem == 0) | (1u << 1) | ((uint32_t)bs << 3);
    o[k++] = (uint8_t)h; o[k++] = (uint8_t)(h >> 8); o[k++] = (uint8_t)(h >> 16); o[k++] = v;
  } while (rem > 0);
  return k;
}
/* a stream of n copies of v for the Blosc1 chunk: synthesised for LZ4/zstd, otherwise stored (cbytes == n means raw) */
static int32_t b1_run(int fmt, uint8_t *o, size_t cap, uint8_t v, int32_t n) {
  int32_t r = -1;
  if (fmt == BLOSC_LZ4_FORMAT) r = b1_run_lz4(o, cap, v, n);
  else if (fmt == BLOSC_ZSTD_FORMAT) r = b1_run_zstd(o, cap, v, n);
  if (r < 0 || r >= n) { if ((size_t)n > cap) return -1; memset(o, v, (size_t)n); r = n; }
  return r;
}
/* Blosc1 header flags from the Blosc2 header: compressor (bits 5-7), dont-split (bit 4) and the shuffle filter */
static int b1_flags(const uint8_t *s, uint8_t *f1) {
  *f1 = (uint8_t)(s[2] & 0xF0);
  for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) {
    uint8_t f = s[16 + i];
    if (f == BLOSC_SHUFFLE) *f1 |= BLOSC_DOSHUFFLE; else if (f == BLOSC_BITSHUFFLE) *f1 |= BLOSC_DOBITSHUFFLE; else if (f != BLOSC_NOFILTER) return -1;
  }
  return 0;
}
/* Blosc1 chunk of nbytes zeros: Blosc2 writes a header-only "special zero" chunk for an all-zero tile */
static int32_t b1_zeros(uint8_t *d, size_t cap, const uint8_t *s, int32_t nbytes, int32_t bsize) {
  uint8_t f1; if (b1_flags(s, &f1)) return -1;
  f1 |= 0x10;                                                 /* one stream per block */
  const int fmt = (s[2] >> 5) & 7;
  int32_t nblocks = nbytes / bsize, leftover = nbytes % bsize; if (leftover) nblocks++;
  size_t out = (size_t)BLOSC_MIN_HEADER_LENGTH + 4 * (size_t)nblocks; if (out > cap) return -1;
  memcpy(d, s, BLOSC_MIN_HEADER_LENGTH); d[0] = BLOSC1_VERSION_FORMAT; d[2] = f1;
  for (int32_t b = 0; b < nblocks; b++) {
    const int32_t bs = (leftover && b == nblocks - 1) ? leftover : bsize;
    b1_wr32(d + BLOSC_MIN_HEADER_LENGTH + 4 * b, (int32_t)out);
    if (out + 4 > cap) return -1;
    int32_t n = b1_run(fmt, d + out + 4, cap - out - 4, 0, bs); if (n < 0) return -1;
    b1_wr32(d + out, n); out += 4 + (size_t)n;
  }
  if (out > (size_t)INT32_MAX) return -1;
  b1_wr32(d + 12, (int32_t)out);
  return (int32_t)out;
}
/* Blosc1 memcpyed chunk: the fallback that is valid for every codec */
static int32_t b1_memcpyed(uint8_t *d, size_t cap, const uint8_t *tile, size_t n, int typesize) {
  if (cap < n + BLOSC_MIN_HEADER_LENGTH || n > (size_t)INT32_MAX - BLOSC_MIN_HEADER_LENGTH) return -1;
  d[0] = BLOSC1_VERSION_FORMAT; d[1] = 1; d[2] = BLOSC_MEMCPYED; d[3] = (uint8_t)typesize;
  b1_wr32(d + 4, (int32_t)n); b1_wr32(d + 8, (int32_t)n); b1_wr32(d + 12, (int32_t)(n + BLOSC_MIN_HEADER_LENGTH));
  memcpy(d + BLOSC_MIN_HEADER_LENGTH, tile, n);
  return (int32_t)(n + BLOSC_MIN_HEADER_LENGTH);
}
/* rewrite the Blosc2 chunk s[0..cs) as a Blosc1 chunk in d; returns its size or <0 if the layout is not the expected one */
static int32_t blosc1_from_blosc2(const uint8_t *s, int32_t cs, uint8_t *d, size_t cap) {
  if (cs < BLOSC_EXTENDED_HEADER_LENGTH || s[0] != BLOSC2_VERSION_FORMAT_STABLE) return -1;
  const uint8_t flags = s[2], ts = s[3];
  const int32_t nbytes = b1_rd32(s + 4), bsize = b1_rd32(s + 8);
  if (nbytes <= 0 || bsize <= 0 || bsize > nbytes || ts == 0) return -1;
  const int special = (s[31] >> 4) & 7;                       /* BLOSC2_SPECIAL_ZERO = 1: header only, all zeros */
  if (special) return (special == 1 && cs == BLOSC_EXTENDED_HEADER_LENGTH) ? b1_zeros(d, cap, s, nbytes, bsize) : -1;
  if (flags & BLOSC_MEMCPYED) {
    if (cs != BLOSC_EXTENDED_HEADER_LENGTH + nbytes) return -1;
    return b1_memcpyed(d, cap, s + BLOSC_EXTENDED_HEADER_LENGTH, (size_t)nbytes, ts);
  }
  if (s[31] & (BLOSC2_USEDICT | BLOSC2_INSTR_CODEC)) return -1;
  uint8_t f1; if (b1_flags(s, &f1)) return -1;
  const int split = !(flags & 0x10);
  if (split && (ts > 16 || bsize / ts < 128)) return -1;       /* c-blosc1 would not split such a block */
  const int fmt = (flags >> 5) & 7;
  int32_t nblocks = nbytes / bsize, leftover = nbytes % bsize; if (leftover) nblocks++;
  if (cs < BLOSC_EXTENDED_HEADER_LENGTH + 4 * nblocks) return -1;
  size_t out = (size_t)BLOSC_MIN_HEADER_LENGTH + 4 * (size_t)nblocks; if (out > cap) return -1;
  memcpy(d, s, BLOSC_MIN_HEADER_LENGTH); d[0] = BLOSC1_VERSION_FORMAT; d[2] = f1;
  for (int32_t b = 0; b < nblocks; b++) {
    const int last_partial = leftover && b == nblocks - 1;
    const int32_t bs = last_partial ? leftover : bsize, nst = (split && !last_partial) ? ts : 1, ne = bs / nst;
    int32_t so = b1_rd32(s + BLOSC_EXTENDED_HEADER_LENGTH + 4 * b);
    if (so < BLOSC_EXTENDED_HEADER_LENGTH || so > cs - 4) return -1;
    b1_wr32(d + BLOSC_MIN_HEADER_LENGTH + 4 * b, (int32_t)out);
    for (int32_t k = 0; k < nst; k++) {
      if (so > cs - 4) return -1;
      int32_t cb = b1_rd32(s + so); so += 4;
      if (cb > 0) {                                             /* compressed or stored stream: verbatim */
        if (cb > ne || cb > cs - so || out + 4 + (size_t)cb > cap) return -1;
        b1_wr32(d + out, cb); memcpy(d + out + 4, s + so, (size_t)cb); so += cb; out += 4 + (size_t)cb;
      } else {                                                  /* Blosc2 run: 0 = zeros, -v = v repeated (+1 token byte) */
        uint8_t v = 0;
        if (cb < 0) { if (cb < -255 || so > cs - 1) return -1; v = (uint8_t)(-cb); so += 1; }
        if (out + 4 > cap) return -1;
        int32_t n = b1_run(fmt, d + out + 4, cap - out - 4, v, ne); if (n < 0) return -1;
        b1_wr32(d + out, n); out += 4 + (size_t)n;
      }
    }
  }
  if (out > (size_t)INT32_MAX) return -1;
  b1_wr32(d + 12, (int32_t)out);
  return (int32_t)out;
}

/* encode one tile that is already contiguous (cz, cy, cx) in `tile`; records the blob for chunk ci in worker w's arena */
static int encode_tile(b4dz_job *j, int w, size_t ci, blosc2_context *ctx, const uint8_t *tile, uint8_t *cbuf, size_t cbuf_cap,
                       uint8_t *cbuf2, size_t cbuf2_cap) {
  const int ts = j->typesize;
  size_t tile_bytes = (size_t)j->cz * j->cy * j->cx * ts;
  size_t off = 0, len = 0; int csize;
  if (j->codec == B4D_ZARR_CODEC_RAW) {
    if (arena_put(j, w, tile, tile_bytes, NULL, 0, &off)) return -1;
    len = tile_bytes;
  } else if (j->codec == B4D_ZARR_CODEC_BLOSC) {            /* Blosc2 context (lock-free) -> Blosc1 layout for the readers */
    csize = blosc2_compress_ctx(ctx, tile, (int32_t)tile_bytes, cbuf, (int32_t)cbuf_cap);
    int32_t c1 = csize > 0 ? blosc1_from_blosc2(cbuf, csize, cbuf2, cbuf2_cap) : -1;
    if (c1 <= 0) { b4dz_fetch_add(&j->blosc1_fallback, 1); c1 = b1_memcpyed(cbuf2, cbuf2_cap, tile, tile_bytes, ts); if (c1 <= 0) return -2; }
    if (arena_put(j, w, cbuf2, (size_t)c1, NULL, 0, &off)) return -1;
    len = (size_t)c1;
  } else {
    csize = blosc2_compress_ctx(ctx, tile, (int32_t)tile_bytes, cbuf, (int32_t)cbuf_cap);
    if (csize <= 0) return -2;
    uint8_t hdr[7 + 24 + 8]; uint8_t *p = hdr;
    memcpy(p, "B4DP", 4); p[4] = 1; p[5] = (uint8_t)(ts == 1 ? 0 : 1); p[6] = 3; p += 7;
    uint64_t shp[3] = { (uint64_t)j->cz, (uint64_t)j->cy, (uint64_t)j->cx }; memcpy(p, shp, 24); p += 24;
    uint32_t np_ = 1, cs = (uint32_t)csize; memcpy(p, &np_, 4); memcpy(p + 4, &cs, 4);
    if (arena_put(j, w, hdr, sizeof hdr, cbuf, (size_t)csize, &off)) return -1;
    len = sizeof hdr + (size_t)csize;
  }
  j->out_worker[ci] = w; j->out_off[ci] = off; j->out_len[ci] = len;
  return 0;
}

/* one task = up to B4DZ_GROUP consecutive x-tiles of one (shard, z-chunk, y-chunk) band:
   gather the band from the block in long runs, then cut the tiles out of the band buffer (cache-resident) */
static int encode_group(b4dz_job *j, int w, size_t gi, blosc2_context *ctx, b4d_params *prm, uint8_t *band, uint8_t *tile, uint8_t *cbuf, size_t cbuf_cap,
                        uint8_t *cbuf2, size_t cbuf2_cap) {
  const int G = j->group;
  size_t groups_per_row = (size_t)(j->ncx + G - 1) / G;
  size_t rows_per_shard = (size_t)j->ncz * j->ncy;
  size_t si = gi / (rows_per_shard * groups_per_row), rem = gi % (rows_per_shard * groups_per_row);
  size_t row = rem / groups_per_row, g = rem % groups_per_row;
  int kz = (int)(row / j->ncy), ky = (int)(row % j->ncy), kx0 = (int)(g * G);
  int ntile = (j->ncx - kx0 < G) ? j->ncx - kx0 : G;
  size_t ki0 = ((size_t)kz * j->ncy + ky) * j->ncx + kx0;
  int bz, by, bx; chunk_origin(j, si, ki0, &bz, &by, &bx);
  const int ts = j->typesize;
  int gw = ntile * j->cx;                                                       /* band width in px */
  /* clipped extents of the band inside block and array */
  int dz = (int)b4dz_min64(j->cz, b4dz_min64(j->nz - bz, j->ashape[0] - (j->z0 + bz)));
  int dy = (int)b4dz_min64(j->cy, b4dz_min64(j->ny - by, j->ashape[1] - (j->y0 + by)));
  int dx = (int)b4dz_min64(gw, b4dz_min64(j->nx - bx, j->ashape[2] - (j->x0 + bx)));
  if (dz <= 0 || dy <= 0 || dx <= 0) {
    for (int t = 0; t < ntile; t++) { j->out_worker[si * rows_per_shard * j->ncx + ki0 + t] = -1; j->out_len[si * rows_per_shard * j->ncx + ki0 + t] = 0; }
    return 0;
  }
  size_t band_row = (size_t)gw * ts;
  if (dz < j->cz || dy < j->cy || dx < gw) memset(band, 0, (size_t)j->cz * j->cy * band_row);   /* zero fill only at a clipped edge */
  for (int z = 0; z < dz; z++) for (int y = 0; y < dy; y++)
    memcpy(band + ((size_t)z * j->cy + y) * band_row, j->block + (((size_t)(bz + z) * j->ny + (by + y)) * j->nx + bx) * ts, (size_t)dx * ts);
  if (j->ds_out && ts == 1) {                                                  /* the same for uint8 */
    int oz0 = bz / j->fz, oy0 = by / j->fy, ox0 = bx / j->fx;
    int noz = (dz + j->fz - 1) / j->fz, noy = (dy + j->fy - 1) / j->fy, nox = (dx + j->fx - 1) / j->fx;
    uint32_t *B4DZ_RESTRICT rowsum = (uint32_t *)(band + (size_t)j->cz * j->cy * band_row);
    for (int oz = 0; oz < noz; oz++) {
      int z0 = oz * j->fz, z1 = z0 + j->fz; if (z1 > dz) z1 = dz;
      for (int oy = 0; oy < noy; oy++) {
        int y0 = oy * j->fy, y1 = y0 + j->fy; if (y1 > dy) y1 = dy;
        uint8_t *B4DZ_RESTRICT out = j->ds_out + ((size_t)(oz0 + oz) * j->ds_oy + (oy0 + oy)) * j->ds_ox + ox0;
        uint32_t nzy = (uint32_t)((z1 - z0) * (y1 - y0));
        {
          const uint8_t *B4DZ_RESTRICT r0 = band + ((size_t)z0 * j->cy + y0) * band_row;
          for (int x = 0; x < dx; x++) rowsum[x] = r0[x];
          for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) {
            if (z == z0 && y == y0) continue;
            const uint8_t *B4DZ_RESTRICT r = band + ((size_t)z * j->cy + y) * band_row;
            for (int x = 0; x < dx; x++) rowsum[x] += r[x];
          }
        }
        if (j->fx == 2) {
          int full = dx / 2; uint32_t n = nzy * 2, half = n / 2;
          if (n == 8)      for (int ox = 0; ox < full; ox++) out[ox] = (uint8_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + 4u) >> 3);
          else if (n == 4) for (int ox = 0; ox < full; ox++) out[ox] = (uint8_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + 2u) >> 2);
          else             for (int ox = 0; ox < full; ox++) out[ox] = (uint8_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + half) / n);
          if (full < nox) out[full] = (uint8_t)((rowsum[2 * full] + nzy / 2) / nzy);
        } else {
          for (int ox = 0; ox < nox; ox++) {
            int x0 = ox * j->fx, x1 = x0 + j->fx; if (x1 > dx) x1 = dx;
            uint32_t acc = 0; for (int x = x0; x < x1; x++) acc += rowsum[x];
            uint32_t n = nzy * (uint32_t)(x1 - x0); out[ox] = (uint8_t)((acc + n / 2) / n);
          }
        }
      }
    }
  }
  if (j->ds_out && ts == 2) {                                                  /* level-1 box mean from the cached band */
    /* the band starts at (bz, by, bx) in the block: multiples of cz, cy, cx*GROUP (all even), so output windows align.
       Vectorisable form: sum the fz*fy input rows of an output row into a 32-bit row (contiguous, no gathers), then
       pair horizontally. The band is L2-resident, so this costs arithmetic only. */
    int oz0 = bz / j->fz, oy0 = by / j->fy, ox0 = bx / j->fx;
    int noz = (dz + j->fz - 1) / j->fz, noy = (dy + j->fy - 1) / j->fy, nox = (dx + j->fx - 1) / j->fx;
    uint32_t *B4DZ_RESTRICT rowsum = (uint32_t *)(band + (size_t)j->cz * j->cy * band_row);   /* scratch after the band: gw uint32 */
    for (int oz = 0; oz < noz; oz++) {
      int z0 = oz * j->fz, z1 = z0 + j->fz; if (z1 > dz) z1 = dz;
      for (int oy = 0; oy < noy; oy++) {
        int y0 = oy * j->fy, y1 = y0 + j->fy; if (y1 > dy) y1 = dy;
        uint16_t *B4DZ_RESTRICT out = (uint16_t *)j->ds_out + ((size_t)(oz0 + oz) * j->ds_oy + (oy0 + oy)) * j->ds_ox + ox0;
        uint32_t nzy = (uint32_t)((z1 - z0) * (y1 - y0));
        /* 1) sum the input rows (vectorises: contiguous uint16 -> uint32 adds) */
        {
          const uint16_t *B4DZ_RESTRICT r0 = (const uint16_t *)(band + ((size_t)z0 * j->cy + y0) * band_row);
          for (int x = 0; x < dx; x++) rowsum[x] = r0[x];
          for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) {
            if (z == z0 && y == y0) continue;
            const uint16_t *B4DZ_RESTRICT r = (const uint16_t *)(band + ((size_t)z * j->cy + y) * band_row);
            for (int x = 0; x < dx; x++) rowsum[x] += r[x];
          }
        }
        /* 2) pair horizontally (fx == 2: the common case; generic fx handled below) */
        if (j->fx == 2) {
          int full = dx / 2; uint32_t n = nzy * 2, half = n / 2;
          if (n == 8)      for (int ox = 0; ox < full; ox++) out[ox] = (uint16_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + 4u) >> 3);
          else if (n == 4) for (int ox = 0; ox < full; ox++) out[ox] = (uint16_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + 2u) >> 2);
          else             for (int ox = 0; ox < full; ox++) out[ox] = (uint16_t)((rowsum[2 * ox] + rowsum[2 * ox + 1] + half) / n);
          if (full < nox) out[full] = (uint16_t)((rowsum[2 * full] + nzy / 2) / nzy);         /* odd last column */
        } else {
          for (int ox = 0; ox < nox; ox++) {
            int x0 = ox * j->fx, x1 = x0 + j->fx; if (x1 > dx) x1 = dx;
            uint32_t acc = 0; for (int x = x0; x < x1; x++) acc += rowsum[x];
            uint32_t n = nzy * (uint32_t)(x1 - x0); out[ox] = (uint16_t)((acc + n / 2) / n);
          }
        }
      }
    }
  }
  size_t tile_row = (size_t)j->cx * ts;
  for (int t = 0; t < ntile; t++) {
    size_t ci = si * rows_per_shard * j->ncx + ki0 + t;
    int tx = t * j->cx;
    if (tx >= dx) { j->out_worker[ci] = -1; j->out_len[ci] = 0; continue; }  /* tile entirely outside the array */
    const uint8_t *src = band;                       /* a one-tile band already has the tile's layout: no copy */
    if (G > 1) {
      for (int z = 0; z < j->cz; z++) for (int y = 0; y < j->cy; y++)
        memcpy(tile + ((size_t)z * j->cy + y) * tile_row, band + ((size_t)z * j->cy + y) * band_row + (size_t)tx * ts, tile_row);
      src = tile;
    }
    if (prm->pixmap) { prm->map_y0 = j->params.map_y0 + (uint32_t)(j->y0 + by); prm->map_x0 = j->params.map_x0 + (uint32_t)(j->x0 + bx + tx); }   /* the context reads prm at each compression */
    int rc = encode_tile(j, w, ci, ctx, src, cbuf, cbuf_cap, cbuf2, cbuf2_cap);
    if (rc) return rc;
  }
  return 0;
}

static void note_group(b4dz_job *j) {
#if defined(_WIN32)
  PROCESSOR_NUMBER pn; GetCurrentProcessorNumberEx(&pn);
  long bit = 1L << (pn.Group & 31);
  for (;;) { long cur = j->groups_mask; if (cur & bit) break; if (InterlockedCompareExchange(&j->groups_mask, cur | bit, cur) == cur) break; }
#else
  (void)j;
#endif
}

static void compress_worker(void *arg) {
  b4dz_job *j = (b4dz_job *)arg;
  const int w = (int)b4dz_fetch_add(&j->next_worker, 1);
  note_group(j);
  const int ts = j->typesize;
  size_t tile_bytes = (size_t)j->cz * j->cy * j->cx * ts;
  size_t band_bytes = tile_bytes * j->group + (size_t)j->cx * j->group * sizeof(uint32_t);   /* + rowsum scratch for the fused downsample */
  size_t cbuf_cap = tile_bytes + BLOSC2_MAX_OVERHEAD + 64;
  size_t cbuf2_cap = j->codec == B4D_ZARR_CODEC_BLOSC ? tile_bytes + tile_bytes / 8 + 4096 : 0;   /* Blosc1 rewrite of the chunk */
  size_t slab_cap = 0, slab_need = B4DZ_AL64(band_bytes) + B4DZ_AL64(tile_bytes) + B4DZ_AL64(cbuf_cap) + B4DZ_AL64(cbuf2_cap);
  uint8_t *slab = wbuf_take(slab_need, &slab_cap);
  uint8_t *band = slab, *tile = slab ? slab + B4DZ_AL64(band_bytes) : NULL, *cbuf = tile ? tile + B4DZ_AL64(tile_bytes) : NULL;
  uint8_t *cbuf2 = (cbuf && cbuf2_cap) ? cbuf + B4DZ_AL64(cbuf_cap) : NULL;
  blosc2_context *ctx = NULL; b4d_params prm = j->params; prm.row_width = (uint32_t)j->cx; prm.plane_rows = (uint32_t)j->cy;
  if (j->codec == B4D_ZARR_CODEC_B4D) {                     /* one context per worker; every tile has the same size */
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
    cp.typesize = ts; cp.compcode = B4D_CODEC_ID; cp.compcode_meta = j->meta; cp.nthreads = 1;
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
    int32_t row_bytes = (int32_t)(j->cx * ts), rows = (1 << 20) / row_bytes; if (rows < 1) rows = 1;
    int32_t bs = rows * row_bytes; if (bs > (int32_t)tile_bytes) bs = (int32_t)tile_bytes; if (bs > ts) bs = bs / ts * ts;
    if (prm.predictor == B4D_PRED_MEAN3D) {                                       /* mean3d: whole planes per Blosc2 block */
      int32_t pb = (int32_t)(j->cy * j->cx * ts), planes = bs / pb; if (planes < 1) planes = 1;
      bs = planes * pb; if (bs > (int32_t)tile_bytes) bs = (int32_t)tile_bytes;
    }
    if (prm.pixmap) bs = (int32_t)tile_bytes;                                     /* per-pixel map: the block must know where its rows sit */
    cp.blocksize = bs; cp.codec_params = &prm;
    ctx = blosc2_create_cctx(cp);
  } else if (j->codec == B4D_ZARR_CODEC_BLOSC) {                /* lz4 / zstd / blosclz / zlib: one context per worker */
    blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;
    int cc = j->cname[0] ? blosc2_compname_to_compcode(j->cname) : BLOSC_LZ4;
    cp.compcode = (uint8_t)(cc < 0 ? BLOSC_LZ4 : cc); cp.clevel = (uint8_t)(j->clevel < 0 ? 0 : j->clevel > 9 ? 9 : j->clevel);
    cp.typesize = j->blosc_typesize; cp.nthreads = 1; cp.blocksize = 0;
    /* measured on this 128-core machine (one context per thread, 262 KB tiles): zstd/zlib with BITshuffle collapse from ~45 GB/s to
       4-9 GB/s once the Blosc block reaches ~24 KB (worse with more threads); 16 KB blocks keep the scaling. Byte shuffle has no cliff. */
    if (j->shuffle == 2 && (cc == BLOSC_ZSTD || cc == BLOSC_ZLIB)) cp.blocksize = 16384;
    for (int i = 0; i < BLOSC2_MAX_FILTERS; i++) cp.filters[i] = BLOSC_NOFILTER;
    cp.filters[BLOSC2_MAX_FILTERS - 1] = (uint8_t)(j->shuffle == 2 ? BLOSC_BITSHUFFLE : j->shuffle == 1 ? BLOSC_SHUFFLE : BLOSC_NOFILTER);
    ctx = blosc2_create_cctx(cp);
  }
  if (!band || !tile || !cbuf || ((j->codec == B4D_ZARR_CODEC_B4D || j->codec == B4D_ZARR_CODEC_BLOSC) && !ctx) ||
      (j->codec == B4D_ZARR_CODEC_BLOSC && !cbuf2)) { b4dz_fetch_add(&j->error, 1); goto done; }
  for (;;) {
    long gi = b4dz_fetch_add(&j->next_chunk, 1);
    if ((size_t)gi >= j->ngroups || j->error) break;
    if (encode_group(j, w, (size_t)gi, ctx, &prm, band, tile, cbuf, cbuf_cap, cbuf2, cbuf2_cap) != 0) { b4dz_fetch_add(&j->error, 1); break; }
  }
done:
  if (ctx) blosc2_free_ctx(ctx);
  wbuf_give(slab, slab_cap);
  b4d_trim_thread_scratch((size_t)8 << 20);                /* keep the per-thread codec scratch warm unless it is large */
}

/* ---- host entropy coding of residual tiles (GPU path with the ANS coder): see b4d_blosc2.h */
static void run_pool(void *arg, b4dz_fn fn, int n);
typedef struct {
  const int16_t *res; int ntiles; uint32_t nelem, cols, ph; int cz, mode, qi; float conv, offset, rn; int pred, entropy; uint8_t meta;
  volatile long next_tile, next_worker, error; int nthreads;
  uint8_t **arena; size_t *alen, *acap;                 /* per worker */
  int *tthr; size_t *toff; int64_t *tlen; int8_t *status; /* per tile */
} rt_job;

static void rt_frame(uint8_t *o, const rt_job *j, uint32_t psize) {
  /* identical to the CUDA kernel's b4dk_phase_write: B4DP container (39 bytes) + Blosc2 chunk header (32) + bstart + csize */
  uint32_t nbytes = j->nelem * 2u, csize = 32u + 4u + 4u + psize;
  o[0] = 'B'; o[1] = '4'; o[2] = 'D'; o[3] = 'P'; o[4] = 1; o[5] = 1; o[6] = 3;
  memset(o + 7, 0, 24); uint32_t cz = (uint32_t)j->cz, cy = j->ph, cx = j->cols;
  memcpy(o + 7, &cz, 4); memcpy(o + 15, &cy, 4); memcpy(o + 23, &cx, 4);
  uint32_t one = 1; memcpy(o + 31, &one, 4); memcpy(o + 35, &csize, 4);
  uint8_t *c = o + 39;
  memset(c, 0, 40); c[0] = 5; c[1] = 3; c[2] = 0xD5; c[3] = 2;
  memcpy(c + 4, &nbytes, 4); memcpy(c + 8, &nbytes, 4); memcpy(c + 12, &csize, 4);
  c[22] = (uint8_t)B4D_CODEC_ID; c[23] = j->meta;
  uint32_t bstart = 36; memcpy(c + 32, &bstart, 4); memcpy(c + 36, &psize, 4);
}

static void rt_worker(void *arg) {
  rt_job *j = (rt_job *)arg;
  const int w = (int)b4dz_fetch_add(&j->next_worker, 1);
  const uint32_t n = j->nelem, nbytes = n * 2u; const size_t cap = (size_t)nbytes + 4096;
  uint8_t *payload = (uint8_t *)malloc(cap);
  if (!payload) { b4dz_fetch_add(&j->error, 1); return; }
  for (;;) {
    long t = b4dz_fetch_add(&j->next_tile, 1);
    if (t >= j->ntiles || j->error) break;
    const int16_t *r = j->res + (size_t)t * n;
    int plain = 1;                                                     /* all residuals after the first zero: all-zero / all-equal tile */
    for (uint32_t i = 1; i < n; i++) if (r[i]) { plain = 0; break; }
    int ps = plain ? 0 : b4d_encode_residual_block2(r, n, (uint8_t)j->mode, 2, (uint8_t)j->qi, j->conv, j->offset, j->rn, j->cols, j->ph,
                                                    j->pred, j->entropy, payload, (int32_t)cap);
    if (plain || ps <= 0 || (size_t)ps + 40 >= nbytes) { j->status[t] = 1; j->tlen[t] = 0; j->tthr[t] = w; j->toff[t] = 0; continue; }
    size_t need = 39 + 40 + (size_t)ps;
    if (j->alen[w] + need > j->acap[w]) {
      size_t nc = j->acap[w] * 2 + need + (1u << 20); uint8_t *na = (uint8_t *)realloc(j->arena[w], nc);
      if (!na) { b4dz_fetch_add(&j->error, 1); break; }
      j->arena[w] = na; j->acap[w] = nc;
    }
    uint8_t *o = j->arena[w] + j->alen[w];
    rt_frame(o, j, (uint32_t)ps); memcpy(o + 79, payload, (size_t)ps);
    j->status[t] = 0; j->tthr[t] = w; j->toff[t] = j->alen[w]; j->tlen[t] = (int64_t)need; j->alen[w] += need;
  }
  free(payload);
  b4d_trim_thread_scratch((size_t)8 << 20);
}

int b4d_zarr_encode_residual_tiles(const int16_t *res, int ntiles, uint32_t nelem, int mode, int qi, float conv, float offset, float rn,
                                   uint32_t cols, uint32_t plane_rows, int predictor, int entropy, uint8_t meta, int nthreads,
                                   uint8_t **blob_out, size_t *blob_len, int64_t *offsets, int64_t *lengths, int8_t *status) {
  if (!res || ntiles <= 0 || nelem == 0 || cols == 0 || plane_rows == 0 || !blob_out || !blob_len || !offsets || !lengths || !status) return -1;
  rt_job j; memset(&j, 0, sizeof j);
  j.res = res; j.ntiles = ntiles; j.nelem = nelem; j.cols = cols; j.ph = plane_rows; j.cz = (int)(nelem / (cols * plane_rows));
  j.mode = mode; j.qi = qi; j.conv = conv; j.offset = offset; j.rn = rn; j.pred = predictor; j.entropy = entropy; j.meta = meta;
  int nt = nthreads < 1 ? 1 : nthreads; if (nt > ntiles) nt = ntiles; j.nthreads = nt;
  j.arena = (uint8_t **)calloc((size_t)nt, sizeof *j.arena); j.alen = (size_t *)calloc((size_t)nt, sizeof *j.alen); j.acap = (size_t *)calloc((size_t)nt, sizeof *j.acap);
  j.tthr = (int *)calloc((size_t)ntiles, sizeof *j.tthr); j.toff = (size_t *)calloc((size_t)ntiles, sizeof *j.toff); j.tlen = lengths; j.status = status;
  if (!j.arena || !j.alen || !j.acap || !j.tthr || !j.toff) { free(j.arena); free(j.alen); free(j.acap); free(j.tthr); free(j.toff); return -1; }
  run_pool(&j, rt_worker, nt);
  int rc = j.error ? -1 : 0;
  size_t total = 0;
  if (rc == 0) {
    for (int t = 0; t < ntiles; t++) total += (size_t)lengths[t];
    uint8_t *out = (uint8_t *)malloc(total ? total : 1);
    if (!out) rc = -1;
    else {
      size_t pos = 0;
      for (int t = 0; t < ntiles; t++) {
        offsets[t] = (int64_t)pos;
        if (lengths[t]) { memcpy(out + pos, j.arena[j.tthr[t]] + j.toff[t], (size_t)lengths[t]); pos += (size_t)lengths[t]; }
      }
      *blob_out = out; *blob_len = total;
    }
  }
  for (int w = 0; w < nt; w++) free(j.arena[w]);
  free(j.arena); free(j.alen); free(j.acap); free(j.tthr); free(j.toff);
  return rc;
}
void b4d_zarr_free_buffer(void *p) { free(p); }

/* ---- shard assembly and file IO */
static int mkdir_p(const char *path) {
  char tmp[4096]; size_t n = strlen(path); if (n >= sizeof tmp) return -1;
  memcpy(tmp, path, n + 1);
  for (size_t i = 1; i < n; i++) if (tmp[i] == '/' || tmp[i] == '\\') { char c = tmp[i]; tmp[i] = 0; b4dz_mkdir(tmp); tmp[i] = c; }
  b4dz_mkdir(tmp);
  struct stat st; return (stat(tmp, &st) == 0) ? 0 : -1;
}

static int write_file_atomic(const char *path, const uint8_t *const *parts, const size_t *lens, size_t nparts, int sync) {
  char tmp[4096]; if (strlen(path) + 8 >= sizeof tmp) return -1;
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb"); if (!f) return -1;
  setvbuf(f, NULL, _IOFBF, 1 << 20);
  for (size_t i = 0; i < nparts; i++) if (lens[i] && fwrite(parts[i], 1, lens[i], f) != lens[i]) { fclose(f); remove(tmp); return -1; }
  if (fflush(f) != 0) { fclose(f); remove(tmp); return -1; }
  if (sync) {
#if defined(_WIN32)
    HANDLE h = (HANDLE)_get_osfhandle(_fileno(f)); if (h != INVALID_HANDLE_VALUE) FlushFileBuffers(h);
#else
    fsync(fileno(f));
#endif
  }
  fclose(f);
  remove(path);                                            /* overwrite semantics */
  return rename(tmp, path) == 0 ? 0 : -1;
}

static int write_shard(b4dz_job *j, size_t si) {
  size_t per_shard = (size_t)j->ncz * j->ncy * j->ncx;
  int szi = (int)(si / ((size_t)j->nsy * j->nsx)), syi = (int)((si / j->nsx) % j->nsy), sxi = (int)(si % j->nsx);
  /* skip shards entirely outside the array */
  if (j->z0 + (int64_t)szi * j->sz >= j->ashape[0] || j->y0 + (int64_t)syi * j->sy >= j->ashape[1] || j->x0 + (int64_t)sxi * j->sx >= j->ashape[2]) return 0;
  const uint8_t **parts = (const uint8_t **)malloc((per_shard + 1) * sizeof *parts);
  size_t *lens = (size_t *)malloc((per_shard + 1) * sizeof *lens);
  uint8_t *index = (uint8_t *)malloc(per_shard * 16 + 4);
  if (!parts || !lens || !index) { free(parts); free(lens); free(index); return -1; }
  uint64_t off = 0; size_t np_ = 0;
  for (size_t k = 0; k < per_shard; k++) {
    size_t ci = si * per_shard + k; uint64_t o = ~0ull, n = ~0ull;
    if (j->out[ci] && j->out_len[ci]) { parts[np_] = j->out[ci]; lens[np_] = j->out_len[ci]; np_++; o = off; n = j->out_len[ci]; off += n; }
    memcpy(index + k * 16, &o, 8); memcpy(index + k * 16 + 8, &n, 8);
  }
  uint32_t crc = crc32c(index, per_shard * 16); memcpy(index + per_shard * 16, &crc, 4);
  parts[np_] = index; lens[np_] = per_shard * 16 + 4; np_++;
  /* key: root/c<sep>Z<sep>Y<sep>X with global shard indices */
  int64_t gz = j->z0 / j->sz + szi, gy = j->y0 / j->sy + syi, gx = j->x0 / j->sx + sxi;
  if (!j->root || !j->root[0]) {                                           /* null sink: assemble only (benchmarking) */
    long long total = 0; for (size_t i = 0; i < np_; i++) total += (long long)lens[i]; j->shard_bytes[si] = total;
    free(parts); free(lens); free(index); return 0;
  }
  char path[4096], dir[4096];
  const char *sep = j->key_sep;
  if (j->key_prefix && j->key_prefix[0])
    snprintf(path, sizeof path, "%s/c%s%s%s%lld%s%lld%s%lld", j->root, sep, j->key_prefix, sep, (long long)gz, sep, (long long)gy, sep, (long long)gx);
  else
    snprintf(path, sizeof path, "%s/c%s%lld%s%lld%s%lld", j->root, sep, (long long)gz, sep, (long long)gy, sep, (long long)gx);
  if (sep[0] == '/') {                                   /* create the key's directory chain */
    size_t n = strlen(path); while (n > 0 && path[n - 1] != '/') n--;
    if (n > 0 && n < sizeof dir) { memcpy(dir, path, n - 1); dir[n - 1] = 0; mkdir_p(dir); }
  }
  int rc = -2;
  if (j->direct) {                                       /* assemble into one aligned buffer, write unbuffered */
    size_t total = 0; for (size_t i = 0; i < np_; i++) total += lens[i];
    size_t cap = 0; uint8_t *buf = iobuf_take(total + B4DZ_SECTOR, &cap);
    if (buf) {
      size_t o = 0; for (size_t i = 0; i < np_; i++) { memcpy(buf + o, parts[i], lens[i]); o += lens[i]; }
      char tmp[4104]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
      int d = write_file_direct(tmp, buf, total);
      iobuf_give(buf, cap);
      if (d == 0) { remove(path); rc = rename(tmp, path) == 0 ? 0 : -1; b4dz_fetch_add(&j->direct_used, 1); }
      else if (d < 0) { remove(tmp); rc = -1; }
      else b4dz_fetch_add(&j->direct_fallback, 1);      /* direct IO unavailable on this volume: buffered path */
    }
  }
  if (rc == -2) rc = write_file_atomic(path, parts, lens, np_, j->sync);
  if (rc == 0) { long long total = 0; for (size_t i = 0; i < np_; i++) total += (long long)lens[i]; j->shard_bytes[si] = total; }
  free(parts); free(lens); free(index);
  return rc;
}

static void shard_worker(void *arg) {
  b4dz_job *j = (b4dz_job *)arg;
  for (;;) {
    long si = b4dz_fetch_add(&j->next_shard, 1);
    if ((size_t)si >= j->nshards || j->error) break;
    if (write_shard(j, (size_t)si) != 0) { b4dz_fetch_add(&j->error, 1); break; }
  }
}

/* ------------------------------------------------------------------ persistent thread pool
   Workers are created once (processor-group assignment at creation), sleep between jobs, and serve a queue of
   jobs; concurrent writes (blocks in flight, pyramid levels) share them. A job asks for n participants; it is
   complete when n workers have run fn(arg) once each (fn itself loops over a shared task counter). */
#if defined(_WIN32)
typedef CRITICAL_SECTION b4dz_mutex; typedef CONDITION_VARIABLE b4dz_cond;
static void mx_init(b4dz_mutex *m) { InitializeCriticalSection(m); }
static void mx_lock(b4dz_mutex *m) { EnterCriticalSection(m); }
static void mx_unlock(b4dz_mutex *m) { LeaveCriticalSection(m); }
static void cv_init(b4dz_cond *c) { InitializeConditionVariable(c); }
static void cv_wait(b4dz_cond *c, b4dz_mutex *m) { SleepConditionVariableCS(c, m, INFINITE); }
static void cv_broadcast(b4dz_cond *c) { WakeAllConditionVariable(c); }
#else
typedef pthread_mutex_t b4dz_mutex; typedef pthread_cond_t b4dz_cond;
static void mx_init(b4dz_mutex *m) { pthread_mutex_init(m, NULL); }
static void mx_lock(b4dz_mutex *m) { pthread_mutex_lock(m); }
static void mx_unlock(b4dz_mutex *m) { pthread_mutex_unlock(m); }
static void cv_init(b4dz_cond *c) { pthread_cond_init(c, NULL); }
static void cv_wait(b4dz_cond *c, b4dz_mutex *m) { pthread_cond_wait(c, m); }
static void cv_broadcast(b4dz_cond *c) { pthread_cond_broadcast(c); }
#endif

typedef struct b4dz_pjob { b4dz_fn fn; void *arg; int slots, want, done; struct b4dz_pjob *next; } b4dz_pjob;
static struct {
  b4dz_mutex mx; b4dz_cond work, finished; int init, created, idle, cap;
  b4dz_pjob *head;
} g_pool;

/* Slot placement (Windows, > 1 processor group). A job's slots are taken by whichever pool workers wake first, and each worker
   was bound to one group when it was created, so a group can get more of a job's threads than it has physical cores (they then
   share cores as SMT siblings while other cores idle). B4D_AFFINITY=core: the worker that takes slot s of a job runs on
   physical core s (both of its logical CPUs), cores taken round-robin over the groups, so a job's first ncores slots never share
   a core. Default (B4D_AFFINITY unset or "group"): a group per worker only. Measured on a 128-core EPYC 9754 (4 groups),
   1-128 writer threads: core pinning was never faster and up to 8% slower in mode 1 at 64-96 threads, so it is opt-in. */
#if defined(_WIN32)
static GROUP_AFFINITY *g_cores = NULL; static int g_ncores = 0, g_affinity = -1;
static void cores_init(void) {
  const char *env = getenv("B4D_AFFINITY");
  g_affinity = (env && !strcmp(env, "core")) ? 1 : 0;
  if (!g_affinity || GetActiveProcessorGroupCount() < 2) { g_affinity = 0; return; }
  DWORD len = 0; GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  uint8_t *buf = (uint8_t *)malloc(len);
  if (!buf || !GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) { free(buf); g_affinity = 0; return; }
  int n = 0, ng = (int)GetActiveProcessorGroupCount();
  for (DWORD off = 0; off < len; off += ((PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off))->Size) n++;
  GROUP_AFFINITY *all = (GROUP_AFFINITY *)calloc((size_t)n, sizeof *all), *out = (GROUP_AFFINITY *)calloc((size_t)n, sizeof *out);
  int *seen = (int *)calloc((size_t)ng, sizeof(int)); int k = 0;
  if (!all || !out || !seen) { free(buf); free(all); free(out); free(seen); g_affinity = 0; return; }
  for (DWORD off = 0; off < len; off += ((PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off))->Size)
    all[k++] = ((PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off))->Processor.GroupMask[0];
  /* interleave: the i-th core of group 0, of group 1, ..., then the (i+1)-th */
  int m = 0;
  for (int round = 0; m < n; round++) {
    int any = 0;
    for (int g = 0; g < ng; g++) {
      int c = 0;
      for (int i = 0; i < n; i++) if (all[i].Group == g && c++ == round) { out[m++] = all[i]; any = 1; break; }
    }
    if (!any) break;
  }
  free(buf); free(all); free(seen);
  g_cores = out; g_ncores = m;
}
static void pin_slot(int s) { if (g_affinity && g_ncores) SetThreadGroupAffinity(GetCurrentThread(), &g_cores[s % g_ncores], NULL); }
#else
static void cores_init(void) {}
static void pin_slot(int s) { (void)s; }
#endif

static void pool_worker(void *unused) {
  (void)unused;
  mx_lock(&g_pool.mx);
  for (;;) {
    b4dz_pjob *jb = g_pool.head;
    while (jb && jb->slots == 0) jb = jb->next;
    if (!jb) { g_pool.idle++; cv_wait(&g_pool.work, &g_pool.mx); g_pool.idle--; continue; }
    const int slot = jb->want - jb->slots;
    jb->slots--;
    mx_unlock(&g_pool.mx);
    pin_slot(slot);
    jb->fn(jb->arg);
    mx_lock(&g_pool.mx);
    jb->done++;
    if (jb->done == jb->want) cv_broadcast(&g_pool.finished);
  }
}

static int b4dz_ncpu(void) {
#if defined(_WIN32)
  return (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 8;
#endif
}

static void pool_ensure_init(void) {
  if (g_pool.init) return;
  arena_lock();                                  /* serialise first-time init */
  if (!g_pool.init) {
    mx_init(&g_pool.mx); cv_init(&g_pool.work); cv_init(&g_pool.finished);
    cores_init();
    g_pool.cap = 2 * b4dz_ncpu();
    const char *env = getenv("B4D_POOL_THREADS");      /* per-process cap on runnable workers (e.g. one of several writer processes) */
    if (env && atoi(env) > 0) g_pool.cap = atoi(env);
    if (g_pool.cap < 4) g_pool.cap = 4;
    g_pool.init = 1;
  }
  arena_unlock();
}

static void run_pool(void *arg, b4dz_fn fn, int n) {
  if (n <= 1) { fn(arg); return; }
  pool_ensure_init();
  b4dz_pjob jb; jb.fn = fn; jb.arg = arg; jb.slots = n; jb.want = n; jb.done = 0;
  mx_lock(&g_pool.mx);
  /* grow the pool so this job does not have to wait for workers busy with another job */
  int need = n - g_pool.idle; int can = g_pool.cap - g_pool.created;
  for (int i = 0; i < need && i < can; i++) {
    b4dz_thread th;
    if (b4dz_spawn(&th, pool_worker, NULL, g_pool.created) == 0) {
      g_pool.created++;
#if defined(_WIN32)
      CloseHandle(th);                           /* detached; workers live for the process */
#else
      pthread_detach(th);
#endif
    } else break;
  }
  jb.next = g_pool.head; g_pool.head = &jb;
  cv_broadcast(&g_pool.work);
  while (jb.done < jb.want) {
    if (jb.slots > 0 && g_pool.idle == 0 && g_pool.created == 0) { /* no workers at all: run inline */
      jb.slots--; mx_unlock(&g_pool.mx); fn(arg); mx_lock(&g_pool.mx); jb.done++; continue;
    }
    cv_wait(&g_pool.finished, &g_pool.mx);
  }
  /* unlink */
  if (g_pool.head == &jb) g_pool.head = jb.next;
  else { b4dz_pjob *p = g_pool.head; while (p && p->next != &jb) p = p->next; if (p) p->next = jb.next; }
  mx_unlock(&g_pool.mx);
}

int b4d_zarr_write_block(const void *block, int typesize, int nz, int ny, int nx,
                         int64_t z0, int64_t y0, int64_t x0, const int64_t ashape[3],
                         int sz, int sy, int sx, int cz, int cy, int cx,
                         int codec, const b4d_params *params, uint8_t meta, const char *cname, int clevel, int shuffle,
                         const uint8_t *const *blobs, const size_t *blob_sizes,
                         const char *root, const char *key_sep, int sync, int nthreads, long long *bytes_written, b4d_zarr_stats *stats,
                         void *ds_out, int fz, int fy, int fx, int direct, const char *key_prefix) {
  if (!crc32c_ready) crc32c_init();
  if (!block && codec != B4D_ZARR_CODEC_BLOBS) return -1;
  if (sz % cz || sy % cy || sx % cx) return -2;
  if (z0 % sz || y0 % sy || x0 % sx) return -3;
  if (nz % sz && z0 + nz < ashape[0]) return -4;          /* block must cover whole shards unless it ends at the array edge */
  if (ny % sy && y0 + ny < ashape[1]) return -4;
  if (nx % sx && x0 + nx < ashape[2]) return -4;
  b4dz_job j; memset(&j, 0, sizeof j);
  j.block = (const uint8_t *)block; j.typesize = typesize; j.nz = nz; j.ny = ny; j.nx = nx;
  j.z0 = z0; j.y0 = y0; j.x0 = x0; memcpy(j.ashape, ashape, sizeof j.ashape);
  j.sz = sz; j.sy = sy; j.sx = sx; j.cz = cz; j.cy = cy; j.cx = cx;
  j.codec = codec; if (params) j.params = *params; j.meta = meta; j.clevel = clevel; j.shuffle = shuffle; j.blosc_typesize = typesize;
  if (cname) { strncpy(j.cname, cname, sizeof j.cname - 1); }
  j.blobs = blobs; j.blob_sizes = blob_sizes;
  if (ds_out) {
    /* fused downsampling requires factors 1 or 2, even chunk/band dimensions and an array-aligned block */
    if ((fz != 1 && fz != 2) || (fy != 1 && fy != 2) || (fx != 1 && fx != 2)) return -7;
    if ((fz == 2 && (cz & 1)) || (fy == 2 && (cy & 1)) || (fx == 2 && (cx & 1))) return -7;
    if ((fz == 2 && (z0 & 1)) || (fy == 2 && (y0 & 1)) || (fx == 2 && (x0 & 1))) return -7;
    j.ds_out = (uint8_t *)ds_out; j.fz = fz; j.fy = fy; j.fx = fx;
    int64_t ey = ashape[1] - y0; if (ey > ny) ey = ny; int64_t ex = ashape[2] - x0; if (ex > nx) ex = nx;
    j.ds_oy = (int)((ey + fy - 1) / fy); j.ds_ox = (int)((ex + fx - 1) / fx);
  }
  j.root = root; j.key_sep = key_sep ? key_sep : "/"; j.sync = sync; j.nthreads = nthreads < 1 ? 1 : nthreads; j.direct = direct;
  j.key_prefix = key_prefix;
  j.nsz = (nz + sz - 1) / sz; j.nsy = (ny + sy - 1) / sy; j.nsx = (nx + sx - 1) / sx;
  j.ncz = sz / cz; j.ncy = sy / cy; j.ncx = sx / cx;
  j.nshards = (size_t)j.nsz * j.nsy * j.nsx; j.nchunks = j.nshards * j.ncz * j.ncy * j.ncx;
  {
    /* a tile of half the band or more is gathered alone and encoded from the band (no tile copy): 64^3 uint16, +1 % at 128
       threads on the EPYC 9754 (measured); smaller tiles keep the long runs (2 per band was 1 % slower for uint8) */
    size_t tile_bytes = (size_t)cz * cy * cx * typesize, g = tile_bytes ? B4DZ_BAND_BYTES / tile_bytes : B4DZ_GROUP;
    if (tile_bytes >= B4DZ_BAND_BYTES / 2) g = 1;
    j.group = g < 1 ? 1 : (g > B4DZ_GROUP ? B4DZ_GROUP : (int)g);
  }
  j.ngroups = j.nshards * (size_t)j.ncz * j.ncy * ((j.ncx + j.group - 1) / j.group);
  j.out = (uint8_t **)calloc(j.nchunks, sizeof *j.out); j.out_len = (size_t *)calloc(j.nchunks, sizeof *j.out_len);
  j.out_worker = (int *)calloc(j.nchunks, sizeof *j.out_worker); j.out_off = (size_t *)calloc(j.nchunks, sizeof *j.out_off);
  j.arena = (uint8_t **)calloc((size_t)j.nthreads, sizeof *j.arena); j.arena_len = (size_t *)calloc((size_t)j.nthreads, sizeof *j.arena_len);
  j.arena_cap = (size_t *)calloc((size_t)j.nthreads, sizeof *j.arena_cap);
  j.shard_bytes = (long long *)calloc(j.nshards, sizeof *j.shard_bytes);
  if (!j.out || !j.out_len || !j.out_worker || !j.out_off || !j.arena || !j.arena_len || !j.arena_cap || !j.shard_bytes) {
    free(j.out); free(j.out_len); free(j.out_worker); free(j.out_off); free(j.arena); free(j.arena_len); free(j.arena_cap); free(j.shard_bytes); return -1; }
  int rc = 0;
  {                                                          /* reuse warmed arenas from previous calls, best fit for this call */
    size_t raw = (size_t)nz * ny * nx * (size_t)typesize;    /* compressed output per worker is well below raw / nthreads: */
    arena_take(j.arena, j.arena_cap, j.nthreads, raw / (size_t)j.nthreads / 8);   /* raw / 8 per worker separates the levels' sizes */
  }
  double t0 = b4dz_now();
  if (codec == B4D_ZARR_CODEC_BLOBS) {
    for (size_t ci = 0; ci < j.nchunks; ci++) { j.out[ci] = (uint8_t *)blobs[ci]; j.out_len[ci] = blob_sizes[ci]; }
  } else {
    run_pool(&j, compress_worker, j.nthreads);
    if (j.error) rc = -5;
    for (size_t ci = 0; ci < j.nchunks && rc == 0; ci++)          /* resolve arena pointers now that arenas are final */
      j.out[ci] = (j.out_worker[ci] >= 0 && j.out_len[ci]) ? j.arena[j.out_worker[ci]] + j.out_off[ci] : NULL;
  }
  double t1 = b4dz_now();
  if (rc == 0) {
    int nio = j.nthreads < (int)j.nshards ? j.nthreads : (int)j.nshards; if (nio < 1) nio = 1;
    run_pool(&j, shard_worker, nio);
    if (j.error) rc = -6;
  }
  double t2 = b4dz_now();
  long long total = 0; for (size_t si = 0; si < j.nshards; si++) total += j.shard_bytes[si];
  arena_give(j.arena, j.arena_cap, j.nthreads);
  free(j.out); free(j.out_len); free(j.out_worker); free(j.out_off); free(j.arena); free(j.arena_len); free(j.arena_cap); free(j.shard_bytes);
  if (bytes_written) *bytes_written = total;
  if (stats) { stats->t_compress = t1 - t0; stats->t_io = t2 - t1; stats->groups_mask = (int)j.groups_mask; stats->bytes_written = total;
               stats->direct_used = (int)j.direct_used; stats->direct_fallback = (int)j.direct_fallback; stats->blosc1_fallback = (int)j.blosc1_fallback; }
  return rc;
}

/* ------------------------------------------------------------------ box downsampling (multiscale pyramids) */
typedef struct {
  const uint8_t *src; uint8_t *dst; int typesize;
  int nz, ny, nx;            /* source */
  int fz, fy, fx;            /* factors (1 or more) */
  int oz, oy, ox;            /* output = ceil(n / f) */
  int rows_per_task;         /* output rows per task (tasks span all planes) */
  long ntasks;
  volatile long next;
} b4dz_ds_job;

/* fast path: fx == 2, interior columns (x1 - x0 == 2), full z/y window; accumulates pairs of columns */
static void ds_rows_u16(const b4dz_ds_job *j, int z, int y, uint32_t *acc) {
  int z0 = z * j->fz, z1 = z0 + j->fz; if (z1 > j->nz) z1 = j->nz;
  int y0 = y * j->fy, y1 = y0 + j->fy; if (y1 > j->ny) y1 = j->ny;
  uint16_t *out = (uint16_t *)j->dst + ((size_t)z * j->oy + y) * j->ox;
  int nzy = (z1 - z0) * (y1 - y0);
  if (j->fx == 2) {
    int full = j->nx / 2;                                              /* outputs with two source columns */
    uint32_t n = (uint32_t)nzy * 2, half = n / 2;
    /* accumulate row sums of column pairs over the window (acc: per-worker buffer of ox entries) */
    memset(acc, 0, (size_t)j->ox * sizeof *acc);
    for (int zz = z0; zz < z1; zz++) for (int yy = y0; yy < y1; yy++) {
      const uint16_t *row = (const uint16_t *)j->src + ((size_t)zz * j->ny + yy) * j->nx;
      for (int x = 0; x < full; x++) acc[x] += (uint32_t)row[2 * x] + row[2 * x + 1];
      if (full < j->ox) acc[full] += row[2 * full];                   /* odd last column */
    }
    if (n == 8)      for (int x = 0; x < full; x++) out[x] = (uint16_t)((acc[x] + 4) >> 3);
    else if (n == 4) for (int x = 0; x < full; x++) out[x] = (uint16_t)((acc[x] + 2) >> 2);
    else             for (int x = 0; x < full; x++) out[x] = (uint16_t)((acc[x] + half) / n);
    if (full < j->ox) { uint32_t m = (uint32_t)nzy; out[full] = (uint16_t)((acc[full] + m / 2) / m); }
    return;
  }
  for (int x = 0; x < j->ox; x++) {                                    /* general factors */
    int x0 = x * j->fx, x1 = x0 + j->fx; if (x1 > j->nx) x1 = j->nx;
    uint32_t sum = 0, n = 0;
    for (int zz = z0; zz < z1; zz++) for (int yy = y0; yy < y1; yy++) {
      const uint16_t *row = (const uint16_t *)j->src + ((size_t)zz * j->ny + yy) * j->nx;
      for (int xx = x0; xx < x1; xx++) { sum += row[xx]; n++; }
    }
    out[x] = (uint16_t)((sum + n / 2) / n);
  }
}

static void ds_rows_u8(const b4dz_ds_job *j, int z, int y, uint32_t *acc) {
  int z0 = z * j->fz, z1 = z0 + j->fz; if (z1 > j->nz) z1 = j->nz;
  int y0 = y * j->fy, y1 = y0 + j->fy; if (y1 > j->ny) y1 = j->ny;
  uint8_t *out = j->dst + ((size_t)z * j->oy + y) * j->ox;
  int nzy = (z1 - z0) * (y1 - y0);
  if (j->fx == 2) {                                                    /* fast path, as ds_rows_u16 (same rounding) */
    int full = j->nx / 2;
    uint32_t n = (uint32_t)nzy * 2, half = n / 2;
    memset(acc, 0, (size_t)j->ox * sizeof *acc);
    for (int zz = z0; zz < z1; zz++) for (int yy = y0; yy < y1; yy++) {
      const uint8_t *row = j->src + ((size_t)zz * j->ny + yy) * j->nx;
      for (int x = 0; x < full; x++) acc[x] += (uint32_t)row[2 * x] + row[2 * x + 1];
      if (full < j->ox) acc[full] += row[2 * full];                   /* odd last column */
    }
    if (n == 8)      for (int x = 0; x < full; x++) out[x] = (uint8_t)((acc[x] + 4) >> 3);
    else if (n == 4) for (int x = 0; x < full; x++) out[x] = (uint8_t)((acc[x] + 2) >> 2);
    else             for (int x = 0; x < full; x++) out[x] = (uint8_t)((acc[x] + half) / n);
    if (full < j->ox) { uint32_t m = (uint32_t)nzy; out[full] = (uint8_t)((acc[full] + m / 2) / m); }
    return;
  }
  for (int x = 0; x < j->ox; x++) {
    int x0 = x * j->fx, x1 = x0 + j->fx; if (x1 > j->nx) x1 = j->nx;
    uint32_t sum = 0, n = 0;
    for (int zz = z0; zz < z1; zz++) for (int yy = y0; yy < y1; yy++) {
      const uint8_t *row = j->src + ((size_t)zz * j->ny + yy) * j->nx;
      for (int xx = x0; xx < x1; xx++) { sum += row[xx]; n++; }
    }
    out[x] = (uint8_t)((sum + n / 2) / n);
  }
}

static void ds_worker(void *arg) {
  b4dz_ds_job *j = (b4dz_ds_job *)arg;
  long tasks_per_plane = (j->oy + j->rows_per_task - 1) / j->rows_per_task;
  uint32_t *acc = (uint32_t *)malloc((size_t)(j->ox > 0 ? j->ox : 1) * sizeof *acc);   /* one per worker, not per row */
  if (!acc) return;
  for (;;) {
    long t = b4dz_fetch_add(&j->next, 1);
    if (t >= j->ntasks) break;
    int z = (int)(t / tasks_per_plane), yb = (int)(t % tasks_per_plane) * j->rows_per_task;
    int ye = yb + j->rows_per_task; if (ye > j->oy) ye = j->oy;
    for (int y = yb; y < ye; y++) (j->typesize == 2) ? ds_rows_u16(j, z, y, acc) : ds_rows_u8(j, z, y, acc);
  }
  free(acc);
}

/* dst must hold ceil(nz/fz) x ceil(ny/fy) x ceil(nx/fx) elements; edge windows average the available pixels */
int b4d_zarr_downsample(const void *src, int typesize, int nz, int ny, int nx, int fz, int fy, int fx, void *dst, int nthreads) {
  if (!src || !dst || (typesize != 1 && typesize != 2) || fz < 1 || fy < 1 || fx < 1) return -1;
  b4dz_ds_job j; memset(&j, 0, sizeof j);
  j.src = (const uint8_t *)src; j.dst = (uint8_t *)dst; j.typesize = typesize;
  j.nz = nz; j.ny = ny; j.nx = nx; j.fz = fz; j.fy = fy; j.fx = fx;
  j.oz = (nz + fz - 1) / fz; j.oy = (ny + fy - 1) / fy; j.ox = (nx + fx - 1) / fx;
  j.rows_per_task = 32; if (j.rows_per_task > j.oy) j.rows_per_task = j.oy > 0 ? j.oy : 1;
  j.ntasks = (long)j.oz * ((j.oy + j.rows_per_task - 1) / j.rows_per_task);
  int n = nthreads < 1 ? 1 : nthreads; if (n > j.ntasks) n = (int)j.ntasks; if (n < 1) n = 1;
  run_pool(&j, ds_worker, n);
  return 0;
}

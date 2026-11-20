/* egrabber_ingest - zero-copy camera ingest engine on top of the Euresys eGrabber SDK (Coaxlink).
 *
 * The engine owns a small number of large, contiguous "block" buffers (frames_per_block frames each).
 * Every frame slot of every block is announced to the frame grabber as its own GenTL buffer (UserMemoryArray),
 * so the grabber's DMA engine writes each camera frame straight into the slot a downstream consumer (the B4D
 * zarr writer) will compress from: no copy between the camera and the compressor.
 *
 * Flow:  eg_open() -> eg_start() -> loop { eg_wait_block(); consume; eg_release_block(); } -> eg_stop() -> eg_close()
 *
 * A block is handed out when all its slots have been filled (or, after eg_stop(), when it holds the final partial
 * run of frames). While a block is held by the consumer its slots are not in the grabber's input queue; if every
 * block is held the grabber has nowhere to put the next frame and the camera's frames are lost (counted in
 * eg_stats: lost_by_timestamp = gaps in the camera timestamps, rejected = grabber "rejected frame" events).
 *
 * C ABI (usable from ctypes); every function returns 0 on success, EG_TIMEOUT or a negative error code;
 * eg_last_error() gives the message of the last failure on the calling handle.
 */
#ifndef EGRABBER_INGEST_H
#define EGRABBER_INGEST_H
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  ifdef EG_INGEST_BUILD
#    define EG_API __declspec(dllexport)
#  else
#    define EG_API __declspec(dllimport)
#  endif
#else
#  define EG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EG_OK          0
#define EG_TIMEOUT     1
#define EG_ERR        -1   /* see eg_last_error() */
#define EG_ERR_ARG    -2
#define EG_ERR_STATE  -3
#define EG_ERR_NOCAM  -4
#define EG_ERR_ALLOC  -5
#define EG_ERR_ORDER  -6   /* a frame landed in a slot other than the predicted one */

typedef struct eg_handle eg_handle;

typedef struct eg_config {
    const char *serial;          /* camera DeviceSerialNumber to open; NULL/"" = first discovered camera */
    const char *pixel_format;    /* remote PixelFormat to select; NULL/"" = leave as is (the VPS-245MX2 only has Mono12) */
    double      fps;             /* AcquisitionFrameRate; <= 0 = the camera's maximum (AcquisitionFrameRate.Max) */
    double      exposure_us;     /* ExposureTime in microseconds; <= 0 = leave as is */
    int         frames_per_block;/* slots per block (= B4D block / shard depth) */
    int         nblocks;         /* blocks in the pool (>= 2; writer blocks in flight + 1) */
    int         large_pages;     /* 1: try 2 MB large pages for the block memory (needs SeLockMemoryPrivilege), else 4 KB */
    int         prefault_threads;/* threads used to touch (zero) the block memory before announcing it; 0 = 16 */
    uint64_t    max_frames;      /* stop acquisition automatically after this many delivered frames; 0 = until eg_stop() */
    int         verbose;         /* log to stderr */
    /* Caller-provided block memory (optional). ext_blocks[nblocks] are the base addresses of nblocks regions of at least
     * frames_per_block * payload bytes each; the engine does not allocate, pre-fault or free them.
     *   ext_kind = 0 : ignored, the engine allocates host memory itself (VirtualAlloc)
     *   ext_kind = 1 : host memory (e.g. CUDA page-locked memory from cudaHostAlloc, so the GPU can DMA from it too)
     *   ext_kind = 2 : NVIDIA device memory (cudaMalloc): announced as GPUDirect RDMA targets (NvidiaRdmaMemory);
     *                  needs a GPU + driver that allow RDMA (on Windows: TCC driver model) */
    void *const *ext_blocks;
    int         ext_kind;
    /* Added for device-only use (a host application that owns the camera settings):
     *   attach_only = 1 : open the camera as configured, change nothing (no PixelFormat / AcquisitionFrameRate / ExposureTime
     *                     writes); fps, exposure_us and pixel_format above are ignored.
     *   nblocks = 0     : open the device only, no buffer pool; allocate one later with eg_pool_alloc(). */
    int         attach_only;
} eg_config;

typedef struct eg_camera_info {
    char     vendor[64], model[64], serial[64], version[64];
    char     pixel_format[32];   /* remote PixelFormat after configuration */
    char     unpacking[32];      /* stream UnpackingMode (Lsb = 12-bit value in the low bits of a 16-bit word) */
    char     interface_id[128];  /* frame grabber */
    char     link_config[32];    /* CxpLinkConfiguration */
    int64_t  width, height;      /* frame size in pixels */
    int      bytes_per_pixel;    /* container size (2 for Mono12 unpacked to 16 bits) */
    uint64_t payload_bytes;      /* bytes per frame (one GenTL buffer) */
    double   fps, fps_max, exposure_us;
    double   link_throughput_limit; /* DeviceLinkThroughputLimit (camera, units as reported) */
} eg_camera_info;

typedef struct eg_frame_info {
    uint64_t frame_id;           /* grabber frame counter (BUFFER_INFO_FRAMEID) */
    uint64_t timestamp_ns;       /* grabber timestamp of the frame (BUFFER_INFO_TIMESTAMP_NS) */
    uint64_t host_ns;            /* host monotonic clock when the new-buffer event was processed */
    uint64_t size_filled;        /* bytes delivered into the slot */
    uint32_t incomplete;         /* BUFFER_INFO_IS_INCOMPLETE */
    uint32_t slot;               /* slot within the block (== position in the block's frame table) */
} eg_frame_info;

typedef struct eg_block_info {
    int                  index;      /* block index in the pool */
    int                  nframes;    /* frames delivered into this block (== frames_per_block except for the final partial block) */
    int                  partial;    /* 1 if this is the final partial block after a stop */
    void                *base;       /* first byte of slot 0 */
    uint64_t             slot_bytes; /* bytes per slot (== payload) */
    uint64_t             block_bytes;/* frames_per_block * slot_bytes */
    const eg_frame_info *frames;     /* nframes entries, valid until eg_release_block(index) */
    uint64_t             first_frame_id, last_frame_id;
    uint64_t             seq;        /* 0-based sequence number of this completed block in the acquisition */
} eg_block_info;

typedef struct eg_stats {
    uint64_t delivered;          /* frames delivered into slots since eg_start() */
    uint64_t incomplete;         /* frames flagged incomplete by the grabber */
    uint64_t lost_by_timestamp;  /* frames missing according to gaps in consecutive camera timestamps */
    uint64_t lost_by_frame_id;   /* frames missing according to gaps in BUFFER_INFO_FRAMEID */
    uint64_t rejected;           /* grabber DataStream "RejectedFrame" events (no buffer available) */
    uint64_t broken;             /* grabber "BrokenFrame" events */
    uint64_t blocks_completed;   /* blocks handed to the consumer (including a final partial one) */
    uint64_t blocks_released;
    double   period_ns_mean;     /* mean camera timestamp spacing */
    double   fps_timestamp;      /* frames/s from the camera timestamps (first to last delivered) */
    double   stream_fps;         /* grabber StatisticsFrameRate */
    double   stream_data_rate_mbs; /* grabber StatisticsDataRate (MB/s) */
    double   seconds_running;    /* host seconds since eg_start() */
    double   t_announce_s;       /* seconds spent announcing (pinning) the block memory in eg_open() */
    double   t_prefault_s;       /* seconds spent touching the block memory in eg_open() */
    int      large_pages_used;
    int      ext_kind;           /* 0 engine-allocated host memory, 1 caller host memory, 2 caller device memory (RDMA) */
    int      ready_blocks;       /* completed blocks waiting for eg_wait_block() */
    int      free_blocks;        /* blocks whose slots are all in the grabber's input queue */
    uint64_t slots_queued;       /* frame slots in the grabber's input queue (free for the camera to fill) */
    uint64_t slots_ready;        /* filled slots not yet released by the consumer (completed blocks + the block being filled) */
    int      triggered;          /* 1 if the acquisition was started with TriggerMode = On (timestamp-gap loss counting is off) */
    uint64_t order_mismatch;     /* frames that landed in a slot other than the predicted one (eg_next_slot) */
} eg_stats;

EG_API int  eg_version(void);
EG_API eg_handle *eg_open(const eg_config *cfg, eg_camera_info *info);  /* discover + configure the camera, allocate, pin and queue the blocks */
EG_API int  eg_camera_info_get(eg_handle *h, eg_camera_info *info);
EG_API int  eg_start(eg_handle *h);                                     /* start the acquisition thread and the camera (config.max_frames) */
EG_API int  eg_start_n(eg_handle *h, uint64_t max_frames);              /* same, stopping by itself after max_frames delivered frames (0 = until eg_stop()).
                                                                           May be called again after eg_stop() / after the run ended by itself, once every block
                                                                           has been released: the pool is reset and reused */
EG_API int  eg_refresh_info(eg_handle *h, eg_camera_info *info);        /* re-read size, payload, frame rate, exposure after the host changed ROI / binning / pixel format */
EG_API int  eg_pool_alloc(eg_handle *h, int frames_per_block, int nblocks, int large_pages, int prefault_threads,
                          void *const *ext_blocks, int ext_kind);       /* allocate, pre-fault, announce and queue a block pool (after eg_open with nblocks = 0) */
EG_API int  eg_pool_free(eg_handle *h);                                 /* stop, revoke the announced buffers and free the pool; the device stays open */
/* Frame-at-a-time consumers that want zero copy (a host whose writer hands out one frame destination at a time):
 * the grabber fills slots in the order of its input FIFO, which the engine mirrors (announce order, then every released
 * block's slots at the tail), so the destination of the next frame is known before it arrives. */
EG_API int  eg_next_slot(eg_handle *h, int *block, int *slot);          /* predicted (block, slot) of the next frame to be delivered; EG_ERR_ORDER while the
                                                                           order is unknown (after a partial block was released; the next eg_start with no block held resets it) */
EG_API int  eg_wait_slot(eg_handle *h, int block, int slot, int timeout_ms, eg_frame_info *out); /* EG_OK once that slot holds a frame; EG_TIMEOUT;
                                                                           EG_ERR_ORDER if a frame landed elsewhere first; EG_ERR_STATE when stopped and drained */
EG_API int  eg_wait_block(eg_handle *h, int timeout_ms, eg_block_info *out); /* EG_OK with a block, EG_TIMEOUT, or EG_ERR_STATE when stopped and drained */
EG_API int  eg_release_block(eg_handle *h, int index);                  /* re-queue the block's slots to the grabber */
EG_API int  eg_block_ptr(eg_handle *h, int index, void **base, uint64_t *bytes);
EG_API int  eg_stop(eg_handle *h);                                      /* stop the camera; a final partial block (if any) becomes available */
EG_API int  eg_is_running(eg_handle *h);                                /* 1 while the acquisition thread runs */
EG_API int  eg_get_stats(eg_handle *h, eg_stats *out);
EG_API const char *eg_last_error(eg_handle *h);                         /* h may be NULL after a failed eg_open(): a process-wide message */
EG_API void eg_close(eg_handle *h);
/* read / write a camera (remote module) or stream feature as a string, for diagnostics (e.g. "DeviceTemperature") */
EG_API int  eg_remote_get(eg_handle *h, const char *feature, char *buf, size_t buflen);
EG_API int  eg_remote_set(eg_handle *h, const char *feature, const char *value);
EG_API int  eg_stream_get(eg_handle *h, const char *feature, char *buf, size_t buflen);

#ifdef __cplusplus
}
#endif
#endif

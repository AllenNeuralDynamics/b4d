// egrabber_ingest.cpp - zero-copy camera ingest engine (Euresys eGrabber / Coaxlink). See include/egrabber_ingest.h.
//
// Build (clang-cl):  python egrabber/build.py        (or see build.py for the exact command line)
//
// Design
//  * One EGrabber<CallbackOnDemand> per camera. Events (new buffer, data-stream) are processed on a dedicated
//    acquisition thread with processEvent<OneOf<NewBufferData, DataStreamData>>; the consumer never touches
//    the grabber directly except to re-queue a block (DSQueueBuffer is thread-safe).
//  * The block pool: nblocks x (frames_per_block x payload) bytes, each block one VirtualAlloc region (large pages
//    when the privilege is available), pre-faulted, announced to the grabber as a UserMemoryArray so every frame slot
//    is a GenTL buffer. The grabber DMAs frame k of a block into slot k; nothing is ever copied.
//  * A block whose slots are all filled is appended to the ready queue; the consumer takes it (eg_wait_block),
//    compresses straight from the memory, then eg_release_block() re-queues its slots.
//  * Loss accounting: frame-id gaps, camera-timestamp gaps (period from the configured frame rate), and the grabber's
//    own RejectedFrame / BrokenFrame events. A buffer underrun (every block held by the consumer) shows up as rejected
//    frames and as timestamp gaps across the block boundary.
#define EG_INGEST_BUILD 1
#include "egrabber_ingest.h"

#include <EGrabber.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

using namespace Euresys;

namespace {

std::string g_global_error;                       // for failures before a handle exists
std::mutex g_global_error_mutex;

uint64_t host_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void copy_str(char *dst, size_t n, const std::string &s) {
    if (!n) return;
    size_t m = std::min(n - 1, s.size());
    memcpy(dst, s.data(), m); dst[m] = 0;
}

// ------------------------------------------------------------------ block memory
struct BlockMemory {
    void *base = nullptr;
    size_t bytes = 0;       // requested (frames * payload)
    size_t alloc_bytes = 0; // rounded for large pages
    bool large = false;
    bool external = false;  // caller-owned: never freed here
    void release() {
        if (external) { base = nullptr; return; }
#ifdef _WIN32
        if (base) VirtualFree(base, 0, MEM_RELEASE);
#else
        if (base) munmap(base, alloc_bytes);
#endif
        base = nullptr;
    }
};

#ifdef _WIN32
bool enable_lock_memory_privilege() {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    TOKEN_PRIVILEGES tp; tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}
#endif

bool alloc_block(BlockMemory &bm, size_t bytes, bool try_large, std::string &why_not_large) {
    bm.bytes = bytes;
#ifdef _WIN32
    if (try_large) {
        static bool priv = enable_lock_memory_privilege();
        if (!priv) {
            why_not_large = "SeLockMemoryPrivilege not available (grant 'Lock pages in memory' to this user to use large pages)";
        } else {
            size_t lp = GetLargePageMinimum();
            if (lp) {
                size_t rounded = (bytes + lp - 1) / lp * lp;
                void *p = VirtualAlloc(nullptr, rounded, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
                if (p) { bm.base = p; bm.alloc_bytes = rounded; bm.large = true; return true; }
                char b[128]; snprintf(b, sizeof b, "VirtualAlloc(MEM_LARGE_PAGES) failed (error %lu), using 4 KB pages", GetLastError());
                why_not_large = b;
            }
        }
    }
    void *p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) return false;
    bm.base = p; bm.alloc_bytes = bytes; bm.large = false; return true;
#else
    (void)try_large; (void)why_not_large;
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    bm.base = p; bm.alloc_bytes = bytes; return true;
#endif
}

void prefault(void *base, size_t bytes, int nthreads) {
    nthreads = std::max(1, nthreads);
    size_t band = (bytes + nthreads - 1) / nthreads;
    band = (band + 4095) & ~size_t(4095);
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) {
        size_t o = (size_t)t * band; if (o >= bytes) break;
        size_t n = std::min(band, bytes - o);
        th.emplace_back([base, o, n]() { memset((char *)base + o, 0, n); });
    }
    for (auto &x : th) x.join();
}

// ------------------------------------------------------------------ the engine
struct Block {
    int index = 0;
    BlockMemory mem;
    BufferIndexRange range;                 // the block's slots as announced to the grabber
    std::vector<eg_frame_info> frames;      // slot -> info
    std::vector<char> filled;               // slot -> holds a frame (since the last release)
    int count = 0;                          // slots filled since the last release
    bool held = false;                      // handed to the consumer
    bool partial = false;
    uint64_t seq = 0;
};

class Engine : public EGrabber<CallbackOnDemand> {
public:
    Engine(EGenTL &gentl, const EGrabberCameraInfo &cam, const eg_config &cfg)
    : EGrabber<CallbackOnDemand>(cam), cfg_(cfg) { (void)gentl; }

    ~Engine() {
        try { stop_internal(); } catch (...) {}
        if (thread_.joinable()) thread_.join();
        try { disableEvent<DataStreamData>(); } catch (...) {}
        try { reallocBuffers(0); } catch (...) {}     // revoke the announced user memory before freeing it
        for (auto &b : blocks_) b.mem.release();
    }

    // ---- configuration (remote device + stream), payload
    double read_fmax() {
        double fmax = 0;
        try { fmax = getFloat<RemoteModule>("AcquisitionFrameRate.Max"); } catch (...) {
            try { fmax = getFloat<RemoteModule>("AcquisitionFrameRate"); } catch (...) {}
        }
        return fmax;
    }

    // write the settings requested in the config (skipped when the host attaches to an already configured camera)
    void apply_settings() {
        if (cfg_.pixel_format && *cfg_.pixel_format) setString<RemoteModule>("PixelFormat", cfg_.pixel_format);
        double fmax = read_fmax();
        if (cfg_.exposure_us > 0) setFloat<RemoteModule>("ExposureTime", cfg_.exposure_us);
        double want = cfg_.fps > 0 ? cfg_.fps : fmax;
        if (want > 0) {
            try { setFloat<RemoteModule>("AcquisitionFrameRate", want); }
            catch (const std::exception &e) {
                if (cfg_.fps > 0) throw;     // explicit request that the camera refuses
                log("AcquisitionFrameRate = %.4f refused (%s); keeping the camera's value", want, e.what());
            }
        }
    }

    void configure(eg_camera_info &info) {
        if (!cfg_.attach_only) apply_settings();
        read_info(info);
    }

    // read the camera's current geometry and rates (also used again after the host changed ROI / binning / pixel format)
    void read_info(eg_camera_info &info) {
        double fmax = read_fmax();
        fps_ = getFloat<RemoteModule>("AcquisitionFrameRate");
        period_ns_ = fps_ > 0 ? 1e9 / fps_ : 0;
        payload_ = getPayloadSize();
        width_ = getInteger<RemoteModule>("Width"); height_ = getInteger<RemoteModule>("Height");
        memset(&info, 0, sizeof info);
        auto rget = [this](const char *f) { try { return getString<RemoteModule>(f); } catch (...) { return std::string(); } };
        auto sget = [this](const char *f) { try { return getString<StreamModule>(f); } catch (...) { return std::string(); } };
        auto iget = [this](const char *f) { try { return getString<InterfaceModule>(f); } catch (...) { return std::string(); } };
        copy_str(info.vendor, sizeof info.vendor, rget("DeviceVendorName"));
        copy_str(info.model, sizeof info.model, rget("DeviceModelName"));
        copy_str(info.serial, sizeof info.serial, rget("DeviceSerialNumber"));
        copy_str(info.version, sizeof info.version, rget("DeviceVersion"));
        copy_str(info.pixel_format, sizeof info.pixel_format, rget("PixelFormat"));
        copy_str(info.unpacking, sizeof info.unpacking, sget("UnpackingMode"));
        copy_str(info.interface_id, sizeof info.interface_id, iget("InterfaceID"));
        copy_str(info.link_config, sizeof info.link_config, rget("CxpLinkConfiguration"));
        info.width = width_; info.height = height_;
        info.payload_bytes = payload_;
        info.bytes_per_pixel = (width_ * height_) ? (int)(payload_ / (uint64_t)(width_ * height_)) : 0;
        info.fps = fps_; info.fps_max = fmax;
        try { info.exposure_us = getFloat<RemoteModule>("ExposureTime"); } catch (...) {}
        try { info.link_throughput_limit = getFloat<RemoteModule>("DeviceLinkThroughputLimit"); } catch (...) {}
        info_ = info;
    }

    // ---- block pool: allocate, pre-fault, announce + queue
    bool has_pool() const { return !blocks_.empty(); }

    void pool_alloc(int fpb, int nb, int large_pages, int prefault_threads, void *const *ext, int ext_kind) {
        std::lock_guard<std::mutex> lk(pool_mu_);
        if (running()) throw std::runtime_error("cannot allocate a pool while the acquisition is running");
        if (has_pool()) throw std::runtime_error("a pool is already allocated (eg_pool_free first)");
        eg_camera_info tmp; read_info(tmp);               // the payload may have changed since open (ROI, binning, pixel format)
        cfg_.frames_per_block = fpb; cfg_.nblocks = nb; cfg_.large_pages = large_pages; cfg_.prefault_threads = prefault_threads;
        cfg_.ext_blocks = ext; cfg_.ext_kind = ext_kind;
        make_blocks();
    }

    void pool_free() {
        std::lock_guard<std::mutex> lk(pool_mu_);
        stop_internal();
        if (!has_pool()) return;
        try { disableEvent<DataStreamData>(); } catch (...) {}
        try { reallocBuffers(0); } catch (...) {}         // revoke the announced user memory before freeing it
        for (auto &b : blocks_) b.mem.release();
        blocks_.clear();
        std::lock_guard<std::mutex> lk2(mu_);
        ready_.clear(); expected_.clear(); ahead_ = 0; slot_consumer_ = false;
    }

    void make_blocks() {
        int fpb = cfg_.frames_per_block, nb = cfg_.nblocks;
        if (fpb <= 0 || nb < 1) throw std::runtime_error("frames_per_block and nblocks must be >= 1");
        blocks_.resize(nb);
        std::string why;
        ext_kind_ = (cfg_.ext_blocks && cfg_.ext_kind) ? cfg_.ext_kind : 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nb; ++i) {
            Block &b = blocks_[i]; b.index = i; b.frames.resize(fpb); b.filled.assign(fpb, 0);
            size_t bytes = (size_t)fpb * payload_;
            if (ext_kind_) {                                   // caller-provided memory: not allocated, touched or freed here
                if (!cfg_.ext_blocks[i]) throw std::runtime_error("ext_blocks[" + std::to_string(i) + "] is NULL");
                b.mem.base = cfg_.ext_blocks[i]; b.mem.bytes = bytes; b.mem.alloc_bytes = 0; b.mem.large = false; b.mem.external = true;
                continue;
            }
            if (!alloc_block(b.mem, bytes, cfg_.large_pages != 0, why))
                throw std::runtime_error("block allocation failed (" + std::to_string(bytes) + " bytes)");
            prefault(b.mem.base, b.mem.alloc_bytes, cfg_.prefault_threads > 0 ? cfg_.prefault_threads : 16);
        }
        t_prefault_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        large_used_ = blocks_[0].mem.large ? 1 : 0;
        if (cfg_.large_pages && !large_used_ && !why.empty() && !ext_kind_) log("%s", why.c_str());
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nb; ++i) {
            Block &b = blocks_[i];
            if (ext_kind_ == 2) {
                // GPUDirect RDMA: one device buffer per slot; userPointer encodes block and slot (BUFFER_INFO_BASE returns
                // the device address, so the address lookup works as well)
                size_t first = 0;
                for (int k = 0; k < fpb; ++k) {
                    void *slot = (char *)b.mem.base + (size_t)k * payload_;
                    BufferIndexRange r = announceAndQueue(NvidiaRdmaMemory(slot, payload_, (void *)(uintptr_t)(((uintptr_t)(i + 1) << 20) | (uintptr_t)k)));
                    if (k == 0) first = r.begin;
                    if (r.size() != 1 || r.begin != first + (size_t)k)
                        throw std::runtime_error("RDMA announce returned a non-contiguous buffer index range");
                }
                b.range = BufferIndexRange(first, first + fpb);
            } else {
                // userPointer = (block index + 1) << 20 (0 would be "no pointer"); the slot comes from the address
                UserMemory um(b.mem.base, (size_t)fpb * payload_, (void *)(uintptr_t)((uintptr_t)(i + 1) << 20));
                b.range = announceAndQueue(UserMemoryArray(um, payload_));
            }
            if (b.range.size() != (size_t)fpb)
                throw std::runtime_error("announceAndQueue returned " + std::to_string(b.range.size()) + " buffers, expected " + std::to_string(fpb));
        }
        t_announce_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        expected_.clear(); order_unknown_ = false; ahead_ = 0; slot_consumer_ = false;
        for (int i = 0; i < nb; ++i) for (int k = 0; k < fpb; ++k) expected_.emplace_back(i, k);   // the input FIFO, in announce order
        log("%d blocks x %d frames x %llu bytes = %.1f GB %s; pre-fault %.2f s, announce (pin) %.2f s",
            nb, fpb, (unsigned long long)payload_, nb * (double)fpb * payload_ / 1e9,
            ext_kind_ == 2 ? "(caller device memory, GPUDirect RDMA)" : ext_kind_ == 1 ? "(caller host memory)" : large_used_ ? "(large pages)" : "(4 KB pages)",
            t_prefault_, t_announce_);
    }

    // ---- acquisition
    void start_acq(uint64_t max_frames) {
        if (thread_.joinable()) thread_.join();            // a previous run that ended by itself
        std::lock_guard<std::mutex> lk(mu_);
        if (running_) throw std::runtime_error("already running");
        if (blocks_.empty()) throw std::runtime_error("no buffer pool (eg_pool_alloc first)");
        for (auto &b : blocks_)
            if (b.held && b.partial) throw std::runtime_error("block " + std::to_string(b.index) + " (a partial block) is still held: release it before restarting");
        cfg_.max_frames = max_frames;
        // the camera may have been reconfigured between runs: re-read the rate. With an external trigger the frame spacing is
        // not the frame period, so timestamp gaps say nothing about loss (frame ids and rejected-frame events still do)
        try { fps_ = getFloat<RemoteModule>("AcquisitionFrameRate"); } catch (...) {}
        triggered_ = false;
        try { triggered_ = getString<RemoteModule>("TriggerMode") == "On"; } catch (...) {}
        period_ns_ = (!triggered_ && fps_ > 0) ? 1e9 / fps_ : 0;
        stop_requested_ = false; drained_ = false; running_ = true;
        delivered_ = incomplete_ = lost_ts_ = lost_fid_ = rejected_ = broken_ = completed_ = released_ = 0;
        have_last_ = false; seq_ = 0; ts_first_ = ts_last_ = 0;
        // blocks held by the consumer (being compressed from a previous run) stay out of the FIFO until released; the
        // grabber's input FIFO itself survives stop/start, so the slot order mirror (expected_) is kept as it is
        for (auto &b : blocks_) if (!b.held) { b.count = 0; b.partial = false; std::fill(b.filled.begin(), b.filled.end(), 0); }
        ready_.clear();
        if (order_unknown_) {
            // a partial block was released since the last start (resetBufferQueue reorders the FIFO): with no block held the
            // whole queue can be reset to announce order, which makes the slot order predictable again
            bool any_held = false; for (auto &b : blocks_) any_held |= b.held;
            if (!any_held) {
                resetBufferQueue();
                for (int i = 0; i < (int)blocks_.size(); ++i) for (int k = 0; k < cfg_.frames_per_block; ++k) expected_.emplace_back(i, k);
                order_unknown_ = false; ahead_ = 0;
            }
        }
        t_start_ = std::chrono::steady_clock::now();
        enableEvent<DataStreamData>();
        thread_ = std::thread([this]() { this->acq_loop(); });
    }

    void acq_loop() {
        try {
            if (cfg_.max_frames) start(cfg_.max_frames); else start();
        } catch (const std::exception &e) {
            set_error(std::string("start failed: ") + e.what());
            std::lock_guard<std::mutex> lk(mu_); running_ = false; drained_ = true; cv_.notify_all(); return;
        }
        bool stopping = false;
        int idle_after_stop = 0;
        for (;;) {
            try {
                processEvent<OneOf<NewBufferData, DataStreamData> >(100);   // 100 ms slices so stop requests are seen
                if (stopping) idle_after_stop = 0;
            } catch (const gentl_error &e) {
                if (e.gc_err == gc::GC_ERR_TIMEOUT) {
                    if (stopping && ++idle_after_stop >= 3) break;          // nothing left in the output queue
                } else if (e.gc_err == gc::GC_ERR_ABORT) {
                    break;
                } else {
                    set_error(std::string("event loop: ") + e.what());
                    break;
                }
            } catch (const std::exception &e) {
                set_error(std::string("event loop: ") + e.what());
                break;
            }
            if (!stopping) {
                bool want_stop = stop_requested_.load();
                if (!want_stop && cfg_.max_frames && delivered_.load() >= cfg_.max_frames) want_stop = true;
                if (want_stop) {
                    stopping = true;
                    try { stop(); } catch (const std::exception &e) { set_error(std::string("stop: ") + e.what()); }
                }
            }
        }
        if (!stopping) { try { stop(); } catch (...) {} }
        // final partial block
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto &b : blocks_) {
                if (!b.held && b.count > 0 && b.count < cfg_.frames_per_block) {
                    b.partial = true; b.held = true; b.seq = seq_++; ready_.push_back(b.index); completed_++;
                }
            }
            running_ = false; drained_ = true;
        }
        cv_.notify_all();
    }

    void request_stop() { stop_requested_ = true; }

    void stop_internal() {
        request_stop();
        if (thread_.joinable()) thread_.join();
    }

    int wait_block(int timeout_ms, eg_block_info *out) {
        std::unique_lock<std::mutex> lk(mu_);
        auto pred = [this]() { return !ready_.empty() || drained_; };
        if (timeout_ms < 0) cv_.wait(lk, pred);
        else if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) return EG_TIMEOUT;
        if (ready_.empty()) return EG_ERR_STATE;      // stopped and drained
        Block &b = blocks_[ready_.front()]; ready_.pop_front();
        memset(out, 0, sizeof *out);
        out->index = b.index; out->nframes = b.count; out->partial = b.partial ? 1 : 0;
        out->base = b.mem.base; out->slot_bytes = payload_; out->block_bytes = (uint64_t)cfg_.frames_per_block * payload_;
        out->frames = b.frames.data();
        out->first_frame_id = b.count ? b.frames[0].frame_id : 0;
        out->last_frame_id = b.count ? b.frames[b.count - 1].frame_id : 0;
        out->seq = b.seq;
        return EG_OK;
    }

    int next_slot(int *block, int *slot) {
        std::lock_guard<std::mutex> lk(mu_);
        if (expected_.empty()) return order_unknown_ ? EG_ERR_ORDER : EG_ERR_STATE;
        slot_consumer_ = true;
        *block = expected_.front().first; *slot = expected_.front().second;
        return EG_OK;
    }

    int wait_slot(int block, int slot, int timeout_ms, eg_frame_info *out) {
        if (block < 0 || block >= (int)blocks_.size() || slot < 0 || slot >= cfg_.frames_per_block) return EG_ERR_ARG;
        std::unique_lock<std::mutex> lk(mu_);
        if (expected_.empty() || expected_.front().first != block || expected_.front().second != slot)
            return EG_ERR_ARG;                               // slots are consumed in order: wait for next_slot()'s answer
        slot_consumer_ = true;
        Block &b = blocks_[block];
        auto pred = [&]() { return b.filled[slot] || drained_ || order_err_; };
        if (timeout_ms < 0) cv_.wait(lk, pred);
        else if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) return EG_TIMEOUT;
        if (b.filled[slot]) {
            *out = b.frames[slot];
            expected_.pop_front(); if (ahead_) ahead_--;      // consumed
            return EG_OK;
        }
        if (order_err_) { order_err_ = false; return EG_ERR_ORDER; }
        return EG_ERR_STATE;
    }

    int release_block(int index) {
        if (index < 0 || index >= (int)blocks_.size()) return EG_ERR_ARG;
        Block &b = blocks_[index];
        bool was_partial = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!b.held) return EG_ERR_STATE;
            was_partial = b.partial;
            b.held = false; b.count = 0; b.partial = false; released_++;
            std::fill(b.filled.begin(), b.filled.end(), 0);
            for (int k = 0; k < cfg_.frames_per_block; ++k) expected_.emplace_back(index, k);   // the slots go to the FIFO tail
        }
        // the whole range goes back to the input FIFO (tail), in slot order. A final partial block still has its unfilled
        // slots queued (the stream is idle after the stop): queueing them again is an error, so reset the range instead
        if (was_partial) {
            resetBufferQueue(b.range);                  // reorders the input FIFO: the slot order is unknown until the next start
            std::lock_guard<std::mutex> lk(mu_); expected_.clear(); order_unknown_ = true; ahead_ = 0;
        } else queue(b.range);
        return EG_OK;
    }

    void stats(eg_stats *s) {
        std::lock_guard<std::mutex> lk(mu_);
        memset(s, 0, sizeof *s);
        s->delivered = delivered_; s->incomplete = incomplete_; s->lost_by_timestamp = lost_ts_; s->lost_by_frame_id = lost_fid_;
        s->rejected = rejected_; s->broken = broken_; s->blocks_completed = completed_; s->blocks_released = released_;
        uint64_t n = delivered_;
        if (n >= 2 && ts_last_ > ts_first_) {
            s->period_ns_mean = (double)(ts_last_ - ts_first_) / (double)(n - 1 + lost_ts_);
            s->fps_timestamp = 1e9 * (double)(n - 1 + lost_ts_) / (double)(ts_last_ - ts_first_);
        }
        try { s->stream_fps = getFloat<StreamModule>("StatisticsFrameRate"); } catch (...) {}
        try { s->stream_data_rate_mbs = getFloat<StreamModule>("StatisticsDataRate"); } catch (...) {}
        s->seconds_running = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start_).count();
        s->t_announce_s = t_announce_; s->t_prefault_s = t_prefault_; s->large_pages_used = large_used_; s->ext_kind = ext_kind_;
        s->ready_blocks = (int)ready_.size();
        int free_ = 0; for (auto &b : blocks_) if (!b.held && b.count == 0) free_++;
        s->free_blocks = free_;
        uint64_t filled = 0;
        for (auto &b : blocks_) filled += (uint64_t)(b.held ? (b.partial ? b.count : cfg_.frames_per_block) : b.count);
        s->slots_ready = filled;
        s->slots_queued = (uint64_t)blocks_.size() * (uint64_t)cfg_.frames_per_block - filled;
        s->triggered = triggered_ ? 1 : 0;
        s->order_mismatch = order_mismatch_;
    }

    bool running() { std::lock_guard<std::mutex> lk(mu_); return running_; }
    int block_ptr(int index, void **base, uint64_t *bytes) {
        if (index < 0 || index >= (int)blocks_.size()) return EG_ERR_ARG;
        *base = blocks_[index].mem.base; *bytes = (uint64_t)cfg_.frames_per_block * payload_; return EG_OK;
    }
    const eg_camera_info &info() const { return info_; }
    uint64_t config_max_frames() const { return cfg_.max_frames; }
    std::string last_error() { std::lock_guard<std::mutex> lk(err_mu_); return error_; }
    void set_error(const std::string &e) { std::lock_guard<std::mutex> lk(err_mu_); error_ = e; if (cfg_.verbose) fprintf(stderr, "[egrabber_ingest] ERROR %s\n", e.c_str()); }

    void log(const char *fmt, ...) {
        if (!cfg_.verbose) return;
        va_list ap; va_start(ap, fmt);
        fprintf(stderr, "[egrabber_ingest] "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); fflush(stderr);
        va_end(ap);
    }

private:
    // ---- event callbacks (acquisition thread)
    virtual void onNewBufferEvent(const NewBufferData &data) {
        Buffer buf(data);                                  // NOT ScopedBuffer: the slot stays out of the input queue until release
        eg_frame_info fi; memset(&fi, 0, sizeof fi);
        fi.host_ns = host_now_ns();
        void *base = buf.getInfo<void *>(*this, gc::BUFFER_INFO_BASE);
        try { fi.frame_id = buf.getInfo<uint64_t>(*this, gc::BUFFER_INFO_FRAMEID); } catch (...) {}
        try { fi.timestamp_ns = buf.getInfo<uint64_t>(*this, gc::BUFFER_INFO_TIMESTAMP_NS); } catch (...) {
            try { fi.timestamp_ns = buf.getInfo<uint64_t>(*this, gc::BUFFER_INFO_TIMESTAMP) * 1000; } catch (...) {}
        }
        try { fi.size_filled = buf.getInfo<size_t>(*this, gc::BUFFER_INFO_SIZE_FILLED); } catch (...) {}
        try { fi.incomplete = buf.getInfo<bool8_t>(*this, gc::BUFFER_INFO_IS_INCOMPLETE) ? 1 : 0; } catch (...) {}
        // locate block + slot from the address (userPointer is a cross-check)
        int bi = -1; uintptr_t up = (uintptr_t)buf.getUserPointer();
        if ((up >> 20) >= 1 && (up >> 20) <= blocks_.size()) bi = (int)(up >> 20) - 1;
        if (bi < 0 || (char *)base < (char *)blocks_[bi].mem.base || (char *)base >= (char *)blocks_[bi].mem.base + blocks_[bi].mem.bytes) {
            bi = -1;
            for (size_t i = 0; i < blocks_.size(); ++i)
                if ((char *)base >= (char *)blocks_[i].mem.base && (char *)base < (char *)blocks_[i].mem.base + blocks_[i].mem.bytes) { bi = (int)i; break; }
        }
        if (bi < 0) { set_error("delivered buffer does not belong to any block"); return; }
        Block &b = blocks_[bi];
        size_t slot = (size_t)((char *)base - (char *)b.mem.base) / payload_;
        fi.slot = (uint32_t)slot;
        bool complete = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (b.held) { set_error("frame delivered into a block held by the consumer (slot re-queued twice?)"); return; }
            if (slot != (size_t)b.count) log("slot %zu arrived while block %d had %d frames (out-of-order fill)", slot, bi, b.count);
            if (slot < b.frames.size()) { b.frames[slot] = fi; b.filled[slot] = 1; }
            if (!expected_.empty()) {
                if (!slot_consumer_) {                       // block consumers: keep the mirror in step with the deliveries
                    if (expected_.front().first == bi && expected_.front().second == (int)slot) expected_.pop_front();
                    else {
                        order_mismatch_++; order_err_ = true;
                        for (auto it = expected_.begin(); it != expected_.end(); ++it)
                            if (it->first == bi && it->second == (int)slot) { expected_.erase(it); break; }
                    }
                } else {                                     // frame consumer: the delivery must be the next undelivered entry
                    if (ahead_ < expected_.size() && expected_[ahead_].first == bi && expected_[ahead_].second == (int)slot) ahead_++;
                    else { order_mismatch_++; order_err_ = true; ahead_++; }
                }
            }
            b.count++;
            delivered_++;
            if (fi.incomplete) incomplete_++;
            if (have_last_) {
                if (fi.frame_id > last_fid_ + 1) lost_fid_ += fi.frame_id - last_fid_ - 1;
                if (period_ns_ > 0 && fi.timestamp_ns > last_ts_) {
                    double k = (double)(fi.timestamp_ns - last_ts_) / period_ns_;
                    long long missing = (long long)(k + 0.5) - 1;
                    if (missing > 0) lost_ts_ += (uint64_t)missing;
                }
            } else { ts_first_ = fi.timestamp_ns; }
            last_fid_ = fi.frame_id; last_ts_ = fi.timestamp_ns; ts_last_ = fi.timestamp_ns; have_last_ = true;
            if (b.count >= cfg_.frames_per_block) {
                b.held = true; b.seq = seq_++; ready_.push_back(bi); completed_++; complete = true;
            }
        }
        cv_.notify_all();                                   // per-frame waiters (wait_slot) and block waiters
        (void)complete;
    }

    virtual void onDataStreamEvent(const DataStreamData &data) {
        switch (data.numid) {
            case ge::EVENT_DATA_NUMID_DATASTREAM_REJECTED_FRAME: rejected_++; break;
            case ge::EVENT_DATA_NUMID_DATASTREAM_BROKEN_FRAME: broken_++; break;
            default: break;
        }
    }

    eg_config cfg_;
    eg_camera_info info_{};
    std::vector<Block> blocks_;
    size_t payload_ = 0; int64_t width_ = 0, height_ = 0;
    double fps_ = 0, period_ns_ = 0;
    double t_prefault_ = 0, t_announce_ = 0; int large_used_ = 0; int ext_kind_ = 0;

    std::thread thread_;
    std::mutex mu_; std::condition_variable cv_;
    std::deque<int> ready_;
    bool running_ = false, drained_ = false, triggered_ = false;
    std::deque<std::pair<int, int> > expected_;   // (block, slot) in input-FIFO order, from the consumer's position on
    size_t ahead_ = 0;                            // frames delivered into expected_[0..ahead_) that the consumer has not taken yet
    bool slot_consumer_ = false;                  // a frame-at-a-time consumer (next_slot / wait_slot) is active
    uint64_t order_mismatch_ = 0; bool order_err_ = false, order_unknown_ = false;
    std::mutex pool_mu_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<uint64_t> delivered_{0};
    uint64_t incomplete_ = 0, lost_ts_ = 0, lost_fid_ = 0, completed_ = 0, released_ = 0, seq_ = 0;
    std::atomic<uint64_t> rejected_{0}, broken_{0};
    bool have_last_ = false; uint64_t last_fid_ = 0, last_ts_ = 0, ts_first_ = 0, ts_last_ = 0;
    std::chrono::steady_clock::time_point t_start_;
    std::mutex err_mu_; std::string error_;
};

} // namespace

// ------------------------------------------------------------------ C API
struct eg_handle {
    std::unique_ptr<EGenTL> gentl;
    std::unique_ptr<Engine> eng;
    std::string error;
};

static void set_global_error(const std::string &s) { std::lock_guard<std::mutex> lk(g_global_error_mutex); g_global_error = s; }
static int fail(eg_handle *h, const std::string &s, int code = EG_ERR) { if (h) h->error = s; else set_global_error(s); return code; }

extern "C" {

EG_API int eg_version(void) { return 1; }

EG_API eg_handle *eg_open(const eg_config *cfg, eg_camera_info *info) {
    if (!cfg) { set_global_error("eg_open: cfg is NULL"); return nullptr; }
    std::unique_ptr<eg_handle> h(new eg_handle());
    try {
        h->gentl.reset(new EGenTL());
        EGrabberDiscovery disc(*h->gentl);
        disc.discover();
        int ncam = disc.cameraCount();
        if (ncam == 0) { set_global_error("no camera detected (eGrabber discovery found 0 cameras)"); return nullptr; }
        int chosen = -1;
        std::string want = cfg->serial ? cfg->serial : "";
        if (want.empty()) chosen = 0;
        else {
            std::vector<std::string> seen;
            for (int i = 0; i < ncam && chosen < 0; ++i) {
                EGrabber<CallbackOnDemand> probe(disc.cameras(i));           // opens the device; destroyed before the engine opens it
                std::string sn;
                try { sn = probe.getString<RemoteModule>("DeviceSerialNumber"); } catch (...) {}
                seen.push_back(sn);
                if (sn == want) chosen = i;
            }
            if (chosen < 0) {
                std::string msg = "camera with serial '" + want + "' not found; detected:";
                for (auto &s : seen) msg += " '" + s + "'";
                set_global_error(msg); return nullptr;
            }
        }
        h->eng.reset(new Engine(*h->gentl, disc.cameras(chosen), *cfg));
        eg_camera_info tmp; h->eng->configure(tmp);
        if (info) *info = tmp;
        h->eng->log("camera %s %s SN %s, %lldx%lld %s (%s), payload %llu bytes, %.4f fps (max %.4f), exposure %.1f us, link %s on %s",
                    tmp.vendor, tmp.model, tmp.serial, (long long)tmp.width, (long long)tmp.height, tmp.pixel_format, tmp.unpacking,
                    (unsigned long long)tmp.payload_bytes, tmp.fps, tmp.fps_max, tmp.exposure_us, tmp.link_config, tmp.interface_id);
        if (cfg->nblocks > 0) h->eng->make_blocks();
    } catch (const std::exception &e) {
        set_global_error(std::string("eg_open: ") + e.what());
        return nullptr;
    }
    return h.release();
}

EG_API int eg_camera_info_get(eg_handle *h, eg_camera_info *info) {
    if (!h || !h->eng || !info) return EG_ERR_ARG;
    *info = h->eng->info(); return EG_OK;
}

EG_API int eg_start_n(eg_handle *h, uint64_t max_frames) {
    if (!h || !h->eng) return EG_ERR_ARG;
    try { h->eng->start_acq(max_frames); } catch (const std::exception &e) { return fail(h, std::string("eg_start: ") + e.what(), EG_ERR_STATE); }
    return EG_OK;
}

EG_API int eg_start(eg_handle *h) { return (h && h->eng) ? eg_start_n(h, h->eng->config_max_frames()) : EG_ERR_ARG; }

EG_API int eg_refresh_info(eg_handle *h, eg_camera_info *info) {
    if (!h || !h->eng || !info) return EG_ERR_ARG;
    try { h->eng->read_info(*info); } catch (const std::exception &e) { return fail(h, std::string("eg_refresh_info: ") + e.what()); }
    return EG_OK;
}

EG_API int eg_pool_alloc(eg_handle *h, int frames_per_block, int nblocks, int large_pages, int prefault_threads, void *const *ext_blocks, int ext_kind) {
    if (!h || !h->eng) return EG_ERR_ARG;
    try { h->eng->pool_alloc(frames_per_block, nblocks, large_pages, prefault_threads, ext_blocks, ext_kind); }
    catch (const std::exception &e) { return fail(h, std::string("eg_pool_alloc: ") + e.what(), EG_ERR_ALLOC); }
    return EG_OK;
}

EG_API int eg_next_slot(eg_handle *h, int *block, int *slot) {
    if (!h || !h->eng || !block || !slot) return EG_ERR_ARG;
    return h->eng->next_slot(block, slot);
}

EG_API int eg_wait_slot(eg_handle *h, int block, int slot, int timeout_ms, eg_frame_info *out) {
    if (!h || !h->eng || !out) return EG_ERR_ARG;
    try { return h->eng->wait_slot(block, slot, timeout_ms, out); } catch (const std::exception &e) { return fail(h, std::string("eg_wait_slot: ") + e.what()); }
}

EG_API int eg_pool_free(eg_handle *h) {
    if (!h || !h->eng) return EG_ERR_ARG;
    try { h->eng->pool_free(); } catch (const std::exception &e) { return fail(h, std::string("eg_pool_free: ") + e.what()); }
    return EG_OK;
}

EG_API int eg_wait_block(eg_handle *h, int timeout_ms, eg_block_info *out) {
    if (!h || !h->eng || !out) return EG_ERR_ARG;
    try { return h->eng->wait_block(timeout_ms, out); } catch (const std::exception &e) { return fail(h, std::string("eg_wait_block: ") + e.what()); }
}

EG_API int eg_release_block(eg_handle *h, int index) {
    if (!h || !h->eng) return EG_ERR_ARG;
    try { return h->eng->release_block(index); } catch (const std::exception &e) { return fail(h, std::string("eg_release_block: ") + e.what()); }
}

EG_API int eg_block_ptr(eg_handle *h, int index, void **base, uint64_t *bytes) {
    if (!h || !h->eng || !base || !bytes) return EG_ERR_ARG;
    return h->eng->block_ptr(index, base, bytes);
}

EG_API int eg_stop(eg_handle *h) {
    if (!h || !h->eng) return EG_ERR_ARG;
    try { h->eng->stop_internal(); } catch (const std::exception &e) { return fail(h, std::string("eg_stop: ") + e.what()); }
    std::string e = h->eng->last_error();
    if (!e.empty()) { h->error = e; }
    return EG_OK;
}

EG_API int eg_is_running(eg_handle *h) { return (h && h->eng && h->eng->running()) ? 1 : 0; }

EG_API int eg_get_stats(eg_handle *h, eg_stats *out) {
    if (!h || !h->eng || !out) return EG_ERR_ARG;
    try { h->eng->stats(out); } catch (const std::exception &e) { return fail(h, std::string("eg_get_stats: ") + e.what()); }
    return EG_OK;
}

EG_API const char *eg_last_error(eg_handle *h) {
    if (!h) { std::lock_guard<std::mutex> lk(g_global_error_mutex); return g_global_error.c_str(); }
    if (h->eng) { std::string e = h->eng->last_error(); if (!e.empty()) h->error = e; }
    return h->error.c_str();
}

EG_API void eg_close(eg_handle *h) {
    if (!h) return;
    h->eng.reset();      // stops, revokes buffers, frees memory
    h->gentl.reset();
    delete h;
}

EG_API int eg_remote_get(eg_handle *h, const char *feature, char *buf, size_t buflen) {
    if (!h || !h->eng || !feature || !buf) return EG_ERR_ARG;
    try { copy_str(buf, buflen, h->eng->getString<RemoteModule>(feature)); } catch (const std::exception &e) { return fail(h, e.what()); }
    return EG_OK;
}
EG_API int eg_remote_set(eg_handle *h, const char *feature, const char *value) {
    if (!h || !h->eng || !feature || !value) return EG_ERR_ARG;
    try { h->eng->setString<RemoteModule>(feature, value); } catch (const std::exception &e) { return fail(h, e.what()); }
    return EG_OK;
}
EG_API int eg_stream_get(eg_handle *h, const char *feature, char *buf, size_t buflen) {
    if (!h || !h->eng || !feature || !buf) return EG_ERR_ARG;
    try { copy_str(buf, buflen, h->eng->getString<StreamModule>(feature)); } catch (const std::exception &e) { return fail(h, e.what()); }
    return EG_OK;
}

} // extern "C"

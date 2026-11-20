// eg_grab - standalone ingest test: configure the camera, stream frames into the zero-copy block pool for a
// while, and report the delivered rate, timestamp spacing, losses and a pixel summary. No compression, no disk
// (optionally dumps one raw frame). Proves the C++ ingest path on its own.
//
//   eg_grab [--serial SN] [--seconds S | --frames N] [--block F] [--blocks B] [--fps F] [--exposure-us E]
//           [--large-pages] [--hold-ms MS] [--dump frame.raw] [--quiet]
//
//   --hold-ms: how long the consumer pretends to work on each block before releasing it (0 = release at once);
//              with (blocks-1) * block / fps < hold the pool underruns and the loss counters must show it.
#include "egrabber_ingest.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    std::string serial, dump; double seconds = 10, fps = 0, exposure = 0; uint64_t frames = 0;
    int block = 8, blocks = 3, hold_ms = 0, large = 0, verbose = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--serial") serial = next(); else if (a == "--seconds") seconds = atof(next());
        else if (a == "--frames") frames = strtoull(next(), nullptr, 10); else if (a == "--block") block = atoi(next());
        else if (a == "--blocks") blocks = atoi(next()); else if (a == "--fps") fps = atof(next());
        else if (a == "--exposure-us") exposure = atof(next()); else if (a == "--hold-ms") hold_ms = atoi(next());
        else if (a == "--dump") dump = next(); else if (a == "--large-pages") large = 1; else if (a == "--quiet") verbose = 0;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    eg_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.serial = serial.empty() ? nullptr : serial.c_str(); cfg.pixel_format = "Mono12";
    cfg.fps = fps; cfg.exposure_us = exposure; cfg.frames_per_block = block; cfg.nblocks = blocks;
    cfg.large_pages = large; cfg.verbose = verbose;
    eg_camera_info ci;
    double t0 = now_s();
    eg_handle *h = eg_open(&cfg, &ci);
    if (!h) { fprintf(stderr, "eg_open failed: %s\n", eg_last_error(nullptr)); return 1; }
    printf("camera: %s %s SN %s (%s) %lldx%lld %s unpacked %s -> %d B/px, payload %.1f MB; %.4f fps (max %.4f), exposure %.1f us; link %s; grabber %s\n",
           ci.vendor, ci.model, ci.serial, ci.version, (long long)ci.width, (long long)ci.height, ci.pixel_format, ci.unpacking,
           ci.bytes_per_pixel, ci.payload_bytes / 1e6, ci.fps, ci.fps_max, ci.exposure_us, ci.link_config, ci.interface_id);
    if (!frames) frames = (uint64_t)std::max(1.0, seconds * ci.fps + 0.5);
    cfg.max_frames = frames;      // (the engine copied cfg already; reopen is not needed: use eg_stop timing instead)
    printf("block pool: %d x %d frames (%.1f GB per block, %.1f GB total); open took %.1f s\n", blocks, block,
           block * (double)ci.payload_bytes / 1e9, blocks * block * (double)ci.payload_bytes / 1e9, now_s() - t0);
    if (eg_start(h) != EG_OK) { fprintf(stderr, "eg_start failed: %s\n", eg_last_error(h)); eg_close(h); return 1; }
    double t_start = now_s(), t_last_status = t_start;
    uint64_t got = 0; std::vector<double> lat; bool dumped = false; uint64_t px_min = ~0ull, px_max = 0; double px_sum = 0; uint64_t px_n = 0;
    while (got < frames) {
        eg_block_info bi;
        int r = eg_wait_block(h, 2000, &bi);
        if (r == EG_TIMEOUT) {
            if (now_s() - t_last_status > 5) { eg_stats s; eg_get_stats(h, &s); printf("  waiting... delivered %llu, lost(ts) %llu, rejected %llu\n", (unsigned long long)s.delivered, (unsigned long long)s.lost_by_timestamp, (unsigned long long)s.rejected); t_last_status = now_s(); }
            if (now_s() - t_start > seconds + 30) { fprintf(stderr, "no block within the expected time; stopping\n"); break; }
            continue;
        }
        if (r != EG_OK) { fprintf(stderr, "eg_wait_block: %d (%s)\n", r, eg_last_error(h)); break; }
        double t_block = now_s();
        // pixel summary of the first frame of the block (subsampled), and the per-frame table
        const uint16_t *f0 = (const uint16_t *)bi.base;
        size_t npx = (size_t)ci.width * ci.height;
        for (size_t i = 0; i < npx; i += 9973) { uint16_t v = f0[i]; px_min = std::min<uint64_t>(px_min, v); px_max = std::max<uint64_t>(px_max, v); px_sum += v; px_n++; }
        double span_ms = bi.nframes > 1 ? (bi.frames[bi.nframes - 1].timestamp_ns - bi.frames[0].timestamp_ns) / 1e6 : 0;
        double age_ms = (double)(bi.frames[bi.nframes - 1].host_ns) / 1e6;   // host ns at last frame event
        (void)age_ms;
        uint32_t inc = 0; for (int k = 0; k < bi.nframes; ++k) inc += bi.frames[k].incomplete;
        printf("  block %llu (pool %d): frames %llu..%llu (%d%s), ts span %.1f ms (%.4f fps within block), incomplete %u, filled %.0f MB/frame\n",
               (unsigned long long)bi.seq, bi.index, (unsigned long long)bi.first_frame_id, (unsigned long long)bi.last_frame_id, bi.nframes,
               bi.partial ? ", partial" : "", span_ms, bi.nframes > 1 ? (bi.nframes - 1) * 1e3 / span_ms : 0.0, inc, bi.frames[0].size_filled / 1e6);
        if (!dump.empty() && !dumped) {
            FILE *fp = fopen(dump.c_str(), "wb");
            if (fp) { fwrite(bi.base, 1, (size_t)bi.slot_bytes, fp); fclose(fp); printf("  dumped frame %llu to %s (%llu bytes, uint16 %lldx%lld)\n", (unsigned long long)bi.first_frame_id, dump.c_str(), (unsigned long long)bi.slot_bytes, (long long)ci.height, (long long)ci.width); }
            dumped = true;
        }
        got += bi.nframes;
        if (hold_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
        lat.push_back(now_s() - t_block);
        if (eg_release_block(h, bi.index) != EG_OK) { fprintf(stderr, "release failed: %s\n", eg_last_error(h)); break; }
        if (bi.partial) break;
        if (!eg_is_running(h) && got < frames) { eg_block_info more; if (eg_wait_block(h, 0, &more) != EG_OK) break; eg_release_block(h, more.index); got += more.nframes; break; }
    }
    double el = now_s() - t_start;
    eg_stop(h);
    eg_stats s; eg_get_stats(h, &s);
    printf("\n%llu frames consumed in %.2f s = %.3f fps consumed; grabber delivered %llu (%.4f fps by camera timestamps, period %.3f ms; grabber stats %.3f fps, %.0f MB/s)\n",
           (unsigned long long)got, el, got / el, (unsigned long long)s.delivered, s.fps_timestamp, s.period_ns_mean / 1e6, s.stream_fps, s.stream_data_rate_mbs);
    printf("losses: by timestamp gaps %llu, by frame-id gaps %llu, grabber rejected %llu, broken %llu, incomplete %llu\n",
           (unsigned long long)s.lost_by_timestamp, (unsigned long long)s.lost_by_frame_id, (unsigned long long)s.rejected, (unsigned long long)s.broken, (unsigned long long)s.incomplete);
    printf("blocks: completed %llu, released %llu; consumer hold per block mean %.1f ms\n", (unsigned long long)s.blocks_completed, (unsigned long long)s.blocks_released,
           lat.empty() ? 0.0 : 1e3 * std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size());
    printf("pixels (subsampled first frames): min %llu max %llu mean %.2f ADU (12-bit values expected <= 4095)\n", (unsigned long long)px_min, (unsigned long long)px_max, px_n ? px_sum / px_n : 0.0);
    printf("memory: pre-fault %.2f s, announce/pin %.2f s, large pages %s\n", s.t_prefault_s, s.t_announce_s, s.large_pages_used ? "yes" : "no");
    bool ok = s.lost_by_timestamp == 0 && s.rejected == 0 && s.incomplete == 0 && got >= frames;
    printf("%s\n", ok ? "INGEST OK: no losses" : "INGEST NOT CLEAN");
    printf("RESULT {\"frames\": %llu, \"seconds\": %.3f, \"fps_consumed\": %.4f, \"fps_timestamp\": %.4f, \"lost_ts\": %llu, \"rejected\": %llu, \"incomplete\": %llu, \"block\": %d, \"blocks\": %d, \"ok\": %s}\n",
           (unsigned long long)got, el, got / el, s.fps_timestamp, (unsigned long long)s.lost_by_timestamp, (unsigned long long)s.rejected, (unsigned long long)s.incomplete, block, blocks, ok ? "true" : "false");
    eg_close(h);
    return ok ? 0 : 3;
}

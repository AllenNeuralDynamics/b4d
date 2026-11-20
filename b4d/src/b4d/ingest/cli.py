"""b4d-record: record any BlockSource into a B4D sharded zarr (benchmark or acquisition).

    b4d-record --source simulated --source-arg shape=2048,2048 --source-arg fps=100 --seconds 10 --sink memory
    b4d-record --source egrabber_camera.source:EGrabberSource --source-arg serial=<serial> --seconds 60 --q 1 \
               --out D:\\data\\run1
    b4d-record --source egrabber_camera.source:EGrabberSource --gpu staged --predictor mean3d --entropy ans \
               --multiscale 6

--source is "simulated" or an import path "module:Class"; --source-arg key=value pairs are passed to its constructor
(ints, floats, booleans and comma tuples are converted); --block / --blocks become frames_per_block / nblocks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from .recorder import BlockRecorder, RecorderConfig, append_csv, load_source


def parse_tuple(s, n=3):
    t = tuple(int(v) for v in str(s).split(","))
    if len(t) != n:
        raise SystemExit(f"expected {n} comma-separated integers, got {s!r}")
    return t


def coerce(v: str):
    """key=value strings -> python values"""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    if "," in v:
        return tuple(coerce(p) for p in v.split(","))
    return v


def build_parser():
    d = RecorderConfig()
    p = argparse.ArgumentParser(
        prog="b4d-record", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument(
        "--source", default="simulated", help="'simulated' or 'module:Class' implementing the BlockSource contract"
    )
    p.add_argument(
        "--source-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="constructor argument for the source (repeatable)",
    )
    p.add_argument(
        "--tuned",
        default=None,
        metavar="recommendation.json",
        help="settings found by b4d-tune (threads, block, blocks, chunks, shards, max-commits, q); flags given "
        "explicitly still win",
    )
    p.add_argument("--block", type=int, default=64, help="frames per block = shard depth (source frames_per_block)")
    p.add_argument("--blocks", type=int, default=3, help="blocks in the source's pool (>= --max-commits + 1)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--frames", type=int, default=0, help="overrides --seconds; rounded up to whole blocks")
    p.add_argument("--max-commits", type=int, default=d.max_commits)
    p.add_argument("--threads", type=int, default=None, help="compression threads per block write (default: physical cores, $B4D_THREADS)")
    p.add_argument("--chunks", default="64,64,64")
    p.add_argument("--shards", default=None, help="Z,Y,X; default (block, <=1280, <=3840) rounded to chunks")
    p.add_argument(
        "--codec",
        default=d.codec,
        choices=["b4d", "zstd", "lz4", "raw"],
        help="level-0 chunk codec (zstd/lz4: zarr blosc codec on the C pool, lossless)",
    )
    p.add_argument("--clevel", type=int, default=d.clevel, help="zstd/lz4 level")
    p.add_argument(
        "--shuffle", default=d.shuffle, choices=["shuffle", "bitshuffle", "noshuffle"], help="zstd/lz4 filter"
    )
    p.add_argument("--q", type=float, default=d.q)
    p.add_argument("--mode", type=int, default=d.mode)
    p.add_argument("--predictor", default=d.predictor, choices=["above", "left_above", "mean3d", "above3"])
    p.add_argument("--entropy", default=d.entropy, choices=["rice", "ans", "huffman"])
    p.add_argument(
        "--dither",
        action="store_true",
        help="seeded subtractive dither (format v8, modes 1 and 2): removes the low-ADU mean bias; CPU writer, or --gpu "
        "with the Rice coder (mode 1 above3/above, mode 2 above)",
    )
    p.add_argument(
        "--gain",
        type=float,
        default=None,
        help="camera gain in e-/ADU (conv = 1/gain); default: the source's camera model",
    )
    p.add_argument("--conv", type=float, default=None, help="DN per e- (overrides --gain)")
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--rn", type=float, default=None)
    p.add_argument("--vst", default="variance", choices=["variance", "std", "anscombe"],
                   help="noise transform: fitted table (variance: rounding adds q^2/12 at every signal; std) or the generalised Anscombe")
    p.add_argument("--calibration", default=None, help="per-pixel .npz from tools/calibrate_camera.py (--mode 1 --predictor above3)")
    p.add_argument("--calibration-origin", default="0,0", help="row,col of the frame's first pixel on the calibration map")
    p.add_argument("--out", default="b4d_recordings", help="directory for the dataset")
    p.add_argument("--name", default=None)
    p.add_argument("--keep", action="store_true", help="keep the dataset (default: deleted after the run)")
    p.add_argument("--sink", default="disk", choices=["disk", "memory"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--direct-io", dest="direct_io", action="store_true", default=True)
    g.add_argument("--buffered-io", dest="direct_io", action="store_false")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument(
        "--multiscale", type=int, default=0, help="pyramid levels below level 0 (OME-NGFF 0.5 group); 0 = plain array"
    )
    p.add_argument("--ms-factors", default="2,2,2")
    p.add_argument("--ms-codec", default="b4d", choices=["b4d", "zstd", "lz4", "raw"])
    p.add_argument(
        "--gpu",
        default="none",
        choices=["none", "staged", "rdma"],
        help="none: native C writer; staged: CUDA encoder, host blocks uploaded; rdma (experimental): device-resident "
        "blocks from the source",
    )
    p.add_argument("--gpu-batch-tiles", type=int, default=d.gpu_batch_tiles)
    p.add_argument("--gpu-slots", type=int, default=d.gpu_slots)
    p.add_argument("--no-gpu-check", action="store_true")
    p.add_argument("--gpu-host-threads", type=int, default=0)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--warmup-blocks", type=int, default=d.warmup_blocks)
    p.add_argument("--status", type=float, default=5.0)
    p.add_argument("--csv", default=None, help="append a results row to this CSV")
    p.add_argument("--notes", default="")
    return p


def source_kwargs(a) -> dict:
    """--source-arg key=value pairs plus --block/--blocks as the constructor convention names them"""
    kwargs = {}
    for kv in a.source_arg:
        k, _, v = kv.partition("=")
        if not _:
            raise SystemExit(f"--source-arg expects key=value, got {kv!r}")
        kwargs[k.strip()] = coerce(v.strip())
    kwargs.setdefault("frames_per_block", a.block)
    kwargs.setdefault("nblocks", a.blocks)
    return kwargs


def tuned_defaults(path) -> dict:
    """parser defaults from a b4d-tune recommendation.json"""
    from b4d.tune import load_recommendation

    rec, src = load_recommendation(path)
    return dict(
        threads=rec["threads"],
        chunks=",".join(map(str, rec["chunks"])),
        shards=",".join(map(str, rec["shards"])),
        max_commits=rec["max_commits"],
        q=rec["q"],
        block=src["frames_per_block"],
        blocks=src["nblocks"],
    )


def main(argv=None):
    parser = build_parser()
    tuned = parser.parse_known_args(argv)[0].tuned
    if tuned:
        parser.set_defaults(**tuned_defaults(tuned))
    a = parser.parse_args(argv)
    import b4d

    source = load_source(a.source, **source_kwargs(a))
    tag = a.source.split(":")[-1].lower()
    gpu_tag = "_" + a.gpu if a.gpu != "none" else ""
    ms_tag = f"_ms{a.multiscale}" if a.multiscale else ""
    name = a.name or f"{tag}_{datetime.now():%Y%m%d_%H%M%S}_q{a.q:g}{gpu_tag}{ms_tag}"
    path = os.path.join(a.out, name + ".zarr")
    conv = a.conv if a.conv else (1.0 / a.gain if a.gain else None)
    chunks = parse_tuple(a.chunks)
    if chunks[0] > a.block:  # a shallow block (small cameras, tests): the chunk depth follows the block depth
        chunks = (a.block,) + chunks[1:]
    cfg = RecorderConfig(
        path=path,
        sink=a.sink,
        direct_io=a.direct_io,
        sync=not a.no_sync,
        threads=a.threads,
        max_commits=a.max_commits,
        chunks=chunks,
        shards=parse_tuple(a.shards) if a.shards else None,
        codec=a.codec,
        clevel=a.clevel,
        shuffle=a.shuffle,
        q=a.q,
        mode=a.mode,
        predictor=a.predictor,
        entropy=a.entropy,
        dither=a.dither,
        conv=conv,
        offset=a.offset,
        rn=a.rn,
        exact_vst=False if a.vst == "anscombe" else a.vst,
        calibration=a.calibration,
        calibration_origin=parse_tuple(a.calibration_origin, 2),
        gpu=a.gpu,
        gpu_batch_tiles=a.gpu_batch_tiles,
        gpu_slots=a.gpu_slots,
        gpu_check=not a.no_gpu_check,
        gpu_host_threads=a.gpu_host_threads,
        warmup_blocks=a.warmup_blocks,
        verify=a.verify,
        notes=a.notes,
        multiscale=a.multiscale,
        ms_factors=parse_tuple(a.ms_factors),
        ms_codec=a.ms_codec,
    )
    print(f"b4d build: {getattr(b4d._b4d, 'COMPILER', '?')}")
    rec = BlockRecorder(source, cfg)
    ci = rec.info
    fb = int(ci.frame_bytes)
    print(
        f"source: {ci.name} SN {ci.serial}, {ci.shape[1]}x{ci.shape[0]} {ci.dtype}, {fb / 1e6:.1f} MB/frame, "
        f"{ci.fps:.4f} fps; {ci.extra}"
    )
    try:
        rec.start(seconds=a.seconds, frames=a.frames or None)
    except Exception as e:
        print(f"START FAILED: {e}", flush=True)
        rec.close()
        return 2
    print(
        f"offered: {ci.fps * fb / 1e6:,.0f} MB/s raw; run {rec.total_frames} frames = {rec.n_blocks} blocks of "
        f"{rec.block} ({rec.block / ci.fps:.1f} s, "
        f"{rec.block * fb / 1e9:.1f} GB each) + {rec.nwarm} warm-up; pool {rec.nblocks} blocks = "
        f"{rec.nblocks * rec.block * fb / 1e9:.1f} GB; source start {rec.t_open:.1f} s"
    )
    codec_desc = (
        f"b4d q={a.q:g} mode {a.mode} {a.predictor}/{a.entropy}"
        if a.codec == "b4d"
        else (
            "raw (bytes)"
            if a.codec == "raw"
            else f"{a.codec} clevel {a.clevel} {a.shuffle} (zarr blosc codec, lossless, C pool)"
        )
    )
    print(
        f"codec: {codec_desc}; camera model conv {rec.conv:.5f} DN/e- "
        f"({1 / rec.conv:.2f} e-/ADU), offset {rec.offset} ADU, read noise {rec.rn} e-"
    )
    writer_desc = (
        f"CUDA encoder ({a.gpu}) + native shard IO"
        if a.gpu != "none"
        else f"native, {rec.threads} threads per block write"
    )
    ms_desc = (
        f"; multiscale {a.multiscale} level(s), factors {a.ms_factors}, pyramid codec {a.ms_codec}"
        if a.multiscale
        else ""
    )
    print(
        f"writer: {writer_desc}, {a.max_commits} block(s) in flight, chunks {cfg.chunks}, shards {rec.shards}, sink "
        f"{a.sink}{' (direct IO)' if a.direct_io and a.sink == 'disk' else ''}{ms_desc}; dataset {path}"
    )
    try:
        res = rec.run(status_every=a.status)
    except KeyboardInterrupt:
        print("interrupted; stopping", flush=True)
        rec.stop()
        res = rec.result()
    print(rec.noise_line())
    print(rec.status_line().replace("  t=", "final t=", 1))
    print()
    print(rec.report(res))
    v = rec.verify() if a.verify else None
    if v is not None:
        print(v["text"])
        res["verify"] = ("OK" if v["ok"] else "FAIL") + (
            f" max {v['max_sigma']:.2f} sigma ({v['max_adu']:.0f} ADU; {v['max_sigma_adj']:.2f} after rounding)"
            if v["ok"] is not None
            else ""
        )
    else:
        res["verify"] = ""
    res["path"] = path if a.keep else ""
    print("RESULT " + json.dumps(res, default=str), flush=True)
    if a.csv:
        append_csv(a.csv, res)
    if a.keep and a.sink == "disk":
        rec.save_sidecar(res)
        print(f"kept {path} (+ frames.npy, acquisition.json)")
    rec.close()
    if not a.keep:
        shutil.rmtree(path, ignore_errors=True)
    return 0 if res["sustained"] else 3


if __name__ == "__main__":
    sys.exit(main())

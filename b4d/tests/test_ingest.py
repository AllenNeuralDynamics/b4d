"""b4d.ingest: the BlockSource contract, the simulated source and BlockRecorder (memory sink, small frames)."""

import os
import shutil
import tempfile

import numpy as np
import pytest

from b4d.ingest import (
    FRAME_DTYPE,
    BlockRecorder,
    FrameBlock,
    RecorderConfig,
    SimulatedCameraSource,
    SourceInfo,
    SourceStats,
    load_source,
    synthetic_frames,
    validate_source,
)
from b4d.ingest.cli import build_parser, coerce

SHAPE = (128, 192)


def small_source(**kw):
    kw.setdefault("shape", SHAPE)
    kw.setdefault("fps", 400.0)
    kw.setdefault("frames_per_block", 16)
    kw.setdefault("nblocks", 3)
    return SimulatedCameraSource(**kw)


def test_frame_dtype_fields():
    assert FRAME_DTYPE.names == ("frame_id", "timestamp_ns", "host_ns", "size_filled", "incomplete", "slot")


def test_validate_source_accepts_simulated_and_rejects_incomplete():
    src = small_source()
    info = validate_source(src)
    assert isinstance(info, SourceInfo) and info.shape == SHAPE and info.fps == 400.0

    class Broken:
        info = src.info
        frames_per_block = 16
        nblocks = 3

        def start(self, max_frames=None):
            pass

    with pytest.raises(TypeError, match="wait_block"):
        validate_source(Broken())


def test_simulated_source_delivers_blocks_in_order_and_stops():
    src = small_source()
    src.start(max_frames=16 * 4 + 5)  # 4 full blocks + a partial one
    seqs, nframes = [], []
    while True:
        try:
            blk = src.wait_block(2000)
        except StopIteration:
            break
        if blk is None:
            continue
        assert isinstance(blk, FrameBlock)
        assert blk.array.shape == (blk.nframes,) + SHAPE and blk.array.dtype == np.uint16
        assert blk.frames.dtype == FRAME_DTYPE and len(blk.frames) == blk.nframes
        seqs.append(blk.seq)
        nframes.append(blk.nframes)
        src.release(blk)
    assert seqs == [0, 1, 2, 3, 4] and nframes == [16, 16, 16, 16, 5]
    st = src.stats()
    assert isinstance(st, SourceStats) and st.delivered == 69 and st.lost == 0 and st.blocks_released == 5
    src.close()


def test_simulated_source_counts_losses_when_blocks_are_held():
    src = small_source(nblocks=2, fps=2000.0)
    src.start(max_frames=16 * 6)
    held = []
    while len(held) < 2:
        try:
            blk = src.wait_block(2000)
        except StopIteration:
            break
        if blk is not None:
            held.append(blk)
    import time

    time.sleep(0.1)  # frames keep coming due while both blocks are held: they are lost, as a grabber would lose them
    src.stop()
    st = src.stats()
    assert st.lost > 0
    for b in held:
        src.release(b)
    src.close()


def test_synthetic_frames_respects_camera_model():
    f = synthetic_frames(SHAPE, 4, conversion=0.5, offset=100.0, read_noise=2.0, peak_e=200.0, seed=3)
    assert f.shape == (4,) + SHAPE and f.dtype == np.uint16
    assert 90 < f.min() and f.max() < 100 + 0.5 * (200 * 3)  # offset + peak in DN, with noise headroom


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_block_recorder_memory_sink(q):
    src = small_source(fps=400.0)
    cfg = RecorderConfig(
        path=os.path.join(tempfile.gettempdir(), "b4d_ingest_test.zarr"),
        sink="memory",
        q=q,
        threads=4,
        max_commits=2,
        chunks=(16, 64, 64),
        shards=(16, 128, 192),
        direct_io=False,
        warmup_blocks=1,
    )
    with BlockRecorder(src, cfg) as rec:
        assert rec.conv == pytest.approx(1 / 7.25) and rec.offset == 2.05  # from the source's camera model
        rec.start(frames=16 * 4)
        assert rec.n_blocks == 4 and rec.total_frames == 64
        res = rec.run(status_every=1e9, status_cb=None)
    assert res["written_frames"] == 64 and res["lost_ts"] == 0 and res["sustained"] is True
    assert res["bytes_written"] > 0 and res["ratio"] > 1.0
    assert res["camera"] == "simulated camera" and res["q"] == q
    assert "noise" in rec.noise_line()
    assert "SUSTAINED" in rec.report(res)


@pytest.mark.parametrize("codec", ["zstd", "lz4", "raw"])
def test_block_recorder_other_codecs_lossless(codec):
    """RecorderConfig(codec=zstd|lz4|raw): zarr blosc / bytes chunks on the native pool, verified exactly"""
    zarr = pytest.importorskip("zarr")
    path = os.path.join(tempfile.mkdtemp(), "rec.zarr")
    src = small_source(fps=400.0, mode="copy")
    cfg = RecorderConfig(
        path=path,
        sink="disk",
        codec=codec,
        clevel=1,
        shuffle="bitshuffle",
        threads=4,
        chunks=(16, 64, 64),
        shards=(16, 128, 192),
        direct_io=False,
        verify=True,
        warmup_blocks=1,
    )
    try:
        with BlockRecorder(src, cfg) as rec:
            assert rec.lossless
            rec.start(frames=32)
            res = rec.run(status_every=1e9, status_cb=None)
            v = rec.verify()
        assert res["sustained"] and v["ok"] is True, v["text"]
        z = zarr.open_array(path, mode="r")
        names = [c["name"] for c in z.metadata.to_dict()["codecs"][0]["configuration"]["codecs"]]
        assert names == (["bytes"] if codec == "raw" else ["bytes", "blosc"])
        assert np.array_equal(z[0], src.bank[0])
        if codec != "raw":
            assert res["bytes_written"] < 32 * src.bank[0].nbytes
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def _calibration_file(rec_shape, cam, noisy=0.03):
    r = np.random.default_rng(4)
    rn = np.full(rec_shape, cam[2] * cam[0] * 0.7)
    rn[r.random(rec_shape) < noisy] *= 6
    path = os.path.join(tempfile.mkdtemp(), "cal.npz")
    np.savez(path, read_noise_dn=rn, offset=np.full(rec_shape, cam[1]))
    return path


def test_block_recorder_calibration_lists_pixels_and_verify_passes():
    pytest.importorskip("zarr")
    cam = small_source().info.camera_model
    path = os.path.join(tempfile.mkdtemp(), "rec.zarr")
    kw = dict(q=1.0, mode=1, predictor="above3", threads=2, max_commits=2, chunks=(16, 64, 64), shards=(16, 128, 192), direct_io=False, verify=True)
    out = {}
    for name, extra in (("plain", {}), ("cal", dict(calibration=_calibration_file(SHAPE, cam)))):
        with BlockRecorder(small_source(fps=400.0, mode="copy"), RecorderConfig(path=path, **kw, **extra)) as rec:
            rec.start(frames=16 * 3)
            res = rec.run(status_every=1e9, status_cb=None)
            v = rec.verify()
        assert v["ok"], v["text"]
        out[name] = res["bytes_written"]
    assert out["cal"] < out["plain"]


def test_recorder_rejects_calibration_without_the_cpu_b4d_path(tmp_path):
    with pytest.raises(ValueError):
        BlockRecorder(small_source(), RecorderConfig(sink="memory", codec="zstd", calibration="x.npz", chunks=(16, 64, 64), shards=(16, 128, 192)))
    with pytest.raises(ValueError):
        BlockRecorder(small_source(), RecorderConfig(sink="memory", gpu="staged", calibration="x.npz", chunks=(16, 64, 64), shards=(16, 128, 192)))


def test_recorder_rejects_gpu_with_other_codecs():
    with pytest.raises(ValueError):
        BlockRecorder(
            small_source(),
            RecorderConfig(sink="memory", codec="zstd", gpu="staged", chunks=(16, 64, 64), shards=(16, 128, 192)),
        )


def test_block_recorder_disk_sink_roundtrip_and_verify():
    zarr = pytest.importorskip("zarr")
    path = os.path.join(tempfile.mkdtemp(), "rec.zarr")
    src = small_source(fps=400.0, mode="copy")
    cfg = RecorderConfig(
        path=path,
        sink="disk",
        q=1.0,
        threads=4,
        chunks=(16, 64, 64),
        shards=(16, 128, 192),
        direct_io=False,
        verify=True,
        warmup_blocks=1,
    )
    try:
        with BlockRecorder(src, cfg) as rec:
            rec.start(frames=48)
            res = rec.run(status_every=1e9, status_cb=None)
            v = rec.verify()
        assert res["sustained"] and v["ok"] is True, v["text"]
        import b4d.zarr3  # noqa: F401

        z = zarr.open_array(path, mode="r")
        assert z.shape == (48,) + SHAPE
        back = z[0]
        assert (
            np.abs(back.astype(float) - src.bank[0].astype(float)).max() <= 3
        )  # q = 1 at these levels: within a few ADU
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def test_recorder_config_rejects_bad_geometry():
    src = small_source()
    with pytest.raises(ValueError, match="shards"):
        BlockRecorder(src, RecorderConfig(sink="memory", shards=(32, 128, 192), chunks=(16, 64, 64)))
    with pytest.raises(ValueError, match="max_commits"):
        BlockRecorder(src, RecorderConfig(sink="memory", max_commits=3, chunks=(16, 64, 64), shards=(16, 128, 192)))


def test_load_source_and_cli_coercion():
    src = load_source("simulated", shape=SHAPE, frames_per_block=8, nblocks=2)
    assert src.frames_per_block == 8
    src2 = load_source("b4d.ingest.simulated:SimulatedCameraSource", shape=SHAPE, frames_per_block=8, nblocks=2)
    assert type(src2) is SimulatedCameraSource
    assert (
        coerce("2048,2048") == (2048, 2048)
        and coerce("true") is True
        and coerce("1.5") == 1.5
        and coerce("Mono12") == "Mono12"
    )
    a = build_parser().parse_args(["--source", "simulated", "--source-arg", "fps=100", "--sink", "memory"])
    assert a.source_arg == ["fps=100"] and a.block == 64

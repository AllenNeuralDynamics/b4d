"""b4d.hardware (machine-derived defaults) and b4d.tune (the b4d-tune helpers that need no simulator run)."""

import json
import os

import pytest

from b4d import hardware, tune
from b4d.ingest.cli import build_parser, tuned_defaults


def test_threads_default_is_physical_cores(monkeypatch):
    monkeypatch.delenv("B4D_THREADS", raising=False)
    n = hardware.auto_threads()
    assert 1 <= n <= hardware.logical_cpus()
    phys = hardware.physical_cores()
    if phys:
        assert n == min(phys, hardware.logical_cpus())


def test_threads_override_order(monkeypatch):
    monkeypatch.setenv("B4D_THREADS", "7")
    assert hardware.auto_threads() == 7
    assert hardware.auto_threads(3) == 3  # an explicit argument beats the environment
    monkeypatch.setenv("B4D_THREADS", "junk")
    assert hardware.auto_threads() >= 1


def test_thread_candidates_start_at_base_and_stay_in_range(monkeypatch):
    monkeypatch.setattr(hardware, "logical_cpus", lambda: 256)
    assert hardware.thread_candidates(128) == [128, 96, 160, 192]  # the reference system's tuning-measurement values
    monkeypatch.setattr(hardware, "logical_cpus", lambda: 8)
    c = hardware.thread_candidates(4)
    assert c[0] == 4 and max(c) <= 8 and len(set(c)) == len(c)


def test_pool_budget_and_block_depth(monkeypatch):
    monkeypatch.setattr(hardware, "host_memory", lambda: (100 * 10**9, 80 * 10**9))
    monkeypatch.delenv("B4D_POOL_FRACTION", raising=False)
    assert hardware.pool_budget_bytes() == 40 * 10**9
    assert hardware.max_block_frames(10**9, 4) == 10
    monkeypatch.setenv("B4D_POOL_FRACTION", "0.25")
    assert hardware.pool_budget_bytes() == 20 * 10**9
    monkeypatch.setattr(hardware, "host_memory", lambda: (None, None))
    assert hardware.max_block_frames(10**9) is None


def test_describe_is_json_serialisable():
    d = hardware.describe()
    json.dumps(d)
    assert d["logical_cpus"] == hardware.logical_cpus()


def test_default_blocks_respect_ram(monkeypatch):
    monkeypatch.setattr(hardware, "host_memory", lambda: (10**12, 10**12))
    assert tune.default_blocks(70, 12800 * 19200 * 2) == [32, 64, 128]
    monkeypatch.setattr(hardware, "host_memory", lambda: (64 * 10**9, 32 * 10**9))  # 16 GB pool / 4 blocks
    assert tune.default_blocks(70, 12800 * 19200 * 2) == [32]  # 4 x 64 x 491 MB would not fit
    monkeypatch.setattr(hardware, "host_memory", lambda: (8 * 10**9, 4 * 10**9))
    assert tune.default_blocks(70, 12800 * 19200 * 2) == [32]  # never empty


def test_shard_for_multiple_of_chunks():
    sy, sx = tune.shard_for(10640, 14192, 64, 64, 1330)
    assert sy % 64 == 0 and sx % 64 == 0


def test_find_stream_sim_explicit_and_missing(tmp_path, monkeypatch):
    f = tmp_path / "sim.py"
    f.write_text("")
    assert tune.find_stream_sim(str(f)) == str(f)
    monkeypatch.delenv("B4D_STREAM_SIM", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tune, "__file__", str(tmp_path / "x" / "tune.py"))
    with pytest.raises(SystemExit):
        tune.find_stream_sim(str(tmp_path / "nope.py"))


def _write_rec(path):
    rec = dict(
        q=2.0,
        settings=dict(threads=96, block=64, bank=64, blocks=3, ring=64, chunks="32,64,64", shards="64,1408,3584", max_commits=2),
    )
    path.write_text(json.dumps(rec))


def test_load_recommendation_and_cli_defaults(tmp_path):
    p = tmp_path / "recommendation.json"
    _write_rec(p)
    rec, src = tune.load_recommendation(p)
    assert rec == dict(threads=96, chunks=(32, 64, 64), shards=(64, 1408, 3584), max_commits=2, q=2.0)
    assert src == dict(frames_per_block=64, nblocks=3)
    parser = build_parser()
    parser.set_defaults(**tuned_defaults(p))
    a = parser.parse_args(["--threads", "32"])  # an explicit flag still wins
    assert (a.threads, a.block, a.blocks, a.chunks, a.shards, a.max_commits, a.q) == (
        32, 64, 3, "32,64,64", "64,1408,3584", 2, 2.0,
    )

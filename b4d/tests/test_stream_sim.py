"""stream_sim must end (FAILED, exit status 3) when a block write raises, not wait for ever for the block to come back.
Regression: a CUDA out-of-memory in a copy-mode 2-GPU imx811 run raised inside the write's done-callback
before the block was returned to the pool, and the simulator hung for 9 h in free.get()."""
import os
import re
import subprocess
import sys

import pytest

SIM = os.environ.get("STREAM_SIM") or os.path.join(os.path.dirname(__file__), "..", "tools", "stream_sim.py")
ARGS = ["--shape", "64,128", "--fps", "400", "--seconds", "0.5", "--min-blocks", "4", "--block", "8", "--bank", "8",
        "--blocks", "3", "--max-commits", "2", "--ring", "16", "--chunks", "8,64,64", "--shards", "8,64,128", "--threads", "2",
        "--copy-threads", "2", "--sink", "memory", "--mode", "1", "--predictor", "above3", "--status", "60", "--phantom", "blobs"]


def run(tmp_path, *extra, timeout=180):
    env = dict(os.environ, B4D_CACHE_DIR=str(tmp_path / "cache"))
    p = subprocess.Popen([sys.executable, SIM, *ARGS, *extra], cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # a venv python.exe is a launcher with the interpreter as its child: kill the whole tree, not just the launcher
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        p.communicate()
        raise
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def test_clean_run_reports_result(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "RESULT " in r.stdout and "FAILED" not in r.stdout


@pytest.mark.parametrize("nth", [1, 2])
def test_failed_block_write_ends_the_run(tmp_path, nth):
    """every write from the nth on fails: the old code lost each block and then waited for one for ever"""
    try:
        r = run(tmp_path, "--inject-write-error", str(nth))
    except subprocess.TimeoutExpired:
        pytest.fail("stream_sim hung after a failed block write")
    assert r.returncode == 3, r.stdout[-2000:] + r.stderr[-2000:]
    assert re.search(r"FAILED: \d+ block write\(s\) raised", r.stdout) and "injected write error" in r.stdout
    assert "RESULT " not in r.stdout


@pytest.mark.parametrize("nth", [1, 3])
def test_stalled_block_write_ends_the_run(tmp_path, nth):
    """A write that never finishes and never raises (an upload event that never completed past the WDDM
    page-locked limit, 2-GPU imx811 copy mode) hung the run in free.get() / close(); the watchdog must end it"""
    try:
        r = run(tmp_path, "--inject-write-stall", str(nth), "--stall-timeout", "3", timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("stream_sim hung after a block write stalled")
    assert r.returncode == 4, r.stdout[-2000:] + r.stderr[-2000:]
    assert re.search(r"FAILED: stalled: no block write completed for \d+ s", r.stdout)
    assert "RESULT " not in r.stdout

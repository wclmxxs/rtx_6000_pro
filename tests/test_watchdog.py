import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "watchdog/main.py"
SPEC = importlib.util.spec_from_file_location("h3_watchdog", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def job(status, now, duration=5, nfe=8):
    return {
        "id": f"{status}-task",
        "status": status,
        "_last_progress_at": now,
        "_business": {"duration": duration, "nfe": nfe},
    }


def test_cuda_oom_is_detected_but_generic_errors_are_not():
    line = MODULE.fatal_oom_line(
        "ok\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
    )
    assert line and "CUDA out of memory" in line
    assert MODULE.fatal_oom_line("RuntimeError: unrelated") is None


def test_queued_job_only_stalls_when_no_task_is_running(monkeypatch):
    monkeypatch.setattr(MODULE, "QUEUED_STALL_SECONDS", 300)
    queued = job("queued", now=100)
    assert "stayed queued" in MODULE.stalled_job_reason([queued], now=401)
    running = job("in_progress", now=390)
    assert MODULE.stalled_job_reason([running, queued], now=401) is None


def test_running_timeout_scales_for_long_high_nfe_jobs(monkeypatch):
    monkeypatch.setattr(MODULE, "RUNNING_STALL_SECONDS", 600)
    monkeypatch.setattr(MODULE, "RUNNING_SECONDS_PER_UNIT", 4.0)
    short = job("in_progress", now=100, duration=5, nfe=8)
    long = job("in_progress", now=100, duration=15, nfe=50)
    assert MODULE.running_timeout(short) == 600
    assert MODULE.running_timeout(long) == 3000
    assert "no running progress" in MODULE.stalled_job_reason([short], now=701)
    assert MODULE.stalled_job_reason([long], now=701) is None


def test_quarantine_marker_is_atomic_and_recoverable(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "SLOTS_ROOT", tmp_path)
    MODULE.quarantine(3, "fatal CUDA OOM", 123.0)
    marker = MODULE.marker_path(3)
    assert marker.is_file()
    assert '"healthy": false' in marker.read_text()
    MODULE.clear_quarantine(3)
    assert not marker.exists()

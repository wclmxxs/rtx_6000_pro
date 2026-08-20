from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


CONFIG_PATH = Path(os.getenv("WATCHDOG_CONFIG", "/config/instances.json"))
STATE_PATH = Path(os.getenv("WATCHDOG_STATE", "/state/status.json"))
SLOTS_ROOT = Path(os.getenv("WATCHDOG_SLOTS_ROOT", "/slots"))
API_KEY = os.getenv("API_KEY", "")
ENABLED = os.getenv("WATCHDOG_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INTERVAL_SECONDS = max(5, int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "15")))
QUEUED_STALL_SECONDS = max(
    60, int(os.getenv("WATCHDOG_QUEUED_STALL_SECONDS", "300"))
)
RUNNING_STALL_SECONDS = max(
    120, int(os.getenv("WATCHDOG_RUNNING_STALL_SECONDS", "600"))
)
RUNNING_SECONDS_PER_UNIT = max(
    1.0, float(os.getenv("WATCHDOG_RUNNING_SECONDS_PER_UNIT", "4"))
)
UNHEALTHY_SECONDS = max(
    30, int(os.getenv("WATCHDOG_UNHEALTHY_SECONDS", "90"))
)
RESTART_COOLDOWN_SECONDS = max(
    60, int(os.getenv("WATCHDOG_RESTART_COOLDOWN_SECONDS", "300"))
)
RECOVERY_GRACE_SECONDS = max(
    5, int(os.getenv("WATCHDOG_RECOVERY_GRACE_SECONDS", "15"))
)

ACTIVE_STATUSES = frozenset({"queued", "in_progress"})
FATAL_OOM_PATTERN = re.compile(
    r"(?:torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory|"
    r"CUDACachingAllocator[^\n]*(?:OOM|out of memory))",
    re.IGNORECASE,
)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def marker_path(gpu_index: int) -> Path:
    return SLOTS_ROOT / str(gpu_index) / "api-data" / "watchdog.json"


def quarantine(gpu_index: int, reason: str, now: float) -> None:
    atomic_write(
        marker_path(gpu_index),
        {"healthy": False, "reason": reason, "timestamp": int(now)},
    )


def clear_quarantine(gpu_index: int) -> None:
    marker_path(gpu_index).unlink(missing_ok=True)


def load_active_jobs(gpu_index: int) -> list[dict[str, Any]]:
    root = SLOTS_ROOT / str(gpu_index) / "api-data" / "jobs"
    jobs: list[dict[str, Any]] = []
    if not root.is_dir():
        return jobs
    for path in root.glob("*.json"):
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") in ACTIVE_STATUSES:
            jobs.append(job)
    return jobs


def running_timeout(job: dict[str, Any]) -> float:
    business = job.get("_business") or {}
    duration = float(business.get("duration") or job.get("seconds") or 5)
    nfe = float(business.get("nfe") or 8)
    return max(RUNNING_STALL_SECONDS, duration * nfe * RUNNING_SECONDS_PER_UNIT)


def stalled_job_reason(jobs: list[dict[str, Any]], now: float) -> str | None:
    running = [job for job in jobs if job.get("status") == "in_progress"]
    if running:
        for job in running:
            last_progress = float(
                job.get("_last_progress_at")
                or job.get("_gpu_started_at")
                or job.get("_queued_at")
                or now
            )
            timeout = running_timeout(job)
            stalled_for = now - last_progress
            if stalled_for >= timeout:
                return (
                    f"task {job.get('id')} made no running progress for "
                    f"{int(stalled_for)}s (limit={int(timeout)}s)"
                )
        # Queued jobs legitimately wait while this single-GPU worker is busy.
        return None

    for job in (item for item in jobs if item.get("status") == "queued"):
        last_progress = float(
            job.get("_last_progress_at")
            or job.get("_queued_at")
            or job.get("created_at")
            or now
        )
        stalled_for = now - last_progress
        if stalled_for >= QUEUED_STALL_SECONDS:
            return (
                f"task {job.get('id')} stayed queued without a running task for "
                f"{int(stalled_for)}s"
            )
    return None


def fatal_oom_line(logs: str) -> str | None:
    matches = [line for line in logs.splitlines() if FATAL_OOM_PATTERN.search(line)]
    return matches[-1][-1000:] if matches else None


def container_health(container: Any) -> tuple[str, str]:
    container.reload()
    state = container.attrs.get("State") or {}
    health = (state.get("Health") or {}).get("Status", "none")
    return str(state.get("Status") or "unknown"), str(health)


def container_started_at(container: Any) -> int:
    value = str((container.attrs.get("State") or {}).get("StartedAt") or "")
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def fail_active_jobs(
    client: httpx.Client, instance: dict[str, Any], reason: str
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    response = client.post(
        f"{str(instance['internal_url']).rstrip('/')}/internal/watchdog/fail-active",
        headers=headers,
        json={"reason": reason},
    )
    response.raise_for_status()
    return response.json()


class SlotTracker:
    def __init__(self, now: float) -> None:
        self.log_since = int(now) - 300
        self.last_oom_line: str | None = None
        self.last_restart_at = 0.0
        self.last_restart_reason: str | None = None
        self.restart_count = 0
        self.unhealthy_since: float | None = None
        self.recovering = False

    def in_cooldown(self, now: float) -> bool:
        return bool(self.last_restart_at) and (
            now - self.last_restart_at < RESTART_COOLDOWN_SECONDS
        )

    def observe_health(self, healthy: bool, now: float) -> None:
        if healthy:
            self.unhealthy_since = None
        elif self.unhealthy_since is None:
            self.unhealthy_since = now

    def unhealthy_reason(self, now: float, state: str, health: str) -> str | None:
        if self.unhealthy_since is None:
            return None
        unhealthy_for = now - self.unhealthy_since
        if unhealthy_for < UNHEALTHY_SECONDS:
            return None
        return (
            f"container remained unhealthy for {int(unhealthy_for)}s "
            f"(state={state}, health={health})"
        )

    def restarted(self, reason: str, now: float) -> None:
        self.last_restart_at = now
        self.last_restart_reason = reason
        self.restart_count += 1
        self.unhealthy_since = now
        self.recovering = True
        self.log_since = int(now) + 1


def restart_worker(
    client: httpx.Client,
    container: Any,
    instance: dict[str, Any],
    tracker: SlotTracker,
    reason: str,
    now: float,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    if tracker.in_cooldown(now):
        return False, None, "restart cooldown is active"
    gpu_index = int(instance["gpu_index"])
    quarantine(gpu_index, reason, now)
    failed: dict[str, Any] | None = None
    fail_error: str | None = None
    try:
        failed = fail_active_jobs(client, instance, reason)
    except Exception as exc:  # orphan detection finishes cleanup after restart
        fail_error = f"{type(exc).__name__}: {exc}"
    print(f"watchdog restarting {container.name}: {reason}", flush=True)
    container.restart(timeout=10)
    tracker.restarted(reason, now)
    return True, failed, fail_error


def main() -> None:
    import docker

    config = load_config()
    instances = config.get("instances") or []
    docker_client = docker.from_env()
    trackers = {
        int(instance["gpu_index"]): SlotTracker(time.time()) for instance in instances
    }
    with httpx.Client(timeout=10) as client:
        while True:
            cycle_started = time.time()
            state: dict[str, Any] = {
                "ok": True,
                "enabled": ENABLED,
                "timestamp": int(cycle_started),
                "instances": [],
            }
            for instance in instances:
                gpu_index = int(instance["gpu_index"])
                tracker = trackers[gpu_index]
                container_name = f"minimax-h3-comfy-{gpu_index}"
                detail: dict[str, Any] = {
                    "gpu_index": gpu_index,
                    "container": container_name,
                    "restarts_by_watchdog": tracker.restart_count,
                    "last_restart_at": int(tracker.last_restart_at) or None,
                    "last_restart_reason": tracker.last_restart_reason,
                }
                try:
                    container = docker_client.containers.get(container_name)
                    container_state, health = container_health(container)
                    healthy = container_state == "running" and health == "healthy"
                    tracker.observe_health(healthy, cycle_started)
                    detail.update(
                        {"container_state": container_state, "health": health}
                    )
                    if not ENABLED:
                        clear_quarantine(gpu_index)
                        state["instances"].append(detail)
                        continue

                    if tracker.recovering and healthy:
                        if cycle_started - tracker.last_restart_at >= RECOVERY_GRACE_SECONDS:
                            clear_quarantine(gpu_index)
                            tracker.recovering = False
                            detail["recovered"] = True
                        else:
                            detail["recovering"] = True

                    reason: str | None = None
                    if container_state == "running":
                        tracker.log_since = max(
                            tracker.log_since, container_started_at(container)
                        )
                        logs = container.logs(
                            since=tracker.log_since,
                            timestamps=True,
                            tail=10000,
                        ).decode(errors="replace")
                        tracker.log_since = int(cycle_started)
                        oom_line = fatal_oom_line(logs)
                        if oom_line and oom_line != tracker.last_oom_line:
                            reason = f"fatal CUDA OOM: {oom_line[-500:]}"
                            tracker.last_oom_line = oom_line

                    jobs = load_active_jobs(gpu_index)
                    detail["active_jobs"] = len(jobs)
                    detail["job_statuses"] = {
                        status: sum(job.get("status") == status for job in jobs)
                        for status in sorted({str(job.get("status")) for job in jobs})
                    }
                    if reason is None and healthy and not tracker.recovering:
                        reason = stalled_job_reason(jobs, cycle_started)
                    if reason is None and not healthy and not tracker.recovering:
                        reason = tracker.unhealthy_reason(
                            cycle_started, container_state, health
                        )

                    if reason:
                        restarted, failed, fail_error = restart_worker(
                            client,
                            container,
                            instance,
                            tracker,
                            reason,
                            cycle_started,
                        )
                        detail.update(
                            {
                                "restart_triggered": restarted,
                                "restart_reason": reason,
                                "failed_jobs": failed,
                                "fail_jobs_error": fail_error,
                            }
                        )
                    elif healthy and not tracker.recovering and marker_path(gpu_index).exists():
                        # Recover from a watchdog process restart after the worker
                        # itself has already become healthy.
                        clear_quarantine(gpu_index)
                        detail["recovered"] = True
                except Exception as exc:  # one bad slot must not hide the others
                    detail["error"] = f"{type(exc).__name__}: {exc}"
                    state["ok"] = False
                    print(f"watchdog GPU {gpu_index} failed: {detail['error']}", flush=True)
                state["instances"].append(detail)
            atomic_write(STATE_PATH, state)
            elapsed = time.time() - cycle_started
            time.sleep(max(1.0, INTERVAL_SECONDS - elapsed))


if __name__ == "__main__":
    main()

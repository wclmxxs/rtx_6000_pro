from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


CONFIG_PATH = Path(os.getenv("REPORTER_CONFIG", "/config/instances.json"))
STATE_PATH = Path(os.getenv("REPORTER_STATE", "/state/status.json"))
CATALOG_URL = os.getenv(
    "REPORT_CATALOG_URL",
    "https://ic-capcut-edit-gateway.capcut.com/ic/capcut/edit_gateway/v1/report_catalog",
)
CATALOG_AUTH = os.getenv(
    "REPORT_CATALOG_AUTH", "bernard-edit-bridge-internal-call"
)
DEPLOYMENT_REPORT_URL = os.getenv("DEPLOYMENT_REPORT_URL", "").strip()
PSM = os.getenv("PSM", "capcut.ai_infra.federation")
SERVICE_ID = os.getenv("SERVICE_ID", "Minimax-H3-AWS-RTX6000")
API_KEY = os.getenv("API_KEY", "")
INTERVAL_SECONDS = max(2, int(os.getenv("REPORT_INTERVAL_SECONDS", "5")))


def write_state(payload: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(STATE_PATH)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def probe_instance(client: httpx.Client, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    alive = False
    error = None
    health: dict[str, Any] = {}
    try:
        response = client.get(
            f"{item['internal_url'].rstrip('/')}/healthz",
            headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
        )
        response.raise_for_status()
        health = response.json()
        alive = bool(health.get("ok")) and int(health.get("healthy_workers", 0)) == 1
    except Exception as exc:  # reporter must keep running through worker restarts
        error = str(exc)

    bernard = {
        "id": item["id"],
        "host": item["host"],
        "ports": [int(item["port"])],
        "state": "TASK_RUNNING",
        "healthCheckResults": [{"alive": alive}],
        "containerInfos": {
            f"h3-gpu-{item['gpu_index']}": {
                "request": {
                    "cpu": int(item.get("cpu", 4)),
                    "memory": int(item.get("memory_mb", 16384)),
                    "nvidia.com/gpu": 1,
                }
            }
        },
    }
    detail = {
        "id": item["id"],
        "gpu_index": item["gpu_index"],
        "gpu_uuid": item["gpu_uuid"],
        "host": item["host"],
        "port": item["port"],
        "alive": alive,
        "health": health,
        "error": error,
    }
    return bernard, detail


def report_catalog(client: httpx.Client, instances: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post(
        CATALOG_URL,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Auth": CATALOG_AUTH,
        },
        json={
            "psm": PSM,
            "service_id": SERVICE_ID,
            "instances_json": json.dumps(instances, separators=(",", ":")),
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") is False:
        raise RuntimeError(payload.get("message") or "ReportCatalog returned success=false")
    return payload


def deployment_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config.get("deployment") or {}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def report_deployment(client: httpx.Client, config: dict[str, Any], details: list[dict[str, Any]]) -> None:
    if not DEPLOYMENT_REPORT_URL:
        return
    response = client.post(
        DEPLOYMENT_REPORT_URL,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Auth": CATALOG_AUTH,
        },
        json={
            "psm": PSM,
            "service_id": SERVICE_ID,
            "node": config.get("node") or {},
            "deployment": config.get("deployment") or {},
            "instances": details,
        },
    )
    response.raise_for_status()


def main() -> None:
    last_deployment = ""
    with httpx.Client(timeout=15) as client:
        while True:
            started = time.time()
            state: dict[str, Any] = {
                "ok": False,
                "catalog_success": False,
                "timestamp": int(started),
                "psm": PSM,
                "service_id": SERVICE_ID,
            }
            try:
                config = load_config()
                probed = [probe_instance(client, item) for item in config["instances"]]
                instances = [item[0] for item in probed]
                details = [item[1] for item in probed]
                catalog = report_catalog(client, instances)
                fingerprint = deployment_fingerprint(config)
                if fingerprint != last_deployment:
                    report_deployment(client, config, details)
                    last_deployment = fingerprint
                state.update(
                    {
                        "ok": True,
                        "catalog_success": True,
                        "catalog_response": catalog,
                        "healthy_instances": sum(item["alive"] for item in details),
                        "instance_count": len(details),
                        "instances": details,
                        "deployment_fingerprint": fingerprint,
                    }
                )
                print(
                    f"catalog reported: healthy={state['healthy_instances']}/{len(details)} "
                    f"response={catalog}",
                    flush=True,
                )
            except Exception as exc:
                state["error"] = str(exc)
                print(f"catalog report failed: {exc}", flush=True)
            write_state(state)
            elapsed = time.time() - started
            time.sleep(max(0.5, INTERVAL_SECONDS - elapsed))


if __name__ == "__main__":
    main()

import importlib.util
import json
from pathlib import Path

import httpx


MODULE_PATH = Path(__file__).resolve().parents[1] / "reporter/main.py"
SPEC = importlib.util.spec_from_file_location("h3_reporter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def instance():
    return {
        "id": "i-test-gpu-2",
        "host": "10.0.0.4",
        "port": 30012,
        "internal_url": "http://h3-api-2:30010",
        "gpu_index": 2,
        "gpu_uuid": "GPU-test",
        "cpu": 8,
        "memory_mb": 32768,
    }


def test_probe_maps_health_to_bernard_instance():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://h3-api-2:30010/healthz"
        return httpx.Response(
            200,
            json={"ok": True, "healthy_workers": 1, "deployment": {"release_id": "r1"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        bernard, detail = MODULE.probe_instance(client, instance())

    assert bernard == {
        "id": "i-test-gpu-2",
        "host": "10.0.0.4",
        "ports": [30012],
        "state": "TASK_RUNNING",
        "healthCheckResults": [{"alive": True}],
        "containerInfos": {
            "h3-gpu-2": {
                "request": {"cpu": 8, "memory": 32768, "nvidia.com/gpu": 1}
            }
        },
    }
    assert detail["alive"] is True
    assert detail["health"]["deployment"]["release_id"] == "r1"


def test_catalog_contract_serializes_instances_json(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"success": True})

    monkeypatch.setattr(MODULE, "CATALOG_URL", "https://gateway.test/report_catalog")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = MODULE.report_catalog(client, [{"id": "slot-0"}])

    assert result == {"success": True}
    assert captured["headers"]["x-internal-auth"] == MODULE.CATALOG_AUTH
    assert captured["body"]["psm"] == "capcut.ai_infra.federation"
    assert captured["body"]["service_id"] == "Minimax-H3-AWS-RTX6000"
    assert json.loads(captured["body"]["instances_json"]) == [{"id": "slot-0"}]


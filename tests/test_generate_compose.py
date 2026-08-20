import importlib.util
import json
import sys
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_compose.py"
SPEC = importlib.util.spec_from_file_location("generate_compose", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_worker_and_api_are_bound_to_requested_gpu():
    gpu = {
        "index": 3,
        "uuid": "GPU-test",
        "name": "RTX PRO 6000 Blackwell",
        "memory_mb": 97887,
    }
    worker = "\n".join(MODULE.worker_service(gpu, "/srv/minimax-h3"))
    api = "\n".join(MODULE.api_service(gpu, "/srv/minimax-h3", "10.0.0.4", 30010))
    assert "device_ids: [\"3\"]" in worker
    assert "minimax-h3-comfy-3" in worker
    assert "command -v cc >/dev/null" in worker
    assert "command -v c++ >/dev/null" in worker
    assert "/usr/include/python3.12/Python.h" in worker
    assert "driver.active.get_current_target()" in worker
    assert "CacheDiT_MiniMax_H3_Advanced_Optimizer" in worker
    assert "MiniMaxH3ScheduledSolAttentionPatch" in worker
    assert "0.0.0.0:30013:30010" in api
    assert "http://10.0.0.4:30013" in api
    assert "CACHE_DIT_ENABLED: ${CACHE_DIT_ENABLED:-true}" in api
    assert "CACHE_DIT_FN_BLOCKS: ${CACHE_DIT_FN_BLOCKS:-1}" in api
    assert "CACHE_DIT_RESIDUAL_DIFF_THRESHOLD: ${CACHE_DIT_RESIDUAL_DIFF_THRESHOLD:-0.24}" in api
    assert "SOL_ATTN_ENABLED: ${SOL_ATTN_ENABLED:-true}" in api
    assert "SOL_ATTN_TAU_START: ${SOL_ATTN_TAU_START:-1.2}" in api
    assert "SOL_ATTN_DENSE_BLOCKS: ${SOL_ATTN_DENSE_BLOCKS:-0-2,-1}" in api


def test_gpu_count_is_discovered_from_nvidia_smi(monkeypatch):
    class Result:
        stdout = (
            "0, GPU-a, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887\n"
            "1, GPU-b, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887\n"
            "2, GPU-c, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887\n"
        )

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Result())
    gpus = MODULE.detect_gpus()
    assert [gpu["index"] for gpu in gpus] == [0, 1, 2]
    assert gpus[2]["uuid"] == "GPU-c"


def test_main_renders_one_service_pair_per_detected_gpu(monkeypatch, tmp_path):
    gpus = [
        {"index": index, "uuid": f"GPU-{index}", "name": "RTX PRO 6000", "memory_mb": 97887}
        for index in range(3)
    ]
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: gpus)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--advertise-host",
            "10.0.0.4",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--worker-image",
            "worker:test",
            "--api-image",
            "api:test",
        ],
    )
    MODULE.main()
    compose = (tmp_path / "compose.yaml").read_text()
    parsed = yaml.safe_load(compose)
    config = json.loads((tmp_path / "instances.json").read_text())
    assert sum(f"\n  h3-comfy-{index}:\n" in compose for index in range(3)) == 3
    assert sum(f"\n  h3-api-{index}:\n" in compose for index in range(3)) == 3
    assert len(parsed["services"]) == 7
    assert parsed["services"]["h3-comfy-2"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["2"]
    api_two = parsed["services"]["h3-api-2"]
    assert "/srv/minimax-h3/slots/2/output:/comfy-output" in api_two["volumes"]
    assert api_two["environment"]["OUTPUT_TTL_SECONDS"] == "${OUTPUT_TTL_SECONDS:-43200}"
    assert api_two["environment"]["CACHE_DIT_ENABLED"] == "${CACHE_DIT_ENABLED:-true}"
    assert api_two["environment"]["CACHE_DIT_WARMUP_STEPS"] == "${CACHE_DIT_WARMUP_STEPS:-2}"
    assert api_two["environment"]["SOL_ATTN_ENABLED"] == "${SOL_ATTN_ENABLED:-true}"
    assert api_two["environment"]["SOL_ATTN_CURVE"] == "${SOL_ATTN_CURVE:-cosine}"
    assert len(config["instances"]) == 3
    assert config["deployment"]["worker_image"] == "worker:test"
    assert config["deployment"]["api_image"] == "api:test"

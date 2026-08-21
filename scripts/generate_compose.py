#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def quote(value: object) -> str:
    return json.dumps(str(value))


def detect_gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    gpus: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mb": int(memory),
            }
        )
    if not gpus:
        raise SystemExit("nvidia-smi returned no GPUs")
    indexes = [gpu["index"] for gpu in gpus]
    if indexes != list(range(len(indexes))):
        raise SystemExit(f"GPU indexes must be contiguous from zero; got {indexes}")
    return gpus


def detect_host_resources(gpu_count: int) -> tuple[int, int]:
    cpu_total = os.cpu_count() or gpu_count
    memory_total_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        with meminfo.open() as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    memory_total_mb = int(line.split()[1]) // 1024
                    break
    if memory_total_mb <= 0:
        memory_total_mb = gpu_count * 16 * 1024
    return max(1, cpu_total // gpu_count), max(1024, memory_total_mb // gpu_count)


def worker_service(gpu: dict[str, object], data_root: str) -> list[str]:
    index = int(gpu["index"])
    slot = f"{data_root}/slots/{index}"
    return [
        f"  h3-comfy-{index}:",
        "    image: ${WORKER_IMAGE}",
        f"    container_name: minimax-h3-comfy-{index}",
        "    restart: unless-stopped",
        "    init: true",
        "    user: ${HOST_UID}:${HOST_GID}",
        "    shm_size: 16gb",
        "    environment:",
        "      PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
        "    command:",
        "      - python",
        "      - main.py",
        "      - --listen",
        "      - 0.0.0.0",
        "      - --port",
        "      - '8188'",
        "      - --models-directory",
        "      - /models",
        "      - --input-directory",
        "      - /input",
        "      - --output-directory",
        "      - /output",
        "      - --temp-directory",
        "      - /temp",
        "      - --user-directory",
        "      - /user",
        "      - --database-url",
        "      - sqlite:////user/comfyui.db",
        "      - --reserve-vram",
        "      - '2'",
        "      - --highvram",
        "      - --use-sage-attention",
        "    volumes:",
        f"      - {data_root}/models:/models:ro",
        f"      - {slot}/input:/input",
        f"      - {slot}/output:/output",
        f"      - {slot}/temp:/temp",
        f"      - {slot}/user:/user",
        "    healthcheck:",
        "      test: ['CMD-SHELL', 'command -v cc >/dev/null && command -v c++ >/dev/null && test -f /usr/include/python3.12/Python.h && (test -f /tmp/.triton-driver-ready || (/opt/venv/bin/python -c \"from triton.runtime import driver; driver.active.get_current_target()\" >/dev/null && touch /tmp/.triton-driver-ready)) && curl -fsS http://127.0.0.1:8188/system_stats >/dev/null && curl -fsS http://127.0.0.1:8188/object_info/CacheDiT_MiniMax_H3_Advanced_Optimizer | grep -q CacheDiT_MiniMax_H3_Advanced_Optimizer && curl -fsS http://127.0.0.1:8188/object_info/MiniMaxH3ScheduledSolAttentionPatch | grep -q MiniMaxH3ScheduledSolAttentionPatch']",
        "      interval: 10s",
        "      timeout: 30s",
        "      retries: 18",
        "      start_period: 90s",
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        f"              device_ids: [{quote(index)}]",
        "              capabilities: [gpu]",
    ]


def api_service(
    gpu: dict[str, object], data_root: str, host: str, base_port: int
) -> list[str]:
    index = int(gpu["index"])
    port = base_port + index
    slot = f"{data_root}/slots/{index}"
    return [
        f"  h3-api-{index}:",
        "    image: ${API_IMAGE}",
        f"    container_name: minimax-h3-api-{index}",
        "    restart: unless-stopped",
        "    init: true",
        "    env_file: ../.env",
        "    depends_on:",
        f"      h3-comfy-{index}:",
        "        condition: service_healthy",
        "    ports:",
        f"      - '0.0.0.0:{port}:30010'",
        f"      - '[::]:{port}:30010'",
        "    environment:",
        f"      COMFY_URL: http://h3-comfy-{index}:8188",
        "      INPUT_ROOT: /input",
        "      MEDIA_ROOT: /input",
        "      DATA_ROOT: /data",
        "      COMFY_OUTPUT_ROOT: /comfy-output",
        "      OUTPUT_TTL_SECONDS: ${OUTPUT_TTL_SECONDS:-43200}",
        "      CLEANUP_INTERVAL_SECONDS: ${CLEANUP_INTERVAL_SECONDS:-60}",
        "      MAX_QUEUE_DEPTH: ${MAX_QUEUE_DEPTH:-2}",
        "      ORPHAN_GRACE_SECONDS: ${ORPHAN_GRACE_SECONDS:-30}",
        "      WATCHDOG_MARKER: /data/watchdog.json",
        "      ALLOW_REMOTE_MEDIA: 'true'",
        "      MAX_MEDIA_BYTES: '4294967296'",
        "      REMOTE_MEDIA_HOST_ALLOWLIST: ${REMOTE_MEDIA_HOST_ALLOWLIST:-.byted.org}",
        "      TURBO_ENABLED: 'true'",
        "      TURBO_LORA: minimax_h3_turbo_v4_step600_ema.safetensors",
        "      CACHE_DIT_ENABLED: ${CACHE_DIT_ENABLED:-true}",
        "      CACHE_DIT_FN_BLOCKS: ${CACHE_DIT_FN_BLOCKS:-1}",
        "      CACHE_DIT_BN_BLOCKS: ${CACHE_DIT_BN_BLOCKS:-0}",
        "      CACHE_DIT_RESIDUAL_DIFF_THRESHOLD: ${CACHE_DIT_RESIDUAL_DIFF_THRESHOLD:-0.24}",
        "      CACHE_DIT_WARMUP_STEPS: ${CACHE_DIT_WARMUP_STEPS:-1}",
        "      CACHE_DIT_PRINT_SUMMARY: ${CACHE_DIT_PRINT_SUMMARY:-true}",
        "      SOL_ATTN_ENABLED: ${SOL_ATTN_ENABLED:-true}",
        "      SOL_ATTN_TAU_START: ${SOL_ATTN_TAU_START:-1.5}",
        "      SOL_ATTN_TAU_END: ${SOL_ATTN_TAU_END:-1.5}",
        "      SOL_ATTN_CURVE: ${SOL_ATTN_CURVE:-cosine}",
        "      SOL_ATTN_MIN_TOKENS: ${SOL_ATTN_MIN_TOKENS:-4096}",
        "      SOL_ATTN_STRICT: ${SOL_ATTN_STRICT:-true}",
        "      SOL_ATTN_DENSE_PERCENT: ${SOL_ATTN_DENSE_PERCENT:-0}",
        "      SOL_ATTN_THRESH_TYPE: ${SOL_ATTN_THRESH_TYPE:-diag}",
        "      SOL_ATTN_INT8_QK: ${SOL_ATTN_INT8_QK:-true}",
        "      SOL_ATTN_INT8_PV: ${SOL_ATTN_INT8_PV:-false}",
        "      SOL_ATTN_SINK_CONDITIONING: ${SOL_ATTN_SINK_CONDITIONING:-exact_kv}",
        '      SOL_ATTN_DENSE_BLOCKS: "${SOL_ATTN_DENSE_BLOCKS-}"',
        "      DEFAULT_NFE: ${DEFAULT_NFE:-8}",
        "      CORS_ORIGINS: '*'",
        f"      PUBLIC_BASE_URL: {quote(f'http://{host}:{port}')}",
        f"      GPU_INDEX: {quote(index)}",
        f"      GPU_UUID: {quote(gpu['uuid'])}",
        "      RELEASE_ID: ${RELEASE_ID}",
        "    volumes:",
        f"      - {slot}/input:/input",
        f"      - {slot}/output:/comfy-output",
        f"      - {slot}/api-data:/data",
        "    healthcheck:",
        "      test: ['CMD-SHELL', 'curl -fsS -H \"Authorization: Bearer $$API_KEY\" http://127.0.0.1:30010/healthz >/dev/null && curl -g -fsS -H \"Authorization: Bearer $$API_KEY\" http://[::1]:30010/healthz >/dev/null']",
        "      interval: 10s",
        "      timeout: 5s",
        "      retries: 12",
        "      start_period: 30s",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".generated")
    parser.add_argument("--data-root", default="/srv/minimax-h3")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--base-port", type=int, default=30010)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--worker-image", default=os.getenv("WORKER_IMAGE", ""))
    parser.add_argument("--api-image", default=os.getenv("API_IMAGE", ""))
    parser.add_argument("--watchdog-image", default=os.getenv("WATCHDOG_IMAGE", ""))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_mount = quote(
        f"{output_dir / 'instances.json'}:/config/instances.json:ro"
    )
    gpus = detect_gpus()
    cpu_per_gpu, memory_per_gpu_mb = detect_host_resources(len(gpus))

    compose = ["name: minimax-h3-rtx6000pro", "", "services:"]
    for gpu in gpus:
        compose.extend(worker_service(gpu, args.data_root))
        compose.append("")
        compose.extend(api_service(gpu, args.data_root, args.advertise_host, args.base_port))
        compose.append("")

    compose.extend(
        [
            "  h3-watchdog:",
            "    image: ${WATCHDOG_IMAGE}",
            "    container_name: minimax-h3-watchdog",
            "    restart: unless-stopped",
            "    init: true",
            "    env_file: ../.env",
            "    environment:",
            "      WATCHDOG_CONFIG: /config/instances.json",
            "      WATCHDOG_STATE: /state/status.json",
            "      WATCHDOG_SLOTS_ROOT: /slots",
            "    volumes:",
            "      - /var/run/docker.sock:/var/run/docker.sock",
            f"      - {instances_mount}",
            f"      - {args.data_root}/slots:/slots",
            f"      - {args.data_root}/watchdog:/state",
            "",
            "  h3-reporter:",
            "    image: ${REPORTER_IMAGE}",
            "    container_name: minimax-h3-reporter",
            "    restart: unless-stopped",
            "    init: true",
            "    env_file: ../.env",
            "    environment:",
            "      REPORTER_CONFIG: /config/instances.json",
            "      REPORTER_STATE: /state/status.json",
            "    volumes:",
            f"      - {quote(f'{output_dir / "instances.json"}:/config/instances.json:ro')}",
            f"      - {args.data_root}/reporter:/state",
        ]
    )

    instances = []
    for gpu in gpus:
        index = int(gpu["index"])
        instances.append(
            {
                "id": f"{args.instance_id}-gpu-{index}",
                "host": args.advertise_host,
                "port": args.base_port + index,
                "internal_url": f"http://h3-api-{index}:30010",
                "gpu_index": index,
                "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"],
                "gpu_memory_mb": gpu["memory_mb"],
                "cpu": cpu_per_gpu,
                "memory_mb": memory_per_gpu_mb,
            }
        )

    model_lock = json.loads((repo_root / "config/models.lock.json").read_text())
    reporter_config = {
        "node": {
            "instance_id": args.instance_id,
            "host": args.advertise_host,
            "gpu_count": len(gpus),
        },
        "deployment": {
            "release_id": args.release_id,
            "worker_image": args.worker_image,
            "api_image": args.api_image,
            "watchdog_image": args.watchdog_image,
            "models": model_lock,
        },
        "instances": instances,
    }
    (output_dir / "compose.yaml").write_text("\n".join(compose) + "\n")
    (output_dir / "instances.json").write_text(
        json.dumps(reporter_config, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "gpu-info.json").write_text(
        json.dumps(gpus, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"generated {len(gpus)} GPU services in {output_dir}")


if __name__ == "__main__":
    main()

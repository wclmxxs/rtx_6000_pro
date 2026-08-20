#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(method: str, url: str, body: dict | None = None, timeout: int = 120) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def warm_one(
    host: str, port: int, release_id: str, marker_root: Path, force: bool = False
) -> dict:
    marker = marker_root / f"gpu-{port}" / f"warm-{release_id}.json"
    if marker.is_file() and not force:
        return json.loads(marker.read_text())
    base = f"http://{host}:{port}"
    payload = {
        "model": "MiniMax-H3",
        "content": [
            {
                "type": "text",
                "text": "Static camera, a red ball resting on a clean studio table, subtle ambient sound.",
            }
        ],
        "resolution": "704P",
        "duration": 4,
        "ratio": "16:9",
        "num_inference_steps": 4,
        "seed": 202608190004,
    }
    task_id = call(
        "POST", f"{base}/ic/capcut/edit_gateway/v2/video_generation", payload
    )["task_id"]
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        result = call(
            "POST",
            f"{base}/ic/capcut/edit_gateway/v2/query/video_generation",
            {"model": "MiniMax-H3", "task_id": task_id},
        )["task"]
        if result["status"] == "succeeded":
            record = {
                "ok": True,
                "port": port,
                "task_id": task_id,
                "release_id": release_id,
                "completed_at": int(time.time()),
            }
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(record, indent=2) + "\n")
            return record
        if result["status"] == "failed":
            raise RuntimeError(f"warmup failed on port {port}: {result.get('error')}")
        time.sleep(2)
    raise TimeoutError(f"warmup timed out on port {port}: {task_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=30010)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--marker-root", default="/srv/minimax-h3/warmup")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run warmup even when a marker from the same release exists",
    )
    args = parser.parse_args()

    ports = [args.base_port + index for index in range(args.gpu_count)]
    failures = []
    parallelism = min(args.parallelism, len(ports))
    for offset in range(0, len(ports), parallelism):
        batch = ports[offset : offset + parallelism]
        print(f"warming ports: {','.join(map(str, batch))}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    warm_one,
                    args.host,
                    port,
                    args.release_id,
                    Path(args.marker_root),
                    args.force,
                ): port
                for port in batch
            }
            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    failures.append(f"port {port}: {exc}")
        if failures:
            # Do not pile more work onto the remaining GPUs when a batch is
            # still running remotely or failed to become ready.
            break
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()

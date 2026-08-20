#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(root: Path, item: dict) -> Path:
    return root / item["local_dir"] / item["filename"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/srv/minimax-h3/models")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "config/models.lock.json"),
    )
    parser.add_argument(
        "--trust-existing-size",
        action="store_true",
        help="accept existing files with the locked size without recomputing SHA256",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    root.mkdir(parents=True, exist_ok=True)
    required = sum(
        int(item["size"])
        for item in manifest["files"]
        if not (
            (destination := destination_for(root, item)).is_file()
            and destination.stat().st_size == int(item["size"])
        )
    )
    free = shutil.disk_usage(root).free
    if free < required + 10 * 1024**3:
        raise SystemExit(
            f"insufficient free space under {root}: need at least "
            f"{(required + 10 * 1024**3) / 1024**3:.1f} GiB, "
            f"have {free / 1024**3:.1f} GiB"
        )

    for item in manifest["files"]:
        destination = destination_for(root, item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == int(item["size"]):
            if args.trust_existing_size:
                print(f"trusted existing size: {destination}", flush=True)
                continue
            print(f"verifying existing: {destination}", flush=True)
            actual = sha256(destination)
            if actual == item["sha256"]:
                print(f"verified existing: {destination}", flush=True)
                continue
            destination.unlink()

        print(f"downloading: {item['repo']}::{item['filename']}", flush=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=item["repo"],
                filename=item["filename"],
                revision=item["revision"],
                local_dir=root / item["local_dir"],
                token=os.getenv("HF_TOKEN") or None,
            )
        )
        if downloaded.resolve() != destination.resolve():
            raise SystemExit(f"unexpected download path: {downloaded}, wanted {destination}")
        if destination.stat().st_size != int(item["size"]):
            raise SystemExit(f"size mismatch: {destination}")
        actual = sha256(destination)
        if actual != item["sha256"]:
            raise SystemExit(
                f"sha256 mismatch for {destination}: expected {item['sha256']}, got {actual}"
            )
        print(f"verified: {destination}", flush=True)


if __name__ == "__main__":
    main()

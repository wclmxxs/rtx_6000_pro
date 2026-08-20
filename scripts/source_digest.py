#!/usr/bin/env python3
"""Build a deterministic digest for the files copied into one Docker image."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


IGNORED_DIRECTORY_NAMES = {".git", "__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def source_files(paths: list[Path], root: Path) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        if candidate.is_file():
            files.add(candidate.resolve())
            continue
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        for item in candidate.rglob("*"):
            if not item.is_file():
                continue
            relative_parts = item.relative_to(root).parts
            if any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts):
                continue
            if item.suffix in IGNORED_SUFFIXES or item.name.startswith("._"):
                continue
            files.add(item.resolve())
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def digest_paths(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in source_files(paths, root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    print(digest_paths([Path(path) for path in args.paths], root))


if __name__ == "__main__":
    main()

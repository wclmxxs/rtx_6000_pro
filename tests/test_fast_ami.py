import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_warmup_module():
    path = ROOT / "scripts" / "warmup.py"
    spec = importlib.util.spec_from_file_location("h3_warmup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_download_module():
    if "huggingface_hub" not in sys.modules:
        huggingface_hub = types.ModuleType("huggingface_hub")
        huggingface_hub.hf_hub_download = lambda **_: None
        sys.modules["huggingface_hub"] = huggingface_hub
    path = ROOT / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("h3_download_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trust_existing_size_skips_hash(monkeypatch, tmp_path, capsys):
    download_models = load_download_module()
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model.bin"
    model.write_bytes(b"trusted-ami-model")
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "repo": "unused/repo",
                        "revision": "unused",
                        "filename": "model.bin",
                        "local_dir": ".",
                        "size": model.stat().st_size,
                        # Deliberately wrong: the fast path trusts the baked size.
                        "sha256": "0" * 64,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        download_models.shutil,
        "disk_usage",
        lambda _: download_models.shutil._ntuple_diskusage(
            100 * 1024**3, 0, 100 * 1024**3
        ),
    )
    monkeypatch.setattr(
        download_models,
        "sha256",
        lambda _: (_ for _ in ()).throw(AssertionError("SHA256 must be skipped")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_models.py",
            "--root",
            str(model_root),
            "--manifest",
            str(manifest),
            "--trust-existing-size",
        ],
    )
    download_models.main()
    assert f"trusted existing size: {model}" in capsys.readouterr().out


def test_forced_warmup_ignores_baked_marker(monkeypatch, tmp_path):
    warmup = load_warmup_module()
    marker = tmp_path / "gpu-30010" / "warm-release.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"ok": True, "task_id": "old-task"}))
    calls = []

    def fake_call(method, url, body=None, timeout=120):
        calls.append((method, url, body))
        if "/query/" not in url:
            return {"task_id": "new-task"}
        return {"task": {"status": "succeeded"}}

    monkeypatch.setattr(warmup, "call", fake_call)
    result = warmup.warm_one(
        "127.0.0.1", 30010, "release", tmp_path, force=True
    )

    assert result["task_id"] == "new-task"
    assert len(calls) == 2
    assert json.loads(marker.read_text())["task_id"] == "new-task"


def test_install_migrates_only_the_previous_sparse_defaults():
    script = (ROOT / "install.sh").read_text()
    assert (
        "migrate_env_default SOL_ATTN_TAU_START 1.2 1.5" in script
    )
    assert (
        "migrate_env_default SOL_ATTN_TAU_END 0.8 1.5" in script
    )
    assert (
        "migrate_env_default SOL_ATTN_DENSE_PERCENT 0.2 0" in script
    )
    assert (
        'migrate_env_default SOL_ATTN_DENSE_BLOCKS 0-2,-1 ""' in script
    )
    assert "migrate_env_default CACHE_DIT_WARMUP_STEPS 2 1" in script

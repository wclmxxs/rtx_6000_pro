import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/source_digest.py"
SPEC = importlib.util.spec_from_file_location("source_digest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_digest_changes_with_tracked_source_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "main.py"
    code.write_text("version = 1\n")
    first = MODULE.digest_paths([Path("source")], tmp_path)

    code.write_text("version = 2\n")
    second = MODULE.digest_paths([Path("source")], tmp_path)

    assert first != second


def test_digest_ignores_python_cache_files(tmp_path):
    source = tmp_path / "source"
    cache = source / "__pycache__"
    cache.mkdir(parents=True)
    (source / "main.py").write_text("version = 1\n")
    first = MODULE.digest_paths([Path("source")], tmp_path)

    (cache / "main.cpython-312.pyc").write_bytes(b"cache")
    second = MODULE.digest_paths([Path("source")], tmp_path)

    assert first == second

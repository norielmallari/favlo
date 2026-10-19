"""HF hub cache snapshot resolution."""

from pathlib import Path

import pytest

from favlo.hf_cache import hub_cache_dir, snapshot_dir


def test_snapshot_dir_reads_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    storage = tmp_path / "models--org--demo"
    commit = "a" * 40
    (storage / "refs").mkdir(parents=True)
    (storage / "refs" / "main").write_text(commit, encoding="utf-8")
    snap = storage / "snapshots" / commit
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")

    assert snapshot_dir("org/demo") == snap
    assert hub_cache_dir() == tmp_path.resolve()

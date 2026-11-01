"""Parked Spark setup still registers a finished record as the default."""

import json
from pathlib import Path

from favlo.registry import default_llm_id, load_registry


def test_a_finished_spark_record_is_the_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAVLO_ROOT", str(tmp_path))
    monkeypatch.setenv("FAVLO_SKIP_PLATE", "1")
    row = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "presence_penalty": 0.0,
    }
    monkeypatch.setattr(
        "favlo.registry.pipeline_sampling",
        lambda *_args, **_kwargs: ({"high": row, "low": dict(row), "none": dict(row)}, "high"),
    )
    record = tmp_path / "models" / "Spark-X2.6-1.7B-fp8"
    record.mkdir(parents=True)
    (record / "weights.bin").write_bytes(b"x")
    (record / "manifest.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    checkout = tmp_path / "models" / "Spark-X2.6-1.7B"
    checkout.mkdir()
    (checkout / "config.json").write_text("{}", encoding="utf-8")

    from spark_x26_1_7b.commands import main

    assert main(["setup", "--console", "plain"]) == 0
    assert not checkout.exists()
    row = load_registry(tmp_path)["models"]["Spark-X2.6-1.7B-fp8"]
    assert row["alias"] == "spark-x26-1-7b"
    assert default_llm_id(tmp_path) == "spark-x26-1-7b:high"

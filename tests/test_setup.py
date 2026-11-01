"""An empty registry on a terminal runs setup. A pipe does not."""

import json
from pathlib import Path
from types import SimpleNamespace

from favlo.registry import default_llm_id, load_registry, save_registry
from favlo.setup import prepare


def test_a_registered_row_skips_setup(tmp_path: Path) -> None:
    save_registry({"version": 1, "models": {"packed": {"pipeline": "qwen36-35b-a3b"}}}, tmp_path)
    calls: list[list[str]] = []

    def _run(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
        del cwd, env
        calls.append(command)
        return SimpleNamespace(returncode=0)

    code = prepare(tmp_path, stdin_tty=True, console="plain", metal=lambda: True, run=_run)
    assert code == 0
    assert calls == []


def test_a_pipe_skips_setup(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _run(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
        del cwd, env
        calls.append(command)
        return SimpleNamespace(returncode=0)

    code = prepare(tmp_path, stdin_tty=False, console="plain", metal=lambda: True, run=_run)
    assert code == 0
    assert calls == []


def test_a_terminal_with_no_rows_runs_setup(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _run(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
        calls.append(command)
        assert cwd == tmp_path
        assert env["FAVLO_SKIP_PLATE"] == "1"
        assert env["FAVLO_ROOT"] == str(tmp_path)
        return SimpleNamespace(returncode=0)

    code = prepare(tmp_path, stdin_tty=True, console="pretty", metal=lambda: True, run=_run)
    assert code == 0
    assert calls[0][-2:] == ["--console", "pretty"]
    assert "--spark-x26-1-7b" in calls[0]
    assert "setup" in calls[0]


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


def test_a_machine_that_cannot_compile_exits(tmp_path: Path, capsys) -> None:
    code = prepare(tmp_path, stdin_tty=True, console="plain", metal=lambda: False, run=_fail)
    assert code == 1
    assert "cannot run here" in capsys.readouterr().err


def _fail(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
    del command, cwd, env
    raise AssertionError("setup should not run")

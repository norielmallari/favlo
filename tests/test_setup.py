"""An empty registry on a terminal runs setup. A pipe does not."""

from pathlib import Path
from types import SimpleNamespace

from favlo.registry import save_registry
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
    assert "setup" in calls[0]


def test_a_machine_that_cannot_compile_exits(tmp_path: Path, capsys) -> None:
    code = prepare(tmp_path, stdin_tty=True, console="plain", metal=lambda: False, run=_fail)
    assert code == 1
    assert "cannot run here" in capsys.readouterr().err


def _fail(command: list[str], cwd: Path, env: dict[str, str]) -> SimpleNamespace:
    del command, cwd, env
    raise AssertionError("setup should not run")

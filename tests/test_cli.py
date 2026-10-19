import sys
from pathlib import Path

from favlo.cli import main, pipeline_names

ROOT = Path(__file__).resolve().parents[1]


def test_a_pipeline_entry_is_the_file_that_names_itself(tmp_path: Path) -> None:
    (tmp_path / "demo-model.py").write_text('PIPELINE = "demo-model"\n', encoding="utf-8")
    (tmp_path / "notes.py").write_text("PIPELINE = 'other'\n", encoding="utf-8")
    assert pipeline_names(tmp_path) == ("demo-model",)


def test_help_lists_the_model_flag(capsys) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--qwen38-flash-next" in captured.out
    assert "--qwen38-27b" in captured.out
    assert "--qwen36-35b-a3b" in captured.out


def test_missing_flag_is_an_error(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "--qwen38-flash-next" in captured.err


def test_unknown_model_is_an_error(capsys) -> None:
    assert main(["--another-model"]) == 2
    captured = capsys.readouterr()
    assert "unknown model flag" in captured.err


def test_serve_loads_the_pipeline_file(monkeypatch, capsys) -> None:
    """The installed command does not put the repo root on sys.path."""
    kept: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            if Path(entry).resolve() == ROOT:
                continue
        except OSError:
            pass
        kept.append(entry)
    monkeypatch.setattr(sys, "path", kept)
    for name in list(sys.modules):
        if name == "qwen38_flash_next" or name.startswith("qwen38_flash_next."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    assert main(["serve", "--help"]) == 0
    captured = capsys.readouterr()
    assert "--host" in captured.out


def test_serve_is_a_process_command(capsys) -> None:
    assert main(["serve", "--help"]) == 0
    captured = capsys.readouterr()
    assert "--host" in captured.out
    assert "--qwen38-flash-next" not in captured.out


def test_pipeline_help_comes_from_the_hyphenated_file(capsys) -> None:
    assert main(["--qwen38-flash-next", "--help"]) == 0
    captured = capsys.readouterr()
    assert "probe" in captured.out
    assert main(["--qwen38-27b", "--help"]) == 0
    captured = capsys.readouterr()
    assert "quantize" in captured.out

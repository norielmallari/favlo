import json
import sys
from pathlib import Path

from favlo.cli import main, pipeline_names

ROOT = Path(__file__).resolve().parents[1]

_DEMO_INIT = '''\
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.language_pipeline import LanguagePipeline
from favlo.pipeline import Runtime


class _Runtime(Runtime):
    pass


class Pipeline(LanguagePipeline):
    def handle_responses(self, reply, runtime, body, selection, args, state):
        del reply, runtime, body, selection, args, state

    def main(self, argv: Sequence[str]) -> int:
        del argv
        return 0

    def open_runtime(self, path: Path, **kwargs: Any) -> Runtime:
        del path, kwargs
        return _Runtime()


pipeline = Pipeline()
'''


def test_a_pipeline_package_names_itself(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "pipelines" / "demo_model"
    runtime = model / "metal41"
    runtime.mkdir(parents=True)
    (model / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "__init__.py").write_text(_DEMO_INIT, encoding="utf-8")
    (runtime / "notes.py").write_text("PIPELINE = 'other'\n", encoding="utf-8")
    monkeypatch.delenv("FAVLO_PIPELINE_ROOTS", raising=False)
    monkeypatch.setattr("favlo.logs.configured_pipeline_roots", lambda settings=None: ())
    assert pipeline_names(tmp_path) == ("demo_model",)


def test_an_extra_plugin_root_is_discovered(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "outside" / "demo_model"
    runtime = model / "metal41"
    runtime.mkdir(parents=True)
    (model / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "__init__.py").write_text(_DEMO_INIT, encoding="utf-8")
    (tmp_path / "pipelines").mkdir()
    monkeypatch.setenv("FAVLO_PIPELINE_ROOTS", str(tmp_path / "outside"))
    monkeypatch.setattr("favlo.logs.configured_pipeline_roots", lambda settings=None: ())
    assert pipeline_names(tmp_path) == ("demo_model",)


def test_load_pipeline_falls_back_when_registry_runtime_is_gone(
    tmp_path: Path, monkeypatch
) -> None:
    from favlo.cli import drop_pipeline_modules, load_pipeline

    model = tmp_path / "pipelines" / "moved_model"
    runtime = model / "portable"
    runtime.mkdir(parents=True)
    (model / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "__init__.py").write_text(_DEMO_INIT, encoding="utf-8")
    monkeypatch.delenv("FAVLO_PIPELINE_ROOTS", raising=False)
    monkeypatch.setattr("favlo.logs.configured_pipeline_roots", lambda settings=None: ())
    drop_pipeline_modules()
    module = load_pipeline("moved_model", tmp_path, runtime="metal41")
    assert module.__name__ == "moved_model.portable"


def test_media_exits_when_serve_is_down(capsys) -> None:
    assert main(
        [
            "media",
            "--pipeline",
            "example",
            "--kind",
            "image",
            "--model",
            "example",
            "--port",
            "1",
            "prompt=a",
        ]
    ) == 1
    assert "not listening" in capsys.readouterr().err


def test_help_lists_the_model_flag(capsys) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "--qwen38_27b" in captured.out
    assert "--qwen36_35b_a3b" in captured.out


def test_missing_flag_is_an_error(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "--qwen36_35b_a3b" in captured.err


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
        if name == "qwen36_35b_a3b" or name.startswith("qwen36_35b_a3b.metal41."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    assert main(["serve", "--help"]) == 0
    captured = capsys.readouterr()
    assert "--host" in captured.out


def test_serve_is_a_process_command(capsys) -> None:
    assert main(["serve", "--help"]) == 0
    captured = capsys.readouterr()
    assert "--host" in captured.out
    assert "--qwen38-flash-next" not in captured.out
    assert "--spark-x26-1-7b" not in captured.out


def test_pipeline_help_comes_from_the_package(capsys) -> None:
    assert main(["--qwen36_35b_a3b", "--help"]) == 0
    captured = capsys.readouterr()
    assert "setup" in captured.out
    assert "quantize" in captured.out
    assert "kernels" in captured.out
    assert main(["--qwen38_27b", "--help"]) == 0
    captured = capsys.readouterr()
    assert "download" in captured.out
    assert "quantize" in captured.out
    assert "kernels" in captured.out


def test_media_uses_the_registry_default_when_model_is_omitted(monkeypatch) -> None:
    from favlo.media_cli import _default_model

    registry = {
        "models": {
            "first": {"pipeline": "z_image_turbo"},
            "chosen": {"pipeline": "z_image_turbo", "alias": "served"},
            "other": {"pipeline": "z_image_base"},
        },
        "media": {"image": "chosen"},
    }
    monkeypatch.setattr("favlo.registry.load_registry", lambda root=None: registry)
    assert _default_model("z_image_turbo", "image") == "served"
    registry["media"]["image"] = "other"
    assert _default_model("z_image_turbo", "image") == "first"


def test_media_body_keeps_numbers() -> None:
    from favlo.media_cli import _body

    body = _body(
        ["guidance=3.5", "steps=28", "shift=6", "cfg_normalization=false", "size=1024x1024"],
        "Z-Image-int8",
        "z_image_base",
        "image",
    )
    assert body["guidance"] == 3.5
    assert body["steps"] == 28
    assert body["shift"] == 6
    assert body["cfg_normalization"] is False
    assert body["size"] == "1024x1024"
    assert body["model"] == "Z-Image-int8"


def test_an_unknown_argument_is_an_error(capsys) -> None:
    assert main(["serve-chat"]) == 2
    captured = capsys.readouterr()
    assert "unknown argument" in captured.err


def test_an_unknown_serve_flag_is_an_error(capsys) -> None:
    assert main(["serve", "--nope"]) == 2
    captured = capsys.readouterr()
    assert captured.err

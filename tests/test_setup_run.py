"""The setup command runner. It does not download or pack a model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from favlo.setup_run import accepted_argv, frees_model, run_setup_command


def _names(root: Path | None = None) -> tuple[str, ...]:
    del root
    return ("qwen36-35b-a3b", "example-pipeline")


def test_a_setup_command_runs_in_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("favlo.setup_run.pipeline_names", _names)
    command = (
        "uv run favlo --example-pipeline setup "
        "--repo https://huggingface.co/example-org/example-model/tree/main"
    )
    argv = accepted_argv(command, tmp_path)
    assert argv[:5] == ["uv", "run", "favlo", "--example-pipeline", "setup"]
    assert frees_model(argv)
    seen: dict[str, object] = {}

    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        seen["argv"] = argv
        seen["cwd"] = kwargs["cwd"]
        seen["shell"] = kwargs["shell"]
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert env["FAVLO_ROOT"] == str(tmp_path.resolve())
        return SimpleNamespace(returncode=0, stdout="packed\n", stderr="")

    text = run_setup_command(command, root=tmp_path, run=run)
    assert seen["cwd"] == tmp_path.resolve()
    assert seen["shell"] is False
    assert text.startswith("exit 0")
    assert "packed" in text


def test_the_image_setup_commands_are_accepted() -> None:
    from favlo.cli import repo_root
    from favlo.guides import read_setup

    root = repo_root()
    guide = read_setup("z-image-turbo")
    commands = (
        "uv sync --extra dev",
        "uv run favlo --z-image-turbo setup",
        "uv run favlo --z-image-turbo quantize",
        "uv run favlo --z-image-turbo quantize --weights models/example-model.safetensors",
        "uv run favlo --z-image-turbo quantize --weights models/example-model.safetensors "
        "--name example-model",
    )
    for command in commands:
        assert command in guide
        argv = accepted_argv(command, root)
        if command != "uv sync --extra dev":
            assert frees_model(argv)
            assert argv[3:5] == ["--z-image-turbo", command.split()[4]]
    tokenizer = (
        "uv run favlo --z-image-turbo quantize --weights models/example-model.safetensors "
        "--tokenizer-source models/Z-Image-Turbo"
    )
    assert accepted_argv(tokenizer, root)[-2:] == [
        "--tokenizer-source",
        "models/Z-Image-Turbo",
    ]


def test_the_video_setup_commands_are_accepted() -> None:
    from favlo.cli import repo_root
    from favlo.guides import read_setup

    root = repo_root()
    guide = read_setup("wan22-i2v")
    commands = (
        "uv run favlo --wan22-i2v setup",
        "uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors "
        "--low models/example-model/low.safetensors",
        "uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors "
        "--low models/example-model/low.safetensors --name example-model",
        "uv run favlo --wan22-i2v quantize --high models/example-model/high.safetensors "
        "--low models/example-model/low.safetensors --name example-model "
        "--high-dir models/example-high --low-dir models/example-low",
        "uv run favlo --wan22-i2v quantize-text",
        "uv run favlo --wan22-i2v quantize-text --name example-model --out models/example-umt5",
        "uv run favlo --wan22-i2v setup --umt5 example-org/example-model",
        "uv run favlo --wan22-i2v quantize-text --name example-model "
        "--umt5 example-org/example-model",
    )
    for command in commands:
        assert command in guide
        argv = accepted_argv(command, root)
        assert frees_model(argv)
        assert argv[3] == "--wan22-i2v"


def test_a_path_and_a_shell_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.setup_run.pipeline_names", _names)
    with pytest.raises(ValueError, match="not a setup step"):
        accepted_argv("uv run favlo --qwen36-35b-a3b kernels", tmp_path)
    with pytest.raises(ValueError, match="not a setup step"):
        accepted_argv("uv run favlo --qwen36-35b-a3b setup && rm -rf .", tmp_path)
    with pytest.raises(ValueError, match="leaves this install"):
        accepted_argv(
            "uv run favlo --qwen36-35b-a3b quantize --src ../outside",
            tmp_path,
        )
    with pytest.raises(ValueError, match="unknown pipeline"):
        accepted_argv("uv run favlo --missing setup", tmp_path)
    delete = accepted_argv("favlo --qwen36-35b-a3b delete example-model:instruct", tmp_path)
    assert delete[-2:] == ["delete", "example-model:instruct"]
    assert accepted_argv("uv sync --extra dev", tmp_path) == ["uv", "sync", "--extra", "dev"]
    packed = accepted_argv(
        "uv run favlo --example-pipeline quantize "
        "--weights models/example-model.safetensors --name example-model",
        tmp_path,
    )
    assert packed[-4:] == [
        "--weights",
        "models/example-model.safetensors",
        "--name",
        "example-model",
    ]
    assert not frees_model(["uv", "sync", "--extra", "dev"])


def test_a_refused_command_does_not_need_the_server() -> None:
    from favlo.talk import _setup_run_tool

    tool = _setup_run_tool("http://127.0.0.1:9/v1")
    text = tool.func("rm -rf .")
    assert text.startswith("refused")


def test_close_posts_before_a_heavy_command() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from favlo.talk import _close_resident

    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, fmt: str, *args: object) -> None:
            del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        assert _close_resident(f"http://{host}:{port}/v1") is None
    finally:
        server.shutdown()
    assert seen == ["/v1/pipelines/close"]

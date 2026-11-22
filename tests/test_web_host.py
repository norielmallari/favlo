"""``favlo web`` refuses a port that is already taken."""

from __future__ import annotations

from favlo.web_host import _expo_command, _registered_model, main


def test_a_taken_api_port_names_that_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: "example-model")
    monkeypatch.setattr("favlo.web_host.port_open", lambda _host, port: port == 8130)
    assert main([]) == 1
    assert "8130" in capsys.readouterr().err


def test_web_does_not_start_without_a_registered_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: None)
    assert main([]) == 1
    assert capsys.readouterr().err == "favlo web did not start. No model is registered.\n"


def test_web_uses_the_registry_default(monkeypatch) -> None:
    monkeypatch.setattr("favlo.registry.default_llm_id", lambda root=None: "example-model")
    monkeypatch.setattr("favlo.registry.resolve", lambda model, root=None: object())
    assert _registered_model() == "example-model"


def test_windows_expo_command_uses_node(monkeypatch, tmp_path) -> None:
    spa = tmp_path / "web-spa"
    cli = spa / "node_modules" / "expo" / "bin" / "cli"
    cli.parent.mkdir(parents=True)
    cli.write_text("// cli\n", encoding="utf-8")
    monkeypatch.setattr("favlo.web_host.os.name", "nt")

    def which(name: str) -> str | None:
        if name == "node":
            return r"C:\node\node.exe"
        return None

    monkeypatch.setattr("favlo.web_host.shutil.which", which)
    argv = _expo_command(spa)
    assert argv[0] == r"C:\node\node.exe"
    assert argv[1] == str(cli)
    assert argv[2:] == ["start", "--web", "--port", "8012", "--host", "localhost"]


def test_posix_expo_command_uses_npx(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("favlo.web_host.os.name", "posix")
    argv = _expo_command(tmp_path)
    assert argv[:2] == ["npx", "expo"]
    assert "--web" in argv


def test_web_refuses_a_registry_default_that_does_not_resolve(monkeypatch) -> None:
    def missing(model: str, root=None):
        del model, root
        raise LookupError("missing")

    monkeypatch.setattr("favlo.registry.default_llm_id", lambda root=None: "example-model")
    monkeypatch.setattr("favlo.registry.resolve", missing)
    assert _registered_model() is None

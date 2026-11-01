"""``favlo web`` refuses a port that is already taken."""

from __future__ import annotations

from favlo.web_host import _registered_model, main


def test_a_taken_api_port_names_that_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: "example-model")
    monkeypatch.setattr("favlo.web_host.port_open", lambda _host, port: port == 8130)
    assert main([]) == 1
    assert "8130" in capsys.readouterr().err


def test_web_does_not_start_without_a_registered_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: None)
    assert main([]) == 1
    assert capsys.readouterr().err == "favlo web did not start. No model is registered.\n"


def test_web_prefers_the_first_pipeline_when_registered(monkeypatch) -> None:
    monkeypatch.setattr("favlo.registry.resolve", lambda model, root=None: object())
    monkeypatch.setattr(
        "favlo.registry.default_llm_id",
        lambda root=None: (_ for _ in ()).throw(AssertionError("default should not run")),
    )
    assert _registered_model() == "qwen35-4b:thinking"


def test_web_falls_back_to_the_registry_default(monkeypatch) -> None:
    def missing(model: str, root=None):
        del model, root
        raise LookupError("missing")

    monkeypatch.setattr("favlo.registry.resolve", missing)
    monkeypatch.setattr("favlo.registry.default_llm_id", lambda root=None: "example-model:instruct")
    assert _registered_model() == "example-model:instruct"

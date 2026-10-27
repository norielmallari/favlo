"""``favlo web`` refuses a port that is already taken."""

from __future__ import annotations

from favlo.web_host import main


def test_a_taken_api_port_names_that_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: "example-model")
    monkeypatch.setattr("favlo.web_host.port_open", lambda _host, port: port == 8130)
    assert main([]) == 1
    assert "8130" in capsys.readouterr().err


def test_web_does_not_start_without_a_registered_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.web_host._registered_model", lambda: None)
    assert main([]) == 1
    assert capsys.readouterr().err == "favlo web did not start. No model is registered.\n"

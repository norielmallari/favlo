"""A terminal ``favlo serve`` is the listener. No model load."""

from __future__ import annotations

import sys

import pytest

from favlo import listener


def test_setup_failure_does_not_open_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("favlo.setup.prepare", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr("favlo.daemon.occupy", lambda *_args, **_kwargs: called.append(1))
    assert listener.serve_main([]) == 1
    assert called == []

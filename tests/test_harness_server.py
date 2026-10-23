"""A harness reuses one listener and stops only a process it started."""

import sys

from harness.server import Listener, close_listener, main, open_listener


def test_status_reports_a_closed_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: False)
    assert main(["status"]) == 1
    assert capsys.readouterr().out.strip() == "down 127.0.0.1:8129"


def test_status_reports_an_open_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: True)
    assert main(["status"]) == 0
    assert capsys.readouterr().out.strip() == "listening 127.0.0.1:8129"


def test_open_reuses_a_listening_port(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: True)

    def _spawn(*_args, **_kwargs):
        raise AssertionError("spawned")

    monkeypatch.setattr("favlo.daemon.ensure", _spawn)
    listener = open_listener()
    assert listener.owned is False
    assert listener.pid is None
    assert "reuse 127.0.0.1:8129" in capsys.readouterr().out


def test_open_starts_when_the_port_is_closed(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("favlo.daemon.ensure", lambda *_args, **_kwargs: (77, True))
    listener = open_listener()
    assert listener == Listener(port=8129, owned=True, pid=77)
    assert "started 127.0.0.1:8129" in capsys.readouterr().out


def test_open_does_not_own_a_daemon_already_starting(monkeypatch, capsys) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("favlo.daemon.ensure", lambda *_args, **_kwargs: (55, False))
    listener = open_listener()
    assert listener.owned is False
    assert listener.pid == 55
    assert "reuse 127.0.0.1:8129" in capsys.readouterr().out


def test_close_leaves_a_reused_listener(monkeypatch) -> None:
    stopped: list[int] = []
    monkeypatch.setattr("favlo.daemon.stop_pid", lambda pid: stopped.append(pid))
    close_listener(Listener(port=8129, owned=False, pid=77))
    assert stopped == []


def test_close_stops_a_started_listener(monkeypatch, capsys) -> None:
    stopped: list[int] = []
    monkeypatch.setattr("favlo.daemon.stop_pid", lambda pid: stopped.append(pid))
    close_listener(Listener(port=8129, owned=True, pid=77))
    assert stopped == [77]
    assert capsys.readouterr().out.strip() == "stopped 127.0.0.1:8129"


def test_run_requires_a_command() -> None:
    assert main(["run"]) == 2


def test_run_stops_a_listener_it_started(monkeypatch, capfd) -> None:
    monkeypatch.setattr(
        "harness.server.open_listener",
        lambda: Listener(port=8129, owned=True, pid=77),
    )
    stopped: list[int] = []
    monkeypatch.setattr("favlo.daemon.stop_pid", lambda pid: stopped.append(pid))
    code = main(["run", "--", sys.executable, "-c", "print('ran')"])
    text = capfd.readouterr().out
    assert code == 0
    assert "ran" in text
    assert stopped == [77]
    assert "stopped 127.0.0.1:8129" in text

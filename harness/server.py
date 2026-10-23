"""One ``favlo serve`` for a harness.

Port ``127.0.0.1:8129`` is the listener. A process already bound there is
reused and left running. A closed port starts the daemon, and that process
is stopped when the session closes. A harness does not bind a second port.

``python -m harness.server status`` prints ``listening`` or ``down``.
``python -m harness.server run -- <command>`` opens the session, runs the
command, and closes it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

HOST = "127.0.0.1"
PORT = 8129


@dataclass(frozen=True)
class Listener:
    """The listener a harness is using."""

    port: int
    owned: bool
    pid: int | None = None
    host: str = HOST


def open_listener(host: str = HOST, port: int = PORT) -> Listener:
    """Reuse a listener on ``host:port``, or start one and own it."""
    from favlo.cli import repo_root
    from favlo.daemon import ensure, port_open

    if port_open(host, port):
        print(f"reuse {host}:{port}", flush=True)
        return Listener(host=host, port=port, owned=False)
    pid, started = ensure(host, port, root=repo_root())
    if not started:
        print(f"reuse {host}:{port}", flush=True)
        return Listener(host=host, port=port, owned=False, pid=pid)
    print(f"started {host}:{port}", flush=True)
    return Listener(host=host, port=port, owned=True, pid=pid)


def close_listener(listener: Listener) -> None:
    """Stop a listener this session started."""
    if not listener.owned or listener.pid is None:
        return
    from favlo.daemon import stop_pid

    stop_pid(listener.pid)
    print(f"stopped {listener.host}:{listener.port}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.server")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "status":
        return _status()
    return _run(list(args.command))


def _status() -> int:
    from favlo.daemon import port_open

    if port_open(HOST, PORT):
        print(f"listening {HOST}:{PORT}")
        return 0
    print(f"down {HOST}:{PORT}")
    return 1


def _run(command: list[str]) -> int:
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("a command is required")
        return 2
    listener = open_listener()
    try:
        completed = subprocess.run(command, check=False)
        return int(completed.returncode)
    finally:
        close_listener(listener)


if __name__ == "__main__":
    sys.exit(main())

"""Image client for the one ``favlo serve`` on ``127.0.0.1:8129``.

``--pipeline`` selects ``docs/<id>/smoke.json``. ``--tool`` is a local name
that pipeline already publishes. The command calls that tool on ``POST /mcp``
and reads the ``file://`` URI. An existing listener is reused. A closed port
starts the daemon, and that process is stopped when the command exits. Exit 0
means those bytes were read. The client does not score the picture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harness.image_smoke_agent import smoke_runs, smoke_note
from harness.mcp_media import generate, published_name, read_link
from harness.server import HOST, close_listener, open_listener


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.image")
    parser.add_argument("--pipeline", required=True, help="pipeline id")
    parser.add_argument("--tool", required=True, help="local tool name")
    parser.add_argument("--model", default=None, help="registry folder")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=8129)
    args = parser.parse_args()
    note = Path("docs") / args.pipeline / "smoke.json"
    if not note.is_file():
        print("smoke.json is missing")
        return 1
    body = smoke_runs(smoke_note(Path("."), args.pipeline), args.model)[0]
    listener = None
    try:
        listener = open_listener(args.host, args.port)
        result = generate(listener.port, published_name(args.pipeline, args.tool), body)
        png = read_link(listener.port, result, "image/png")
    finally:
        if listener is not None:
            close_listener(listener)
    if not png:
        print("response has no image")
        return 1
    print(f"image bytes {len(png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Image client for the one ``favlo serve`` on ``127.0.0.1:8129``.

The command calls ``generate_image`` on ``POST /mcp`` and reads the PNG with
``resources/read``. The prompt, size, and seed come from
``docs/z-image-turbo/smoke.json``. An existing listener is reused. A closed
port starts the daemon, and that process is stopped when the command exits.
Exit 0 means those bytes were read. The client does not score the picture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.mcp_media import generate, read_link
from harness.server import HOST, close_listener, open_listener


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.image")
    parser.add_argument(
        "--model",
        default=None,
        help="registry folder; omit for the first z-image-turbo folder",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=8129)
    args = parser.parse_args()
    note = Path("docs/z-image-turbo/smoke.json")
    if not note.is_file():
        print("smoke.json has no recorded latent")
        return 1
    stored = json.loads(note.read_text(encoding="utf-8"))
    body = {
        "prompt": stored["prompt"],
        "size": f"{stored['width']}x{stored['height']}",
        "seed": stored["seed"],
    }
    if args.model:
        body["model"] = args.model
    listener = None
    try:
        listener = open_listener(args.host, args.port)
        result = generate(listener.port, "generate_image", body)
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

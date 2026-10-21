"""Image client for a running ``favlo serve``.

The command posts a fixed prompt list to ``POST /v1/images/generations`` and
compares each latent only when ``docs/z-image-turbo/smoke.json`` has a hash
for that prompt. A prompt with no stored hash fails. The client does not
start a second server and does not score the picture.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.image")
    parser.add_argument("--model", default="Z-Image-Turbo-fp4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8129)
    args = parser.parse_args()
    note = Path("docs/z-image-turbo/smoke.json")
    if not note.is_file():
        print("smoke.json has no recorded latent")
        return 1
    stored = json.loads(note.read_text(encoding="utf-8"))
    body = {
        "model": args.model,
        "prompt": stored["prompt"],
        "size": f"{stored['width']}x{stored['height']}",
        "n": 1,
        "seed": stored["seed"],
    }
    request = urllib.request.Request(
        f"http://{args.host}:{args.port}/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read())
    if "b64_json" not in payload["data"][0]:
        print("response has no image")
        return 1
    print(f"image bytes {len(payload['data'][0]['b64_json'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

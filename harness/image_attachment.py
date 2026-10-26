"""Send the horse still as an OpenAI chat image and require a description.

The part is ``type`` ``image_url`` with a ``data:image/png;base64,`` URL and
``detail`` ``auto``. The question does not name the animal. Exit 0 means
``prompt_tokens`` covers the merged patch grid and the reply names the horse
and two marks that are only in the picture.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any

from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
MODEL = "qwen38-27b"
HORSE = Path("artifacts") / "favlo-horse.png"
QUESTION = "Describe the picture. Name the animal and what it is doing."
_PATCH = 16
_MERGE = 2
_MARKS = ("blue", "desk", "blaze", "bandana", "chestnut", "mane", "hoof")
_REFUSAL = (
    "can't see",
    "cannot see",
    "can't view",
    "cannot view",
    "don't see",
    "do not see",
    "no image",
    "unable to",
    "not able to see",
    "without an image",
    "problem with analysis",
)


def png_size(blob: bytes) -> tuple[int, int]:
    """Width and height from an IHDR."""
    if not blob.startswith(b"\x89PNG\r\n\x1a\n") or len(blob) < 24:
        message = "horse image is not a PNG"
        raise RuntimeError(message)
    width, height = struct.unpack(">II", blob[16:24])
    return int(width), int(height)


def merged_tokens(width: int, height: int) -> int:
    """Merged patch tokens for a side that is already a multiple of 32."""
    if width % (_PATCH * _MERGE) or height % (_PATCH * _MERGE) or width < 1 or height < 1:
        message = f"horse image {width}x{height} is not on the patch grid"
        raise RuntimeError(message)
    return (height // _PATCH // _MERGE) * (width // _PATCH // _MERGE)


def chat_body(png: bytes) -> dict[str, Any]:
    """One user turn: the question, then the still as an OpenAI image part."""
    encoded = base64.b64encode(png).decode("ascii")
    return {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "enable_thinking": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + encoded,
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    }


def sees_horse(text: str) -> bool:
    """True when the reply names the horse and two picture marks, and does not refuse."""
    folded = text.lower()
    if any(phrase in folded for phrase in _REFUSAL):
        return False
    if "horse" not in folded:
        return False
    return sum(mark in folded for mark in _MARKS) >= 2


def post_json(port: int, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise RuntimeError(message)
    return payload


def reply_text(payload: dict[str, Any]) -> str:
    message = payload["choices"][0]["message"]
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    return content if content.strip() else reasoning


def prompt_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    count = usage.get("prompt_tokens")
    return int(count) if isinstance(count, int) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that 27B describes the horse still.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    path = args.root.resolve() / HORSE
    if not path.is_file():
        print(f"missing {HORSE.as_posix()}", flush=True)
        return 1
    png = path.read_bytes()
    width, height = png_size(png)
    needed = merged_tokens(width, height)
    listener = None
    try:
        listener = open_listener()
        print(f"checking {MODEL} on {listener.port} image {width}x{height}", flush=True)
        payload = post_json(listener.port, chat_body(png))
        counted = prompt_tokens(payload)
        text = reply_text(payload)
        print(f"prompt tokens {counted}", flush=True)
        print(text, flush=True)
        if counted < needed:
            print(f"prompt tokens {counted} are below the merged grid {needed}", flush=True)
            return 1
        if not sees_horse(text):
            print("the reply does not describe the horse", flush=True)
            return 1
        print("horse described", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())

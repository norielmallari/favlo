"""Call ``POST /v1/embeddings`` on the listener and check the dense vector.

The command does not take the machine lock. It reuses ``127.0.0.1:8129`` or
starts that listener and stops the process it started. An in-flight chat is
left running. This call waits for it, then runs before a chat that has not
started.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from typing import Any

from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
MODEL = "bge-m3-w8a8"
TEXT = "A stone bridge under a clear sky."
HIDDEN = 1024


def check_payload(payload: dict[str, Any]) -> None:
    """The OpenAI embedding object for one string."""
    if payload.get("object") != "list":
        message = "embeddings response is not a list"
        raise RuntimeError(message)
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1:
        message = "embeddings response does not contain one vector"
        raise RuntimeError(message)
    row = data[0]
    if not isinstance(row, dict) or row.get("object") != "embedding":
        message = "embeddings row is not an embedding object"
        raise RuntimeError(message)
    vector = row.get("embedding")
    if not isinstance(vector, list) or len(vector) != HIDDEN:
        message = f"embedding length is not {HIDDEN}"
        raise RuntimeError(message)
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if abs(norm - 1.0) > 1e-3:
        message = f"embedding norm is {norm}"
        raise RuntimeError(message)
    usage = payload.get("usage")
    if not isinstance(usage, dict) or int(usage.get("prompt_tokens", 0)) < 2:
        message = "embeddings usage is missing the token count"
        raise RuntimeError(message)


def post_embedding(port: int, text: str = TEXT, model: str = MODEL) -> dict[str, Any]:
    body = json.dumps({"model": model, "input": text}).encode()
    request = urllib.request.Request(
        f"http://{HOST}:{port}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise RuntimeError(message)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.embeddings_agent")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args(argv)
    listener = open_listener()
    try:
        payload = post_embedding(listener.port, args.text, args.model)
        check_payload(payload)
    finally:
        close_listener(listener)
    usage = payload["usage"]
    print(f"tokens {usage['prompt_tokens']} dim {HIDDEN}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

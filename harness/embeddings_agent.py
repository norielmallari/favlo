"""Call ``POST /v1/embeddings`` and check that paraphrases outscore other topics.

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
MODEL = "bge-m3-int8"
HIDDEN = 1024
SENTENCES: tuple[tuple[str, str], ...] = (
    ("bridge", "A stone bridge under a clear sky."),
    ("bridge_near", "A stone bridge beneath a clear sky."),
    ("cat", "A cat sits on a woven mat."),
    ("cat_near", "A cat is sitting on a woven mat."),
    ("orbit", "The satellite orbits the planet once each day."),
    ("wheat", "Wheat prices rose after the dry season."),
)
NEAR: tuple[tuple[str, str], ...] = (
    ("bridge", "bridge_near"),
    ("cat", "cat_near"),
)
FAR: tuple[tuple[str, str], ...] = (
    ("bridge", "orbit"),
    ("bridge", "wheat"),
    ("bridge_near", "orbit"),
    ("cat", "orbit"),
    ("cat", "wheat"),
    ("cat_near", "wheat"),
)


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


def cosine(left: list[float], right: list[float]) -> float:
    """Dot product of two dense rows. The server returns unit vectors."""
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def check_scores(vectors: dict[str, list[float]]) -> list[tuple[str, str, str, float]]:
    """Every paraphrase cosine is higher than every cross-topic cosine."""
    rows: list[tuple[str, str, str, float]] = []
    for kind, pairs in (("near", NEAR), ("far", FAR)):
        for left, right in pairs:
            rows.append((kind, left, right, cosine(vectors[left], vectors[right])))
    near_min = min(score for kind, _left, _right, score in rows if kind == "near")
    far_max = max(score for kind, _left, _right, score in rows if kind == "far")
    if near_min <= far_max:
        message = f"near minimum {near_min} does not exceed far maximum {far_max}"
        raise RuntimeError(message)
    return rows


def post_embedding(port: int, text: str, model: str = MODEL) -> dict[str, Any]:
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
    args = parser.parse_args(argv)
    listener = open_listener()
    vectors: dict[str, list[float]] = {}
    try:
        for name, text in SENTENCES:
            payload = post_embedding(listener.port, text, args.model)
            check_payload(payload)
            row = payload["data"][0]["embedding"]
            if not isinstance(row, list):
                message = "embedding row is not a list"
                raise RuntimeError(message)
            vectors[name] = row
        scores = check_scores(vectors)
    finally:
        close_listener(listener)
    for kind, left, right, score in scores:
        print(f"{kind} {left} {right} {score:.6f}", flush=True)
    near_min = min(score for kind, _left, _right, score in scores if kind == "near")
    far_max = max(score for kind, _left, _right, score in scores if kind == "far")
    print(f"sentences {len(SENTENCES)} near {near_min:.6f} far {far_max:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

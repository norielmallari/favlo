"""Shader checks run on every pytest. Set ZIMAGE_SMOKE=1 to generate the recorded image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models" / "Z-Image-Turbo-fp4"
NOTE = ROOT / "docs" / "z-image-turbo" / "smoke.json"
PROMPT = "a red cube on a white table"
SEED = 0
WIDTH = 512
HEIGHT = 512


def test_kernels() -> None:
    from z_image_turbo.commands import kernels

    assert kernels() == 0


@pytest.mark.skipif(
    os.environ.get("ZIMAGE_SMOKE") != "1",
    reason="set ZIMAGE_SMOKE=1 to generate",
)
def test_generate_matches_the_recorded_latent() -> None:
    if not (RECORD / "manifest.json").is_file():
        pytest.skip("packed record is not present")
    from z_image_turbo.runtime import Runtime

    runtime = Runtime(RECORD)
    try:
        _png, digest, timing = runtime.generate(PROMPT, WIDTH, HEIGHT, 8, SEED)
        times = timing["steps"]
    finally:
        runtime.close()
    assert times
    if not NOTE.is_file():
        NOTE.write_text(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "seed": SEED,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "latent_sha256": digest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    stored = json.loads(NOTE.read_text(encoding="utf-8"))
    assert stored["latent_sha256"] == digest
    assert len(digest) == hashlib.sha256().digest_size * 2

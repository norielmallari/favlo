"""Shader checks run on every pytest. Set ZIMAGE_SMOKE=1 to generate the recorded images."""

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
SIZES = ((512, 512), (1024, 1024))


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
    digests: dict[tuple[int, int], str] = {}
    try:
        for width, height in SIZES:
            _png, digest, timing = runtime.generate(PROMPT, width, height, 8, SEED)
            assert timing["steps"]
            assert len(digest) == hashlib.sha256().digest_size * 2
            digests[(width, height)] = digest
    finally:
        runtime.close()
    picture = digests[(512, 512)]
    if not NOTE.is_file():
        NOTE.write_text(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "seed": SEED,
                    "width": 512,
                    "height": 512,
                    "latent_sha256": picture,
                    "sizes": [
                        {
                            "width": width,
                            "height": height,
                            "latent_sha256": digests[(width, height)],
                        }
                        for width, height in SIZES
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    stored = json.loads(NOTE.read_text(encoding="utf-8"))
    assert stored["width"] == 512
    assert stored["height"] == 512
    assert stored["latent_sha256"] == picture
    recorded = [(int(item["width"]), int(item["height"])) for item in stored["sizes"]]
    assert recorded == list(SIZES)
    for width, height in SIZES:
        match = next(item for item in stored["sizes"] if int(item["width"]) == width)
        assert int(match["height"]) == height
        assert match["latent_sha256"] == digests[(width, height)]

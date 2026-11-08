"""Agent that relights a transparent subject onto a generated background.

An existing ``artifacts/lighting-subject.png`` and
``artifacts/lighting-background.png`` are the inputs. A missing file is drawn
with one Microsoft Agent Framework turn that calls ``generate_image``.
``--fresh`` draws both again. The last turn calls ``relight_image``. Vision
confirms the same woman and the same street. The face color temperature must
move toward the plate. Exit 0 means the subject is RGBA, the relit file is
RGB, and both checks accept.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from agent_framework import Agent, tool
from pydantic import Field

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_smoke_agent import png_color_type
from harness.mcp_media import generate, read_uri
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
SIZE = "512x512"
SEED = 0
EVAL_TOKENS = 80
SUBJECT = (
    "One young woman standing full length and facing the camera, arms relaxed, "
    "wearing a plain red wool coat, dark trousers, and black shoes. Short black hair, "
    "calm closed mouth, even skin. Flat even studio light with no cast shadow and no rim light. "
    "The figure is centered, full body in frame, shoes included, on a plain white backdrop. "
    "correct human anatomy, natural hands, no extra limbs, no text, no watermark, no logo."
)
BACKGROUND = (
    "An empty cobblestone street between honey-colored stone walls at late afternoon. "
    "Warm sunlight comes from the upper left and lays long soft shadows toward the right. "
    "Dry clear weather, no people, no animals, no vehicles, eye-level, the street fills the frame. "
    "sharp focus, no text, no watermark, no logo."
)
RELIGHT = "a young woman, detailed face, warm late-afternoon sunlight from the left"
PLACE_QUESTION = (
    "The first picture is an empty street. The second places a woman on that street. "
    "Reply with one JSON object and nothing else: "
    '{"background_kept": true}. '
    "background_kept means the place is still that street."
)
PERSON_QUESTION = (
    "The first picture is a cutout of a woman. The second should show that same woman. "
    "Reply with one JSON object and nothing else: "
    '{"same_person": true}.'
)


def verdict_ok(text: str) -> bool:
    """True when vision keeps the street and the same woman."""
    found: dict[str, Any] = {}
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            body = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            found.update(body)
    return found.get("background_kept") is True and found.get("same_person") is True


def light_moved(subject: bytes, background: bytes, relit: bytes) -> bool:
    """True when the face color temperature moves toward the plate.

    The window is the upper body of the transparent subject. Red minus blue
    there is compared on the cutout, on the plate, and on the relit still.
    A flat paste stays at the cutout. A relight closes the gap by more than 15.
    """
    import cv2

    def rgb(raw: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cut = rgb(subject)
    plate = rgb(background)[:, :, :3]
    done = rgb(relit)[:, :, :3]
    if cut.shape[2] != 4 or cut.shape[:2] != plate.shape[:2] or cut.shape[:2] != done.shape[:2]:
        return False
    alpha = cut[:, :, 3] > 200
    rows = np.flatnonzero(alpha.any(axis=1))
    if rows.size == 0:
        return False
    top = int(rows[0])
    face = alpha.copy()
    face[: top + 40] = False
    face[top + 130 :] = False
    if int(face.sum()) < 32:
        return False

    def warmth(image: np.ndarray) -> float:
        pix = image[face].astype(np.float32)
        return float(pix[:, 0].mean() - pix[:, 2].mean())

    street = warmth(plate)
    before = warmth(cut[:, :, :3])
    after = warmth(done)
    print(
        f"face warmth street {street:.1f} cutout {before:.1f} relit {after:.1f}",
        flush=True,
    )
    return abs(after - street) + 15 < abs(before - street)


def post_json(port: int, route: str, body: dict[str, Any]) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        f"http://{HOST}:{port}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise RuntimeError(message)
    return payload


def _pngs(port: int, result: dict[str, Any]) -> list[bytes]:
    content = result.get("content")
    if not isinstance(content, list):
        message = "the task result has no content"
        raise RuntimeError(message)
    found: list[bytes] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "resource_link":
            continue
        uri = block.get("uri")
        if not isinstance(uri, str) or not uri.startswith("favlo://"):
            continue
        if block.get("mimeType") != "image/png":
            continue
        found.append(read_uri(port, uri))
    if not found:
        message = "the task result has no image/png link"
        raise RuntimeError(message)
    return found


def _stored_png(path: Path, colors: frozenset[int]) -> bytes | None:
    """The PNG at ``path`` when its color type is one of ``colors``."""
    if not path.is_file():
        return None
    blob = path.read_bytes()
    try:
        color = png_color_type(blob)
    except ValueError:
        return None
    if color not in colors:
        return None
    return blob


def _image_body(prompt: str, *, transparent: bool) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "size": SIZE,
        "seed": SEED,
        "transparent": transparent,
        "quality": "low",
    }


async def _once(
    base_url: str,
    model: str,
    name: str,
    instructions: str,
    user: str,
    function_name: str,
    tools: list[object],
) -> None:
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name=name,
        instructions=instructions,
        tools=tools,
        default_options={"temperature": 0},
    )
    await agent.run(
        user,
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": function_name},
        },
    )


def _draw_tool(
    port: int,
    body: dict[str, Any],
    slot: list[bytes],
    function_name: str,
    prompt: str,
) -> object:
    @tool(name=function_name, approval_mode="never_require")
    def draw(
        prompt_text: Annotated[str, Field(description="Exact sentence to draw.")],
    ) -> str:
        """Generate one image from the exact prompt."""
        if prompt_text.strip() != prompt:
            return f"use this prompt exactly: {prompt}"
        result = generate(port, "generate_image", body)
        slot.append(_pngs(port, result)[-1])
        return "image ready"

    return draw


def _relight_tool(
    port: int, subject: list[bytes], background: list[bytes], slot: list[bytes]
) -> object:
    @tool(name="relight_image", approval_mode="never_require")
    def relight_image(
        prompt_text: Annotated[str, Field(description="Exact lighting sentence.")],
    ) -> str:
        """Relight the saved subject onto the saved background."""
        if prompt_text.strip() != RELIGHT:
            return f"use this prompt exactly: {RELIGHT}"
        if not subject or not background:
            return "the subject and the background are not ready"
        result = generate(
            port,
            "relight_image",
            {
                "source": {
                    "type": "image",
                    "data": base64.b64encode(subject[-1]).decode("ascii"),
                },
                "target": {
                    "type": "image",
                    "data": base64.b64encode(background[-1]).decode("ascii"),
                },
                "prompt": RELIGHT,
                "seed": SEED,
            },
        )
        images = _pngs(port, result)
        if len(images) < 2:
            return "the relight returned one image"
        slot.append(images[0])
        slot.append(images[1])
        return "relight ready"

    return relight_image


def _ask(port: int, model: str, text: str, images: list[bytes]) -> str:
    def part(blob: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(blob).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encoded}",
            "detail": "auto",
        }

    payload = post_json(
        port,
        "/v1/responses",
        {
            "model": model,
            "temperature": 0,
            "max_output_tokens": EVAL_TOKENS,
            "reasoning": {"effort": "none"},
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        *[part(blob) for blob in images],
                    ],
                }
            ],
        },
    )
    return _reply_text(payload)


def _reply_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                parts.append(str(block.get("text") or ""))
    return "".join(parts)


def judge(port: int, subject: bytes, background: bytes, relit: bytes, model: str) -> str:
    place = _ask(port, model, PLACE_QUESTION, [background, relit])
    person = _ask(port, model, PERSON_QUESTION, [subject, relit])
    return f"{place}\n{person}"


async def _run(
    base: str, port: int, model: str, out: Path, *, reuse: bool
) -> tuple[bytes, bytes, bytes, bytes]:
    subject: list[bytes] = []
    background: list[bytes] = []
    relit: list[bytes] = []
    if reuse:
        stored = _stored_png(out / "lighting-subject.png", frozenset({6}))
        if stored is not None:
            subject.append(stored)
            print("reuse lighting-subject.png", flush=True)
        stored = _stored_png(out / "lighting-background.png", frozenset({2, 6}))
        if stored is not None:
            background.append(stored)
            print("reuse lighting-background.png", flush=True)
    if not subject:
        await _once(
            base,
            model,
            "lighting-subject",
            "Call generate_subject once with the exact prompt. Then stop.",
            f"Create a transparent image of this exact sentence: {SUBJECT}",
            "generate_subject",
            [
                _draw_tool(
                    port,
                    _image_body(SUBJECT, transparent=True),
                    subject,
                    "generate_subject",
                    SUBJECT,
                )
            ],
        )
    if not background:
        await _once(
            base,
            model,
            "lighting-background",
            "Call generate_background once with the exact prompt. Then stop.",
            f"Create a background of this exact sentence: {BACKGROUND}",
            "generate_background",
            [
                _draw_tool(
                    port,
                    _image_body(BACKGROUND, transparent=False),
                    background,
                    "generate_background",
                    BACKGROUND,
                )
            ],
        )
    await _once(
        base,
        model,
        "lighting-relight",
        "Call relight_image once with the exact prompt. Then stop.",
        f"Relight the subject onto the background with this exact sentence: {RELIGHT}",
        "relight_image",
        [_relight_tool(port, subject, background, relit)],
    )
    if not subject or not background or len(relit) < 2:
        message = "the agent did not finish the composite"
        raise RuntimeError(message)
    return subject[-1], background[-1], relit[0], relit[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Relight a subject onto a generated plate.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--judge", default=None, help="chat model; default is the registry default")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="draw the subject and the background again",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        print(f"lighting agent on {port}", flush=True)
        out = root / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        subject, background, foreground, relit = asyncio.run(
            _run(base, port, judge_id, out, reuse=not args.fresh)
        )
        paths = {
            "lighting-subject.png": subject,
            "lighting-background.png": background,
            "lighting-foreground.png": foreground,
            "lighting-output.png": relit,
        }
        for name, blob in paths.items():
            path = out / name
            path.write_bytes(blob)
            print(f"wrote {name} bytes {path.stat().st_size}", flush=True)
        if png_color_type(subject) != 6:
            print("subject is not RGBA", flush=True)
            return 1
        if png_color_type(relit) != 2:
            print("relit image is not RGB", flush=True)
            return 1
        print("checking the composite", flush=True)
        if not light_moved(subject, background, relit):
            print("the face light did not move toward the street", flush=True)
            return 1
        reply = judge(port, subject, background, relit, judge_id)
        print(reply, flush=True)
        if not verdict_ok(reply):
            print("the composite was not accepted", flush=True)
            return 1
        print("lighting matches", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())

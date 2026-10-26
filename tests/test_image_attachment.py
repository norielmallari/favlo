"""OpenAI image part for the horse still. No listener."""

from __future__ import annotations

import pytest

from harness.image_attachment import QUESTION, chat_body, merged_tokens, sees_horse
from qwen38_27b.commands import installed_record
from qwen38_27b.vision import IMAGE_PAD


def test_the_request_is_an_openai_image_part() -> None:
    body = chat_body(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    assert body["model"] == "qwen38-27b"
    assert body["enable_thinking"] is False
    part = body["messages"][0]["content"]
    assert part[0] == {"type": "text", "text": QUESTION}
    assert "horse" not in QUESTION.lower()
    image = part[1]
    assert image["type"] == "image_url"
    assert set(image["image_url"]) == {"url", "detail"}
    assert image["image_url"]["detail"] == "auto"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_square_still_counts_merged_tokens() -> None:
    assert merged_tokens(1280, 1280) == 1600


def test_the_reply_must_name_the_horse_and_two_marks() -> None:
    assert sees_horse("A chestnut horse with blue eyes sits at a wooden desk.")
    assert not sees_horse("A chestnut horse stands in a field.")
    assert not sees_horse("I have a problem with analysis and cannot see an image of a horse at a desk.")


def test_an_openai_image_part_is_one_image_pad() -> None:
    root = installed_record()
    if root is None or not (root / "tokenizer.json").is_file():
        pytest.skip("quantized record is not installed")
    from qwen38_27b.chat import encode_chat

    messages = chat_body(b"png")["messages"]
    ids = encode_chat(root, messages, None, False)
    assert ids.count(IMAGE_PAD) == 1

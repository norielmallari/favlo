"""Harness MCP client prefixes local tool names."""

from __future__ import annotations

from harness.mcp_media import published_call


def test_a_local_name_is_published_with_the_pipeline_prefix() -> None:
    name, body = published_call("generate_image", {"prompt": "cube"})
    assert name == "z_image_turbo__generate_image"
    assert body == {"prompt": "cube"}


def test_a_folder_model_selects_that_pipeline() -> None:
    name, body = published_call(
        "generate_image",
        {"prompt": "cube", "model": "Z-Image-Turbo-int8"},
    )
    assert name == "z_image_turbo__generate_image"
    assert body["model"] == "Z-Image-Turbo-int8"


def test_a_published_name_is_unchanged() -> None:
    name, body = published_call("z_image_turbo__generate_image", {"prompt": "cube"})
    assert name == "z_image_turbo__generate_image"
    assert body == {"prompt": "cube"}

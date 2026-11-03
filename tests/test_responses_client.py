"""Shared Responses image-load helpers."""

from __future__ import annotations

from pathlib import Path

from favlo.responses_client import image_data_url, normalize_v1_base, output_text


def test_normalize_v1_base() -> None:
    assert normalize_v1_base("http://127.0.0.1:8129") == "http://127.0.0.1:8129/v1"
    assert normalize_v1_base("http://127.0.0.1:8129/") == "http://127.0.0.1:8129/v1"
    assert normalize_v1_base("http://127.0.0.1:8129/v1") == "http://127.0.0.1:8129/v1"
    assert normalize_v1_base("http://127.0.0.1:8129/v1/") == "http://127.0.0.1:8129/v1"


def test_image_data_url_from_path(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n")
    url = image_data_url(path)
    assert url is not None
    assert url.startswith("data:image/png;base64,")


def test_image_data_url_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hi")
    assert image_data_url(path) is None


def test_output_text_from_message_parts() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
    }
    assert output_text(payload) == "hello"

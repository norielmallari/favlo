"""A favlo:// link streams the stored file."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from favlo.images import store_image  # noqa: E402
from web_api.api import create_app  # noqa: E402
from web_api.routes.media import media_response  # noqa: E402


async def _reply(**_kwargs: object):
    yield "Hello"


def test_an_image_link_streams_the_png(tmp_path: Path, monkeypatch) -> None:
    store_image("image_abc", b"\x89PNG\r\n\x1a\n", home=tmp_path)
    monkeypatch.setattr("web_api.routes.media.favlo_home", lambda: tmp_path)
    app = create_app(tmp_path, reply=_reply)
    with TestClient(app) as client:
        found = client.get("/media/images/image_abc.png")
        assert found.status_code == 200
        assert found.headers["content-type"].startswith("image/png")
        assert found.content.startswith(b"\x89PNG")
        missing = client.get("/media/images/image_missing.png")
        assert missing.status_code == 404
        assert media_response("images", "../secret.png", tmp_path) is None
        assert media_response("images", "image_abc.mp4", tmp_path) is None

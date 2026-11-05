"""Gallery index wake helper and path-only sidecars when vision is unavailable."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.gallery_embed import request_wake, run_once  # noqa: E402
from web_api.gallery_metadata import METADATA_DIR, read_sidecar  # noqa: E402


def test_request_wake_never_raises_without_lock() -> None:
    class State:
        gallery_index_running = True

    class App:
        state = State()

    assert request_wake(App(), "workspace_x") is False


def test_request_wake_schedules_pass_for_workspace(tmp_path: Path, monkeypatch) -> None:
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    channel = store.channels()[0]
    root = tmp_path / "photos"
    root.mkdir()
    status, workspace, _ = store.create_workspace(
        "key",
        {
            "channel_id": channel["id"],
            "name": "Lib",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert status in {200, 201}
    workspace_id = workspace["id"]

    class State:
        def __init__(self) -> None:
            self.store = store
            self.gallery_index_running: dict[str, bool] = {}
            self.gallery_index_lock = threading.Lock()
            self.gallery_index = None
            self.tasks: set = set()

    class App:
        state = State()

    store.gallery_index_app = App()
    started: list[str] = []

    def fake_pass(_store, _index, wid: str, _vision: bool = True) -> int:
        started.append(wid)
        return 0

    monkeypatch.setattr("web_api.gallery_embed._pass", fake_pass)

    async def main() -> None:
        assert request_wake(App(), workspace_id) is True
        assert request_wake(App(), workspace_id) is False
        await asyncio.sleep(0.05)
        assert started == [workspace_id]
        assert request_wake(App(), workspace_id) is True
        await asyncio.sleep(0.05)
        assert started == [workspace_id, workspace_id]

    asyncio.run(main())


def test_request_wake_allows_other_workspace(tmp_path: Path, monkeypatch) -> None:
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    channel = store.channels()[0]
    ids: list[str] = []
    for name in ("A", "B"):
        root = tmp_path / name
        root.mkdir()
        status, workspace, _ = store.create_workspace(
            f"key-{name}",
            {
                "channel_id": channel["id"],
                "name": name,
                "type": "gallery",
                "location": str(root),
            },
        )
        assert status in {200, 201}
        ids.append(workspace["id"])

    gate = threading.Event()
    entered = threading.Event()

    class State:
        def __init__(self) -> None:
            self.store = store
            self.gallery_index_running: dict[str, bool] = {}
            self.gallery_index_lock = threading.Lock()
            self.gallery_index = None
            self.tasks: set = set()

    class App:
        state = State()

    store.gallery_index_app = App()

    def fake_pass(_store, _index, wid: str, _vision: bool = True) -> int:
        if wid == ids[0]:
            entered.set()
            gate.wait(timeout=2.0)
        return 0

    monkeypatch.setattr("web_api.gallery_embed._pass", fake_pass)

    async def main() -> None:
        assert request_wake(App(), ids[0]) is True
        for _ in range(50):
            if entered.is_set():
                break
            await asyncio.sleep(0.02)
        assert entered.is_set()
        assert request_wake(App(), ids[0]) is False
        assert request_wake(App(), ids[1]) is True
        gate.set()
        await asyncio.sleep(0.05)

    asyncio.run(main())


def test_run_once_writes_path_sidecar_without_chat_model(tmp_path: Path, monkeypatch) -> None:
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    monkeypatch.setattr(store, "gallery_vision_model", lambda workspace_id=None: (None, None))
    monkeypatch.setattr("web_api.gallery_embed._embed", lambda _text: [0.0] * 1024)

    channel = store.channels()[0]
    root = tmp_path / "photos"
    root.mkdir()
    (root / "still.png").write_bytes(b"png")
    (root / "track.mp3").write_bytes(b"id3")
    status, workspace, _ = store.create_workspace(
        "key",
        {
            "channel_id": channel["id"],
            "name": "Lib",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert status in {200, 201}

    worked = run_once(store, workspace_id=workspace["id"])
    assert worked >= 2
    assert read_sidecar(root, "still.png") is not None
    assert read_sidecar(root, "track.mp3") is not None
    assert (root / METADATA_DIR / "still.png.json").is_file()
    assert not (root / "still.png.json").exists()


def test_gallery_vision_uses_the_agent_model_or_the_global_default(tmp_path: Path) -> None:
    from web_api.db import LOCAL_PROVIDER_ID
    from web_api.ids import FAVLO_AGENT_ID
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    default = store.add_model(
        "vision-default",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "qwen36-35b-a3b:thinking",
            "display_name": "35B",
        },
    )
    assert default[0] == 201
    other = store.add_model(
        "vision-other",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "qwen35-4b:thinking",
            "display_name": "4B",
        },
    )
    assert other[0] == 201
    settings = store.settings()
    store.update_settings(
        settings["revision"],
        default[1]["id"],
        10,
        other[1]["id"],
    )
    channel = next(row for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID)
    root = tmp_path / "photos"
    root.mkdir()
    _status, workspace, _events = store.create_workspace(
        "vision-model",
        {
            "channel_id": channel["id"],
            "name": "Lib",
            "type": "gallery",
            "location": str(root),
        },
    )
    model, base = store.gallery_vision_model(workspace["id"])
    assert model == "qwen36-35b-a3b:thinking"
    assert base
    favlo = next(row for row in store.agents() if row["id"] == FAVLO_AGENT_ID)
    store.update_agent(
        favlo["id"],
        favlo["revision"],
        favlo["name"],
        str(favlo.get("persona") or ""),
        other[1]["id"],
    )
    model, _base = store.gallery_vision_model(workspace["id"])
    assert model == "qwen35-4b:thinking"


def test_provider_model_throws_when_settings_has_no_model(tmp_path: Path) -> None:
    import pytest
    from web_api.ids import FAVLO_AGENT_ID
    from web_api.store import Rejected, Store

    store = Store(tmp_path / "web" / "app.db")
    with pytest.raises(Rejected, match="model"):
        store.chat_model(FAVLO_AGENT_ID)


def test_run_once_skips_embed_when_prompt_hash_matches(tmp_path: Path, monkeypatch) -> None:
    from web_api.gallery_index import GalleryIndex
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    monkeypatch.setattr(store, "gallery_vision_model", lambda workspace_id=None: (None, None))
    embeds: list[str] = []

    def fake_embed(text: str) -> list[float]:
        embeds.append(text)
        return [0.0] * 1024

    monkeypatch.setattr("web_api.gallery_embed._embed", fake_embed)

    channel = store.channels()[0]
    root = tmp_path / "photos"
    root.mkdir()
    (root / "still.png").write_bytes(b"png")
    status, workspace, _ = store.create_workspace(
        "key",
        {
            "channel_id": channel["id"],
            "name": "Lib",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert status in {200, 201}
    index = GalleryIndex(home / "gallery_index.db")

    assert run_once(store, index, workspace["id"]) == 1
    assert len(embeds) == 1
    assert run_once(store, index, workspace["id"]) == 0
    assert len(embeds) == 1
    row = index.row(workspace["id"], "still.png")
    assert row is not None
    assert row["prompt_hash"]


def test_vision_waits_120_seconds_for_a_new_generated_image(tmp_path: Path, monkeypatch) -> None:
    import json
    import os
    import time

    from web_api.gallery_embed import write_vision_sidecars
    from web_api.gallery_metadata import write_sidecar
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    monkeypatch.setattr(
        store,
        "gallery_vision_model",
        lambda workspace_id=None: ("example-model", "http://127.0.0.1:9"),
    )
    monkeypatch.setattr(store, "gallery_vision_effort", lambda workspace_id=None: None)
    monkeypatch.setattr("web_api.gallery_embed._embed", lambda _text: [0.0] * 1024)
    calls: list[str] = []

    def fake_analyze(path, *, relative, model, base_url, existing=None, reasoning_effort=None):
        calls.append(relative)
        return {
            "filename": "shot.png",
            "kind": "image",
            "title": "Square",
            "description": "A red square.",
            "tags": ["square"],
            "objects": ["square"],
            "style": "flat",
            "source": "index",
        }

    monkeypatch.setattr("web_api.gallery_embed.analyze_image", fake_analyze)

    channel = store.channels()[0]
    root = tmp_path / "photos"
    root.mkdir()
    image = root / "shot.png"
    image.write_bytes(b"png")
    write_sidecar(
        root,
        "shot.png",
        {
            "version": 1,
            "filename": "shot.png",
            "kind": "image",
            "description": "a red square",
            "source": "generate",
            "generation": {"prompt": "a red square", "model": "example-model"},
            "style": "",
            "objects": [],
        },
    )
    status, workspace, _events = store.create_workspace(
        "fresh-gen",
        {
            "channel_id": channel["id"],
            "name": "Fresh",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert status in {200, 201}

    assert run_once(store, workspace_id=workspace["id"]) >= 1
    assert calls == []
    side = json.loads((root / METADATA_DIR / "shot.png.json").read_text())
    assert side["source"] == "generate"

    stamped = time.time() - 121
    os.utime(image, (stamped, stamped))
    run_once(store, workspace_id=workspace["id"])
    assert calls == ["shot.png"]

    calls.clear()
    other = root / "new.png"
    other.write_bytes(b"png")
    write_sidecar(
        root,
        "new.png",
        {
            "version": 1,
            "filename": "new.png",
            "kind": "image",
            "description": "a blue circle",
            "source": "generate",
            "generation": {"prompt": "a blue circle", "model": "example-model"},
            "style": "",
            "objects": [],
        },
    )
    write_vision_sidecars(store, workspace["id"])
    assert "new.png" in calls

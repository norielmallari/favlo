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
    monkeypatch.setattr(store, "gallery_vision_model", lambda: (None, None))
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


def test_gallery_vision_model_prefers_settings_over_default(tmp_path: Path) -> None:
    from web_api.db import LOCAL_PROVIDER_ID
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    default = store.add_model(
        "vision-default",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "qwen36-35b-a3b",
            "display_name": "35B",
        },
    )
    assert default[0] == 201
    other = store.add_model(
        "vision-other",
        {
            "provider_id": LOCAL_PROVIDER_ID,
            "model_id": "qwen35-4b",
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
    model, base = store.gallery_vision_model()
    assert model == "qwen35-4b"
    assert base
    cleared = store.settings()
    store.update_settings(cleared["revision"], default[1]["id"], 10, None)
    model, _ = store.gallery_vision_model()
    assert model == "qwen36-35b-a3b"


def test_run_once_skips_embed_when_prompt_hash_matches(tmp_path: Path, monkeypatch) -> None:
    from web_api.gallery_index import GalleryIndex
    from web_api.store import Store

    home = tmp_path / "web"
    store = Store(home / "app.db")
    monkeypatch.setattr(store, "gallery_vision_model", lambda: (None, None))
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

"""Workspace placement validation and the place tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.api import create_app  # noqa: E402
from web_api.placement import initial_for, validate  # noqa: E402
from web_api.store import Store  # noqa: E402

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


async def _reply(**_kwargs: object):
    yield "Hello"


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path, reply=_reply)
    with TestClient(app) as test_client:
        yield test_client



def _loc(tmp_path: Path, name: str = "gallery-root") -> str:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    return str(root)


def test_initial_gallery_has_preview_and_empty_filmstrip() -> None:
    state = initial_for("gallery")
    elements = state["favlo_workspace"]["elements"]
    assert [item["element"] for item in elements] == ["preview", "filmstrip"]
    strip = next(item for item in elements if item["element"] == "filmstrip")
    assert strip["props"]["items"] == []
    assert state["favlo_workspace"]["selection"] is None


def test_validate_keeps_prior_when_elements_all_dropped() -> None:
    prior = initial_for("gallery")
    state, ignored = validate(
        {
            "favlo_workspace": {
                "version": 1,
                "guide": "gallery",
                "elements": [{"key": "bad", "element": "terminal_log", "slot": "stage", "props": {}}],
                "selection": None,
            }
        },
        guide_id="gallery",
        prior=prior,
    )
    assert [item["element"] for item in state["favlo_workspace"]["elements"]] == ["preview", "filmstrip"]
    assert any("empty elements" in item or "unknown element" in item for item in ignored)


def test_validate_selects_first_filmstrip_path_when_selection_missing() -> None:
    state, _ignored = validate(
        {
            "favlo_workspace": {
                "version": 1,
                "guide": "gallery",
                "elements": [
                    {
                        "key": "preview",
                        "element": "preview",
                        "slot": "stage",
                        "props": {"fit": "contain"},
                    },
                    {
                        "key": "filmstrip",
                        "element": "filmstrip",
                        "slot": "rail",
                        "props": {
                            "items": [{"path": "a.png"}, {"path": "b.png"}],
                        },
                    },
                ],
                "selection": None,
            }
        },
        guide_id="gallery",
    )
    assert state["favlo_workspace"]["selection"] == {"kind": "file", "path": "a.png"}


def test_validate_moves_selection_onto_strip_when_path_is_absent() -> None:
    state, _ignored = validate(
        {
            "favlo_workspace": {
                "version": 1,
                "guide": "gallery",
                "elements": [
                    {
                        "key": "preview",
                        "element": "preview",
                        "slot": "stage",
                        "props": {"fit": "contain"},
                    },
                    {
                        "key": "filmstrip",
                        "element": "filmstrip",
                        "slot": "rail",
                        "props": {"items": [{"path": "a.png"}]},
                    },
                ],
                "selection": {"kind": "file", "path": "gone.png"},
            }
        },
        guide_id="gallery",
    )
    assert state["favlo_workspace"]["selection"] == {"kind": "file", "path": "a.png"}


def test_validate_clears_selection_when_filmstrip_is_empty() -> None:
    state, _ignored = validate(
        {
            "favlo_workspace": {
                "version": 1,
                "guide": "gallery",
                "elements": [
                    {
                        "key": "preview",
                        "element": "preview",
                        "slot": "stage",
                        "props": {"fit": "contain"},
                    },
                    {
                        "key": "filmstrip",
                        "element": "filmstrip",
                        "slot": "rail",
                        "props": {"items": []},
                    },
                ],
                "selection": {"kind": "file", "path": "a.png"},
            }
        },
        guide_id="gallery",
    )
    assert state["favlo_workspace"]["selection"] is None


def test_validate_restores_preview_when_filmstrip_has_items() -> None:
    payload = {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "frame.png"}]},
                },
            ],
            "selection": None,
        }
    }
    prior = {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": []},
                },
            ],
            "selection": None,
        }
    }
    state, ignored = validate(payload, guide_id="gallery", prior=prior)
    names = [item["element"] for item in state["favlo_workspace"]["elements"]]
    assert names == ["preview", "filmstrip"]
    assert state["favlo_workspace"]["selection"] == {"kind": "file", "path": "frame.png"}
    assert any("restored preview" in item for item in ignored)


def test_validate_allows_removing_preview_after_fill() -> None:
    prior = {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "frame.png"}]},
                },
            ],
            "selection": {"kind": "file", "path": "frame.png"},
        }
    }
    payload = {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "frame.png"}]},
                },
            ],
            "selection": None,
        }
    }
    state, ignored = validate(payload, guide_id="gallery", prior=prior)
    names = [item["element"] for item in state["favlo_workspace"]["elements"]]
    assert names == ["filmstrip"]
    assert not any("restored preview" in item for item in ignored)


def test_validate_drops_unknown_elements_and_wrong_slots() -> None:
    payload = {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {"key": "bad", "element": "terminal_log", "slot": "stage", "props": {}},
                {"key": "mis-slot", "element": "filmstrip", "slot": "stage", "props": {"items": []}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "a.png"}, {"path": "a.png"}]},
                },
            ],
            "selection": {"kind": "file", "path": "a.png"},
        }
    }
    state, ignored = validate(payload, guide_id="gallery")
    names = [item["element"] for item in state["favlo_workspace"]["elements"]]
    assert names == ["preview", "filmstrip"]
    strip = next(item for item in state["favlo_workspace"]["elements"] if item["element"] == "filmstrip")
    assert strip["props"]["items"] == [{"path": "a.png"}]
    assert any("unknown element" in item for item in ignored)
    assert any("wrong slot" in item for item in ignored)


def test_workspace_place_persists_existing_media(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    root = tmp_path / "library"
    root.mkdir()
    (root / "horse.png").write_bytes(PNG)
    nested = root / "Albums"
    nested.mkdir()
    (nested / "zebra.png").write_bytes(PNG)
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-place"},
        json={
            "channel_id": channel_id,
            "name": "Place",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert created.status_code == 201
    workspace = created.json()

    from web_api.workspace_tools import tools as place_tools

    store: Store = client.app.state.store
    [place] = place_tools(store, workspace["id"], guide_id="gallery")
    result = json.loads(
        place.func(
            elements=[
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {
                        "items": [
                            {"path": "horse.png"},
                            {"path": "Albums/zebra.png"},
                            {"path": "missing.png"},
                        ]
                    },
                },
            ],
            selection={"kind": "file", "path": "Albums/zebra.png"},
        )
    )
    items = result["state"]["favlo_workspace"]["elements"][1]["props"]["items"]
    assert [item["path"] for item in items] == ["horse.png", "Albums/zebra.png"]
    assert result["state"]["favlo_workspace"]["selection"]["path"] == "Albums/zebra.png"
    assert any("missing path" in item for item in result["ignored"])
    snap = client.get(f"/sessions/{workspace['session_id']}").json()
    assert snap["state"]["favlo_workspace"]["selection"]["path"] == "Albums/zebra.png"

    # Omitted selection keeps the prior path when it is still on the strip.
    without = json.loads(
        place.func(
            elements=[
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "horse.png"}, {"path": "Albums/zebra.png"}]},
                },
            ],
            selection=None,
        )
    )
    assert without["state"]["favlo_workspace"]["selection"]["path"] == "Albums/zebra.png"

    # A strip that drops the prior path falls back to the first remaining item.
    moved = json.loads(
        place.func(
            elements=[
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "horse.png"}]},
                },
            ],
            selection=None,
        )
    )
    assert moved["state"]["favlo_workspace"]["selection"]["path"] == "horse.png"

    # Basename-only paths resolve when unique under the folder.
    by_name = json.loads(
        place.func(
            elements=[
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"items": [{"path": "zebra.png"}, {"path": "horse.png"}]},
                },
            ],
            selection={"kind": "file", "path": "zebra.png"},
        )
    )
    items = by_name["state"]["favlo_workspace"]["elements"][1]["props"]["items"]
    assert [item["path"] for item in items] == ["Albums/zebra.png", "horse.png"]
    assert by_name["state"]["favlo_workspace"]["selection"]["path"] == "Albums/zebra.png"


def test_workspace_file_serves_relative_media(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    root = tmp_path / "files"
    root.mkdir()
    (root / "horse.png").write_bytes(PNG)
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-file"},
        json={
            "channel_id": channel_id,
            "name": "File",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert created.status_code == 201
    workspace_id = created.json()["id"]
    response = client.get(f"/workspaces/{workspace_id}/file", params={"path": "horse.png"})
    assert response.status_code == 200
    assert response.content == PNG
    assert "image/png" in response.headers["content-type"]

"""Gallery workspace: one local root, tools only on that session."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.api import create_app  # noqa: E402
from web_api.folder_port import LocalFolder  # noqa: E402
from web_api.store import Rejected, Store  # noqa: E402

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


def _gallery(client: TestClient, tmp_path: Path) -> dict:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-1"},
        json={"channel_id": channel_id, "name": "Stills", "type": "gallery"},
    )
    assert created.status_code == 201
    root = tmp_path / "gallery"
    root.mkdir()
    saved = client.put(
        f"/workspaces/{created.json()['id']}/root",
        headers={"Idempotency-Key": "root-1"},
        json={"name": "Stills", "kind": "local", "location": str(root)},
    )
    assert saved.status_code == 200
    return created.json() | {"root_path": root, "channel_id": channel_id, "root": saved.json()}


def test_profile_lists_disabled_cloud_kinds(client: TestClient) -> None:
    body = client.get("/profile").json()
    assert body["profile"] == "local"
    assert body["folder_kinds"] == [
        {"kind": "local", "enabled": True},
        {"kind": "cloud", "provider": "onedrive", "enabled": False},
        {"kind": "cloud", "provider": "dropbox", "enabled": False},
    ]


def test_a_gallery_needs_a_local_root_before_images(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-empty"},
        json={"channel_id": channel_id, "name": "Empty", "type": "gallery"},
    )
    workspace_id = created.json()["id"]
    listed = client.get(f"/workspaces?channel_id={channel_id}").json()["workspaces"]
    assert listed[0]["root"] is None
    missing = client.get(f"/workspaces/{workspace_id}/images")
    assert missing.status_code == 400
    assert "root" in missing.json()["message"].lower()
    cloud = client.put(
        f"/workspaces/{workspace_id}/root",
        headers={"Idempotency-Key": "cloud"},
        json={"kind": "cloud", "provider": "onedrive"},
    )
    assert cloud.status_code == 400
    assert "onedrive" in cloud.json()["message"].lower()
    absent = client.put(
        f"/workspaces/{workspace_id}/root",
        headers={"Idempotency-Key": "missing-dir"},
        json={"kind": "local", "location": str(tmp_path / "nope")},
    )
    assert absent.status_code == 400


def test_pick_root_returns_the_chosen_directory(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-pick"},
        json={"channel_id": channel_id, "name": "Stills", "type": "gallery"},
    )
    workspace_id = created.json()["id"]
    chosen = tmp_path / "picked"
    chosen.mkdir()
    monkeypatch.setattr("web_api.routes.workspaces.choose_directory", lambda: str(chosen))
    picked = client.post(f"/workspaces/{workspace_id}/root/pick")
    assert picked.status_code == 200
    assert picked.json() == {"cancelled": False, "location": str(chosen)}
    monkeypatch.setattr("web_api.routes.workspaces.choose_directory", lambda: None)
    cancelled = client.post(f"/workspaces/{workspace_id}/root/pick")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": True}


def test_images_and_folders_stay_inside_the_root(client: TestClient, tmp_path: Path) -> None:
    gallery = _gallery(client, tmp_path)
    workspace_id = gallery["id"]
    encoded = base64.b64encode(PNG).decode()
    written = client.post(
        f"/workspaces/{workspace_id}/images",
        json={"path": "horse.png", "bytes": encoded, "note": {"prompt": "a horse"}},
    )
    assert written.status_code == 200
    assert written.json()["path"] == "horse.png"
    assert written.json()["note"] == {"prompt": "a horse"}
    listing = client.get(f"/workspaces/{workspace_id}/images").json()
    assert [item["name"] for item in listing["images"]] == ["horse.png"]
    png = client.get(f"/workspaces/{workspace_id}/file?path=horse.png")
    assert png.status_code == 200
    assert png.content == PNG
    nested = client.post(
        f"/workspaces/{workspace_id}/folders",
        json={"path": "shots"},
    )
    assert nested.status_code == 201
    moved = client.patch(
        f"/workspaces/{workspace_id}/folders",
        json={"path": "shots", "to": "takes"},
    )
    assert moved.status_code == 200
    renamed = client.patch(
        f"/workspaces/{workspace_id}/folders",
        json={"path": "takes", "name": "stills"},
    )
    assert renamed.json()["path"] == "stills"
    escape = client.post(
        f"/workspaces/{workspace_id}/images",
        json={"path": "../outside.png", "bytes": encoded},
    )
    assert escape.status_code == 400
    root = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders",
        json={"path": "/"},
    )
    assert root.status_code == 400
    removed = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders",
        json={"path": "stills"},
    )
    assert removed.status_code == 200
    assert (gallery["root_path"] / "horse.png").is_file()


def test_gallery_tools_are_only_on_the_workspace_session(tmp_path: Path) -> None:
    from web_api.chat import turn_setup
    from web_api.db import FAVLO_SESSION_ID

    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-tools",
            {"channel_id": channel_id, "name": "Tools", "type": "gallery"},
        )
        root = tmp_path / "root"
        root.mkdir()
        store.set_root(
            "root-tools",
            workspace["id"],
            {"kind": "local", "location": str(root), "name": "Tools"},
        )
        home_instructions, home_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            ["session"],
            "http://127.0.0.1:8129/v1",
            store,
            "agent_favlo",
            FAVLO_SESSION_ID,
            None,
        )
        gallery_instructions, gallery_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            ["session"],
            "http://127.0.0.1:8129/v1",
            store,
            "agent_favlo",
            workspace["session_id"],
            None,
        )
    finally:
        store.close()
    home_names = {_tool_name(item) for item in home_tools}
    gallery_names = {_tool_name(item) for item in gallery_tools}
    assert "gallery_create_folder" not in home_names
    assert "gallery_write_image" not in home_names
    assert "gallery_create_folder" in gallery_names
    assert "gallery_write_image" in gallery_names
    assert "gallery_rename_folder" in gallery_names
    assert "gallery_move_folder" in gallery_names
    assert "gallery_delete_folder" in gallery_names
    assert "operating in a gallery workspace" in gallery_instructions.lower()
    assert "gallery root" in gallery_instructions.lower()
    assert "operating in a gallery workspace" not in home_instructions.lower()
    assert "gallery root" not in home_instructions.lower()


def test_a_workspace_session_gets_the_type_prompt_before_the_root(tmp_path: Path) -> None:
    from web_api.chat import turn_setup
    from web_api.workspace import INSTRUCTIONS as WORKSPACE_INSTRUCTIONS

    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-prompt",
            {"channel_id": channel_id, "name": "Prompt", "type": "gallery"},
        )
        instructions, tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            "agent_favlo",
            workspace["session_id"],
            None,
        )
    finally:
        store.close()
    names = {_tool_name(item) for item in tools}
    assert WORKSPACE_INSTRUCTIONS["gallery"] in instructions
    assert WORKSPACE_INSTRUCTIONS["coding"] not in instructions
    assert "gallery_create_folder" not in names
    assert "gallery root" not in instructions.lower()


def test_workspace_prompt_is_fixed_per_type() -> None:
    from web_api.workspace import INSTRUCTIONS, text_for

    assert "gallery workspace" in text_for("gallery")
    assert "coding workspace" in text_for("coding")
    assert text_for("gallery") != text_for("coding")
    assert text_for("gallery") == INSTRUCTIONS["gallery"]
    assert text_for("unknown") is None


def test_workspace_skills_are_name_and_description(client: TestClient) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-skill"},
        json={"channel_id": channel_id, "name": "Skills", "type": "gallery"},
    )
    workspace_id = created.json()["id"]
    assert created.json()["skills"] == []
    added = client.post(
        f"/workspaces/{workspace_id}/skills",
        headers={"Idempotency-Key": "skill-1"},
        json={"name": "Photo captions", "description": "Write a one-line caption."},
    )
    assert added.status_code == 201
    skill = added.json()
    assert skill["name"] == "Photo captions"
    assert skill["description"] == "Write a one-line caption."
    listed = client.get(f"/workspaces/{workspace_id}/skills").json()["skills"]
    assert [item["name"] for item in listed] == ["Photo captions"]
    duplicate = client.post(
        f"/workspaces/{workspace_id}/skills",
        headers={"Idempotency-Key": "skill-dup"},
        json={"name": "Photo captions", "description": "Other"},
    )
    assert duplicate.status_code == 400
    patched = client.patch(
        f"/workspaces/{workspace_id}/skills/{skill['id']}",
        json={
            "expected_revision": skill["revision"],
            "name": "Album notes",
            "description": "Name the folder first.",
        },
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Album notes"
    removed = client.delete(f"/workspaces/{workspace_id}/skills/{skill['id']}")
    assert removed.status_code == 200
    empty = client.get(f"/workspaces/{workspace_id}").json()["workspace"]["skills"]
    assert empty == []


def test_workspace_skills_join_the_system_prompt(tmp_path: Path) -> None:
    from web_api.chat import turn_setup
    from web_api.workspace import skills_text

    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-skill-prompt",
            {"channel_id": channel_id, "name": "Prompt", "type": "gallery"},
        )
        store.create_skill(
            "skill-caption",
            workspace["id"],
            {"name": "Photo captions", "description": "Write a one-line caption."},
        )
        store.create_skill(
            "skill-color",
            workspace["id"],
            {"name": "Color notes", "description": "Prefer the folder palette."},
        )
        store.create_agent_skill(
            "agent-skill",
            {
                "agent_id": "agent_favlo",
                "name": "Global captions",
                "description": "Keep names short.",
            },
        )
        instructions, _tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            "agent_favlo",
            workspace["session_id"],
            None,
        )
        agent_skills = store.list_agent_skills("agent_favlo")
        stored = agent_skills + store.list_skills(workspace["id"])
        from web_api.db import FAVLO_SESSION_ID

        home, _home_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            "agent_favlo",
            FAVLO_SESSION_ID,
            None,
        )
    finally:
        store.close()
    block = skills_text(stored)
    assert block is not None
    assert block.startswith("# Skills")
    assert block.index("## Global captions") < block.index("## Photo captions")
    assert block in instructions
    home_block = skills_text(agent_skills)
    assert home_block is not None
    assert home_block in home
    assert "## Photo captions" not in home
    assert skills_text([]) is None


def test_agent_skills_are_name_and_description(client: TestClient) -> None:
    created = client.post(
        "/agent-skills",
        headers={"Idempotency-Key": "global-skill"},
        json={
            "agent_id": "agent_favlo",
            "name": "Photo captions",
            "description": "Write a one-line caption.",
        },
    )
    assert created.status_code == 201
    skill = created.json()
    assert skill["name"] == "Photo captions"
    listed = client.get("/agent-skills").json()["skills"]
    assert [row["id"] for row in listed if row["id"] == skill["id"]] == [skill["id"]]
    duplicate = client.post(
        "/agent-skills",
        headers={"Idempotency-Key": "global-skill-dup"},
        json={
            "agent_id": "agent_favlo",
            "name": "Photo captions",
            "description": "Other",
        },
    )
    assert duplicate.status_code == 400
    patched = client.patch(
        f"/agent-skills/{skill['id']}",
        json={
            "expected_revision": skill["revision"],
            "name": "Album notes",
            "description": "Name the folder first.",
        },
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Album notes"
    removed = client.delete(f"/agent-skills/{skill['id']}")
    assert removed.status_code == 200
    empty = client.get("/agent-skills").json()["skills"]
    assert skill["id"] not in [row["id"] for row in empty]


def test_deleting_a_workspace_keeps_the_files(client: TestClient, tmp_path: Path) -> None:
    gallery = _gallery(client, tmp_path)
    path = gallery["root_path"] / "keep.png"
    path.write_bytes(PNG)
    deleted = client.delete(f"/workspaces/{gallery['id']}")
    assert deleted.status_code == 200
    listed = client.get(f"/workspaces?channel_id={gallery['channel_id']}").json()["workspaces"]
    assert listed == []
    assert path.is_file()
    missing = client.get(f"/workspaces/{gallery['id']}")
    assert missing.status_code == 404


def test_local_folder_rejects_an_escape(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "ok.png").write_bytes(PNG)
    folder = LocalFolder(str(root))
    with pytest.raises(Rejected):
        folder.read("../ok.png")
    with pytest.raises(Rejected):
        folder.write("/tmp/out.png", PNG)
    with pytest.raises(Rejected):
        folder.create_folder("..")
    with pytest.raises(Rejected):
        folder.create_folder("")
    listed = folder.list("")
    assert listed["images"][0]["name"] == "ok.png"


def test_workspace_session_refuses_messages_before_the_root(client: TestClient) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-msg"},
        json={"channel_id": channel_id, "name": "Later", "type": "gallery"},
    )
    refused = client.post(
        f"/workspaces/{created.json()['id']}/session/messages",
        headers={"Idempotency-Key": "msg"},
        json={"text": "Hello"},
    )
    assert refused.status_code == 400


def _tool_name(item: object) -> str:
    return str(getattr(item, "name", None) or getattr(item, "__name__", ""))

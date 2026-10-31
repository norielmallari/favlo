"""Gallery workspace: many folders, tools only on that session."""

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
    saved = client.post(
        f"/workspaces/{created.json()['id']}/folders",
        headers={"Idempotency-Key": "root-1"},
        json={"name": "Stills", "kind": "local", "location": str(root)},
    )
    assert saved.status_code == 201
    return created.json() | {"root_path": root, "channel_id": channel_id, "folder": saved.json()}


def test_profile_lists_disabled_cloud_kinds(client: TestClient) -> None:
    body = client.get("/profile").json()
    assert body["profile"] == "local"
    assert body["folder_kinds"] == [
        {"kind": "local", "enabled": True},
        {"kind": "cloud", "provider": "onedrive", "enabled": False},
        {"kind": "cloud", "provider": "dropbox", "enabled": False},
    ]


def test_a_gallery_stores_many_folders(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-empty"},
        json={"channel_id": channel_id, "name": "Empty", "type": "gallery"},
    )
    workspace_id = created.json()["id"]
    assert client.get(f"/workspaces/{workspace_id}/folders").json()["folders"] == []
    cloud = client.post(
        f"/workspaces/{workspace_id}/folders",
        headers={"Idempotency-Key": "cloud"},
        json={"name": "Tuesday", "kind": "cloud", "provider": "onedrive", "remote_id": "remote"},
    )
    assert cloud.status_code == 201
    assert cloud.json()["location"] is None
    assert cloud.json()["parent_id"] is None
    nested = client.post(
        f"/workspaces/{workspace_id}/folders",
        headers={"Idempotency-Key": "nested"},
        json={
            "name": "Inside",
            "kind": "cloud",
            "provider": "onedrive",
            "parent_id": cloud.json()["id"],
        },
    )
    assert nested.status_code == 201
    children = client.get(
        f"/workspaces/{workspace_id}/folders",
        params={"parent_id": cloud.json()["id"]},
    ).json()["folders"]
    assert [item["name"] for item in children] == ["Inside"]
    absent = client.post(
        f"/workspaces/{workspace_id}/folders",
        headers={"Idempotency-Key": "missing-dir"},
        json={"name": "Missing", "kind": "local", "location": str(tmp_path / "nope")},
    )
    assert absent.status_code == 400
    local = tmp_path / "local"
    local.mkdir()
    saved = client.post(
        f"/workspaces/{workspace_id}/folders",
        headers={"Idempotency-Key": "local"},
        json={"name": "Camera", "kind": "local", "location": str(local), "parent_id": cloud.json()["id"]},
    )
    assert saved.status_code == 400
    assert "upload" in saved.json()["message"].lower() or "root" in saved.json()["message"].lower()
    placed = client.post(
        f"/workspaces/{workspace_id}/folders",
        headers={"Idempotency-Key": "placed"},
        json={"name": "Camera", "kind": "local", "location": str(local)},
    )
    assert placed.status_code == 201
    moved = client.patch(
        f"/workspaces/{workspace_id}/folders/{placed.json()['id']}",
        json={"expected_revision": 1, "move": True, "parent_id": cloud.json()["id"]},
    )
    assert moved.status_code == 400
    assert "upload" in moved.json()["message"].lower()
    onto_local = client.patch(
        f"/workspaces/{workspace_id}/folders/{nested.json()['id']}",
        json={"expected_revision": 1, "move": True, "parent_id": placed.json()["id"]},
    )
    assert onto_local.status_code == 400
    assert "download" in onto_local.json()["message"].lower()
    cycle = client.patch(
        f"/workspaces/{workspace_id}/folders/{cloud.json()['id']}",
        json={"expected_revision": 1, "move": True, "parent_id": nested.json()["id"]},
    )
    assert cycle.status_code == 400
    assert "itself" in cycle.json()["message"].lower()


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
    picked = client.post(f"/workspaces/{workspace_id}/folders/pick")
    assert picked.status_code == 200
    assert picked.json() == {"cancelled": False, "location": str(chosen)}
    monkeypatch.setattr("web_api.routes.workspaces.choose_directory", lambda: None)
    cancelled = client.post(f"/workspaces/{workspace_id}/folders/pick")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": True}


def test_images_and_folders_stay_inside_the_local_folder(client: TestClient, tmp_path: Path) -> None:
    gallery = _gallery(client, tmp_path)
    workspace_id = gallery["id"]
    folder_id = gallery["folder"]["id"]
    encoded = base64.b64encode(PNG).decode()
    written = client.post(
        f"/workspaces/{workspace_id}/folders/{folder_id}/images",
        json={"path": "horse.png", "bytes": encoded, "note": {"prompt": "a horse"}},
    )
    assert written.status_code == 200
    assert written.json()["path"] == "horse.png"
    assert written.json()["note"] == {"prompt": "a horse"}
    listing = client.get(f"/workspaces/{workspace_id}/folders/{folder_id}/images").json()
    assert [item["name"] for item in listing["images"]] == ["horse.png"]
    png = client.get(f"/workspaces/{workspace_id}/folders/{folder_id}/file?path=horse.png")
    assert png.status_code == 200
    assert png.content == PNG
    nested = client.post(
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "shots"},
    )
    assert nested.status_code == 201
    moved = client.patch(
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "shots", "to": "takes"},
    )
    assert moved.status_code == 200
    renamed = client.patch(
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "takes", "name": "stills"},
    )
    assert renamed.json()["path"] == "stills"
    image = client.post(
        f"/workspaces/{workspace_id}/folders/{folder_id}/images/move",
        json={"path": "horse.png", "to": "stills/horse.png"},
    )
    assert image.status_code == 200
    assert image.json()["path"] == "stills/horse.png"
    occupied = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "stills"},
    )
    assert occupied.status_code == 400
    deleted = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders/{folder_id}/images",
        json={"path": "stills/horse.png"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["name"] == "horse.png"
    assert not (gallery["root_path"] / "stills" / "horse.png").exists()
    listing = client.get(f"/workspaces/{workspace_id}/folders/{folder_id}/images?path=stills").json()
    assert listing["empty"] is True
    escape = client.post(
        f"/workspaces/{workspace_id}/folders/{folder_id}/images",
        json={"path": "../outside.png", "bytes": encoded},
    )
    assert escape.status_code == 400
    root = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "/"},
    )
    assert root.status_code == 400
    removed = client.request(
        "DELETE",
        f"/workspaces/{workspace_id}/folders/{folder_id}/directories",
        json={"path": "stills"},
    )
    assert removed.status_code == 200
    assert not (gallery["root_path"] / "stills").exists()


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
        store.add_folder(
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
    assert "gallery_move_image" in gallery_names
    assert "gallery_delete_image" in gallery_names
    assert "gallery_delete_folder" in gallery_names
    assert "operating in a gallery workspace" in gallery_instructions.lower()
    assert "local folder" in gallery_instructions.lower()
    assert "operating in a gallery workspace" not in home_instructions.lower()
    assert "local folder" not in home_instructions.lower()


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
    assert "this session is using" not in instructions.lower()


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
    assert [item["name"] for item in created.json()["skills"]] == ["manage-gallery"]
    assert "gallery_write_image" in created.json()["skills"][0]["definition"]
    assert "gallery_delete_image" in created.json()["skills"][0]["definition"]
    assert "open folder" in created.json()["skills"][0]["definition"]
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
    assert [item["name"] for item in listed] == ["manage-gallery", "Photo captions"]
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
    assert [item["name"] for item in empty] == ["manage-gallery"]


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
        store.create_global_skill(
            "catalog-only",
            {"name": "Catalog only", "description": "Not on a turn."},
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
    assert "Catalog only" not in instructions
    assert "Catalog only" not in home
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


def test_a_catalog_skill_copies_without_following_later_edits(client: TestClient) -> None:
    created = client.post(
        "/global-skills",
        headers={"Idempotency-Key": "catalog-skill"},
        json={"name": "Photo captions", "description": "Write a one-line caption."},
    )
    assert created.status_code == 201
    skill = created.json()
    listed = client.get("/global-skills").json()["skills"]
    names = [row["name"] for row in listed]
    assert names == ["Photo captions", "session-clear", "session-remember"]
    seeded = {row["name"]: row for row in listed}
    assert seeded["session-clear"]["add_by_default"] is True
    assert seeded["session-remember"]["add_by_default"] is True
    assert "clear, reset, or forget" in seeded["session-clear"]["description"]
    assert "session_clear_session" in seeded["session-clear"]["definition"]
    assert "`remember`" in seeded["session-remember"]["definition"]
    assert "`recall`" in seeded["session-remember"]["definition"]
    assert "`forget`" in seeded["session-remember"]["definition"]
    favlo_skills = client.get("/agent-skills").json()["skills"]
    assert {"session-clear", "session-remember"} <= {row["name"] for row in favlo_skills}
    from web_api.workspace import skills_text

    favlo = next(row for row in favlo_skills if row["name"] == "session-clear")
    injected = skills_text([favlo])
    assert favlo["description"] in injected
    assert favlo["definition"] in injected
    copied = client.post(
        f"/global-skills/{skill['id']}/copy",
        headers={"Idempotency-Key": "copy-agent"},
        json={"agent_id": "agent_favlo"},
    )
    assert copied.status_code == 201
    assert copied.json()["agent_id"] == "agent_favlo"
    assert copied.json()["description"] == "Write a one-line caption."
    edited = client.patch(
        f"/global-skills/{skill['id']}",
        json={
            "expected_revision": skill["revision"],
            "name": "Photo captions",
            "description": "Changed in the catalog.",
        },
    )
    assert edited.status_code == 200
    stored = client.get("/agent-skills").json()["skills"]
    match = next(row for row in stored if row["id"] == copied.json()["id"])
    assert match["description"] == "Write a one-line caption."
    again = client.post(
        f"/global-skills/{skill['id']}/copy",
        headers={"Idempotency-Key": "copy-agent-again"},
        json={"agent_id": "agent_favlo"},
    )
    assert again.status_code == 400
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    workspace = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "catalog-workspace"},
        json={"channel_id": channel_id, "name": "Catalog", "type": "gallery"},
    )
    assert workspace.status_code == 201
    copied_workspace = client.post(
        f"/global-skills/{skill['id']}/copy",
        headers={"Idempotency-Key": "copy-workspace"},
        json={"workspace_id": workspace.json()["id"]},
    )
    assert copied_workspace.status_code == 201
    assert copied_workspace.json()["workspace_id"] == workspace.json()["id"]
    missing = client.post(
        f"/global-skills/{skill['id']}/copy",
        headers={"Idempotency-Key": "copy-neither"},
        json={},
    )
    assert missing.status_code == 400
    removed = client.delete(f"/global-skills/{skill['id']}")
    assert removed.status_code == 200
    remaining = client.get("/agent-skills").json()["skills"]
    assert copied.json()["id"] in [row["id"] for row in remaining]
    names = [row["name"] for row in client.get("/workspaces").json()["workspaces"]]
    assert "Catalog" in names


def test_a_new_agent_receives_default_global_skills(client: TestClient) -> None:
    created = client.post(
        "/agents",
        headers={"Idempotency-Key": "default-skills-agent"},
        json={"name": "River", "persona": "A short persona."},
    )
    assert created.status_code == 201
    agent_id = created.json()["agent"]["id"]
    skills = client.get("/agent-skills").json()["skills"]
    names = {row["name"] for row in skills if row["agent_id"] == agent_id}
    assert names == {"session-clear", "session-remember"}
    catalog = client.get("/global-skills").json()["skills"]
    remembered = next(row for row in catalog if row["name"] == "session-remember")
    turned_off = client.patch(
        f"/global-skills/{remembered['id']}",
        json={
            "expected_revision": remembered["revision"],
            "name": remembered["name"],
            "description": remembered["description"],
            "add_by_default": False,
        },
    )
    assert turned_off.status_code == 200
    assert turned_off.json()["add_by_default"] is False
    later = client.post(
        "/agents",
        headers={"Idempotency-Key": "default-skills-later"},
        json={"name": "Cedar", "persona": "A short persona."},
    )
    assert later.status_code == 201
    later_id = later.json()["agent"]["id"]
    later_skills = client.get("/agent-skills").json()["skills"]
    later_names = {row["name"] for row in later_skills if row["agent_id"] == later_id}
    assert later_names == {"session-clear"}


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


def test_an_old_root_row_gains_a_parent(tmp_path: Path) -> None:
    import sqlite3

    from web_api.db import SCHEMA, migrate

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE folder (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            kind TEXT NOT NULL,
            provider TEXT,
            location TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            remote_id TEXT,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO folder (
            id, workspace_id, name, role, kind, provider, location,
            machine_id, remote_id, revision, created_at
        )
        VALUES ('folder_old', 'workspace_old', 'Stills', 'root', 'local', NULL, '/tmp/gallery', '', NULL, 1, '2026-01-01')
        """
    )
    conn.commit()
    conn.close()
    again = sqlite3.connect(path)
    again.row_factory = sqlite3.Row
    migrate(again)
    row = again.execute("SELECT * FROM folder WHERE id = 'folder_old'").fetchone()
    again.close()
    assert row["parent_id"] is None
    assert row["name"] == "Stills"
    assert "folder (" in SCHEMA


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


def test_workspace_session_refuses_messages_before_a_local_folder(client: TestClient) -> None:
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


def test_delete_image_removes_the_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.png").write_bytes(PNG)
    folder = LocalFolder(str(root))
    deleted = folder.delete_image("ok.png")
    assert deleted["name"] == "ok.png"
    assert folder.list("")["images"] == []


def test_a_sent_turn_records_the_open_folder_and_image(tmp_path: Path) -> None:
    from web_api.store import Store

    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.png").write_bytes(PNG)
    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-turn",
            {"channel_id": channel_id, "name": "Turn", "type": "gallery"},
        )
        store.add_folder("root-turn", workspace["id"], {"kind": "local", "location": str(root), "name": "Turn"})
        _status, started, _events = store.begin_message(
            "draw",
            workspace["session_id"],
            "generate an image",
            "",
            "ok.png",
        )
        context = store.run_context(started["run_id"])
    finally:
        store.close()
    assert context["user_text"] == "generate an image"
    assert context["gallery_folder"] == ""
    assert context["gallery_image"] == "ok.png"


def test_seeded_manage_gallery_is_refreshed_once(tmp_path: Path) -> None:
    import sqlite3

    from web_api.db import _PREVIOUS_MANAGE_GALLERY, _refresh_manage_gallery
    from web_api.store import Store

    path = tmp_path / "app.db"
    store = Store(path)
    try:
        channel_id = store.channels()[0]["id"]
        store.create_workspace("ws-refresh", {"channel_id": channel_id, "name": "Refresh", "type": "gallery"})
    finally:
        store.close()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "UPDATE skill SET definition = ? WHERE name = 'manage-gallery'",
        (_PREVIOUS_MANAGE_GALLERY,),
    )
    conn.execute("DELETE FROM meta WHERE key LIKE 'workspace_skill_refresh:%'")
    _refresh_manage_gallery(conn)
    refreshed = conn.execute("SELECT definition FROM skill WHERE name = 'manage-gallery'").fetchone()
    assert "gallery_delete_image" in refreshed["definition"]
    from web_api.db import _MANAGE_GALLERY_BEFORE_EMPTY, _refresh_empty_folder_rule

    conn.execute(
        "UPDATE skill SET definition = ? WHERE name = 'manage-gallery'",
        (_MANAGE_GALLERY_BEFORE_EMPTY,),
    )
    conn.execute("DELETE FROM meta WHERE key LIKE 'workspace_skill_empty:%'")
    _refresh_empty_folder_rule(conn)
    emptied = conn.execute("SELECT definition FROM skill WHERE name = 'manage-gallery'").fetchone()
    assert "Delete it only when it is empty." in emptied["definition"]
    conn.execute("UPDATE skill SET definition = ? WHERE name = 'manage-gallery'", (_MANAGE_GALLERY_BEFORE_EMPTY,))
    _refresh_empty_folder_rule(conn)
    kept_empty = conn.execute("SELECT definition FROM skill WHERE name = 'manage-gallery'").fetchone()
    assert kept_empty["definition"] == _MANAGE_GALLERY_BEFORE_EMPTY
    conn.execute("UPDATE skill SET definition = 'Custom gallery notes.' WHERE name = 'manage-gallery'")
    conn.execute("DELETE FROM meta WHERE key LIKE 'workspace_skill_refresh:%'")
    _refresh_manage_gallery(conn)
    edited = conn.execute("SELECT definition FROM skill WHERE name = 'manage-gallery'").fetchone()
    assert edited["definition"] == "Custom gallery notes."
    conn.execute("UPDATE skill SET definition = ? WHERE name = 'manage-gallery'", (_PREVIOUS_MANAGE_GALLERY,))
    _refresh_manage_gallery(conn)
    kept = conn.execute("SELECT definition FROM skill WHERE name = 'manage-gallery'").fetchone()
    assert kept["definition"] == _PREVIOUS_MANAGE_GALLERY
    conn.close()


def _tool_name(item: object) -> str:
    return str(getattr(item, "name", None) or getattr(item, "__name__", ""))

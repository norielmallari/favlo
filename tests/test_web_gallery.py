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
from web_api.ids import FAVLO_AGENT_ID  # noqa: E402
from web_api.folder_port import LocalFolder  # noqa: E402
from web_api.store import Rejected, Store  # noqa: E402

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _drop_instead_of_trash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite from filling the Trash. The real move is tested beside this."""

    def drop(path: Path) -> Path:
        path.unlink()
        return path

    monkeypatch.setattr("web_api.folder_port.trash_path", drop)


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


def _gallery(client: TestClient, tmp_path: Path) -> dict:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    root = tmp_path / "gallery"
    root.mkdir()
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-1"},
        json={
            "channel_id": channel_id,
            "name": "Stills",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert created.status_code == 201
    folders = client.get(f"/workspaces/{created.json()['id']}/folders").json()["folders"]
    local = next(item for item in folders if item["kind"] == "local")
    return created.json() | {"root_path": root, "channel_id": channel_id, "folder": local}


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
        json={"channel_id": channel_id, "name": "Empty", "type": "gallery", "location": _loc(tmp_path)},
    )
    workspace_id = created.json()["id"]
    roots = client.get(f"/workspaces/{workspace_id}/folders").json()["folders"]
    assert len(roots) == 1
    assert roots[0]["kind"] == "local"
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
        json={"channel_id": channel_id, "name": "Stills", "type": "gallery", "location": _loc(tmp_path)},
    )
    workspace_id = created.json()["id"]
    chosen = tmp_path / "picked"
    chosen.mkdir()
    monkeypatch.setattr("web_api.routes.workspaces.choose_directory", lambda: str(chosen))
    picked = client.post("/folders/pick")
    assert picked.status_code == 200
    assert picked.json() == {"cancelled": False, "location": str(chosen)}
    workspace_pick = client.post(f"/workspaces/{workspace_id}/folders/pick")
    assert workspace_pick.status_code == 200
    assert workspace_pick.json() == {"cancelled": False, "location": str(chosen)}
    monkeypatch.setattr("web_api.routes.workspaces.choose_directory", lambda: None)
    cancelled = client.post("/folders/pick")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"cancelled": True}


def test_gallery_create_requires_a_local_folder(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    missing = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-no-folder"},
        json={"channel_id": channel_id, "name": "NoFolder", "type": "gallery"},
    )
    assert missing.status_code == 400
    assert "local folder" in missing.json()["message"].lower()
    absent = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-missing-dir"},
        json={
            "channel_id": channel_id,
            "name": "MissingDir",
            "type": "gallery",
            "location": str(tmp_path / "nope"),
        },
    )
    assert absent.status_code == 400


def test_images_and_folders_stay_inside_the_local_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _drop_instead_of_trash(monkeypatch)
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

    store = Store(tmp_path / "app.db")
    try:
        home = next(
            row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID
        )
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-tools",
            {"channel_id": channel_id, "name": "Tools", "type": "gallery", "location": _loc(tmp_path)},
        )
        home_instructions, home_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            FAVLO_AGENT_ID,
            home,
            None,
        )
        gallery_instructions, gallery_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            FAVLO_AGENT_ID,
            workspace["session_id"],
            None,
        )
    finally:
        store.close()
    home_names = {_tool_name(item) for item in home_tools}
    gallery_names = {_tool_name(item) for item in gallery_tools}
    assert "gallery_create_folder" not in home_names
    assert "gallery_write_image" not in home_names
    assert "gallery_write_image" in gallery_names
    assert "gallery_search_unlike_media" in gallery_names
    assert "gallery_create_folder" in gallery_names
    assert "gallery_load_folder_tools" not in gallery_names
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
            {"channel_id": channel_id, "name": "Prompt", "type": "gallery", "location": _loc(tmp_path)},
        )
        instructions, tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            FAVLO_AGENT_ID,
            workspace["session_id"],
            None,
        )
    finally:
        store.close()
    names = {_tool_name(item) for item in tools}
    assert WORKSPACE_INSTRUCTIONS["gallery"] in instructions
    assert WORKSPACE_INSTRUCTIONS["coding"] not in instructions
    assert "gallery_load_folder_tools" not in names
    assert "gallery_find_media" in names
    assert "gallery_search_media" in names
    assert "gallery_get_metadata" in names
    assert "gallery_load_image" in names
    assert "gallery_search_unlike_media" in names
    assert "gallery_create_folder" in names
    assert "workspace_place" in names
    assert "gallery_set_theme" in names
    assert "gallery_update_metadata" in names
    assert "gallery_list_folder" not in names
    assert "this session is using" not in instructions.lower()
    assert "gallery_get_metadata" in instructions
    assert "gallery_load_image" in instructions
    assert "When the person asks" not in instructions


def test_workspace_prompt_is_fixed_per_type() -> None:
    from web_api.workspace import INSTRUCTIONS, text_for

    assert "gallery workspace" in text_for("gallery")
    assert "coding workspace" in text_for("coding")
    assert text_for("gallery") != text_for("coding")
    assert text_for("gallery") == INSTRUCTIONS["gallery"]
    assert text_for("unknown") is None


def test_workspace_skills_are_name_and_description(client: TestClient, tmp_path: Path) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-skill"},
        json={"channel_id": channel_id, "name": "Skills", "type": "gallery", "location": _loc(tmp_path)},
    )
    workspace_id = created.json()["id"]
    assert [item["name"] for item in created.json()["skills"]] == ["manage-gallery"]
    assert "gallery_write_image" in created.json()["skills"][0]["definition"]
    assert "gallery_find_media" in created.json()["skills"][0]["definition"]
    assert "gallery_get_metadata" in created.json()["skills"][0]["definition"]
    assert "gallery_load_image" in created.json()["skills"][0]["definition"]
    assert "gallery_update_metadata" in created.json()["skills"][0]["definition"]
    assert "gallery_search_unlike_media" in created.json()["skills"][0]["definition"]
    assert "workspace_place" in created.json()["skills"][0]["definition"]
    assert "gallery_list_folder" not in created.json()["skills"][0]["definition"]
    assert "When the person asks" not in created.json()["skills"][0]["definition"]
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
            {"channel_id": channel_id, "name": "Prompt", "type": "gallery", "location": _loc(tmp_path)},
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
                "agent_id": FAVLO_AGENT_ID,
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
            FAVLO_AGENT_ID,
            workspace["session_id"],
            None,
        )
        agent_skills = store.list_agent_skills(FAVLO_AGENT_ID)
        stored = agent_skills + store.list_skills(workspace["id"])
        home_id = next(
            row["session_id"] for row in store.channels() if row["agent_id"] == FAVLO_AGENT_ID
        )
        home, _home_tools = turn_setup(
            "You are Favlo.",
            "Favlo",
            [],
            "http://127.0.0.1:8129/v1",
            store,
            FAVLO_AGENT_ID,
            home_id,
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
            "agent_id": FAVLO_AGENT_ID,
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
            "agent_id": FAVLO_AGENT_ID,
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


def test_a_catalog_skill_copies_without_following_later_edits(
    client: TestClient, tmp_path: Path
) -> None:
    created = client.post(
        "/global-skills",
        headers={"Idempotency-Key": "catalog-skill"},
        json={"name": "Photo captions", "description": "Write a one-line caption."},
    )
    assert created.status_code == 201
    skill = created.json()
    listed = client.get("/global-skills").json()["skills"]
    names = [row["name"] for row in listed]
    assert names == ["Photo captions", "session-remember"]
    seeded = {row["name"]: row for row in listed}
    assert "session-clear" not in seeded
    assert seeded["session-remember"]["add_by_default"] is True
    assert "session_clear_session" not in seeded["session-remember"]["definition"]
    assert "`remember`" in seeded["session-remember"]["definition"]
    assert "`recall`" in seeded["session-remember"]["definition"]
    assert "`forget`" in seeded["session-remember"]["definition"]
    favlo_skills = client.get("/agent-skills").json()["skills"]
    assert "session-clear" not in {row["name"] for row in favlo_skills}
    assert "session-remember" in {row["name"] for row in favlo_skills}
    from web_api.workspace import skills_text

    favlo = next(row for row in favlo_skills if row["name"] == "session-remember")
    injected = skills_text([favlo])
    assert favlo["description"] in injected
    assert favlo["definition"] in injected
    copied = client.post(
        f"/global-skills/{skill['id']}/copy",
        headers={"Idempotency-Key": "copy-agent"},
        json={"agent_id": FAVLO_AGENT_ID},
    )
    assert copied.status_code == 201
    assert copied.json()["agent_id"] == FAVLO_AGENT_ID
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
        json={"agent_id": FAVLO_AGENT_ID},
    )
    assert again.status_code == 400
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    workspace = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "catalog-workspace"},
        json={"channel_id": channel_id, "name": "Catalog", "type": "gallery", "location": _loc(tmp_path)},
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
    assert names == {"session-remember"}
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
    assert later_names == set()


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

    from sqlalchemy import inspect

    from web_api.db import SCHEMA, migrate
    from web_api.infrastructure.db.database import open_session

    assert callable(migrate)
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
        VALUES (
            'folder_old', 'workspace_old', 'Stills', 'root', 'local',
            NULL, '/tmp/gallery', '', NULL, 1, '2026-01-01'
        )
        """
    )
    conn.commit()
    conn.close()
    session = open_session(path)
    try:
        columns = {column["name"] for column in inspect(session.get_bind()).get_columns("folder")}
        nullable = {
            column["name"]: column["nullable"]
            for column in inspect(session.get_bind()).get_columns("folder")
        }
    finally:
        session.close()
        session.get_bind().dispose()
    assert "parent_id" in columns
    assert nullable["parent_id"] is True
    assert "name" in columns
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


def test_workspace_session_accepts_messages_when_created_with_a_local_folder(
    client: TestClient, tmp_path: Path
) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-msg"},
        json={
            "channel_id": channel_id,
            "name": "Ready",
            "type": "gallery",
            "location": _loc(tmp_path),
        },
    )
    assert created.status_code == 201
    accepted = client.post(
        f"/workspaces/{created.json()['id']}/session/messages",
        headers={"Idempotency-Key": "msg"},
        json={"text": "Hello"},
    )
    assert accepted.status_code in {200, 201, 202}


def test_delete_image_moves_the_file_to_the_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.png").write_bytes(PNG)
    (root / "ok.png.json").write_text("{}\n", encoding="utf-8")
    moved: list[str] = []

    def remember(path: Path) -> Path:
        moved.append(path.name)
        path.unlink()
        return path

    monkeypatch.setattr("web_api.folder_port.trash_path", remember)
    folder = LocalFolder(str(root))
    deleted = folder.delete_image("ok.png")
    assert deleted["name"] == "ok.png"
    assert moved == ["ok.png", "ok.png.json"]
    assert folder.list("")["images"] == []


def test_trash_path_moves_a_file_into_the_macos_trash(tmp_path: Path) -> None:
    from Foundation import NSURL, NSFileManager

    from web_api.folder_port import trash_path

    target = tmp_path / "favlo-trash-check.png"
    target.write_bytes(PNG)
    landed = trash_path(target)
    try:
        assert not target.exists()
        assert landed.name.startswith("favlo-trash-check")
        assert landed.parent == Path.home() / ".Trash"
    finally:
        url = NSURL.fileURLWithPath_(str(landed))
        NSFileManager.defaultManager().removeItemAtURL_error_(url, None)


def test_a_named_image_in_the_message_is_read_from_the_gallery_folder(tmp_path: Path) -> None:
    from web_api.chat import user_turn
    from web_api.gallery_tools import images_in_message, named_images

    assert named_images("see zebra.png and photos/ok.jpg") == ["zebra.png", "photos/ok.jpg"]
    assert named_images("favlo://images/abc.png") == []
    assert named_images("https://example.com/a.png") == []

    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "photos").mkdir()
    (root / "a" / "zebra.png").write_bytes(PNG)
    (root / "b" / "zebra.png").write_bytes(PNG + b"b")
    (root / "photos" / "my photo.png").write_bytes(b"mine")
    (root / "photo.png").write_bytes(b"other")
    (root / "note.txt").write_bytes(b"no")
    folder = LocalFolder(str(root))
    assert folder.find_images(["zebra.png", "missing.png", "note.txt"]) == ["a/zebra.png"]
    assert folder.find_images(["b/zebra.png"]) == ["b/zebra.png"]

    store = Store(tmp_path / "app.db")
    try:
        channel = store.channels()[0]
        _status, workspace, _events = store.create_workspace(
            "ws-named",
            {
                "channel_id": channel["id"],
                "name": "Named",
                "type": "gallery",
                "location": str(root),
            },
        )
        home = images_in_message(store, channel["session_id"], "see zebra.png")
        found = images_in_message(store, workspace["session_id"], "look at zebra.png")
        exact = images_in_message(store, workspace["session_id"], "photos/my photo.png")
        missing = images_in_message(store, workspace["session_id"], "missing.png")
    finally:
        store.close()

    assert home == []
    assert found == [(PNG, "image/png")]
    assert exact == [(b"mine", "image/png")]
    assert missing == []
    prompt = user_turn("look at zebra.png", found)
    assert prompt.contents[0].text == "look at zebra.png"
    assert prompt.contents[1].media_type == "image/png"
    assert str(prompt.contents[1].uri).startswith("data:image/png;base64,")
    assert user_turn("plain", None) == "plain"


def test_an_image_already_in_the_session_is_not_attached_again() -> None:
    from agent_framework import AgentSession, Content, Message

    from web_api.chat import unsent_images

    png = b"\x89PNG\r\n"
    session = AgentSession()
    session.state["in_memory"] = {
        "messages": [
            Message("user", ["see a.png", Content.from_data(data=png, media_type="image/png")]),
        ]
    }
    fresh = unsent_images([(png, "image/png"), (b"jpg", "image/jpeg")], session)
    assert fresh == [(b"jpg", "image/jpeg")]
    assert unsent_images([(png, "image/png")], AgentSession()) == [(png, "image/png")]


def test_a_sent_turn_records_the_open_folder_and_image(tmp_path: Path) -> None:
    from web_api.store import Store

    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.png").write_bytes(PNG)
    store = Store(tmp_path / "app.db")
    store.ensure_initial_model("example-model")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-turn",
            {"channel_id": channel_id, "name": "Turn", "type": "gallery", "location": _loc(tmp_path)},
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
        assert context["gallery_attachment"] is None
        store.complete_run(started["run_id"], "Done")
        _status, attached, _events = store.begin_message(
            "draw-file",
            workspace["session_id"],
            "look at this",
            "",
            "ok.png",
            "ok.png",
        )
        attached_context = store.run_context(attached["run_id"])
    finally:
        store.close()
    assert context["user_text"] == "generate an image"
    assert context["gallery_folder"] == ""
    assert context["gallery_image"] == "ok.png"
    assert attached_context["gallery_image"] == "ok.png"
    assert attached_context["gallery_attachment"] == "ok.png"


def test_create_coding_workspace(client: TestClient) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-coding"},
        json={"channel_id": channel_id, "name": "Repos", "type": "coding"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["type"] == "coding"
    assert body["name"] == "Repos"
    listed = client.get(f"/workspaces?channel_id={channel_id}").json()["workspaces"]
    assert any(row["id"] == body["id"] and row["type"] == "coding" for row in listed)


def test_delete_workspace_keeps_disk_files(client: TestClient, tmp_path: Path) -> None:
    gallery = _gallery(client, tmp_path)
    still = gallery["root_path"] / "horse.png"
    still.write_bytes(PNG)
    removed = client.delete(f"/workspaces/{gallery['id']}")
    assert removed.status_code == 200
    assert still.exists()
    assert still.read_bytes() == PNG
    listed = client.get(f"/workspaces?channel_id={gallery['channel_id']}").json()["workspaces"]
    assert all(row["id"] != gallery["id"] for row in listed)


def test_shells_are_listed_from_the_database(client: TestClient) -> None:
    shells = client.get("/shells").json()["shells"]
    ids = [row["id"] for row in shells]
    assert ids == ["gallery", "coding"]
    gallery = shells[0]
    assert gallery["label"] == "Gallery"
    assert gallery["icon"] == "image"
    assert gallery["setup"] == "local_folder"
    assert gallery["renderer"] == "media"
    assert [item["name"] for item in gallery["elements"]] == ["preview", "filmstrip"]
    assert shells[1]["renderer"] == "chat"
    assert shells[1]["elements"] == []


def test_gallery_greeting_does_not_place_the_filmstrip() -> None:
    from web_api.provision import _cue

    text = _cue({"type": "gallery", "name": "Pixel"})
    assert "Do not call tools." in text
    assert "gallery_find_media" not in text
    assert "workspace_place" not in text
    assert "Do not send a second greeting." in text


def test_opening_filmstrip_keeps_the_placement_clamps(client: TestClient, tmp_path: Path) -> None:
    import time

    from web_api.gallery_metadata import write_sidecar
    from web_api.workspace_tools import ensure_opening_filmstrip

    gallery = _gallery(client, tmp_path)
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = client.get(f"/sessions/{gallery['session_id']}").json()
        if snap["messages"]:
            break
        time.sleep(0.05)
    root = gallery["root_path"]
    for index in range(61):
        (root / f"frame-{index:02d}.png").write_bytes(PNG)
    write_sidecar(
        root,
        "frame-00.png",
        {
            "filename": "frame-00.png",
            "kind": "image",
            "title": "H" * 120,
            "description": "A long description that must not replace the title.",
            "tags": ["frame"],
            "source": "index",
        },
    )
    ensure_opening_filmstrip(client.app.state.store, gallery["id"])
    snap = client.get(f"/sessions/{gallery['session_id']}").json()
    items = snap["state"]["favlo_workspace"]["elements"][1]["props"]["items"]
    assert len(items) == 60
    assert items[0]["path"] == "frame-00.png"
    assert items[0]["caption"] == "H" * 80
    assert snap["state"]["favlo_workspace"]["selection"]["path"] == "frame-00.png"


def test_find_media_rows_are_path_and_kind_only(tmp_path: Path) -> None:
    import json

    from web_api.gallery_metadata import write_sidecar
    from web_api.gallery_tools import _slim_media_rows

    root = tmp_path / "photos"
    root.mkdir()
    (root / "horse.png").write_bytes(PNG)
    write_sidecar(
        root,
        "horse.png",
        {
            "filename": "horse.png",
            "kind": "image",
            "title": "Horse",
            "description": "A painted horse in a field.",
            "tags": ["horse"],
            "source": "index",
        },
    )
    folder = LocalFolder(root)
    listing = _slim_media_rows(folder.walk_media())
    row = listing["media"][0]
    assert row == {"path": "horse.png", "kind": "image"}
    assert "images" not in listing
    assert "title" not in row
    assert "description" not in json.dumps(listing)


def test_selection_announcement_only_when_path_changes() -> None:
    from web_api.gallery_tools import (
        announced_selection,
        announced_selection_from_history,
        selection_announcement,
        selection_cue,
    )

    first = selection_announcement("horse.png", None, "A painted horse.")
    assert "horse.png" in first
    assert "A painted horse." in first
    prior = selection_cue("horse.png", "A painted horse.")
    assert announced_selection(prior) == "horse.png"
    assert selection_announcement("horse.png", "horse.png", "ignored") == ""
    cleared = selection_announcement(None, "horse.png")
    assert cleared == "No media is selected on this turn."
    assert announced_selection(cleared) == ""
    from web_api.host_cues import fence_cue, host_cue_message

    history = [
        {"role": "user", "content": "show this"},
        host_cue_message(prior, message_id="cue-1"),
        {"role": "assistant", "content": "ok"},
    ]
    assert announced_selection_from_history(history) == "horse.png"
    assert announced_selection_from_history([{"role": "user", "content": prior}]) == "horse.png"
    assert announced_selection_from_history([{"role": "user", "content": "show this"}]) is None
    # Legacy developer/system rows still count.
    assert announced_selection_from_history(
        [{"role": "developer", "content": prior}]
    ) == "horse.png"
    assert announced_selection_from_history(
        [*history, {"role": "system", "content": cleared}]
    ) == ""
    assert announced_selection_from_history(
        [*history, {"role": "user", "content": fence_cue(cleared)}]
    ) == ""


def test_silent_server_cue_is_a_fenced_host_snapshot_row(tmp_path: Path) -> None:
    from web_api.agui import _SNAPSHOT_SCOPE
    from web_api.host_cues import fence_cue, host_cue_message
    from web_api.store import Store

    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-silent-cue",
            {
                "channel_id": channel_id,
                "name": "Silent",
                "type": "gallery",
                "location": _loc(tmp_path),
            },
        )
        thread_id = str(workspace["session_id"])
        cue = (
            "The user just created this gallery, named Silent. "
            "The filmstrip already shows the images. "
            "Greet them once in one or two sentences."
        )
        row = host_cue_message(cue, message_id="cue-1")
        store.append_thread_snapshot_message(_SNAPSHOT_SCOPE, thread_id, row)
        document = store.thread_snapshot(_SNAPSHOT_SCOPE, thread_id)
        assert document is not None
        assert document["messages"] == [row]
        # Second append keeps history; nothing is deleted after the run.
        store.append_thread_snapshot_message(
            _SNAPSHOT_SCOPE,
            thread_id,
            {"id": "u-pad", "role": "user", "content": ""},
        )
        store.append_thread_snapshot_message(
            _SNAPSHOT_SCOPE,
            thread_id,
            {"id": "a1", "role": "assistant", "content": "Welcome."},
        )
        kept = store.thread_snapshot(_SNAPSHOT_SCOPE, thread_id)["messages"]
        assert kept[0]["role"] == "user"
        assert kept[0]["content"] == fence_cue(cue)
        assert kept[0]["metadata"]["favlo"]["ui"] == "hidden"
        assert kept[1]["content"] == ""
        assert kept[2]["role"] == "assistant"
    finally:
        store.close()


def _placement(path: str | None) -> dict:
    selection = None if path is None else {"kind": "file", "path": path}
    return {
        "favlo_workspace": {
            "version": 1,
            "guide": "gallery",
            "elements": [],
            "selection": selection,
        }
    }


def test_selection_metadata_rides_the_model_suffix(tmp_path: Path) -> None:
    from agent_framework_ag_ui._agent_run import _split_service_session_input

    from web_api.agui import _images, _with_selection_cue
    from web_api.gallery_metadata import prompt_text, read_sidecar
    from web_api.snapshots import project_messages

    root = Path(_loc(tmp_path, "cue-root"))
    (root / "horse.png").write_bytes(PNG)
    (root / "zebra.png").write_bytes(PNG)
    meta = root / ".favlo_metadata"
    meta.mkdir()
    (meta / "horse.png.json").write_text(
        '{"version":1,"filename":"horse.png","kind":"image","description":"A painted horse."}\n',
        encoding="utf-8",
    )
    (meta / "zebra.png.json").write_text(
        '{"version":1,"filename":"zebra.png","kind":"image","description":"A striped zebra."}\n',
        encoding="utf-8",
    )
    store = Store(tmp_path / "app.db")
    try:
        channel_id = store.channels()[0]["id"]
        _status, workspace, _events = store.create_workspace(
            "ws-selection-cue",
            {
                "channel_id": channel_id,
                "name": "Cue",
                "type": "gallery",
                "location": str(root),
            },
        )
        thread_id = str(workspace["session_id"])
        store.save_agui_state(thread_id, _placement("horse.png"))
        incoming = [{"id": "u1", "role": "user", "content": "Show the meadow"}]
        zebra = read_sidecar(root, "zebra.png")
        assert zebra is not None
        out, fresh = _with_selection_cue(store, thread_id, incoming, _placement("zebra.png"))
        assert fresh == "zebra.png"
        assert incoming[0]["content"] == "Show the meadow"
        assert out[0]["content"].startswith("Show the meadow")
        assert "zebra.png" in out[0]["content"]
        assert prompt_text(zebra) in out[0]["content"]
        assert "horse.png" not in out[0]["content"]
        stored = [{"id": "greet", "role": "assistant", "content": "Welcome."}]
        suffix, _full = _split_service_session_input(stored, out)
        assert len(suffix) == 1
        assert "A striped zebra." in suffix[0]["content"]
        pixels = _images(store, thread_id, f"Show the meadow\n{fresh}", [])
        assert pixels and pixels[0][1] == "image/png"
        shown = project_messages([*stored, suffix[0]])
        assert shown[-1]["body"] == "Show the meadow"
        assert "zebra" not in shown[-1]["body"]
        store.save_thread_snapshot(
            "favlo",
            thread_id,
            {"messages": [*stored, suffix[0]], "state": None, "interrupt": None, "session_state": None},
        )
        again, skipped = _with_selection_cue(store, thread_id, incoming, _placement("zebra.png"))
        assert skipped is None
        assert again[0]["content"] == "Show the meadow"
        assert _images(store, thread_id, "Show the meadow", [suffix[0]]) == []
        missing, none = _with_selection_cue(store, thread_id, incoming, None)
        assert none is None
        assert missing[0]["content"] == "Show the meadow"
    finally:
        store.close()


def test_gallery_creation_captions_before_the_greeting(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    import json
    import time

    calls: list[str] = []

    def fake_analyze(path, *, relative, model, base_url, existing=None, reasoning_effort=None):
        calls.append(model)
        return {
            "filename": "a.png",
            "kind": "image",
            "title": "Horse",
            "description": "A painted horse in a field.",
            "tags": ["horse"],
            "source": "index",
        }

    monkeypatch.setattr("web_api.gallery_embed.analyze_image", fake_analyze)
    monkeypatch.setattr("web_api.gallery_embed._embed", lambda _text: [0.0] * 1024)

    def vision_model(workspace_id=None):
        return ("qwen36-35b-a3b:thinking", "http://127.0.0.1:9")

    client.app.state.store.gallery_vision_model = vision_model
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    root = tmp_path / "captioned"
    root.mkdir()
    (root / "a.png").write_bytes(PNG)
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-vision"},
        json={
            "channel_id": channel_id,
            "name": "Captioned",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert created.status_code == 201
    body = created.json()
    deadline = time.time() + 10
    row = body
    while time.time() < deadline:
        row = client.get(f"/workspaces/{body['id']}").json()["workspace"]
        if row["provisioning_state"] == "provisioned":
            break
        time.sleep(0.05)
    assert row["provisioning_state"] == "provisioned"
    assert calls == ["qwen36-35b-a3b:thinking"]
    side = json.loads((root / ".favlo_metadata" / "a.png.json").read_text())
    assert side["description"] == "A painted horse in a field."
    snap = client.get(f"/sessions/{body['session_id']}").json()
    items = snap["state"]["favlo_workspace"]["elements"][1]["props"]["items"]
    assert [item["path"] for item in items] == ["a.png"]
    assert items[0]["caption"] == "Horse"
    assert snap["state"]["favlo_workspace"]["selection"]["path"] == "a.png"


def test_create_provisions_then_shows_the_failure_in_chat(client: TestClient, tmp_path: Path) -> None:
    import time

    channel_id = client.get("/channels").json()["channels"][0]["id"]
    root = tmp_path / "prov"
    root.mkdir()
    (root / "a.png").write_bytes(PNG)
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-provision"},
        json={
            "channel_id": channel_id,
            "name": "Fresh",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["provisioning_state"] == "provisioning"
    assert body["shell"]["slots"] == ["stage", "rail"]
    deadline = time.time() + 5
    row = body
    snap = {"messages": []}
    while time.time() < deadline:
        row = client.get(f"/workspaces/{body['id']}").json()["workspace"]
        snap = client.get(f"/sessions/{body['session_id']}").json()
        if snap["messages"]:
            break
        time.sleep(0.05)
    assert row["provisioning_state"] == "provisioning"
    assert (root / ".favlo_metadata" / "a.png.json").is_file()
    snap = client.get(f"/sessions/{body['session_id']}").json()
    assert [item["role"] for item in snap["messages"]] == ["assistant"]
    assert "model" in snap["messages"][0]["body"].lower()
    names = [item["element"] for item in snap["state"]["favlo_workspace"]["elements"]]
    assert names == ["preview", "filmstrip"]
    coding = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-provision-coding"},
        json={"channel_id": channel_id, "name": "Code", "type": "coding"},
    )
    assert coding.status_code == 201
    coding_id = coding.json()["id"]
    deadline = time.time() + 10
    coding_row = coding.json()
    while time.time() < deadline:
        coding_row = client.get(f"/workspaces/{coding_id}").json()["workspace"]
        if coding_row["provisioning_state"] == "provisioned":
            break
        time.sleep(0.05)
    assert coding_row["provisioning_state"] == "provisioned"
    coding_snap = client.get(f"/sessions/{coding.json()['session_id']}").json()
    assert coding_snap["state"]["favlo_workspace"]["elements"] == []
    replay = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-provision"},
        json={
            "channel_id": channel_id,
            "name": "Fresh",
            "type": "gallery",
            "location": str(root),
        },
    )
    assert replay.status_code == 201
    assert replay.json()["id"] == body["id"]
    again = client.get(f"/sessions/{body['session_id']}").json()
    assert len(again["messages"]) == 1


def test_create_workspace_rejects_unknown_type(client: TestClient) -> None:
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-bad-type"},
        json={"channel_id": channel_id, "name": "Odd", "type": "music"},
    )
    assert created.status_code == 400


def test_gallery_theme_round_trip(client: TestClient, tmp_path: Path) -> None:
    import asyncio
    import json

    from web_api.gallery_tools import tools

    gallery = _gallery(client, tmp_path)
    assert client.get(f"/workspaces/{gallery['id']}").json()["workspace"]["theme"] == "rose-pine"
    updated = client.put(f"/workspaces/{gallery['id']}/theme", json={"theme": "rose-pine"})
    assert updated.status_code == 200
    assert updated.json()["theme"] == "rose-pine"
    assert client.get(f"/workspaces/{gallery['id']}").json()["workspace"]["theme"] == "rose-pine"
    rejected = client.put(f"/workspaces/{gallery['id']}/theme", json={"theme": "neon"})
    assert rejected.status_code == 400

    class _Bus:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def publish(self, event: dict[str, object]) -> None:
            self.events.append(event)

    bus = _Bus()
    tool = next(
        item
        for item in tools(client.app.state.store, gallery["id"], bus)
        if _tool_name(item) == "gallery_set_theme"
    )
    body = json.loads(asyncio.run(tool.invoke(arguments={"theme": "catppuccin"}, skip_parsing=True)))
    assert body["theme"] == "catppuccin"
    assert bus.events[0]["type"] == "workspace.theme_set"
    assert client.get(f"/workspaces/{gallery['id']}").json()["workspace"]["theme"] == "catppuccin"
    missing = json.loads(asyncio.run(tool.invoke(arguments={"theme": "neon"}, skip_parsing=True)))
    assert "error" in missing


def _tool_json(tool: object, arguments: dict) -> dict:
    import asyncio
    import json

    result = asyncio.run(tool.invoke(arguments=arguments))
    text = result if isinstance(result, str) else result[0].text
    loaded = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded


def test_update_metadata_changes_only_the_named_fields(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    import time

    from web_api.gallery_metadata import read_sidecar
    from web_api.gallery_tools import tools

    monkeypatch.setattr("web_api.gallery_embed._embed", lambda _text: [0.0] * 1024)
    root = tmp_path / "meta"
    root.mkdir()
    (root / "horse.png").write_bytes(PNG)
    (root / "clip.mp3").write_bytes(b"id3")
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-meta"},
        json={"channel_id": channel_id, "name": "Meta", "type": "gallery", "location": str(root)},
    )
    assert created.status_code == 201
    tool = next(
        item
        for item in tools(client.app.state.store, created.json()["id"], None)
        if _tool_name(item) == "gallery_update_metadata"
    )
    schema = tool.parameters()
    assert schema["required"] == ["path"]
    assert "description" in schema["properties"]
    deadline = time.time() + 5
    while time.time() < deadline and not (root / ".favlo_metadata" / "horse.png.json").is_file():
        time.sleep(0.05)
    empty = _tool_json(tool, {"path": "horse.png"})
    assert empty["error"] == "missing_fields"
    updated = _tool_json(
        tool,
        {"path": "horse.png", "title": "Painted horse", "tags": ["horse"], "nsfw": True},
    )
    assert updated["title"] == "Painted horse"
    assert updated["tags"] == ["horse"]
    assert updated["nsfw"] is True
    assert updated["indexed"] is True
    side = read_sidecar(root, "horse.png")
    assert side is not None
    assert side["title"] == "Painted horse"
    assert side["description"]
    assert side["description"] != "Painted horse"
    audio = _tool_json(tool, {"path": "clip.mp3", "title": "Nope"})
    assert audio["error"] == "not_an_image"
    missing = _tool_json(tool, {"path": "gone.png", "title": "Nope"})
    assert missing["error"] == "missing"
    coerced = _tool_json(tool, {"path": "horse.png", "nsfw": "false"})
    assert coerced["nsfw"] is False


def test_get_metadata_and_load_image(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import time

    from agent_framework import Content

    from web_api.gallery_tools import tools

    monkeypatch.setattr("web_api.gallery_embed._embed", lambda _text: [0.0] * 1024)
    root = tmp_path / "detail"
    root.mkdir()
    (root / "horse.png").write_bytes(PNG)
    channel_id = client.get("/channels").json()["channels"][0]["id"]
    created = client.post(
        "/workspaces",
        headers={"Idempotency-Key": "ws-detail"},
        json={"channel_id": channel_id, "name": "Detail", "type": "gallery", "location": str(root)},
    )
    assert created.status_code == 201
    workspace_id = created.json()["id"]
    deadline = time.time() + 5
    while time.time() < deadline and not (root / ".favlo_metadata" / "horse.png.json").is_file():
        time.sleep(0.05)
    meta = next(
        item
        for item in tools(client.app.state.store, workspace_id, None)
        if _tool_name(item) == "gallery_get_metadata"
    )
    body = _tool_json(meta, {"path": "horse.png"})
    assert body["path"] == "horse.png"
    assert body["description"]
    assert "title" in body
    load = next(
        item
        for item in tools(client.app.state.store, workspace_id, None)
        if _tool_name(item) == "gallery_load_image"
    )
    loaded = asyncio.run(load.invoke(arguments={"path": "horse.png"}, skip_parsing=True))
    assert isinstance(loaded, Content)
    assert loaded.media_type == "image/png"


def test_vision_request_uses_the_agent_effort(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    from web_api.gallery_vision import analyze_image
    from web_api.ids import FAVLO_AGENT_ID

    gallery = _gallery(client, tmp_path)
    store = client.app.state.store
    assert store.gallery_vision_effort(gallery["id"]) is None
    agent = next(item for item in client.get("/agents").json()["agents"] if item["id"] == FAVLO_AGENT_ID)
    updated = client.patch(
        f"/agents/{FAVLO_AGENT_ID}",
        json={
            "expected_revision": agent["revision"],
            "name": agent["name"],
            "persona": agent["persona"],
            "default_model_id": agent["default_model_id"],
            "tool_groups": agent["tool_groups"],
            "reasoning_effort": "high",
        },
    )
    assert updated.status_code == 200
    assert store.gallery_vision_effort(gallery["id"]) == "high"

    seen: dict[str, object] = {}

    def fake_post(_base: str, body: dict[str, object]) -> dict[str, str]:
        seen["body"] = body
        return {"output_text": '{"title":"Horse","description":"A horse."}'}

    monkeypatch.setattr("favlo.responses_client.post_responses", fake_post)
    image = tmp_path / "a.png"
    image.write_bytes(PNG)
    analyze_image(image, relative="a.png", model="example-model", base_url="http://127.0.0.1:9")
    body = seen["body"]
    assert isinstance(body, dict)
    assert "reasoning" not in body
    assert "enable_thinking" not in body
    analyze_image(
        image,
        relative="a.png",
        model="example-model",
        base_url="http://127.0.0.1:9",
        reasoning_effort="high",
    )
    sent = seen["body"]
    assert isinstance(sent, dict)
    assert sent["reasoning"] == {"effort": "high"}
    assert "enable_thinking" not in sent


def _tool_name(item: object) -> str:
    return str(getattr(item, "name", None) or getattr(item, "__name__", ""))

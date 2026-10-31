"""Gallery workspaces, folder rows, and the workspace session."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import Response

from web_api.folder_pick import choose_directory
from web_api.folder_port import folder_for
from web_api.gallery_tools import image_bytes
from web_api.http import json_body, key
from web_api.profile import document
from web_api.runs import perform
from web_api.schemas import (
    FocusUpdate,
    FolderCreate,
    FolderDelete,
    FolderPatch,
    FolderRecord,
    FolderRecordUpdate,
    ImageDelete,
    ImageMove,
    ImageWrite,
    MessageCreate,
    SkillCreate,
    SkillUpdate,
    WorkspaceCreate,
)
from web_api.store import Rejected

router = APIRouter()


@router.get("/profile")
async def profile():
    return json_body(200, document())


@router.get("/workspaces")
async def list_workspaces(request: Request, channel_id: str | None = None):
    rows = await asyncio.to_thread(request.app.state.store.list_workspaces, channel_id)
    return json_body(200, {"workspaces": rows})


@router.post("/workspaces")
async def create_workspace(body: WorkspaceCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.create_workspace,
        key(request),
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.get("/workspaces/{workspace_id}")
async def read_workspace(workspace_id: str, request: Request):
    row = await asyncio.to_thread(request.app.state.store.workspace, workspace_id)
    body = document()
    body["workspace"] = row
    return json_body(200, body)


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, request: Request):
    result, events, run_ids = await asyncio.to_thread(
        request.app.state.store.delete_workspace,
        workspace_id,
    )
    for run_id in run_ids:
        pending = request.app.state.cancels.get(run_id)
        if pending is not None:
            pending.set()
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, result)


@router.get("/workspaces/{workspace_id}/skills")
async def list_skills(workspace_id: str, request: Request):
    rows = await asyncio.to_thread(request.app.state.store.list_skills, workspace_id)
    return json_body(200, {"skills": rows})


@router.post("/workspaces/{workspace_id}/skills")
async def create_skill(workspace_id: str, body: SkillCreate, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.create_skill,
        key(request),
        workspace_id,
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.patch("/workspaces/{workspace_id}/skills/{skill_id}")
async def update_skill(workspace_id: str, skill_id: str, body: SkillUpdate, request: Request):
    updated, events = await asyncio.to_thread(
        request.app.state.store.update_skill,
        workspace_id,
        skill_id,
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, updated)


@router.delete("/workspaces/{workspace_id}/skills/{skill_id}")
async def delete_skill(workspace_id: str, skill_id: str, request: Request):
    result, events = await asyncio.to_thread(
        request.app.state.store.delete_skill,
        workspace_id,
        skill_id,
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, result)


@router.get("/workspaces/{workspace_id}/folders")
async def list_folder_rows(workspace_id: str, request: Request, parent_id: str | None = None):
    rows = await asyncio.to_thread(
        request.app.state.store.list_folders,
        workspace_id,
        parent_id,
    )
    return json_body(200, {"folders": rows})


@router.post("/workspaces/{workspace_id}/folders")
async def create_folder_row(workspace_id: str, body: FolderRecord, request: Request):
    status, result, events = await asyncio.to_thread(
        request.app.state.store.add_folder,
        key(request),
        workspace_id,
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(status, result)


@router.post("/workspaces/{workspace_id}/folders/pick")
async def pick_folder(workspace_id: str, request: Request):
    await asyncio.to_thread(request.app.state.store.workspace, workspace_id)
    location = await asyncio.to_thread(choose_directory)
    if location is None:
        return json_body(200, {"cancelled": True})
    return json_body(200, {"cancelled": False, "location": location})


@router.put("/workspaces/{workspace_id}/focus")
async def put_focus(workspace_id: str, body: FocusUpdate, request: Request):
    result = await asyncio.to_thread(
        request.app.state.store.set_focus,
        workspace_id,
        body.folder_id,
    )
    return json_body(200, result)


@router.patch("/workspaces/{workspace_id}/folders/{folder_id}")
async def patch_folder_row(
    workspace_id: str, folder_id: str, body: FolderRecordUpdate, request: Request
):
    result, events = await asyncio.to_thread(
        request.app.state.store.update_folder,
        workspace_id,
        folder_id,
        body.model_dump(),
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, result)


@router.delete("/workspaces/{workspace_id}/folders/{folder_id}")
async def delete_folder_row(workspace_id: str, folder_id: str, request: Request):
    result, events = await asyncio.to_thread(
        request.app.state.store.delete_folder_row,
        workspace_id,
        folder_id,
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, result)


@router.get("/workspaces/{workspace_id}/folders/{folder_id}/images")
async def list_images(workspace_id: str, folder_id: str, request: Request, path: str = ""):
    listing = await asyncio.to_thread(
        folder_for(request.app.state.store, workspace_id, folder_id).list,
        path,
    )
    return json_body(200, listing)


@router.get("/workspaces/{workspace_id}/folders/{folder_id}/file")
async def read_file(workspace_id: str, folder_id: str, request: Request, path: str):
    folder = folder_for(request.app.state.store, workspace_id, folder_id)
    payload = await asyncio.to_thread(folder.read, path)
    stat = await asyncio.to_thread(folder.stat, path)
    return Response(content=payload, media_type=stat["media_type"])


@router.post("/workspaces/{workspace_id}/folders/{folder_id}/images")
async def write_image(workspace_id: str, folder_id: str, body: ImageWrite, request: Request):
    folder = folder_for(request.app.state.store, workspace_id, folder_id)
    payload = image_bytes(source=body.source, data=body.data)
    written = await asyncio.to_thread(
        folder.write,
        body.path,
        payload,
        body.note,
        body.expected_revision,
    )
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, written)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, written)


@router.post("/workspaces/{workspace_id}/folders/{folder_id}/images/move")
async def move_image(workspace_id: str, folder_id: str, body: ImageMove, request: Request):
    folder = folder_for(request.app.state.store, workspace_id, folder_id)
    moved = await asyncio.to_thread(folder.move, body.path, body.to)
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, moved)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, moved)


@router.delete("/workspaces/{workspace_id}/folders/{folder_id}/images")
async def delete_image(workspace_id: str, folder_id: str, body: ImageDelete, request: Request):
    folder = folder_for(request.app.state.store, workspace_id, folder_id)
    deleted = await asyncio.to_thread(folder.delete_image, body.path)
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, deleted)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, deleted)


@router.post("/workspaces/{workspace_id}/folders/{folder_id}/directories")
async def create_directory(workspace_id: str, folder_id: str, body: FolderCreate, request: Request):
    created = await asyncio.to_thread(
        folder_for(request.app.state.store, workspace_id, folder_id).create_folder,
        body.path,
    )
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, created)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(201, created)


@router.patch("/workspaces/{workspace_id}/folders/{folder_id}/directories")
async def patch_directory(workspace_id: str, folder_id: str, body: FolderPatch, request: Request):
    folder = folder_for(request.app.state.store, workspace_id, folder_id)
    if body.to:
        changed = await asyncio.to_thread(folder.move_folder, body.path, body.to)
    elif body.name:
        changed = await asyncio.to_thread(folder.rename_folder, body.path, body.name)
    else:
        raise Rejected("Provide name or to.")
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, changed)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, changed)


@router.delete("/workspaces/{workspace_id}/folders/{folder_id}/directories")
async def delete_directory(workspace_id: str, folder_id: str, body: FolderDelete, request: Request):
    deleted = await asyncio.to_thread(
        folder_for(request.app.state.store, workspace_id, folder_id).delete_folder,
        body.path,
    )
    events = await asyncio.to_thread(request.app.state.store.note_gallery, workspace_id, deleted)
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, deleted)


@router.get("/workspaces/{workspace_id}/session")
async def workspace_session(workspace_id: str, request: Request):
    row = await asyncio.to_thread(request.app.state.store.workspace, workspace_id)
    snapshot = await asyncio.to_thread(
        request.app.state.store.session_snapshot,
        row["session_id"],
    )
    snapshot.update(document())
    snapshot["workspace"] = row
    return json_body(200, snapshot)


@router.post("/workspaces/{workspace_id}/session/messages")
async def send(workspace_id: str, body: MessageCreate, request: Request):
    row = await asyncio.to_thread(request.app.state.store.workspace, workspace_id)
    ready = await asyncio.to_thread(request.app.state.store.has_local_folder, workspace_id)
    if not ready:
        raise Rejected("Save a local folder first.")
    folder = body.folder
    image = body.image.strip() if body.image else None
    if folder is not None or image:
        port = folder_for(request.app.state.store, workspace_id)
        if folder is not None:
            await asyncio.to_thread(port.resolve, folder, directory=True)
        if image:
            await asyncio.to_thread(port.resolve, image, file=True)
            if folder and not image.startswith(f"{folder}/"):
                raise Rejected("That image is not in the open folder.")
    status, result, events = await asyncio.to_thread(
        request.app.state.store.begin_message,
        key(request),
        row["session_id"],
        body.text,
        folder,
        image,
    )
    bus = request.app.state.bus
    for event in events:
        await bus.publish(event)
    if events:
        cancel = asyncio.Event()
        request.app.state.cancels[result["run_id"]] = cancel
        task = asyncio.create_task(
            perform(
                request.app.state.store,
                bus,
                cancel,
                result["run_id"],
                request.app.state.reply,
                request.app.state.tasks,
            )
        )
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
    return json_body(status, result)


@router.post("/workspaces/{workspace_id}/session/clear")
async def clear_session(workspace_id: str, request: Request):
    row = await asyncio.to_thread(request.app.state.store.workspace, workspace_id)
    events = await asyncio.to_thread(
        request.app.state.store.clear_session_messages,
        row["session_id"],
    )
    for event in events:
        await request.app.state.bus.publish(event)
    return json_body(200, {"status": "cleared"})

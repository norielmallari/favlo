"""Finish a new workspace after create has returned.

A gallery writes path sidecars and captions every image. The opening
placement then puts those images on the filmstrip. The model does not
call a tool for that. The workspace stays provisioning until the captions
exist. Any other shell only greets. A failure after the captions still
marks the workspace provisioned and leaves the failure in the chat.
"""

from __future__ import annotations

import asyncio
from typing import Any

from web_api.ids import guid
from web_api.reply_error import spoken_error
from web_api.runs import perform
from web_api.store import Rejected

def schedule(app: Any, workspace_id: str) -> None:
    job = asyncio.create_task(provision(app, workspace_id))
    app.state.tasks.add(job)
    job.add_done_callback(app.state.tasks.discard)


async def provision(app: Any, workspace_id: str) -> None:
    store = app.state.store
    bus = app.state.bus
    held = False
    captions_ready = False
    try:
        workspace = await asyncio.to_thread(store.workspace, workspace_id)
    except Exception:
        return
    captions_ready = workspace.get("type") != "gallery"
    try:
        if workspace.get("type") == "gallery":
            from web_api.gallery_embed import (
                hold_index,
                uncaptioned_images,
                write_path_sidecars,
                write_vision_sidecars,
            )

            held = await asyncio.to_thread(hold_index, app, workspace_id)
            if not held:
                raise Rejected("The gallery index is busy.")
            await asyncio.to_thread(write_path_sidecars, store, workspace_id)
            await asyncio.to_thread(write_vision_sidecars, store, workspace_id)
            pending = await asyncio.to_thread(uncaptioned_images, store, workspace_id)
            if pending:
                raise Rejected("Image captions are not ready.")
            captions_ready = True
        await _greet(app, workspace)
    except Exception as exc:
        await _say(app, workspace["session_id"], spoken_error(str(exc)))
    finally:
        if held:
            from web_api.gallery_embed import release_index

            await asyncio.to_thread(release_index, app, workspace_id)
        if captions_ready and workspace.get("type") == "gallery":
            from web_api.workspace_tools import ensure_opening_filmstrip

            try:
                await asyncio.to_thread(ensure_opening_filmstrip, store, workspace_id)
            except Exception as exc:
                await _say(app, workspace["session_id"], spoken_error(str(exc)))
        if captions_ready:
            try:
                events = await asyncio.to_thread(store.finish_provisioning, workspace_id)
            except Exception:
                events = []
            for event in events:
                await bus.publish(event)


async def _greet(app: Any, workspace: dict[str, Any]) -> None:
    store = app.state.store
    bus = app.state.bus
    session_id = str(workspace["session_id"])
    snapshot = await asyncio.to_thread(store.session_snapshot, session_id)
    agent_id = str(snapshot["session"].get("participant_id") or "")
    await asyncio.to_thread(store.chat_model, agent_id)
    run_id = guid()
    started = await asyncio.to_thread(store.begin_greeting, session_id, run_id, _cue(workspace))
    for event in started["events"]:
        await bus.publish(event)
    cancel = asyncio.Event()
    app.state.cancels[run_id] = cancel
    try:
        await perform(
            store,
            bus,
            cancel,
            run_id,
            reply=app.state.reply,
            pending=app.state.tasks,
        )
    finally:
        app.state.cancels.pop(run_id, None)
    after = await asyncio.to_thread(store.session_snapshot, session_id)
    failed = next(
        (row for row in reversed(after["runs"]) if row.get("id") == run_id and row.get("status") == "failed"),
        None,
    )
    if failed and failed.get("error") and not any(row.get("role") == "assistant" for row in after["messages"]):
        await _say(app, session_id, str(failed["error"]))


def _cue(workspace: dict[str, Any]) -> str:
    name = str(workspace.get("name") or "this workspace")
    if workspace.get("type") == "gallery":
        return (
            f"The person just created this gallery, named {name}. "
            "The filmstrip already shows the images. "
            "Greet them once in one or two sentences. "
            "Do not call tools. Do not send a second greeting. Do not repeat the greeting."
        )
    return (
        f"The person just created this workspace, named {name}. "
        "Greet them in one or two sentences. Do not call tools."
    )


async def _say(app: Any, session_id: str, text: str) -> None:
    events = await asyncio.to_thread(app.state.store.record_assistant, session_id, text)
    for event in events:
        await app.state.bus.publish(event)

"""Start each enabled heartbeat when its cron minute arrives."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from web_api.runs import perform


def seconds_until_next_minute(moment: datetime) -> float:
    remain = 60 - (moment.second + moment.microsecond / 1_000_000)
    if remain <= 0:
        return 60.0
    return remain


async def watch(app: Any) -> None:
    stop: asyncio.Event = app.state.heartbeat_stop
    while not stop.is_set():
        try:
            await tick(app, datetime.now(UTC))
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed check waits for the next minute.
            pass
        try:
            await asyncio.wait_for(stop.wait(), seconds_until_next_minute(datetime.now()))
        except TimeoutError:
            continue


async def tick(app: Any, moment: datetime) -> None:
    started = await asyncio.to_thread(app.state.store.dispatch_heartbeats, moment)
    bus = app.state.bus
    for result, events in started:
        for event in events:
            await bus.publish(event)
        cancel = asyncio.Event()
        app.state.cancels[result["run_id"]] = cancel
        task = asyncio.create_task(
            perform(
                app.state.store,
                bus,
                cancel,
                result["run_id"],
                app.state.reply,
                app.state.tasks,
            )
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

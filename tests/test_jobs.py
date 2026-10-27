"""Media queue and the pipeline process. No GPU."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from favlo.console import Console
from favlo.pipelines import run_job
from favlo.memory import memory
from favlo.pipeline_worker import ChildExited, PipelineChild, _ProtocolStream
from favlo.reply import MemoryReply


def _selection() -> SimpleNamespace:
    return SimpleNamespace(
        folder="example-model",
        path=Path("example-model"),
        pipeline="example",
        alias=None,
        requested_id="example-model",
        parent="",
        variant=None,
        sampling=None,
        kv_cache="e4m3",
        resident_percent=None,
        max_prompt_tokens=None,
        context_length=131072,
        variants=None,
        default_variant=None,
        runtime="metal41",
        custom=False,
        parts=None,
    )


def test_one_job_runs_at_a_time() -> None:
    order: list[int] = []
    started = threading.Event()
    release = threading.Event()

    def first() -> str:
        order.append(1)
        started.set()
        release.wait(2)
        order.append(2)
        return "first"

    def second() -> str:
        order.append(3)
        return "second"

    first_thread = threading.Thread(target=lambda: run_job("a", first))
    second_thread = threading.Thread(target=lambda: run_job("b", second))
    first_thread.start()
    assert started.wait(2)
    second_thread.start()
    release.set()
    first_thread.join(2)
    second_thread.join(2)
    assert order == [1, 2, 3]


def test_a_crashed_child_drops_its_reported_owners() -> None:
    child = PipelineChild(_selection(), None, stub=True)
    try:
        child.open()
        memory.report(child.counts)
        assert memory.reported().get("stub") == 32
        reply = MemoryReply()
        try:
            child.proxy.handle_image(reply, child, {"crash": True}, None, None, {})
        except ChildExited:
            pass
        else:
            message = "the stub should exit"
            raise AssertionError(message)
    finally:
        child.close()
    assert "stub" not in memory.reported()


def test_a_live_stub_returns_a_result() -> None:
    child = PipelineChild(_selection(), None, stub=True)
    try:
        child.open()
        reply = MemoryReply()
        child.proxy.handle_image(reply, child, {"prompt": "ok"}, None, None, {})
        assert reply.payload == {"status": "completed", "stub": True}
    finally:
        child.close()


def test_console_events_reach_the_daemon_stream() -> None:
    captured = io.BytesIO()

    class _Proto:
        def write(self, blob: bytes) -> None:
            captured.write(blob)

    child_ui = Console(
        "json",
        pipeline="wan22-i2v",
        stream=_ProtocolStream(_Proto()),
        start_thread=False,
    )
    daemon_stream = io.StringIO()
    daemon = Console("json", pipeline="wan22-i2v", stream=daemon_stream, start_thread=False)
    child = PipelineChild(_selection(), None, console=daemon)
    row = child_ui.begin("sample", total=8, unit="blocks", detail="encode")
    row.done = 3
    child_ui.sample()
    child_ui.end(row, "encode done")
    for raw in captured.getvalue().splitlines():
        message = json.loads(raw)
        child._apply_console(message["event"])
    events = [json.loads(line) for line in daemon_stream.getvalue().splitlines() if line]
    assert [event["type"] for event in events] == ["begin", "tick", "end"]
    assert events[1]["done"] == 3
    assert events[1]["phase"] == "sample"
    assert events[2]["summary"] == "encode done"

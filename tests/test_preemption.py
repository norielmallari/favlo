"""Admission seams and media checkpoints. No GPU."""

from __future__ import annotations

import threading
import time

import pytest

from favlo.bulk_media_pipeline import BulkMediaPipeline
from favlo.embeddings_pipeline import EmbeddingsPipeline
from favlo.language_pipeline import LanguagePipeline
from favlo.media_pipeline import MediaCall, MediaPaused, MediaPipeline, checkpoint
from favlo.schedule import Scheduler, disconnect_event, seam_for


def _wait_until(ready, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.01)
    assert ready()


def test_seam_for_maps_http_kinds_onto_the_four_classes() -> None:
    assert seam_for("chat") is LanguagePipeline
    assert seam_for("responses") is LanguagePipeline
    assert seam_for("embed") is EmbeddingsPipeline
    assert seam_for("bulk") is BulkMediaPipeline
    for kind in ("media", "image", "video", "music", "voice"):
        assert seam_for(kind) is MediaPipeline
    with pytest.raises(ValueError, match="unknown queue kind"):
        seam_for("nope")


def test_each_seam_owns_its_admission_flags() -> None:
    assert LanguagePipeline.preempts_media is True
    assert LanguagePipeline.preempts_bulk is True
    assert LanguagePipeline.drop_if_disconnected is True
    assert LanguagePipeline.debounce_after_release is True
    assert LanguagePipeline.resident == "exclusive"
    assert EmbeddingsPipeline.preempts_media is True
    assert EmbeddingsPipeline.preempts_bulk is True
    assert EmbeddingsPipeline.drop_if_disconnected is False
    assert EmbeddingsPipeline.debounce_after_release is False
    assert EmbeddingsPipeline.resident == "beside"
    assert EmbeddingsPipeline.waits_for_memory is False
    assert MediaPipeline.preempts_media is False
    assert MediaPipeline.preempts_bulk is True
    assert MediaPipeline.drop_if_disconnected is False
    assert MediaPipeline.debounce_after_release is False
    assert MediaPipeline.resident == "exclusive"
    assert BulkMediaPipeline.preempts_media is False
    assert BulkMediaPipeline.preempts_bulk is False
    assert BulkMediaPipeline.admission == frozenset({"bulk"})
    assert LanguagePipeline.waits_for_memory is True
    assert MediaPipeline.waits_for_memory is True
    assert BulkMediaPipeline.waits_for_memory is True


def test_disconnect_event_is_only_for_a_language_waiter() -> None:
    flag = threading.Event()
    assert disconnect_event("chat", flag) is flag
    assert disconnect_event("responses", flag) is flag
    assert disconnect_event("embed", flag) is None
    assert disconnect_event("image", flag) is None
    assert disconnect_event("chat", None) is None
    assert disconnect_event("chat", object()) is None
    assert disconnect_event("nope", flag) is None


def test_responses_share_the_chat_queue_and_set_preempt() -> None:
    schedule = Scheduler(chat_debounce=0)
    media = schedule.acquire("media")
    assert schedule.preempt.is_set() is False
    started = threading.Event()

    def wait_responses() -> None:
        ticket = schedule.acquire("responses")
        started.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_responses)
    thread.start()
    _wait_until(lambda: schedule.waiting("responses") == 1)
    assert schedule.preempt.is_set()
    schedule.release(media)
    assert started.wait(2)
    thread.join(2)
    assert schedule.preempt.is_set() is False


def test_a_second_media_ticket_does_not_set_preempt() -> None:
    schedule = Scheduler(chat_debounce=0)
    first = schedule.acquire("media")
    entered = threading.Event()

    def wait_media() -> None:
        ticket = schedule.acquire("media")
        entered.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_media)
    thread.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    assert schedule.preempt.is_set() is False
    schedule.release(first)
    assert entered.wait(2)
    thread.join(2)


def test_bulk_sets_preempt_for_embeddings_chat_and_media() -> None:
    schedule = Scheduler(chat_debounce=0)
    bulk = schedule.acquire("bulk")
    assert schedule.preempt.is_set() is False

    def wait(kind: str, started: threading.Event) -> None:
        ticket = schedule.acquire(kind)
        started.set()
        schedule.release(ticket)

    for kind in ("embed", "chat", "media"):
        started = threading.Event()
        thread = threading.Thread(target=wait, args=(kind, started))
        thread.start()
        _wait_until(lambda k=kind: schedule.waiting(k) == 1)
        assert schedule.preempt.is_set()
        schedule.release(bulk)
        assert started.wait(2)
        thread.join(2)
        assert schedule.preempt.is_set() is False
        bulk = schedule.acquire("bulk")
    schedule.release(bulk)


def test_a_second_bulk_ticket_does_not_set_preempt() -> None:
    schedule = Scheduler(chat_debounce=0)
    first = schedule.acquire("bulk")
    entered = threading.Event()

    def wait_bulk() -> None:
        ticket = schedule.acquire("bulk")
        entered.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_bulk)
    thread.start()
    _wait_until(lambda: schedule.waiting("bulk") == 1)
    assert schedule.preempt.is_set() is False
    schedule.release(first)
    assert entered.wait(2)
    thread.join(2)


def test_an_embedding_sets_preempt_on_interactive_media() -> None:
    schedule = Scheduler(chat_debounce=0)
    media = schedule.acquire("media")
    started = threading.Event()

    def wait_embed() -> None:
        ticket = schedule.acquire("embed")
        started.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_embed)
    thread.start()
    _wait_until(lambda: schedule.waiting("embed") == 1)
    assert schedule.preempt.is_set()
    schedule.release(media)
    assert started.wait(2)
    thread.join(2)
    assert schedule.preempt.is_set() is False


def test_a_second_media_ticket_still_leaves_interactive_media_preempt_clear() -> None:
    schedule = Scheduler(chat_debounce=0)
    media = schedule.acquire("media")
    started = threading.Event()

    def wait_media() -> None:
        ticket = schedule.acquire("media")
        started.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_media)
    thread.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    assert schedule.preempt.is_set() is False
    schedule.release(media)
    assert started.wait(2)
    thread.join(2)


def test_a_chat_ticket_still_sets_preempt_on_interactive_media() -> None:
    schedule = Scheduler(chat_debounce=0)
    media = schedule.acquire("media")
    started = threading.Event()

    def wait_chat() -> None:
        ticket = schedule.acquire("chat")
        started.set()
        schedule.release(ticket)

    thread = threading.Thread(target=wait_chat)
    thread.start()
    _wait_until(lambda: schedule.waiting("chat") == 1)
    assert schedule.preempt.is_set()
    schedule.release(media)
    assert started.wait(2)
    thread.join(2)
    assert schedule.preempt.is_set() is False


def test_media_queue_for_follows_the_handler_class(monkeypatch: pytest.MonkeyPatch) -> None:
    from favlo.pipelines import media_queue_for

    class _Fields:
        kind = "video"
        id_prefix = "video_"
        object_name = "video"
        table_name = "videos"

        def parse(self, body: dict) -> dict:
            return body

        def describe(self, request: dict) -> dict:
            return {}

        def identity(self, request: dict) -> str:
            return ""

        def matches(self, request: dict, identity: object) -> bool:
            return True

        def generate(self, runtime, request, hold, job, stop, call: MediaCall):
            del runtime, request, hold, job, stop, call
            return None

        def pack(self, runtime, request, result, job, home, table, call):
            del runtime, request, result, job, home, table, call
            return None

        def main(self, argv):
            del argv
            return 0

        def open_runtime(self, path, **kwargs):
            del path, kwargs
            return _Runtime()

        def mcp_tools(self):
            return ()

    class _Interactive(_Fields, MediaPipeline):
        pass

    class _Bulk(_Fields, BulkMediaPipeline):
        pass

    interactive = _Interactive()
    bulk = _Bulk()
    monkeypatch.setattr("favlo.pipelines.pipeline_module", lambda selection: object())
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, **kwargs: type("Sel", (), {"pipeline": "example"})(),
    )
    monkeypatch.setattr(
        "favlo.pipelines.handler_for",
        lambda kind, module, handler=None: interactive,
    )
    assert media_queue_for("video", {"model": "example"}) == "media"
    monkeypatch.setattr(
        "favlo.pipelines.handler_for",
        lambda kind, module, handler=None: bulk,
    )
    assert media_queue_for("video", {"model": "example"}) == "bulk"


def test_served_video_handlers_are_bulk_instances() -> None:
    from wan22_i2v.metal41.chat import handle_video as i2v

    assert isinstance(i2v, BulkMediaPipeline)
    assert isinstance(i2v, MediaPipeline)


def test_an_embedding_does_not_set_preempt_or_clear_a_language_debounce() -> None:
    schedule = Scheduler(chat_debounce=1)
    chat = schedule.acquire("chat")
    schedule.release(chat)
    assert schedule.foreground()
    embed = schedule.acquire("embed")
    assert schedule.preempt.is_set() is False
    assert schedule.foreground()
    schedule.release(embed)
    assert schedule.foreground()


def test_a_disconnected_chat_leaves_and_an_embedding_does_not() -> None:
    schedule = Scheduler(chat_debounce=0)
    media = schedule.acquire("media")
    gone = threading.Event()
    gone.set()
    chat_left = threading.Event()

    def wait_chat() -> None:
        from favlo.schedule import ClientClosed

        try:
            schedule.acquire("chat", cancel=gone)
        except ClientClosed:
            chat_left.set()

    chat = threading.Thread(target=wait_chat)
    chat.start()
    assert chat_left.wait(2)
    assert schedule.waiting("chat") == 0
    embed_running = threading.Event()

    def wait_embed() -> None:
        ticket = schedule.acquire("embed", cancel=disconnect_event("embed", gone))
        embed_running.set()
        schedule.release(ticket)

    embed = threading.Thread(target=wait_embed)
    embed.start()
    _wait_until(lambda: schedule.waiting("embed") == 1)
    assert embed_running.is_set() is False
    schedule.release(media)
    assert embed_running.wait(2)
    chat.join(2)
    embed.join(2)


def test_checkpoint_cancel_beats_a_hold_and_a_quiet_stop_returns() -> None:
    with pytest.raises(MediaPaused) as caught:
        checkpoint(lambda: "preempt", {"stage": "step"})
    assert caught.value.reason == "preempt"
    assert caught.value.hold == {"stage": "step"}
    with pytest.raises(MediaPaused) as cancelled:
        checkpoint(lambda: "cancel", {"stage": "step"})
    assert cancelled.value.reason == "cancel"
    assert cancelled.value.hold is None
    checkpoint(lambda: "preempt", {"stage": "step"}, preempt=False)
    with pytest.raises(MediaPaused) as still:
        checkpoint(lambda: "cancel", {"stage": "step"}, preempt=False)
    assert still.value.hold is None
    checkpoint("cancel", {"stage": "step"})
    checkpoint(lambda: "other", {"stage": "step"})


class _Reply:
    def __init__(self) -> None:
        self.status = 0
        self.payload: dict | None = None

    def json(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = payload


class _Runtime:
    def __init__(self) -> None:
        self.closed = 0

    def ensure_open(self, request: object) -> None:
        del request

    def close(self) -> None:
        self.closed += 1

    def generate(self) -> bytes:
        return b"clip"


class _Check(MediaPipeline):
    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"

    def parse(self, body: dict) -> dict:
        return body

    def describe(self, request: dict) -> dict:
        return {"prompt": request.get("prompt", "")}

    def identity(self, request: dict) -> str:
        return str(request.get("prompt", ""))

    def matches(self, request: dict, identity: object) -> bool:
        return identity == self.identity(request)

    def generate(self, runtime, request, hold, job, stop, call: MediaCall):
        del request, call
        if hold == {"stage": "text"}:
            return runtime.generate()
        checkpoint(stop, {"stage": "text"})
        return runtime.generate()

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, result, job, home, table, call
        return None

    def main(self, argv):
        del argv
        return 0

    def open_runtime(self, path, **kwargs):
        del path, kwargs
        return _Runtime()

    def mcp_tools(self):
        return ()


def _state() -> dict:
    return {"preempt": threading.Event(), "cancel": threading.Event()}


def test_a_checkpoint_preempt_stores_the_hold_and_closes() -> None:
    pipe = _Check()
    state = _state()
    state["preempt"].set()
    runtime = _Runtime()
    reply = _Reply()
    pipe(reply, runtime, {"prompt": "move"}, None, None, state)
    assert reply.payload is not None
    assert reply.payload["status"] == "in_progress"
    assert state["video_hold"]["hold"] == {"stage": "text"}
    assert state["video_hold"]["job"]["id"] == reply.payload["id"]
    assert runtime.closed == 1


def test_a_checkpoint_cancel_clears_the_hold_and_closes() -> None:
    pipe = _Check()
    state = _state()
    state["cancel"].set()
    runtime = _Runtime()
    reply = _Reply()
    pipe(reply, runtime, {"prompt": "move"}, None, None, state)
    assert reply.payload is not None
    assert reply.payload["status"] == "in_progress"
    assert state["video_hold"] is None
    assert runtime.closed == 1


def test_a_checkpoint_resume_finishes_the_same_job() -> None:
    pipe = _Check()
    state = _state()
    state["preempt"].set()
    runtime = _Runtime()
    first = _Reply()
    pipe(first, runtime, {"prompt": "move"}, None, None, state)
    assert first.payload is not None
    job_id = first.payload["id"]
    state["preempt"].clear()
    second = _Reply()
    pipe(second, runtime, {"prompt": "move"}, None, None, state)
    assert second.payload is not None
    assert second.payload["id"] == job_id
    assert second.payload["status"] == "completed"
    assert state.get("video_hold") is None
    assert runtime.closed == 2


def test_language_on_open_assigns_the_stop_callable() -> None:
    class _Box:
        def handle_responses(self, reply, runtime, body, selection, args, state) -> None:
            del reply, runtime, body, selection, args, state

        def main(self, argv):
            del argv
            return 0

        def open_runtime(self, path, **kwargs):
            del path, kwargs
            return object()

    class _Qwen(_Box, LanguagePipeline):
        def on_open(self, runtime, reply, selection) -> None:
            super().on_open(runtime, reply, selection)
            runtime._halted = False

    class _Gemma(_Box, LanguagePipeline):
        def on_open(self, runtime, reply, selection) -> None:
            del reply, selection
            runtime._stop = False

    def stopped() -> str | None:
        return None

    qwen_runtime = type("Runtime", (), {})()
    _Qwen().on_open(qwen_runtime, type("Reply", (), {"stopped": staticmethod(stopped)})(), None)
    assert qwen_runtime._stop is stopped
    assert qwen_runtime._halted is False
    gemma_runtime = type("Runtime", (), {})()
    _Gemma().on_open(gemma_runtime, type("Reply", (), {"stopped": staticmethod(stopped)})(), None)
    assert gemma_runtime._stop is False


def test_lazy_embed_open_uses_the_beside_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.pipelines._open_embed_for", lambda body, state, args: "beside")
    from favlo.pipelines import open_for

    class _Args:
        lazy = True

    assert open_for("embed", {}, {}, _Args()) == "beside"


def test_a_chat_call_does_not_unpickle_a_media_hold(tmp_path) -> None:
    import pickle

    from favlo.pipeline_worker import _child_state
    from z_image.metal41.hold import ImageHold

    path = tmp_path / "image_hold.pkl"
    path.write_bytes(pickle.dumps(ImageHold(stage="text")))
    message = {"holds": {"image_hold": str(path)}}
    control = {"preempt": threading.Event(), "cancel": threading.Event()}
    chat = _child_state(message, control, "chat")
    assert "image_hold" not in chat
    image = _child_state(message, control, "image")
    assert image["image_hold"].stage == "text"
    assert path.is_file()


def test_a_hold_export_is_a_unique_file_removed_on_load() -> None:
    from pathlib import Path

    from favlo.pipeline_worker import _child_state, _export_holds
    from z_image.metal41.hold import ImageHold

    state = {"image_hold": ImageHold(stage="text")}
    first = _export_holds(state)
    second = _export_holds(state)
    assert first["image_hold"] != second["image_hold"]
    path = Path(first["image_hold"])
    other = Path(second["image_hold"])
    assert path.name.startswith("favlo-image_hold-")
    assert path.suffix == ".pkl"
    assert path.is_file()
    message = {"holds": first}
    control = {"preempt": threading.Event(), "cancel": threading.Event()}
    image = _child_state(message, control, "image")
    assert image["image_hold"].stage == "text"
    assert not path.exists()
    other.unlink(missing_ok=True)

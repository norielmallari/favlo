"""MediaPipeline listener contract. No GPU."""

from __future__ import annotations

import threading

from favlo.media_pipeline import MediaCall, MediaPaused, MediaPipeline, MediaRequestError


class _Reply:
    def __init__(self) -> None:
        self.status = 0
        self.payload: dict | None = None

    def json(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = payload


class _Runtime:
    def __init__(self, result: object = b"clip", error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.console = None
        self._halted = False
        self.finished = 0

    def generate(self) -> object:
        if self.error is not None:
            raise self.error
        return self.result


class _Pipe(MediaPipeline):
    kind = "video"
    id_prefix = "video_"
    object_name = "video"
    table_name = "videos"
    same = True

    def parse(self, body: dict) -> dict:
        if body.get("bad"):
            raise MediaRequestError("prompt is required", "prompt")
        return body

    def describe(self, request: dict) -> dict:
        return {"prompt": request.get("prompt", "")}

    def identity(self, request: dict) -> str:
        return str(request.get("prompt", ""))

    def matches(self, request: dict, identity: object) -> bool:
        if not self.same:
            return False
        return identity == self.identity(request)

    def generate(self, runtime, request, hold, job, stop, call: MediaCall):
        del request, call
        reason = stop()
        if reason:
            raise MediaPaused(reason, {"stage": "text", "prior": hold})
        if hold == "again":
            reason = stop()
            if reason:
                raise MediaPaused(reason, {"stage": "pack"})
        return runtime.generate()

    def pack(self, runtime, request, result, job, home, table, call):
        del runtime, request, job, home, table, call
        if isinstance(result, dict) and result.get("replace"):
            return {"id": "custom", "status": "completed"}
        return None

    def on_finish(self, runtime, call) -> None:
        del call
        runtime.finished += 1


def _state() -> dict:
    return {"preempt": threading.Event(), "cancel": threading.Event()}


def test_stop_prefers_cancel_and_a_halt() -> None:
    pipe = _Pipe()
    state = _state()
    runtime = _Runtime()
    state["preempt"].set()
    state["cancel"].set()
    assert pipe.stop(state, runtime) == "cancel"
    state["cancel"].clear()
    assert pipe.stop(state, runtime) == "preempt"

    class _Halted(_Pipe):
        def halted(self, runtime) -> bool:
            return bool(getattr(runtime, "_halted", False))

    runtime._halted = True
    state["preempt"].clear()
    assert _Halted().stop(state, runtime) == "cancel"


def test_preempt_stores_the_hold_and_stays_in_progress() -> None:
    pipe = _Pipe()
    state = _state()
    state["preempt"].set()
    reply = _Reply()
    pipe(reply, _Runtime(), {"prompt": "move"}, None, None, state)
    assert reply.status == 200
    assert reply.payload is not None
    assert reply.payload["status"] == "in_progress"
    assert state["video_hold"]["hold"]["stage"] == "text"
    assert state["videos"][reply.payload["id"]] is state["video_hold"]["job"]
    assert reply.payload["prompt"] == "move"


def test_cancel_drops_the_hold_and_stays_in_progress() -> None:
    pipe = _Pipe()
    state = _state()
    state["cancel"].set()
    reply = _Reply()
    pipe(reply, _Runtime(), {"prompt": "move"}, None, None, state)
    assert reply.payload is not None
    assert reply.payload["status"] == "in_progress"
    assert state["video_hold"] is None


def test_a_failed_generate_sets_error_type_and_drops_the_hold() -> None:
    pipe = _Pipe()
    state = _state()
    reply = _Reply()
    pipe(reply, _Runtime(error=RuntimeError("device")), {"prompt": "move"}, None, None, state)
    assert reply.payload is not None
    assert reply.payload["status"] == "failed"
    assert reply.payload["error"] == {"message": "device", "type": "RuntimeError"}
    assert state["video_hold"] is None


def test_a_bad_body_is_a_400_and_stores_no_hold() -> None:
    pipe = _Pipe()
    state = _state()
    reply = _Reply()
    pipe(reply, _Runtime(), {"bad": True}, None, None, state)
    assert reply.status == 400
    assert reply.payload is not None
    assert reply.payload["error"]["type"] == "invalid_request_error"
    assert reply.payload["error"]["param"] == "prompt"
    assert state.get("video_hold") is None


def test_a_mismatch_mints_a_new_id() -> None:
    pipe = _Pipe()
    state = _state()
    state["preempt"].set()
    first = _Reply()
    pipe(first, _Runtime(), {"prompt": "move"}, None, None, state)
    kept = first.payload["id"]
    pipe.same = False
    state["preempt"].clear()
    second = _Reply()
    pipe(second, _Runtime(), {"prompt": "move"}, None, None, state)
    assert second.payload is not None
    assert second.payload["id"] != kept
    assert second.payload["status"] == "completed"
    assert state["video_hold"] is None


def test_pack_can_replace_the_payload() -> None:
    pipe = _Pipe()
    state = _state()
    reply = _Reply()
    pipe(reply, _Runtime(result={"replace": True}), {"prompt": "move"}, None, None, state)
    assert reply.payload == {"id": "custom", "status": "completed"}
    assert state["video_hold"] is None


def test_on_finish_runs_after_success_pause_and_failure() -> None:
    pipe = _Pipe()
    ok = _Runtime()
    pipe(_Reply(), ok, {"prompt": "move"}, None, None, _state())
    paused = _Runtime()
    state = _state()
    state["preempt"].set()
    pipe(_Reply(), paused, {"prompt": "move"}, None, None, state)
    failed = _Runtime(error=RuntimeError("device"))
    pipe(_Reply(), failed, {"prompt": "move"}, None, None, _state())
    assert (ok.finished, paused.finished, failed.finished) == (1, 1, 1)

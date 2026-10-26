"""Listener shells for chat and embeddings. They do not open a device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from favlo.embeddings_pipeline import EmbeddingsPipeline
from favlo.language_pipeline import LanguagePipeline
from favlo.media_pipeline import MediaPipeline
from favlo.pipeline import ClientClosed, Pipeline


class _Reply:
    def __init__(self, stopped: bool = False) -> None:
        self.stopped = stopped
        self.status = 0
        self.payload: dict[str, object] | None = None

    def json(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self.payload = payload


def test_media_pipeline_extends_the_daemon_base() -> None:
    assert issubclass(MediaPipeline, Pipeline)
    assert issubclass(LanguagePipeline, Pipeline)
    assert issubclass(EmbeddingsPipeline, Pipeline)


def test_a_closed_client_clears_stop_and_does_not_fail() -> None:
    notes: list[str] = []

    class Pipe(LanguagePipeline):
        def complete(self, reply, runtime, body, selection, args, state) -> None:
            del reply, body, selection, args, state
            raise ClientClosed()

        def on_disconnect(self, runtime) -> None:
            notes.append("closed")
            super().on_disconnect(runtime)

    runtime = SimpleNamespace(_stop=True, _status=notes.append)
    Pipe().handle(_Reply(), runtime, {}, None, None, {})
    assert runtime._stop is None
    assert notes == ["closed", "client closed"]


def test_an_unrelated_oserror_propagates_and_clears_stop() -> None:
    class Pipe(LanguagePipeline):
        def complete(self, reply, runtime, body, selection, args, state) -> None:
            del reply, runtime, body, selection, args, state
            raise OSError("disk")

    runtime = SimpleNamespace(_stop=True)
    with pytest.raises(OSError, match="disk"):
        Pipe().handle(_Reply(), runtime, {}, None, None, {})
    assert runtime._stop is None


def test_a_named_oserror_can_count_as_the_client() -> None:
    class Pipe(LanguagePipeline):
        def complete(self, reply, runtime, body, selection, args, state) -> None:
            del reply, runtime, body, selection, args, state
            raise OSError("client closed")

        def client_closed(self, exc: OSError) -> bool:
            return str(exc) == "client closed"

    runtime = SimpleNamespace(_stop=True, _status=lambda _message: None)
    Pipe().handle(_Reply(), runtime, {}, None, None, {})
    assert runtime._stop is None


def test_qwen38_27b_on_open_resets_halted() -> None:
    from qwen38_27b.chat import Pipeline as Chat

    runtime = SimpleNamespace(_halted=True, _stop=True)
    Chat().on_open(runtime, _Reply(stopped=False), None)
    assert runtime._halted is False
    assert runtime._stop is False


def test_qwen36_treats_only_the_client_message_as_closed() -> None:
    from qwen36_35b_a3b.chat import Pipeline as Chat

    pipe = Chat()
    assert pipe.client_closed(OSError("client closed"))
    assert not pipe.client_closed(OSError("disk"))


def test_flash_next_treats_any_oserror_as_closed() -> None:
    from qwen38_flash_next.commands import Pipeline as Chat

    assert Chat().client_closed(OSError("disk"))


def test_a_blank_embedding_is_a_400() -> None:
    class Pipe(EmbeddingsPipeline):
        def embed(self, runtime, text):
            del runtime, text
            raise AssertionError("embed")

    reply = _Reply()
    Pipe().handle(reply, None, {"input": "  "}, None, None, {})
    assert reply.status == 400
    error = reply.payload["error"]
    assert error["message"] == "input must be a string"
    assert error["param"] == "input"
    assert error["code"] is None


def test_an_embed_value_error_is_a_400_without_a_param() -> None:
    class Pipe(EmbeddingsPipeline):
        def embed(self, runtime, text):
            del runtime, text
            raise ValueError("too long")

    reply = _Reply()
    Pipe().handle(reply, None, {"input": "hello"}, None, None, {})
    error = reply.payload["error"]
    assert error["message"] == "too long"
    assert error["param"] is None


def test_an_embedding_reply_uses_the_selection_id() -> None:
    class Pipe(EmbeddingsPipeline):
        def embed(self, runtime, text):
            del runtime
            return [float(len(text))], 3

    reply = _Reply()
    selection = SimpleNamespace(requested_id="bge-m3-int8")
    Pipe().handle(reply, None, {"input": "hello", "model": "other"}, selection, None, {})
    assert reply.payload["model"] == "bge-m3-int8"
    assert reply.payload["data"][0]["embedding"] == [5.0]
    assert reply.payload["usage"] == {"prompt_tokens": 3, "total_tokens": 3}


def test_an_embedding_without_a_selection_has_an_empty_model() -> None:
    class Pipe(EmbeddingsPipeline):
        def embed(self, runtime, text):
            del runtime, text
            return [1.0], 2

    reply = _Reply()
    Pipe().handle(reply, None, {"input": "hello", "model": "other"}, None, None, {})
    assert reply.payload["model"] == ""

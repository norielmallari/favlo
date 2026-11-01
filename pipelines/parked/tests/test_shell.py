"""Parked pipeline host checks. Outside the default pytest run."""

from __future__ import annotations

from types import SimpleNamespace

from favlo.console import Progress


def test_flash_next_treats_any_oserror_as_closed() -> None:
    from qwen38_flash_next.commands import Pipeline as Chat

    assert Chat().client_closed(OSError("disk"))


def test_usage_excludes_the_cached_prefix() -> None:
    from qwen38_flash_next import commands

    runtime = SimpleNamespace(_cached_tokens=900, _prefill_seconds=2.0, _decode_seconds=4.0)
    usage = commands._usage(runtime, 1187, 480, commands.DEFAULT_MAX_PROMPT_TOKENS)
    timings = usage["timings"]
    assert usage["prompt_tokens"] == 1187
    assert timings["prompt_n"] == 287
    assert timings["predicted_n"] == 480
    assert timings["prompt_per_second"] == 143.5
    assert timings["predicted_per_second"] == 120.0
    assert usage["max_prompt_tokens"] == commands.DEFAULT_MAX_PROMPT_TOKENS
    assert usage["context_length"] == commands.DEFAULT_CONTEXT_LENGTH
    assert usage["prompt_tokens_details"]["cached_tokens"] == 900
    idle = commands._usage(SimpleNamespace(), 10, 4)
    assert idle["timings"]["prompt_per_second"] == 0.0
    assert idle["timings"]["predicted_per_second"] == 0.0
    assert idle["timings"]["prompt_n"] == 10


def test_prefill_done_only_moves_forward() -> None:
    from qwen38_flash_next.runtime import Qwen38Runtime

    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    runtime.layers = 4
    runtime._prefill_progress = Progress(total=100, done=0)
    Qwen38Runtime._note_prefill(runtime, 0, 50, 100)
    assert runtime._prefill_progress.done == 12
    Qwen38Runtime._note_prefill(runtime, 1, 0, 100)
    assert runtime._prefill_progress.done == 25
    Qwen38Runtime._note_prefill(runtime, 0, 10, 100)
    assert runtime._prefill_progress.done == 25
    Qwen38Runtime._note_prefill(runtime, 3, 100, 100)
    assert runtime._prefill_progress.done == 100

"""Console standard. No device and no checkpoint."""

import io
import json
import time

import pytest

from favlo.console import PHASES, UNITS, VERSION, Console, Progress


class Clock:
    def __init__(self) -> None:
        self.now = 1_760_496_000.0

    def __call__(self) -> float:
        return self.now


def _ui(mode: str, *, width: int = 120, color: int = 0) -> tuple[Console, io.StringIO, Clock]:
    stream = io.StringIO()
    clock = Clock()
    ui = Console(
        mode,
        pipeline="qwen38-flash-next",
        stream=stream,
        clock=clock,
        width=width,
        color=color,
        start_thread=False,
    )
    return ui, stream, clock


def test_a_traceback_is_not_painted_over() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    progress = ui.begin("prefill", total=50, unit="tok")
    progress.done = 3
    ui.sample()
    ui.report("Traceback (most recent call last):\nRuntimeError: boom\n")
    ui.sample()
    text = stream.getvalue()
    assert text.rstrip().endswith("RuntimeError: boom")
    assert "\x1b[" not in text.split("RuntimeError: boom")[-1]


def test_pretty_listening_line_has_no_mark() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.log("listening on http://127.0.0.1:8129")
    assert stream.getvalue() == "listening on http://127.0.0.1:8129\n"


def test_plain_lines_match_the_harness_text() -> None:
    ui, stream, _clock = _ui("plain")
    ui.log("listening on http://127.0.0.1:8129")
    progress = ui.begin("prefill", total=1187, unit="tok", request=7)
    ui.end(progress, "prefill 98.03 tok/s (1187 tok in 12.11s)")
    assert stream.getvalue() == (
        "listening on http://127.0.0.1:8129\nprefill 98.03 tok/s (1187 tok in 12.11s)\n"
    )


def test_pretty_bar_spinner_gauge_and_erase() -> None:
    ui, stream, clock = _ui("pretty", color=0)
    ui.plate()
    assert "F A V L O" in stream.getvalue()
    progress = ui.begin("prefill", total=100, unit="tok")
    progress.done = 0
    ui.sample()
    progress.done = 50
    clock.now += 1
    ui.sample()
    progress.done = 100
    clock.now += 1
    ui.sample()
    text = stream.getvalue()
    assert "0%" in text
    assert "50%" in text
    assert "100%" in text
    assert "50/100 tok" in text
    ui.end(progress, "prefill 50.00 tok/s (50 tok in 1.00s)")
    finished = stream.getvalue()
    assert "prefill 50.00 tok/s (50 tok in 1.00s)" in finished
    assert "\r\x1b[2K" in finished
    open_row = ui.begin("decode", total=0, unit="tok", detail="req #7")
    open_row.done = 214
    open_row.gauge = ("resident", 9700)
    clock.now += 2
    ui.sample()
    drawn = stream.getvalue()
    assert "req #7" in drawn
    assert "resident 9,700" in drawn
    assert any(frame in drawn for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    ui.end(open_row, "decode 11.41 tok/s (214 tok in 18.10s)")


def test_narrow_terminal_drops_the_tail() -> None:
    ui, stream, clock = _ui("pretty", width=24, color=0)
    progress = ui.begin("prefill", total=1000, unit="tok")
    progress.done = 10
    ui.sample()
    clock.now += 1
    progress.done = 20
    ui.sample()
    row = [line for line in stream.getvalue().splitlines() if "prefill" in line][-1]
    visible = row.split("\x1b")[0]
    assert "ETA" not in visible
    assert len(visible) <= 24
    ui.end(progress, "")


def test_palette_falls_back() -> None:
    pretty, stream, _clock = _ui("pretty", color=2)
    pretty.plate()
    assert "\x1b[38;5;75m" in stream.getvalue()
    basic, basic_stream, _clock = _ui("pretty", color=1)
    basic.plate()
    assert "\x1b[1;36m" in basic_stream.getvalue()
    plain, plain_stream, _clock = _ui("pretty", color=0)
    plain.plate()
    assert "\x1b" not in plain_stream.getvalue()


def test_a_daemon_tick_paints_the_row() -> None:
    ui, stream, clock = _ui("pretty", color=0)
    ui.apply_event(
        {"type": "begin", "phase": "load", "total": 100, "unit": "bytes", "detail": "record"}
    )
    ui.apply_event({"type": "tick", "phase": "load", "done": 0, "total": 100, "unit": "bytes"})
    ui.sample()
    clock.now += 1
    ui.apply_event({"type": "tick", "phase": "load", "done": 50, "total": 100, "unit": "bytes"})
    ui.sample()
    assert "50%" in stream.getvalue()
    ui.apply_event({"type": "end", "phase": "load", "summary": "loaded record"})
    assert "loaded record" in stream.getvalue()
    assert ui.shows_bars
    plain, _plain_stream, _clock = _ui("plain")
    assert not plain.shows_bars
    assert Console("json", pipeline="", stream=io.StringIO(), start_thread=False).shows_bars


def test_a_daemon_load_begin_opens_its_own_row() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    local = ui.begin("load", total=0, unit="bytes")
    ui.apply_event(
        {
            "type": "begin",
            "phase": "load",
            "total": 100,
            "unit": "bytes",
            "detail": "record",
            "request": 1,
        }
    )
    assert local.request is None
    assert not local.closed
    assert [item.phase for item in ui._handles] == ["load", "load"]
    assert "load" in stream.getvalue()


def test_a_reply_streams_without_a_second_copy() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    progress = ui.begin("decode", total=10, unit="tok")
    ui.draft("favlo: He")
    ui.draft("favlo: Hello")
    ui.sample()
    ui.end(progress, "decoded 2 tok")
    ui.say("favlo: Hello")
    text = stream.getvalue()
    assert text.count("favlo: He") == 1
    assert "favlo: He\n" not in text
    assert "decoded 2 tok" in text
    assert "\x1b[?7h" in text


def test_a_reply_keeps_the_decode_rate_under_the_text() -> None:
    ui, stream, clock = _ui("pretty", color=0)
    progress = ui.begin("decode", total=0, unit="tok")
    progress.done = 4
    clock.now += 1
    progress.done = 12
    ui.draft("favlo: He")
    ui.sample()
    text = stream.getvalue()
    assert "favlo: He" in text
    assert "tok/s" in text
    assert text.index("favlo: He") < text.rindex("tok/s")
    ui.end(progress, "decoded 12 tok")


def test_plain_plate_is_one_line() -> None:
    ui, stream, _clock = _ui("plain")
    ui.plate()
    assert stream.getvalue() == f"favlo {VERSION}\n"


def test_every_unit_renders() -> None:
    ui, stream, clock = _ui("pretty", color=0)
    for unit in UNITS:
        progress = ui.begin("load", total=10, unit=unit)
        progress.done = 4
        clock.now += 1
        ui.sample()
        ui.end(progress, f"done {unit}")
    text = stream.getvalue()
    for unit in UNITS:
        assert unit in text


def test_json_events_are_self_contained() -> None:
    ui, stream, clock = _ui("json")
    ui.plate()
    progress = ui.begin("prefill", total=1187, unit="tok", request=7, detail="req #7")
    progress.done = 200
    ui.sample()
    progress.done = 452
    progress.gauge = ("resident", 9700)
    clock.now += 1
    ui.sample()
    ui.end(progress, "prefill 98.03 tok/s (1187 tok in 12.11s)")
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    kinds = [row["type"] for row in rows]
    assert kinds[0] == "log"
    assert "begin" in kinds and "tick" in kinds and "end" in kinds
    for row in rows:
        assert row["v"] == 1
        assert row["pipeline"] == "qwen38-flash-next"
    tick = [row for row in rows if row["type"] == "tick"][-1]
    assert tick["total"] == 1187
    assert tick["unit"] == "tok"
    assert tick["request"] == 7
    assert tick["done"] == 452
    assert tick["gauge"] == ["resident", 9700]
    assert tick["eta"] is not None


def test_two_handles_stack_in_begin_order() -> None:
    ui, stream, clock = _ui("json")
    first = ui.begin("prefill", total=10, unit="tok", request=1)
    second = ui.begin("decode", total=0, unit="tok", request=2)
    clock.now += 1
    ui.sample()
    rows = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    ticks = [row for row in rows if row["type"] == "tick"]
    assert [row["phase"] for row in ticks] == ["prefill", "decode"]
    assert [row["request"] for row in ticks] == [1, 2]
    ui.end(first, "prefill done")
    ui.end(second, "decode done")


def test_phase_context_restores_on_error() -> None:
    ui, stream, _clock = _ui("plain")
    with pytest.raises(RuntimeError):
        with ui.phase("quantize", total=4, unit="layers") as progress:
            progress.summary = "quantize stopped"
            raise RuntimeError("stop")
    assert stream.getvalue() == "quantize stopped\n"


def test_unknown_phase_is_rejected() -> None:
    ui, _stream, _clock = _ui("plain")
    with pytest.raises(ValueError, match="phase"):
        ui.begin("warmup", total=1, unit="tok")


def _tty_stream() -> io.StringIO:
    stream = io.StringIO()

    def isatty() -> bool:
        return True

    stream.isatty = isatty  # type: ignore[method-assign]
    return stream


def test_from_env_forces_plain_for_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    ui = Console.from_env("auto", pipeline="qwen38-flash-next", stream=_tty_stream())
    assert ui.mode == "plain"


def test_from_env_forces_plain_for_a_dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    ui = Console.from_env("auto", pipeline="qwen38-flash-next", stream=_tty_stream())
    assert ui.mode == "plain"


def test_from_env_is_pretty_on_a_color_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    ui = Console.from_env("auto", pipeline="qwen38-flash-next", stream=_tty_stream())
    assert ui.mode == "pretty"


@pytest.mark.parametrize("mode", ["plain", "pretty", "json"])
@pytest.mark.parametrize("phase", PHASES)
def test_every_phase_ends_in_every_mode(mode: str, phase: str) -> None:
    ui, stream, clock = _ui(mode, color=0)
    progress = ui.begin(phase, total=10, unit="tok", request=3)
    progress.done = 4
    clock.now += 1
    ui.sample()
    ui.end(progress, f"{phase} done")
    text = stream.getvalue()
    if mode == "plain":
        assert text == f"{phase} done\n"
    elif mode == "json":
        rows = [json.loads(line) for line in text.splitlines()]
        assert {row["phase"] for row in rows} == {phase}
        assert {row["type"] for row in rows} == {"begin", "tick", "end"}
    else:
        assert phase in text
        assert f"{phase} done" in text


def test_prefill_tok_rate_is_the_cumulative_average() -> None:
    ui, stream, clock = _ui("json")
    progress = ui.begin("prefill", total=5000, unit="tok")
    progress.done = 0
    ui.sample()
    clock.now += 1
    progress.done = 1000
    ui.sample()
    clock.now += 1
    progress.done = 1100
    ui.sample()
    clock.now += 0.1
    ui.sample()
    ticks = [json.loads(line) for line in stream.getvalue().splitlines() if '"tick"' in line]
    assert ticks[0]["rate"] is None
    assert ticks[1]["rate"] == pytest.approx(1000)
    assert ticks[2]["rate"] == pytest.approx(1100 / 2)
    assert ticks[3]["rate"] == pytest.approx(1100 / 2.1)
    ui.end(progress, "")


def test_retarget_keeps_a_same_unit_window() -> None:
    ui, _stream, clock = _ui("json")
    progress = ui.begin("quantize", total=48, unit="layers", detail="experts")
    progress.done = 0
    ui.sample()
    clock.now += 1
    progress.done = 4
    ui.sample()
    ui.retarget(progress, total=48, unit="layers", detail="mtp", done=4)
    clock.now += 1
    progress.done = 8
    ui.sample()
    rows = [json.loads(line) for line in _stream.getvalue().splitlines()]
    ticks = [row for row in rows if row["type"] == "tick"]
    assert ticks[-1]["rate"] == pytest.approx(4.0)
    assert ticks[-1]["detail"] == "mtp"


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


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_each_long_phase_updates_before_it_ends() -> None:
    """A phase that is still open keeps painting. The summary is later."""
    for phase in PHASES:
        stream = io.StringIO()
        ui = Console("pretty", pipeline="favlo", stream=stream, color=0, start_thread=True)
        try:
            progress = ui.begin(phase, total=100, unit="tok")
            assert phase in stream.getvalue()
            progress.done = 41

            def painted(raw: io.StringIO = stream) -> bool:
                return "41/100 tok" in raw.getvalue()

            assert _wait_until(painted)
            assert f"{phase} settled" not in stream.getvalue()
            assert not progress.closed
            ui.end(progress, f"{phase} settled")
            assert f"{phase} settled" in stream.getvalue()
        finally:
            ui.close()


def test_json_ticks_while_a_phase_is_still_open() -> None:
    for phase in PHASES:
        stream = io.StringIO()
        ui = Console("json", pipeline="favlo", stream=stream, start_thread=True)
        try:
            progress = ui.begin(phase, total=100, unit="tok")
            progress.done = 41

            def painted(raw: io.StringIO = stream, name: str = phase) -> bool:
                for line in raw.getvalue().splitlines():
                    if '"tick"' not in line:
                        continue
                    row = json.loads(line)
                    if row.get("phase") == name and row.get("done") == 41:
                        return True
                return False

            assert _wait_until(painted)
            assert not progress.closed
        finally:
            ui.close()


def test_a_mirrored_tick_shows_before_the_phase_ends() -> None:
    for phase, unit in (
        ("load", "bytes"),
        ("download", "bytes"),
        ("quantize", "layers"),
        ("prefill", "tok"),
        ("decode", "tok"),
    ):
        stream = io.StringIO()
        ui = Console("pretty", pipeline="", stream=stream, color=0, start_thread=True)
        try:
            ui.apply_event(
                {"type": "begin", "phase": phase, "total": 100, "unit": unit, "detail": phase}
            )
            ui.apply_event(
                {"type": "tick", "phase": phase, "done": 41, "total": 100, "unit": unit}
            )
            def painted(raw: io.StringIO = stream) -> bool:
                return "41" in raw.getvalue()

            assert _wait_until(painted)
            assert f"{phase} settled" not in stream.getvalue()
            ui.apply_event({"type": "end", "phase": phase, "summary": f"{phase} settled"})
            assert f"{phase} settled" in stream.getvalue()
        finally:
            ui.close()


def test_a_reply_shows_the_rate_before_the_phase_ends() -> None:
    for phase in ("prefill", "decode"):
        stream = io.StringIO()
        ui = Console("pretty", pipeline="", stream=stream, color=0, start_thread=True)
        try:
            ui.apply_event({"type": "begin", "phase": phase, "total": 0, "unit": "tok"})
            ui.draft("favlo: He")
            ui.apply_event({"type": "tick", "phase": phase, "done": 41, "total": 0, "unit": "tok"})
            def painted(raw: io.StringIO = stream) -> bool:
                text = raw.getvalue()
                return "favlo: He" in text and "41 tok" in text

            assert _wait_until(painted)
            assert f"{phase} settled" not in stream.getvalue()
            ui.apply_event({"type": "end", "phase": phase, "summary": f"{phase} settled"})
        finally:
            ui.close()


def test_the_load_row_shows_before_the_reply() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.hold(True)
    ui.apply_event(
        {"type": "begin", "phase": "load", "total": 100, "unit": "bytes", "detail": "record"}
    )
    ui.apply_event({"type": "tick", "phase": "load", "done": 41, "total": 100, "unit": "bytes"})
    ui.say("favlo: Hello")
    assert "favlo: Hello" not in stream.getvalue()
    assert "41" in stream.getvalue()
    ui.apply_event({"type": "end", "phase": "load", "summary": "loaded record"})
    ui.hold(False)
    text = stream.getvalue()
    assert text.index("41") < text.index("loaded record")
    assert text.index("loaded record") < text.index("favlo: Hello")


def test_another_clients_request_prints_the_prompt_again() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.prompt()
    ui.apply_event({"type": "log", "line": "request open"})
    ui.apply_event(
        {"type": "begin", "phase": "load", "total": 100, "unit": "bytes", "detail": "record"}
    )
    ui.apply_event({"type": "tick", "phase": "load", "done": 41, "total": 100, "unit": "bytes"})
    ui.apply_event({"type": "end", "phase": "load", "summary": "loaded record"})
    ui.apply_event({"type": "log", "line": "prefix store 10 tokens"})
    assert not stream.getvalue().rstrip().endswith("you:")
    ui.apply_event({"type": "log", "line": "request closed"})
    text = stream.getvalue()
    assert "request open" not in text
    assert "request closed" not in text
    assert text.index("41") < text.index("loaded record")
    assert text.index("prefix store 10 tokens") < text.rindex("you:")
    assert text.rstrip().endswith("you:")


def test_a_closed_request_leaves_an_untouched_prompt() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.prompt()
    ui.apply_event({"type": "log", "line": "request open"})
    ui.apply_event({"type": "log", "line": "request closed"})
    assert stream.getvalue() == "you: "


def test_a_held_turn_does_not_print_the_prompt_when_the_request_closes() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.prompt()
    ui.hold(True)
    ui.apply_event({"type": "log", "line": "request open"})
    ui.apply_event({"type": "begin", "phase": "decode", "total": 0, "unit": "tok"})
    ui.apply_event({"type": "end", "phase": "decode", "summary": "decode settled"})
    ui.apply_event({"type": "log", "line": "request closed"})
    assert stream.getvalue().count("you:") == 1


def test_the_prompt_starts_on_the_line_after_telemetry() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    progress = ui.begin("decode", total=0, unit="tok")
    progress.done = 12
    ui.end(progress, "decode 12.00 tok/s (12 tok in 1.00s)")
    ui.log("prefix store 10 tokens")
    ui.prompt(True)
    text = stream.getvalue()
    assert "prefix store 10 tokens\n" in text
    assert text.index("prefix store 10 tokens") < text.index("you:")
    assert "tokensyou:" not in text


def test_the_frame_waits_until_every_request_closes() -> None:
    import threading

    ui, stream, _clock = _ui("pretty", color=0)
    ui.hold(True)
    ui.note_caught_up()
    ui.apply_event({"type": "log", "line": "request open"})
    ui.apply_event({"type": "log", "line": "prefix store 10 tokens"})
    ui.say("favlo: Hello")
    assert "prefix store 10 tokens" in stream.getvalue()
    assert "favlo: Hello" not in stream.getvalue()

    def finish() -> None:
        time.sleep(0.05)
        ui.apply_event({"type": "log", "line": "request closed"})
        ui.note_caught_up()

    threading.Thread(target=finish).start()
    ui.hold(False)
    ui.prompt(True)
    text = stream.getvalue()
    assert text.index("prefix store 10 tokens") < text.index("favlo: Hello")
    assert text.index("favlo: Hello") < text.index("you:")
    assert "request open" not in text
    assert "request closed" not in text


def test_a_turn_prints_every_request_after_the_last_close() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.hold(True)
    for label in ("prefix store 1 tokens", "prefix store 2 tokens"):
        ui.apply_event({"type": "log", "line": "request open"})
        ui.apply_event({"type": "log", "line": label})
        ui.apply_event({"type": "log", "line": "request closed"})
    ui.say("favlo: Hello")
    assert "prefix store 1 tokens" in stream.getvalue()
    assert "favlo: Hello" not in stream.getvalue()
    ui.hold(False)
    text = stream.getvalue()
    reply_at = text.index("favlo: Hello")
    assert text.index("prefix store 1 tokens") < reply_at
    assert text.index("prefix store 2 tokens") < reply_at


def test_a_wait_line_is_shown_while_the_frame_is_open() -> None:
    ui, stream, _clock = _ui("pretty", color=0)
    ui.hold(True)
    ui.apply_event({"type": "log", "line": "waiting for the open request"})
    assert "waiting for the open request" in stream.getvalue()
    assert "you:" not in stream.getvalue()
    ui.hold(False)


def test_vocabulary_is_closed() -> None:
    assert "prefill" in PHASES
    assert "tok" in UNITS
    progress = Progress()
    assert progress.done == 0

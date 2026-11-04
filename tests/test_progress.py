"""The progress sanitizer. No device and no checkpoint."""

from favlo.progress import Sanitizer


def _sanitizer() -> Sanitizer:
    phases = frozenset({"load", "prefill", "decode"})
    units = frozenset({"tok", "bytes"})
    return Sanitizer(phases, units)


def test_a_count_before_begin_is_dropped() -> None:
    painted = _sanitizer().accept(
        {"type": "tick", "phase": "prefill", "done": 4, "total": 10, "unit": "tok"}
    )
    assert painted == []


def test_a_count_after_end_is_dropped_and_the_summary_remains() -> None:
    ui = _sanitizer()
    assert ui.accept({"type": "begin", "phase": "prefill", "total": 10, "unit": "tok"})
    assert ui.accept({"type": "end", "phase": "prefill", "summary": "prefill settled"})
    assert ui.accept({"type": "tick", "phase": "prefill", "done": 10, "total": 10, "unit": "tok"}) == []
    again = _sanitizer().accept({"type": "end", "phase": "decode", "summary": "decode settled"})
    assert again == [{"type": "log", "line": "decode settled"}]


def test_two_pipelines_keep_separate_rows() -> None:
    ui = _sanitizer()
    ui.accept(
        {"type": "begin", "phase": "load", "total": 10, "unit": "bytes", "source": 1}
    )
    ui.accept(
        {"type": "begin", "phase": "load", "total": 10, "unit": "bytes", "source": 2}
    )
    first = ui.accept(
        {"type": "tick", "phase": "load", "done": 3, "total": 10, "unit": "bytes", "source": 1}
    )
    second = ui.accept(
        {"type": "tick", "phase": "load", "done": 8, "total": 10, "unit": "bytes", "source": 2}
    )
    assert first[0]["done"] == 3
    assert second[0]["done"] == 8
    assert first[0]["source"] == 1
    assert second[0]["source"] == 2


def test_a_pretty_traceback_stays_off_the_console() -> None:
    ui = _sanitizer()
    dropped = ui.accept(
        {"type": "log", "line": "Traceback (most recent call last):"},
        pretty=True,
    )
    kept = ui.accept(
        {"type": "log", "line": "Traceback (most recent call last):"},
        pretty=False,
    )
    assert dropped == []
    assert kept == [{"type": "log", "line": "Traceback (most recent call last):"}]

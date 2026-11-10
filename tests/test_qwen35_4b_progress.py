"""Progress helpers enqueue through favlo.console. No device."""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35_4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from favlo.console import Console  # noqa: E402
from qwen35_4b.metal41 import progress  # noqa: E402


def test_phase_helpers_use_format_rate() -> None:
    stream = io.StringIO()
    ui = Console("plain", pipeline="qwen35_4b", stream=stream, start_thread=False)
    prefill = progress.begin_phase(ui, "prefill", 10)
    progress.note_prefill(prefill, 4)
    assert prefill.done == 4
    progress.end_phase(ui, prefill, progress.rate("prefill", 10, 2.0))
    decode = progress.begin_phase(ui, "decode", 0, live=True)
    assert decode is None  # plain has no live decode spinner
    progress.note(ui, "prompt", tokens=10)
    progress.note(ui, "cache", hit=0)
    text = stream.getvalue()
    assert "prefill 5.00 tok/s (10 tok in 2.00s)" in text
    assert "prompt tokens 10" in text
    assert "cache miss" in text


def test_prefill_chunk_waits_before_the_progress_tick() -> None:
    """A non-blocking prefill end made the live bar track host encode time."""
    path = (
        Path(__file__).resolve().parents[1]
        / "pipelines"
        / "metal41"
        / "qwen35_4b"
        / "qwen35_4b"
        / "forward.py"
    )
    text = path.read_text(encoding="utf-8")
    start = text.index("def _prefill_chunk")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    assert "session.end(wait=True)" in body
    assert "end(wait=False)" not in body

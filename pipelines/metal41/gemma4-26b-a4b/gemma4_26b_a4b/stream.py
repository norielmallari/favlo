"""Decode Gemma 4 tokens away from the sampler.

The sampler only appends token ids. This process turns those ids into text
pieces. The request process writes the pieces as response events.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from tokenizers import Tokenizer

from gemma4_26b_a4b.tools import StreamText

_READY_S = 120.0
_POLL_S = 0.5

Decode = Callable[[list[int]], str]


class TokenStream:
    """A token buffer the sampler fills, and the pieces a stream process returns."""

    def __init__(
        self,
        tokens: Any,
        pieces: Any,
        *,
        proc: Any = None,
        thread: threading.Thread | None = None,
    ) -> None:
        self._tokens = tokens
        self._pieces = pieces
        self._proc = proc
        self._thread = thread

    def put(self, token: int) -> None:
        """Append one token id. This is the sampler's only stream call."""
        self._tokens.put(int(token))

    def finish(self) -> None:
        """Mark the last token. The stream process then flushes its tail."""
        self._tokens.put(None)

    def take(self, timeout: float = _POLL_S) -> tuple[Any, ...] | None:
        """The next piece, or None when the process has finished."""
        return self._pieces.get(timeout=timeout)

    def running(self) -> bool:
        proc = self._proc
        if proc is not None:
            return bool(proc.is_alive())
        thread = self._thread
        return thread is not None and thread.is_alive()

    def close(self) -> None:
        proc = self._proc
        if proc is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)


def launch(root: Path, lead: str, channel_open: bool) -> TokenStream:
    """Start the stream process and wait until its tokenizer is loaded."""
    ctx = multiprocessing.get_context("spawn")
    tokens: Any = ctx.Queue()
    pieces: Any = ctx.Queue()
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_process_main,
        args=(str(root), lead, channel_open, tokens, pieces, child),
        name="favlo-gemma-stream",
        daemon=True,
    )
    proc.start()
    child.close()
    if not parent.poll(_READY_S):
        proc.terminate()
        proc.join(timeout=1)
        message = "gemma stream process did not start"
        raise RuntimeError(message)
    try:
        status = parent.recv()
    except EOFError:
        status = "gemma stream process exited"
    parent.close()
    if status:
        proc.join(timeout=2)
        raise RuntimeError(str(status))
    return TokenStream(tokens, pieces, proc=proc)


def launch_local(
    root: Path,
    lead: str,
    channel_open: bool,
    decode: Callable[[Path, list[int]], str],
) -> TokenStream:
    """Run the same consumer on a thread. Tests use this so a patched decode stays visible."""
    tokens: queue.Queue[int | None] = queue.Queue()
    pieces: queue.Queue[tuple[Any, ...] | None] = queue.Queue()

    def bound(ids: list[int]) -> str:
        return decode(root, ids) if ids else ""

    def run() -> None:
        consume(lead, channel_open, tokens, pieces, bound)

    thread = threading.Thread(target=run, name="favlo-gemma-stream", daemon=True)
    thread.start()
    return TokenStream(tokens, pieces, thread=thread)


def consume(
    lead: str,
    channel_open: bool,
    tokens: Any,
    pieces: Any,
    decode: Decode,
) -> None:
    """Read token ids and push ``("delta", channel, text)`` pieces, then None."""
    try:
        _read(lead, channel_open, tokens, pieces, decode)
    except BaseException as exc:
        pieces.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        pieces.put(None)


def _process_main(
    root: str,
    lead: str,
    channel_open: bool,
    tokens: Any,
    pieces: Any,
    started: Any,
) -> None:
    try:
        tokenizer = Tokenizer.from_file(str(Path(root) / "tokenizer.json"))

        def decode_ids(ids: list[int], _tokenizer: Tokenizer = tokenizer) -> str:
            if not ids:
                return ""
            return _tokenizer.decode(ids, skip_special_tokens=False)

    except BaseException as exc:
        started.send(f"{type(exc).__name__}: {exc}")
        return
    started.send(None)
    consume(lead, channel_open, tokens, pieces, decode_ids)


def _read(
    lead: str,
    channel_open: bool,
    tokens: Any,
    pieces: Any,
    decode: Decode,
) -> None:
    produced: list[int] = []
    shown = ""
    gate = StreamText(channel_open=channel_open)
    while True:
        token = tokens.get()
        final = token is None
        if not final:
            produced.append(int(token))
        decoded = lead + (decode(produced) if produced else "")
        piece, shown = _stream_delta(decoded, shown, final=final)
        for channel, visible in gate.push(piece):
            if visible:
                pieces.put(("delta", channel, visible))
        if final:
            for channel, visible in gate.finish():
                if visible:
                    pieces.put(("delta", channel, visible))
            return


def _stream_delta(decoded: str, shown: str, *, final: bool) -> tuple[str, str]:
    """Text that is safe to send, and the piece after ``shown``.

    The tokenizer writes U+FFFD when a token is only some of the bytes of a
    character. That mark is held back. The next token replaces it.
    """
    if not final and decoded.endswith("\ufffd"):
        decoded = decoded[:-1]
    return decoded[len(shown) :], decoded

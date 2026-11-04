"""The end-to-end rate of one cold prompt.

Prefill seconds and decode seconds are the clocks the listener already
records on the hot path. Their sum is the time a user waits for that
completion. The token count is the uncached prompt plus the sampled tokens.
"""

from __future__ import annotations

from harness.file_session import RequestRate, summed_rate


def cold_prefill(request: RequestRate) -> bool:
    """True when this request did not reuse a stored prefix."""
    if request.prefix_load is not None:
        return False
    if request.missed:
        return request.prefix_hit in (None, 0)
    return request.prefix_hit == 0


def measured_problems(request: RequestRate, tokens: int) -> list[str]:
    """Problems when the server did not prefill exactly ``tokens`` from a miss."""
    problems: list[str] = []
    if request.prompt_tokens != tokens:
        problems.append(f"prompt {request.prompt_tokens} expected {tokens}")
    if not cold_prefill(request):
        problems.append("prefill reused a prefix")
    if not request.prefill or not request.decode:
        problems.append("the request did not finish")
    prefill = summed_rate([request.prefill]) if request.prefill else None
    if prefill is not None and prefill[0] != tokens:
        problems.append(f"prefill counted {prefill[0]} expected {tokens}")
    return problems


def end_to_end(request: RequestRate) -> str:
    """Prompt tokens plus sampled tokens, over prefill time plus decode time."""
    prefill = summed_rate([request.prefill]) if request.prefill else None
    decode = summed_rate([request.decode]) if request.decode else None
    if prefill is None or decode is None:
        return "end to end pending"
    tokens = prefill[0] + decode[0]
    seconds = prefill[1] + decode[1]
    if tokens <= 0 or seconds <= 0:
        return "end to end pending"
    return f"end to end {tokens / seconds:.2f} tok/s ({tokens} tok in {seconds:.2f}s)"

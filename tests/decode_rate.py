"""The billed decode rate a smoke must report."""


def billed_decode(runtime: object) -> float:
    """Tokens divided by the seconds those tokens were billed.

    A smoke fails when decode did not record both a token and a duration.
    """
    tokens = int(runtime._decode_tokens)
    seconds = float(runtime._decode_seconds)
    assert tokens > 0 and seconds > 0, f"decode was not billed ({tokens} tok in {seconds}s)"
    rate = tokens / seconds
    print(f"decode {rate:.2f} tok/s ({tokens} tok in {seconds:.2f}s)")
    return rate

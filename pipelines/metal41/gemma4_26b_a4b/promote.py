"""Choose the routed experts stored as fp8.

The score is the fp4 restoration error. Gate-up and down move together. An
equal score keeps the earlier layer, then the lower expert index. The count
is the largest whole number that does not exceed the percentage.
"""

from __future__ import annotations


def promotion_count(total: int, percent: int) -> int:
    if percent <= 0:
        return 0
    if percent >= 100:
        return total
    return (total * percent) // 100


def select_promoted(
    scores: list[tuple[float, int, int]],
    percent: int,
) -> list[tuple[int, int]]:
    """Return ``(layer, expert)`` pairs, highest score first."""
    count = promotion_count(len(scores), percent)
    ordered = sorted(scores, key=lambda row: (-row[0], row[1], row[2]))
    return [(layer, expert) for _score, layer, expert in ordered[:count]]

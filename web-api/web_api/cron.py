"""Five-field cron expressions and IANA timezones for a heartbeat."""

from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_LIMITS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


class ScheduleError(ValueError):
    pass


def check_cron(expression: str) -> str:
    fields = expression.strip().split()
    if len(fields) != 5:
        raise ScheduleError("A cron expression has five fields.")
    for field, bounds in zip(fields, _LIMITS, strict=True):
        _field(field, bounds[0], bounds[1])
    return " ".join(fields)


def check_timezone(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ScheduleError("A timezone is required.")
    try:
        ZoneInfo(cleaned)
    except ZoneInfoNotFoundError as exc:
        raise ScheduleError("That timezone is not recognized.") from exc
    return cleaned


def _field(field: str, start: int, limit: int) -> None:
    if not field:
        raise ScheduleError("That cron field is not valid.")
    for piece in field.split(","):
        _piece(piece, start, limit)


def _piece(piece: str, start: int, limit: int) -> None:
    step = piece
    if "/" in piece:
        step, count = piece.split("/", 1)
        if not count.isdigit() or int(count) < 1:
            raise ScheduleError("That cron field is not valid.")
    if step == "*":
        return
    if "-" in step:
        left, right = step.split("-", 1)
        low = _number(left, start, limit)
        high = _number(right, start, limit)
        if low > high:
            raise ScheduleError("That cron field is not valid.")
        return
    _number(step, start, limit)


def _number(text: str, start: int, limit: int) -> int:
    if not text.isdigit():
        raise ScheduleError("That cron field is not valid.")
    value = int(text)
    if value < start or value > limit:
        raise ScheduleError("That cron field is not valid.")
    return value

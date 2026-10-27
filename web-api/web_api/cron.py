"""Five-field cron expressions and IANA timezones for a heartbeat."""

from __future__ import annotations

from datetime import datetime
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


def due(expression: str, moment: datetime) -> bool:
    """True when ``moment`` falls in a minute the expression names.

    ``moment`` carries the heartbeat's timezone. A restricted day-of-month and a
    restricted weekday match when either one does. A star leaves that field open.
    """
    if moment.tzinfo is None:
        raise ScheduleError("A heartbeat time needs a timezone.")
    minute, hour, day, month, weekday = expression.split()
    if not _hit(minute, moment.minute, 0, 59):
        return False
    if not _hit(hour, moment.hour, 0, 23):
        return False
    if not _hit(month, moment.month, 1, 12):
        return False
    dom = _hit(day, moment.day, 1, 31)
    cron_dow = moment.isoweekday() % 7
    if cron_dow == 0:
        dow = _hit(weekday, 0, 0, 7) or _hit(weekday, 7, 0, 7)
    else:
        dow = _hit(weekday, cron_dow, 0, 7)
    if day != "*" and weekday != "*":
        return dom or dow
    return dom and dow


def check_timezone(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ScheduleError("A timezone is required.")
    try:
        ZoneInfo(cleaned)
    except ZoneInfoNotFoundError as exc:
        raise ScheduleError("That timezone is not recognized.") from exc
    return cleaned


def _hit(field: str, value: int, start: int, limit: int) -> bool:
    return any(_piece_hit(piece, value, start, limit) for piece in field.split(","))


def _piece_hit(piece: str, value: int, start: int, limit: int) -> bool:
    step = 1
    base = piece
    if "/" in piece:
        base, count = piece.split("/", 1)
        step = int(count)
    if base == "*":
        low, high = start, limit
    elif "-" in base:
        left, right = base.split("-", 1)
        low, high = int(left), int(right)
    else:
        low = int(base)
        high = limit if "/" in piece else low
    if value < low or value > high:
        return False
    return (value - low) % step == 0


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

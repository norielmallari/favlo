"""Match a workspace path against a glob. ``*`` does not cross a slash."""

from __future__ import annotations

import re


def glob_match(relative: str, pattern: str) -> bool:
    """True when ``relative`` matches ``pattern``.

    A pattern with no slash matches the file name in any directory. ``**``
    matches across directories. A leading dot is not special.
    """
    if "/" not in pattern:
        name = relative.rsplit("/", 1)[-1]
        return _full_match(name, pattern)
    return _full_match(relative, pattern)


def _full_match(text: str, pattern: str) -> bool:
    return re.fullmatch(_regex(pattern), text) is not None


def _regex(pattern: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            parts.append(".*")
            index += 2
            continue
        char = pattern[index]
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)

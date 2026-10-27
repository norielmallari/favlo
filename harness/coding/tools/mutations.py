"""Serialize file edits against each other and against a running shell."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class MutationCoordinator:
    """One shell job holds the workspace. File tools wait, then lock their path.

    Status, input, and cancel do not take this lock. A new shell waits until
    file edits and the previous shell have finished, then returns while its
    own command is still running.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._files: set[str] = set()
        self._shells = 0

    @contextmanager
    def file(self, path: Path) -> Iterator[None]:
        key = str(path)
        with self._condition:
            while self._shells or key in self._files:
                self._condition.wait()
            self._files.add(key)
        try:
            yield
        finally:
            with self._condition:
                self._files.discard(key)
                self._condition.notify_all()

    def acquire_shell(self) -> None:
        with self._condition:
            while self._shells or self._files:
                self._condition.wait()
            self._shells += 1

    def release_shell(self) -> None:
        with self._condition:
            if self._shells:
                self._shells -= 1
            self._condition.notify_all()

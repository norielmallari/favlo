"""Open and close gates shared by every media runtime.

``MediaPipeline`` requires the runtime to call these before the first allocate
and after sweeping an owner. Packages do not keep a private reclaim wait.
"""

from __future__ import annotations

from favlo.memory import memory, reclaimable_bytes


def stamp_owner(stem: str, owner: str | None, runtime_id: int) -> str:
    """One owner string for this opening. The listener's mint wins when present."""
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    return f"{stem}:{runtime_id}"


def open_gate(need: int, *, detail: str = "") -> None:
    """Block until prior releases settle, then refuse when ``need`` exceeds room."""
    memory.wait_released()
    room = int(memory.room())
    required = max(0, int(need))
    if required <= room:
        return
    suffix = f" ({detail})" if detail else ""
    message = f"minimum working set does not fit{suffix}: need {required}, room {room}"
    raise RuntimeError(message)


def _owned(owner: str) -> int:
    """Live spans plus bytes this owner reported. Report only bytes that are not spans."""
    label = owner if isinstance(owner, str) else ""
    if not label:
        return 0
    spans = int(memory.owner_bytes().get(label, 0))
    reported = int(memory.reported().get(label, 0))
    return spans + reported


def begin_release(owner: str) -> tuple[int, int]:
    """Snapshot reclaimable pages and this owner's live bytes before a release."""
    return reclaimable_bytes(), _owned(owner)


def confirm_release(owner: str, before: int, held: int) -> None:
    """Wait until the drop in this owner's live bytes is reclaimable.

    Use after a mid-run resident drop that already freed spans. Do not sweep.
    Reported bytes count once the pipeline has cleared that report.
    """
    freed = max(0, int(held) - _owned(owner))
    if freed <= 0:
        return
    if not memory.confirm_released(int(before), freed):
        memory.note_returning(freed)


def finish_release(owner: str, before: int, held: int) -> None:
    """Sweep the owner, then wait until those wired bytes are reclaimable."""
    label = owner if isinstance(owner, str) else ""
    if label:
        memory.sweep(label)
    confirm_release(label, before, held)

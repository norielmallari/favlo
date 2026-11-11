"""Daemon admission queues. No GPU."""

from __future__ import annotations

import threading
import time

from favlo.schedule import ClientClosed, Scheduler


def _run(schedule: Scheduler, kind: str, order: list[str]) -> None:
    ticket = schedule.acquire(kind)
    order.append(kind)
    schedule.release(ticket)


def _park(schedule: Scheduler, kind: str, entered: threading.Event, release: threading.Event) -> None:
    ticket = schedule.acquire(kind)
    entered.set()
    assert release.wait(2)
    schedule.release(ticket)


def _wait_until(ready, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.01)
    assert ready()


def test_an_embedding_does_not_preempt_media_and_runs_before_the_chat() -> None:
    schedule = Scheduler(chat_debounce=0)
    order: list[str] = []
    media = schedule.acquire("media")
    embed = threading.Thread(target=_run, args=(schedule, "embed", order))
    chat = threading.Thread(target=_run, args=(schedule, "chat", order))
    embed.start()
    _wait_until(lambda: schedule.waiting("embed") == 1)
    assert schedule.preempt.is_set() is False
    assert schedule.foreground() is True
    chat.start()
    _wait_until(lambda: schedule.waiting("chat") == 1)
    assert schedule.preempt.is_set() is True
    schedule.release(media)
    embed.join(2)
    chat.join(2)
    assert order == ["embed", "chat"]
    assert schedule.preempt.is_set() is False
    again = schedule.acquire("media")
    schedule.release(again)


def test_each_later_chat_preempts_the_media_slice_it_meets() -> None:
    schedule = Scheduler(chat_debounce=0)
    first = schedule.acquire("media")
    chat = threading.Thread(target=lambda: schedule.release(schedule.acquire("chat")))
    chat.start()
    assert schedule.preempt.wait(2)
    schedule.release(first)
    chat.join(2)
    second = schedule.acquire("media")
    assert schedule.preempt.is_set() is False
    chat_again = threading.Thread(target=lambda: schedule.release(schedule.acquire("chat")))
    chat_again.start()
    assert schedule.preempt.wait(2)
    schedule.release(second)
    chat_again.join(2)
    assert schedule.running() is None


def test_a_disconnected_chat_leaves_the_queue() -> None:
    schedule = Scheduler()
    held = schedule.acquire("chat")
    cancel = threading.Event()
    order: list[str] = []

    def later() -> None:
        try:
            ticket = schedule.acquire("chat", cancel=cancel)
        except ClientClosed:
            order.append("closed")
            return
        order.append("ran")
        schedule.release(ticket)

    worker = threading.Thread(target=later)
    worker.start()
    _wait_until(lambda: schedule.waiting("chat") == 1)
    cancel.set()
    worker.join(2)
    assert order == ["closed"]
    assert schedule.waiting("chat") == 0
    assert worker.is_alive() is False
    schedule.release(held)


def test_a_call_waits_until_the_memory_hold_lifts() -> None:
    schedule = Scheduler()
    order: list[str] = []
    schedule.hold_memory()
    worker = threading.Thread(target=_run, args=(schedule, "chat", order))
    worker.start()
    _wait_until(lambda: schedule.waiting("chat") == 1)
    assert order == []
    schedule.release_memory()
    worker.join(2)
    assert order == ["chat"]
    assert worker.is_alive() is False


def test_an_embedding_does_not_wait_for_a_memory_hold(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr("favlo.listener._log", lines.append)
    schedule = Scheduler()
    order: list[str] = []
    schedule.hold_memory("z_image_turbo")
    worker = threading.Thread(target=_run, args=(schedule, "embed", order))
    worker.start()
    worker.join(2)
    assert order == ["embed"]
    assert worker.is_alive() is False
    assert lines == []
    schedule.release_memory()


def test_waiting_for_memory_names_the_pipeline(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr("favlo.listener._log", lines.append)
    schedule = Scheduler()
    order: list[str] = []
    schedule.hold_memory("z_image_turbo")
    worker = threading.Thread(target=_run, args=(schedule, "chat", order))
    worker.start()
    _wait_until(lambda: schedule.waiting("chat") == 1 and lines)
    assert lines == ["waiting for memory for z_image_turbo"]
    schedule.release_memory()
    worker.join(2)
    assert order == ["chat"]


def test_an_embedding_does_not_preempt_a_running_chat() -> None:
    schedule = Scheduler(chat_debounce=0)
    order: list[str] = []
    chat = schedule.acquire("chat")
    embed = threading.Thread(target=_run, args=(schedule, "embed", order))
    nxt = threading.Thread(target=_run, args=(schedule, "chat", order))
    embed.start()
    nxt.start()
    _wait_until(lambda: schedule.waiting("embed") == 1 and schedule.waiting("chat") == 1)
    assert schedule.preempt.is_set() is False
    assert schedule.running() == "chat"
    schedule.release(chat)
    embed.join(2)
    nxt.join(2)
    assert order == ["embed", "chat"]
    assert schedule.preempt.is_set() is False


def test_waiting_work_runs_embeddings_then_chats_then_media_fifo() -> None:
    schedule = Scheduler()
    order: list[str] = []
    schedule.hold_memory()
    threads = [
        threading.Thread(target=_run, args=(schedule, kind, order))
        for kind in ("media", "chat", "embed", "media", "chat", "embed")
    ]
    for thread in threads:
        thread.start()
    _wait_until(
        lambda: order.count("embed") == 2
        and schedule.waiting("chat") == 2
        and schedule.waiting("media") == 2
    )
    assert schedule.foreground() is True
    assert schedule.preempt.is_set() is False
    assert order == ["embed", "embed"]
    schedule.release_memory()
    for thread in threads:
        thread.join(2)
        assert thread.is_alive() is False
    assert order == ["embed", "embed", "chat", "chat", "media", "media"]


def test_a_second_media_waits_and_does_not_set_preempt() -> None:
    schedule = Scheduler()
    order: list[str] = []
    first_in = threading.Event()
    first_go = threading.Event()
    holder = threading.Thread(target=_park, args=(schedule, "media", first_in, first_go))
    follower = threading.Thread(target=_run, args=(schedule, "media", order))
    holder.start()
    assert first_in.wait(2)
    follower.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    assert schedule.preempt.is_set() is False
    assert schedule.foreground() is False
    assert schedule.running() == "media"
    first_go.set()
    holder.join(2)
    follower.join(2)
    assert order == ["media"]


def test_media_waits_behind_a_chat_without_preempt() -> None:
    schedule = Scheduler(chat_debounce=0)
    order: list[str] = []
    chat = schedule.acquire("chat")
    media = threading.Thread(target=_run, args=(schedule, "media", order))
    media.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    assert schedule.preempt.is_set() is False
    assert schedule.foreground() is True
    schedule.release(chat)
    media.join(2)
    assert order == ["media"]
    assert schedule.running() is None


def test_chat_debounce_keeps_media_from_unloading_the_language_model() -> None:
    """Media must not admit between a chat release and the next chat acquire."""
    schedule = Scheduler(chat_debounce=1.0)
    order: list[str] = []
    first = schedule.acquire("chat")
    media = threading.Thread(target=_run, args=(schedule, "media", order))
    media.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    schedule.release(first)
    assert schedule.foreground() is True
    assert schedule.waiting("media") == 1
    assert order == []
    # The follow-up /v1/responses that carries the tool result wins the gap.
    reply = schedule.acquire("chat")
    assert order == []
    assert schedule.running() == "chat"
    assert schedule.waiting("media") == 1
    schedule.release(reply)
    with schedule._cond:
        schedule._chat_debounce_until = time.monotonic() + 0.05
        schedule._cond.notify_all()
    _wait_until(lambda: order == ["media"], timeout=2.0)
    media.join(2)


def test_chat_debounce_expires_so_media_can_resume() -> None:
    schedule = Scheduler(chat_debounce=0.05)
    order: list[str] = []
    chat = schedule.acquire("chat")
    media = threading.Thread(target=_run, args=(schedule, "media", order))
    media.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    schedule.release(chat)
    assert schedule.foreground() is True
    _wait_until(lambda: order == ["media"], timeout=2.0)
    media.join(2)
    assert schedule.foreground() is False


def test_preempt_clears_when_the_chat_leaves_and_an_embedding_is_still_waiting() -> None:
    schedule = Scheduler()
    media_in = threading.Event()
    media_go = threading.Event()
    embed_in = threading.Event()
    embed_go = threading.Event()
    cancel = threading.Event()
    closed = threading.Event()
    media = threading.Thread(target=_park, args=(schedule, "media", media_in, media_go))
    embed = threading.Thread(target=_park, args=(schedule, "embed", embed_in, embed_go))

    def leave() -> None:
        try:
            schedule.acquire("chat", cancel=cancel)
        except ClientClosed:
            closed.set()

    media.start()
    assert media_in.wait(2)
    embed.start()
    chat = threading.Thread(target=leave)
    chat.start()
    _wait_until(lambda: schedule.waiting("embed") == 1 and schedule.waiting("chat") == 1)
    assert schedule.preempt.is_set() is True
    cancel.set()
    assert closed.wait(2)
    chat.join(2)
    assert schedule.preempt.is_set() is False
    assert schedule.waiting("embed") == 1
    media_go.set()
    assert embed_in.wait(2)
    assert schedule.running() == "embed"
    assert schedule.preempt.is_set() is False
    embed_go.set()
    media.join(2)
    embed.join(2)


def test_preempt_clears_when_the_only_waiting_chat_disconnects() -> None:
    schedule = Scheduler()
    media = schedule.acquire("media")
    cancel = threading.Event()
    closed = threading.Event()

    def leave() -> None:
        try:
            schedule.acquire("chat", cancel=cancel)
        except ClientClosed:
            closed.set()

    chat = threading.Thread(target=leave)
    chat.start()
    assert schedule.preempt.wait(2)
    cancel.set()
    assert closed.wait(2)
    chat.join(2)
    assert schedule.preempt.is_set() is False
    assert schedule.running() == "media"
    assert schedule.foreground() is False
    schedule.release(media)


def test_a_disconnected_chat_does_not_block_the_chat_behind_it() -> None:
    schedule = Scheduler()
    cancel = threading.Event()
    order: list[str] = []
    schedule.hold_memory()

    def gone() -> None:
        try:
            schedule.acquire("chat", cancel=cancel)
        except ClientClosed:
            order.append("closed")

    def kept() -> None:
        ticket = schedule.acquire("chat")
        order.append("kept")
        schedule.release(ticket)

    first = threading.Thread(target=gone)
    second = threading.Thread(target=kept)
    first.start()
    _wait_until(lambda: schedule.waiting("chat") == 1)
    second.start()
    _wait_until(lambda: schedule.waiting("chat") == 2)
    cancel.set()
    _wait_until(lambda: schedule.waiting("chat") == 1 and order == ["closed"])
    schedule.release_memory()
    first.join(2)
    second.join(2)
    assert order == ["closed", "kept"]
    assert schedule.running() is None


def test_memory_hold_is_a_count() -> None:
    schedule = Scheduler()
    order: list[str] = []
    schedule.hold_memory()
    schedule.hold_memory()
    worker = threading.Thread(target=_run, args=(schedule, "media", order))
    worker.start()
    _wait_until(lambda: schedule.waiting("media") == 1)
    schedule.release_memory()
    time.sleep(0.05)
    assert order == []
    assert schedule.running() is None
    schedule.release_memory()
    worker.join(2)
    assert order == ["media"]
    assert worker.is_alive() is False


def test_an_unknown_kind_does_not_enter_the_queue() -> None:
    schedule = Scheduler()
    try:
        schedule.acquire("video")
    except ValueError:
        refused = True
    else:
        refused = False
    assert refused
    assert schedule.running() is None
    assert schedule.foreground() is False


def test_flush_drops_every_waiter_and_the_running_request_stays() -> None:
    schedule = Scheduler()
    media = schedule.acquire("media")
    closed: list[str] = []

    def wait(kind: str) -> None:
        try:
            schedule.acquire(kind)
        except ClientClosed:
            closed.append(kind)

    threads = [threading.Thread(target=wait, args=(kind,)) for kind in ("embed", "chat", "media")]
    for thread in threads:
        thread.start()
    _wait_until(
        lambda: schedule.waiting("embed") == 1
        and schedule.waiting("chat") == 1
        and schedule.waiting("media") == 1
    )
    assert schedule.preempt.is_set() is True
    epoch = schedule.epoch
    schedule.flush()
    for thread in threads:
        thread.join(2)
    assert sorted(closed) == ["chat", "embed", "media"]
    assert schedule.waiting("embed") == schedule.waiting("chat") == schedule.waiting("media") == 0
    assert schedule.preempt.is_set() is False
    assert schedule.running() == "media"
    assert schedule.epoch == epoch + 1
    schedule.release(media)
    chat = schedule.acquire("chat")
    assert schedule.running() == "chat"
    schedule.release(chat)


def test_a_second_release_leaves_the_queue_idle() -> None:
    schedule = Scheduler()
    ticket = schedule.acquire("chat")
    schedule.release(ticket)
    schedule.release(ticket)
    assert schedule.running() is None
    media = schedule.acquire("media")
    assert schedule.running() == "media"
    schedule.release(media)

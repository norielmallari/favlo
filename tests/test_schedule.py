"""Daemon admission queues. No GPU."""

from __future__ import annotations

import threading
import time

from favlo.schedule import ClientClosed, Scheduler


def _run(schedule: Scheduler, kind: str, order: list[str]) -> None:
    ticket = schedule.acquire(kind)
    order.append(kind)
    schedule.release(ticket)


def _wait_until(ready) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.01)
    assert ready()


def test_an_embedding_preempts_media_and_runs_before_the_chat() -> None:
    schedule = Scheduler()
    order: list[str] = []
    media = schedule.acquire("media")
    embed = threading.Thread(target=_run, args=(schedule, "embed", order))
    chat = threading.Thread(target=_run, args=(schedule, "chat", order))
    embed.start()
    assert schedule.preempt.wait(2)
    chat.start()
    _wait_until(lambda: schedule.waiting("embed") == 1 and schedule.waiting("chat") == 1)
    schedule.release(media)
    embed.join(2)
    chat.join(2)
    assert order == ["embed", "chat"]
    assert schedule.preempt.is_set() is False
    again = schedule.acquire("media")
    schedule.release(again)


def test_each_later_chat_preempts_the_media_slice_it_meets() -> None:
    schedule = Scheduler()
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


def test_an_embedding_does_not_preempt_a_running_chat() -> None:
    schedule = Scheduler()
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

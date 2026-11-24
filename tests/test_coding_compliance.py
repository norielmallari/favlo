"""Chat-contract rules for the coding harness. No listener."""

from __future__ import annotations

import asyncio
import json

from agent_framework import Content, Message
from harness.coding.compliance import (
    CODE,
    LABEL,
    STOP_WORD,
    TEXT_EFFORTS,
    WORD,
    ToolRound,
    choice_problems,
    finish_problem,
    image_content,
    image_floor,
    image_message,
    image_turn_problems,
    length_problems,
    raw_closed_problems,
    raw_reply_problems,
    reasoning_level_problems,
    reasoning_problems,
    recall_problems,
    reply_problems,
    search_calls,
    stop_problems,
    stored_reasoning,
    stream_problems,
    tool_problems,
    tool_rounds,
    tool_stream_problems,
    usage_problems,
    vision_problems,
    web_search_problems,
)
from harness.image_attachment import QUESTION


def test_a_reply_keeps_markup_and_reasoning_out() -> None:
    assert reply_problems("Apple.", "") == []
    assert any(WORD in item for item in reply_problems("pear", ""))
    assert any("think" in item for item in reply_problems("apple <think>hidden</think>", ""))
    assert any("repeats" in item for item in reply_problems("look", "look"))


def test_every_text_effort_is_a_reasoning_check() -> None:
    assert TEXT_EFFORTS == ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert reasoning_level_problems("oats", "", stored="", effort="none") == []
    assert reasoning_level_problems("oats", "", stored="", effort="minimal") == []
    missing = reasoning_level_problems("oats", "", stored="", effort="max")
    assert any("no reasoning" in item for item in missing)


def test_reasoning_must_stay_on_the_session() -> None:
    assert reasoning_problems("Oats.", "look", stored="look") == []
    problems = reasoning_problems("Oats.", "", stored="")
    assert any("no reasoning" in item for item in problems)
    assert any("did not store" in item for item in problems)


def test_a_stored_turn_keeps_reasoning_off_the_reply() -> None:
    message = Message("assistant", [Content.from_text_reasoning(text="look"), "Oats."])
    session = type("Session", (), {"state": {"in_memory": {"messages": [message]}}})()
    assert stored_reasoning(session) == "look"


def test_the_tool_result_is_bound_to_the_call_id() -> None:
    arguments = f'{{"code": "{CODE}"}}'
    call = Content.from_function_call(call_id="c1", name="locker", arguments=arguments)
    result = Content.from_function_result(call_id="c1", result=LABEL)
    rounds = tool_rounds([Message("assistant", [call]), Message("tool", [result])])
    assert rounds == [ToolRound("c1", CODE, LABEL)]
    assert tool_problems(rounds, LABEL) == []
    unbound = Content.from_function_result(call_id="other", result=LABEL)
    missed = tool_rounds([Message("assistant", [call]), Message("tool", [unbound])])
    assert missed[0].result is None
    assert any("call id" in item for item in tool_problems(missed, LABEL))
    assert tool_problems([], "done")


def test_web_search_requires_the_server_call() -> None:
    call = Content.from_search_tool_call(call_id="ws_1", tool_name="web_search", arguments={})
    response = type("Response", (), {"messages": [Message("assistant", [call, "The card is public."])]})()
    assert search_calls(response) == ["ws_1"]
    assert web_search_problems("The card is public.", ["ws_1"]) == []
    assert any("no web_search" in item for item in web_search_problems("The card is public.", []))


def test_a_stored_follow_up_keeps_the_secret_word() -> None:
    assert recall_problems("apple") == []
    assert any(WORD in item for item in recall_problems("pear"))


def test_tool_choice_none_rejects_a_call() -> None:
    call = ToolRound("c1", CODE, LABEL)
    assert choice_problems([], "Apple.", "") == []
    assert any("tool_choice" in item for item in choice_problems([call], "Apple.", ""))


def test_a_thinking_stream_shows_the_tool_name_then_the_arguments() -> None:
    held = tool_stream_problems(
        saw_reasoning=True,
        name="locker",
        pieces=['{"code": "pebble-7"', "}", '{"code": "pebble-7"}'],
    )
    assert held == []
    late = tool_stream_problems(
        saw_reasoning=False,
        name="locker",
        pieces=['{"code": "pebble-7"}'],
    )
    assert any("reasoning" in item for item in late)
    assert any("one piece" in item for item in late)


def test_a_raw_responses_stream_needs_more_than_one_piece() -> None:
    assert raw_reply_problems(deltas=["one ", "two"], text="one two") == []
    assert any("did not stream" in item for item in raw_reply_problems(deltas=["one two"], text="one two"))
    assert any("one piece" in item for item in raw_reply_problems(deltas=["one two", ""], text="one two"))
    closed = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "one two"}]}]}
    assert raw_closed_problems(closed) == []
    assert any("one object" in item for item in raw_closed_problems("data: {}"))
    assert any("empty" in item for item in raw_closed_problems({"status": "completed", "output": []}))


def test_length_and_stop_replies() -> None:
    assert length_problems("A") == []
    assert length_problems("")
    assert stop_problems(f"alpha {STOP_WORD} omega")
    assert stop_problems("alpha ") == []
    assert stream_problems(["App", "le."], "Apple.") == []
    assert stream_problems([], "Apple.")


def test_usage_is_prompt_plus_completion() -> None:
    held = {"input_token_count": 3, "output_token_count": 2, "total_token_count": 5}
    assert usage_problems(held) == []
    assert usage_problems(None)
    assert any("total_tokens" in item for item in usage_problems({
        "input_token_count": 3,
        "output_token_count": 2,
        "total_token_count": 9,
    }))


def test_the_vision_tower_runs_once() -> None:
    held = (
        "vision image 1 patches 64 grid 8x8 merged 16\n"
        "vision patches 64\n"
        "vision merged 16\n"
        "vision cached 16\n"
    )
    assert vision_problems(held) == []
    assert vision_problems(held + "vision cache 16\n") == []
    assert vision_problems("vision cache 16\n") == []
    assert vision_problems("prompt tokens 1\n")
    twice = "vision patches 64\nvision patches 64\nvision cached 16\n"
    assert any("ran 2" in item for item in vision_problems(twice))
    early = "vision cache 16\nvision patches 64\n"
    assert any("preceded" in item for item in vision_problems(early))
    covered = "vision cached 16\nvision patches 64\nvision cache 16\n"
    assert vision_problems(covered) == []


def test_the_image_turn_covers_the_pooled_tokens() -> None:
    floor = image_floor(1280, 1280)
    assert floor == 256
    assert floor < 1600
    small = image_floor(32, 32)
    assert image_turn_problems("A chestnut horse with a dark mane is galloping.", small, 32, 32) == []
    assert image_turn_problems("a horse", 0, 32, 32)
    assert any("horse" in item for item in image_turn_problems("a desk", 100, 32, 32))


def test_finish_reason_must_match() -> None:
    assert finish_problem("stop", "stop") is None
    assert finish_problem("tool_calls", "stop") is not None
    assert finish_problem("length", "length") is None


def test_the_image_part_is_an_openai_data_url() -> None:
    from favlo.chat_client import close_chat_client, open_chat_client

    picture = image_content(b"\x89PNG\r\n\x1a\n")
    message = image_message(b"\x89PNG\r\n\x1a\n")
    assert message.contents[0].text == QUESTION
    client = open_chat_client("example-model", "http://127.0.0.1:8129/v1")
    try:
        part = client._prepare_content_for_openai("user", picture)
    finally:
        asyncio.run(close_chat_client(client))
    assert part["type"] == "input_image"
    assert part["detail"] == "auto"
    assert part["image_url"].startswith("data:image/png;base64,")


def test_chat_roles_and_tool_calls_match_the_completions_object() -> None:
    from favlo.chat_client import close_chat_client, open_chat_client

    call = Content.from_function_call(call_id="c1", name="locker", arguments={"code": CODE})
    result = Content.from_function_result(call_id="c1", result=LABEL)
    client = open_chat_client("example-model", "http://127.0.0.1:8129/v1")
    try:
        developer = client._prepare_message_for_openai(Message("developer", "The reply is apple."))
        assistant = client._prepare_message_for_openai(
            Message("assistant", [call]),
            request_uses_service_side_storage=False,
        )
        tool = client._prepare_message_for_openai(
            Message("tool", [result]),
            request_uses_service_side_storage=False,
        )
    finally:
        asyncio.run(close_chat_client(client))
    assert developer == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "The reply is apple."}],
        }
    ]
    assert assistant[0]["type"] == "function_call"
    assert assistant[0]["name"] == "locker"
    assert assistant[0]["call_id"] == "c1"
    assert tool[0]["type"] == "function_call_output"
    assert tool[0]["call_id"] == "c1"
    assert LABEL in str(tool[0].get("output"))

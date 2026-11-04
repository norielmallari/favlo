"""The listener rebuilds one transcript before the pipeline encodes."""

from __future__ import annotations

from favlo.response_chain import finish_turn, joined_input, present, remember


def test_a_stored_chain_includes_reasoning() -> None:
    state: dict[str, object] = {}
    response = {
        "id": "resp_one",
        "object": "response",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_one",
                "content": [{"type": "reasoning_text", "text": "think"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        ],
    }
    remember(state, {"input": "Hi", "store": True}, response)
    presented, turn = present(
        {"input": "Next", "previous_response_id": "resp_one", "store": True},
        state,
    )
    assert turn is not None
    assert "store" not in presented
    assert "previous_response_id" not in presented
    kinds = [item.get("type") for item in presented["input"]]
    assert kinds == ["message", "reasoning", "message", "message"]


def test_a_missing_response_id_starts_a_new_chain() -> None:
    presented, turn = present(
        {"input": "Next", "previous_response_id": "resp_gone", "store": True},
        {},
    )
    assert turn is not None
    assert presented["input"] == "Next"
    assert "previous_response_id" not in presented


def test_store_false_leaves_the_body_alone() -> None:
    state: dict[str, object] = {}
    body = {"input": "Next", "previous_response_id": "resp_one", "store": False}
    presented, turn = present(body, state)
    assert turn is None
    assert presented is body
    assert joined_input({"input": "Next", "previous_response_id": "resp_one"}, state) == "Next"


def test_a_replayed_image_stays_once_and_a_tool_result_stays() -> None:
    image = {
        "type": "input_image",
        "image_url": "data:image/png;base64,aaaa",
        "detail": "auto",
    }
    user = {
        "type": "message",
        "role": "user",
        "content": [image, {"type": "input_text", "text": "What is this?"}],
    }
    assistant = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "A horse"}],
    }
    state: dict[str, object] = {}
    remember(
        state,
        {"input": [user], "store": True},
        {"id": "resp_one", "object": "response", "output": [assistant]},
    )
    result = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "6",
    }
    replay = [user, assistant, result, {"type": "message", "role": "user", "content": "Next"}]
    presented, _turn = present(
        {"input": replay, "previous_response_id": "resp_one", "store": True},
        state,
    )
    images = [
        part
        for item in presented["input"]
        if isinstance(item, dict)
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    assert len(images) == 1
    kinds = [item.get("type") for item in presented["input"]]
    assert kinds[-2:] == ["function_call_output", "message"]


def test_a_repeated_tool_result_stays() -> None:
    result = {
        "type": "function_call_output",
        "call_id": "call_0",
        "output": "Error: unexpected argument status",
    }
    state: dict[str, object] = {}
    remember(
        state,
        {"input": ["Hi", result], "store": True},
        {"id": "resp_one", "object": "response", "output": []},
    )
    presented, _turn = present(
        {
            "input": [result],
            "previous_response_id": "resp_one",
            "store": True,
        },
        state,
    )
    outputs = [
        item
        for item in presented["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 2


def test_finish_turn_keeps_the_response_the_pipeline_wrote() -> None:
    state: dict[str, object] = {}
    _presented, turn = present({"input": "Hi", "store": True}, state)

    class _Reply:
        payload = {"id": "resp_one", "object": "response", "output": []}

    finish_turn(turn, _Reply(), state)
    again, _turn = present(
        {"input": "Next", "previous_response_id": "resp_one", "store": True},
        state,
    )
    texts = [item.get("content") for item in again["input"]]
    assert "Hi" in texts
    assert "Next" in texts

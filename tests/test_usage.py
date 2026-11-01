"""Chat completion results carry the OpenAI token fields."""

from favlo.reply import MemoryReply
from favlo.usage import complete_chat_usage


def test_a_chat_completion_usage_includes_the_standard_fields() -> None:
    payload = {
        "object": "chat.completion",
        "model": "example-model",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "context_length": 131072,
            "prompt_tokens_details": {"cached_tokens": 3},
            "timings": {"prompt_n": 7},
        },
    }
    complete_chat_usage(payload)
    usage = payload["usage"]
    assert usage["max_prompt_tokens"] == 131072
    assert usage["prompt_tokens_details"] == {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 3,
        "image_tokens": 0,
        "text_tokens": 10,
    }
    assert usage["completion_tokens_details"] == {
        "accepted_prediction_tokens": 0,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
        "text_tokens": 4,
    }
    assert usage["timings"] == {"prompt_n": 7}
    assert usage["context_length"] == 131072


def test_a_pipeline_cap_wins_over_the_context_length() -> None:
    payload = {
        "object": "chat.completion.chunk",
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
            "context_length": 131072,
            "max_prompt_tokens": 4096,
        },
    }
    complete_chat_usage(payload)
    assert payload["usage"]["max_prompt_tokens"] == 4096


def test_a_reply_fills_usage_before_it_is_recorded() -> None:
    reply = MemoryReply("example-model")
    reply.json(
        200,
        {
            "object": "chat.completion",
            "model": "example-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    assert "max_prompt_tokens" not in reply.body["usage"]
    assert reply.body["usage"]["prompt_tokens_details"]["text_tokens"] == 1
    reply.event(
        {
            "object": "chat.completion.chunk",
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "context_length": 1024,
            },
        }
    )
    assert reply.events[0]["usage"]["max_prompt_tokens"] == 1024


def test_other_objects_keep_their_body() -> None:
    payload = {"object": "list", "data": []}
    complete_chat_usage(payload)
    assert payload == {"object": "list", "data": []}

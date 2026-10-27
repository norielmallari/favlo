"""The shared chat client sends reasoning back on the next request."""

from favlo.chat_client import history_message, prepare_messages, read_reasoning


class _Part:
    def __init__(self, kind: str, text: str) -> None:
        self.type = kind
        self.text = text
        self.protected_data = None


class _Message:
    def __init__(self, contents: list[_Part]) -> None:
        self.role = "assistant"
        self.contents = contents


def test_reasoning_returns_on_its_own_field() -> None:
    message = _Message(
        [
            _Part("text_reasoning", "look\n"),
            _Part("text", "<tool_call>\n<function=read_file>\n"),
        ]
    )
    built = [
        {"role": "assistant", "content": [{"type": "text", "text": "look\n"}]},
        {
            "role": "assistant",
            "content": "<tool_call>\n<function=read_file>\n",
            "tool_calls": [{"id": "call_0"}],
        },
    ]
    prepared = prepare_messages(message, built)
    assert prepared == [
        {
            "role": "assistant",
            "content": "<tool_call>\n<function=read_file>\n",
            "reasoning_content": "look\n",
        }
    ]


def test_reasoning_delta_is_not_reply_text() -> None:
    class _Delta:
        model_extra = {"reasoning_content": "look"}

    parsed = read_reasoning(_Delta(), [])
    assert parsed[0].type == "text_reasoning"
    assert parsed[0].text == "look"


def test_stored_reasoning_returns_on_the_next_request() -> None:
    message = history_message("assistant", "Hi", "look\n")
    prepared = prepare_messages(message, [{"role": "assistant", "content": "Hi"}])
    assert prepared == [
        {"role": "assistant", "content": "Hi", "reasoning_content": "look\n"}
    ]

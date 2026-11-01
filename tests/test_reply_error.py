"""A failed reply is stored as a sentence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web-api"))

from web_api.reply_error import spoken_error  # noqa: E402

RAW = (
    "(\"<class 'agent_framework_openai._chat_completion_client.OpenAIChatCompletionClient'>"
    " service failed to complete the prompt: Error code: 503 - "
    "{'error': {'message': 'the gpu is held for a pipeline smoke', 'type': 'busy'}}\", "
    "InternalServerError(\"Error code: 503 - "
    "{'error': {'message': 'the gpu is held for a pipeline smoke', 'type': 'busy'}}\"))"
)


def test_a_smoke_hold_is_a_sentence() -> None:
    assert spoken_error(RAW) == (
        "The GPU is held for a pipeline smoke. Send again when it is free."
    )


def test_a_plain_failure_stays_as_written() -> None:
    assert spoken_error("The model returned an empty reply.") == (
        "The model returned an empty reply."
    )

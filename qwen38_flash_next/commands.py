"""quantize, bf16, complete, and serve."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.checks import check_state_kernels
from qwen38_flash_next.convert import importance_hot_ids, quantize_checkpoint
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.grove import write_grove
from qwen38_flash_next.probe import matmul, probe
from qwen38_flash_next.runtime import Qwen38Runtime
from qwen38_flash_next.session import _append_transcript, _log, _prompt_text
from qwen38_flash_next.tools import (
    PENALTY_WINDOW,
    ContentGate,
    ThinkStream,
    call_closes,
    parse_tool_calls,
    required_function,
    split_thinking,
    tool_names,
)


def _checkpoint(path: Path | None) -> Path:
    if path is None:
        message = "pass --source with the checkpoint directory"
        raise FileNotFoundError(message)
    if not (path / "config.json").is_file():
        message = f"checkpoint config is missing: {path}"
        raise FileNotFoundError(message)
    return path


def _record(path: Path | None) -> Path:
    return path or DEFAULT_RECORD


def quantize_main(args: Any) -> int:
    try:
        source = _checkpoint(args.source)
        hot = args.hot
        if args.importance is not None:
            if hot is not None:
                message = "pass either --hot or --importance"
                raise RuntimeError(message)
            config = json.loads((source / "config.json").read_text())
            text = config["text_config"]
            graph = json.loads(args.importance.read_text())
            chosen = importance_hot_ids(
                graph,
                len(text["layer_types"]),
                int(text["num_experts"]),
                int(args.hot_count),
            )
            args.output.mkdir(parents=True, exist_ok=True)
            hot = args.output / "hot-ids.json"
            hot.write_text(json.dumps({str(layer): row for layer, row in enumerate(chosen)}))
        packed = quantize_checkpoint(
            source,
            args.output,
            hot,
            args.super,
            hot_bits=int(args.hot_bits),
            cold_bits=int(args.cold_bits),
        )
        write_grove(args.output, replace=packed)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def bf16_complete(args: Any) -> int:
    """Same forward as complete, with Metal dots of the checkpoint BF16 values."""
    try:
        runtime = Qwen38Runtime(
            _checkpoint(args.source), bf16=True, kv_cache=args.kv_cache
        )
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=args.thinking)
        _log(f"prompt tokens {len(prompt)} ids {prompt}")
        logits = runtime.forward(prompt)
        order = np.argsort(logits)[::-1]
        top = " ".join(
            f"{float(logits[i]):.2f}:{int(i)}:{runtime.decode([int(i)])!r}" for i in order[:8]
        )
        _log(f"top {top}")
        four = runtime.tokenizer.encode("4").ids
        if len(four) == 1:
            rank = int(np.flatnonzero(order == four[0])[0])
            _log(f"4 rank {rank} logit {float(logits[four[0]]):.2f}")
        produced: list[int] = []
        shown = ""
        temperature, top_p, top_k = _cli_sampling(args)
        for token in runtime._new_tokens(logits, args.max_tokens, temperature, top_p, top_k):
            produced.append(token)
            text = runtime.decode(produced)
            sys.stdout.write(text[len(shown) :])
            sys.stdout.flush()
            shown = text
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _append_transcript(args.transcript, args.prompt, shown)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def complete(args: Any) -> int:
    try:
        runtime = Qwen38Runtime(
            _record(args.record),
            resident_percent=args.resident_percent,
            kv_cache=args.kv_cache,
        )
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=args.thinking)
        _log(f"prompt tokens {len(prompt)} ids {prompt}")
        logits = runtime.forward(prompt)
        order = np.argsort(logits)[::-1]
        top = " ".join(
            f"{float(logits[i]):.2f}:{int(i)}:{runtime.decode([int(i)])!r}" for i in order[:8]
        )
        _log(f"top {top}")
        four = runtime.tokenizer.encode("4").ids
        if len(four) == 1:
            rank = int(np.flatnonzero(order == four[0])[0])
            _log(f"4 rank {rank} logit {float(logits[four[0]]):.2f}")
        produced: list[int] = []
        shown = ""
        temperature, top_p, top_k = _cli_sampling(args)
        for token in runtime._new_tokens(logits, args.max_tokens, temperature, top_p, top_k):
            produced.append(token)
            text = runtime.decode(produced)
            sys.stdout.write(text[len(shown) :])
            sys.stdout.flush()
            shown = text
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _append_transcript(args.transcript, args.prompt, shown)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


class _SingleRequestServer(HTTPServer):
    """Accept the next connection only after the current request returns."""

    request_queue_size = 128


def serve(args: Any) -> int:
    try:
        runtime = Qwen38Runtime(
            _record(args.record),
            resident_percent=args.resident_percent,
            kv_cache=args.kv_cache,
        )
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *log_args: Any) -> None:
            _log(fmt % log_args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/models":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen38-flash-next",
                            "object": "model",
                            "owned_by": "madmetal",
                        }
                    ],
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._started = False
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                messages = body["messages"]
                tools = body.get("tools")
                enable_thinking = _enable_thinking(body)
                max_tokens = _request_limit(body)
                temperature, top_p, top_k = _sampling(body, enable_thinking)
                stream = bool(body.get("stream", False))
                with lock:
                    runtime.tool_names = tool_names(tools)
                    runtime.required_function = required_function(body.get("tool_choice"))
                    runtime.required_closes = call_closes(tools, runtime.required_function)
                    close = runtime.tokenizer.encode("</think>").ids
                    if enable_thinking and len(close) == 1:
                        runtime._avoid_token = int(close[0])
                        runtime._avoid_until = 8
                    else:
                        runtime._avoid_token = None
                        runtime._avoid_until = 0
                    penalty = body.get("presence_penalty")
                    runtime._presence_penalty = 0.0 if penalty is None else float(penalty)
                    frequency = body.get("frequency_penalty")
                    runtime._frequency_penalty = 0.0 if frequency is None else float(frequency)
                    runtime._penalty_window = _penalty_window(body)
                    try:
                        prompt = runtime.encode_chat(messages, tools, enable_thinking)
                        _log(f"prompt tokens {len(prompt)}")
                        if stream:
                            self._stream(
                                prompt,
                                max_tokens,
                                temperature,
                                top_p,
                                top_k,
                                _prompt_text(messages),
                                enable_thinking,
                            )
                        else:
                            tokens = runtime.generate(
                                prompt, max_tokens, temperature, top_p, top_k
                            )
                            text = runtime.decode(tokens)
                            reasoning, reply = split_thinking(text, enable_thinking)
                            content, calls = parse_tool_calls(reply if enable_thinking else text)
                            _log(f"completion: {text}")
                            _append_transcript(args.transcript, _prompt_text(messages), text)
                            if calls:
                                reason = "tool_calls"
                            else:
                                reason = "length" if runtime._hit_limit else "stop"
                                if not enable_thinking:
                                    content = text
                            self._json(
                                200,
                                _completion(
                                    content,
                                    len(prompt),
                                    len(tokens),
                                    reason,
                                    calls,
                                    reasoning if enable_thinking else "",
                                ),
                            )
                    finally:
                        runtime.tool_names = []
                        runtime.required_function = None
                        runtime.required_closes = False
            except Exception as exc:
                traceback.print_exc()
                if self._started:
                    return
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

        def _stream(
            self,
            prompt: list[int],
            max_tokens: int | None,
            temperature: float,
            top_p: float,
            top_k: int,
            prompt_text: str,
            enable_thinking: bool,
        ) -> None:
            self._started = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._event(
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            )
            start = runtime.align(prompt)
            if start == len(prompt):
                if runtime.last_logits is None:
                    message = "the cached prefix has no logits"
                    raise RuntimeError(message)
                logits = runtime.last_logits
                _log("prefill 0 tok")
            else:
                logits = runtime.forward(prompt[start:])

            gate = ContentGate()
            thinking = ThinkStream(enable_thinking)

            def emit_channel(channel: str, visible: str) -> None:
                if channel == "content":
                    visible = gate.push(visible)
                if not visible:
                    return
                field = "reasoning_content" if channel == "reasoning" else "content"
                self._event(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {field: visible},
                                "finish_reason": None,
                            }
                        ]
                    }
                )

            def emit(piece: str) -> None:
                for channel, visible in thinking.push(piece):
                    emit_channel(channel, visible)

            previous = _drain_stream(
                runtime,
                runtime._new_tokens(logits, max_tokens, temperature, top_p, top_k),
                emit,
            )
            for channel, visible in thinking.finish():
                emit_channel(channel, visible)
            tail = gate.finish()
            if tail:
                emit_content = {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": tail},
                            "finish_reason": None,
                        }
                    ]
                }
                self._event(emit_content)
            _content, calls = parse_tool_calls(previous)
            if calls:
                for index, call in enumerate(calls):
                    self._event(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": call["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": call["function"]["name"],
                                                    "arguments": call["function"]["arguments"],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                reason = "tool_calls"
            else:
                reason = "length" if runtime._hit_limit else "stop"
            self._event({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            runtime._log_decode()
            _append_transcript(args.transcript, prompt_text, previous)

        def _event(self, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self.wfile.write(b"data: " + blob + b"\n\n")
            self.wfile.flush()

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    server = _SingleRequestServer((args.host, args.port), Handler)
    _log(f"listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


# Qwen team, Qwen3.8-Flash-Next model card.
# https://huggingface.co/Qwen/Qwen3.8-Flash-Next
THINKING_TEMPERATURE = 1.0
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
INSTRUCT_TEMPERATURE = 0.7
INSTRUCT_TOP_P = 0.8
INSTRUCT_TOP_K = 20


def _drain_stream(runtime: Any, tokens: Iterator[int], emit: Any) -> str:
    """Enqueue sampled ids and keep stepping. Detokenize and emit on a side thread."""
    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    text = ""

    def write() -> None:
        nonlocal text
        produced: list[int] = []
        previous = ""
        try:
            while True:
                token = pending.get()
                if token is None:
                    break
                produced.append(token)
                decoded = runtime.decode(produced)
                piece = decoded[len(previous) :]
                previous = decoded
                if piece:
                    emit(piece)
        except BaseException as exc:
            failed.append(exc)
        text = previous

    writer = threading.Thread(target=write, name="madmetal-stream")
    writer.start()
    try:
        for token in tokens:
            if failed:
                break
            pending.put(token)
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    return text


def _enable_thinking(body: dict[str, Any]) -> bool:
    """Leave `<think>` open unless the request closes it.

    The chat template writes the reasoning during decode when this is true.
    An explicit false closes the block in the prompt, so decode starts at the answer.
    """
    if "enable_thinking" in body and body["enable_thinking"] is not None:
        return bool(body["enable_thinking"])
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is not None:
        return bool(kwargs["enable_thinking"])
    return True


def _sampling(body: dict[str, Any], thinking: bool) -> tuple[float, float, int]:
    """Model-card sampling for the mode, unless the request sets a field.

    Temperature 0 stays an argmax. Thinking is 1.0 / 0.95 / 20. Instruct is 0.7 / 0.8 / 20.
    """
    temperature = body.get("temperature", None)
    top_p = body.get("top_p", None)
    top_k = body.get("top_k", None)
    if thinking:
        defaults = (THINKING_TEMPERATURE, THINKING_TOP_P, THINKING_TOP_K)
    else:
        defaults = (INSTRUCT_TEMPERATURE, INSTRUCT_TOP_P, INSTRUCT_TOP_K)
    return (
        defaults[0] if temperature is None else float(temperature),
        defaults[1] if top_p is None else float(top_p),
        defaults[2] if top_k is None else int(top_k),
    )


def _cli_sampling(args: Any) -> tuple[float, float, int]:
    body: dict[str, Any] = {}
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        body["top_k"] = args.top_k
    return _sampling(body, bool(args.thinking))


def _penalty_window(body: dict[str, Any]) -> int:
    """Last tokens a presence or frequency penalty sees. Zero is the whole completion."""
    window = body.get("penalty_window")
    if window is None:
        return PENALTY_WINDOW
    return int(window)


def _request_limit(body: dict[str, Any]) -> int | None:
    """A missing cap runs until a stop id. Zero is a cap of zero, not a missing cap."""
    if "max_tokens" in body and body["max_tokens"] is not None:
        return int(body["max_tokens"])
    if "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        return int(body["max_completion_tokens"])
    return None


def _completion(
    text: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str = "",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-madmetal",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen38-flash-next",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _add_kv_cache(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kv-cache",
        choices=("e4m3", "bf16"),
        default="e4m3",
        help="K and V width. e4m3 keeps a BF16 tail on the open block of 4",
    )


def _add_resident_percent(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resident-percent",
        type=float,
        default=None,
        help="percent of the hot expert tier to keep resident",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="madmetal --qwen38-flash-next")
    sub = parser.add_subparsers(dest="command", required=True)
    probe_parser = sub.add_parser("probe", help="measure one block of 32 on this GPU")
    probe_parser.add_argument("--json", action="store_true", help="print the measurement as JSON")
    sub.add_parser("matmul", help="check int4 E8M0 prefill tiles against the decode dot")
    sub.add_parser("kernels", help="check the GDN delta and the attention shader")
    quant_parser = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant_parser.add_argument("--source", type=Path, default=None)
    quant_parser.add_argument("--output", type=Path, default=DEFAULT_RECORD)
    quant_parser.add_argument("--hot", type=Path, default=None)
    quant_parser.add_argument("--importance", type=Path, default=None)
    quant_parser.add_argument("--hot-count", type=int, default=128)
    quant_parser.add_argument("--hot-bits", type=int, default=4)
    quant_parser.add_argument("--cold-bits", type=int, default=2)
    quant_parser.add_argument("--super", type=Path, default=None)
    serve_parser = sub.add_parser("serve", help="serve the quantized record")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8129)
    _add_resident_percent(serve_parser)
    _add_kv_cache(serve_parser)
    serve_parser.add_argument("--record", type=Path, default=None)
    serve_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append each prompt and completion to this file",
    )
    bf16_parser = sub.add_parser("bf16", help="complete from the BF16 checkpoint with Metal dots")
    bf16_parser.add_argument("--prompt", required=True)
    bf16_parser.add_argument("--max-tokens", type=int, default=8)
    bf16_parser.add_argument("--temperature", type=float, default=None)
    bf16_parser.add_argument("--top-p", type=float, default=None)
    bf16_parser.add_argument("--top-k", type=int, default=None)
    bf16_parser.add_argument(
        "--thinking", action="store_true", help="enable thinking in the chat template"
    )
    bf16_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    bf16_parser.add_argument("--source", type=Path, default=None)
    _add_kv_cache(bf16_parser)
    bf16_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append the prompt and the completion to this file",
    )
    complete_parser = sub.add_parser("complete", help="print one completion from the record")
    complete_parser.add_argument("--prompt", required=True)
    complete_parser.add_argument("--max-tokens", type=int, default=8)
    complete_parser.add_argument("--temperature", type=float, default=None)
    complete_parser.add_argument("--top-p", type=float, default=None)
    complete_parser.add_argument("--top-k", type=int, default=None)
    complete_parser.add_argument(
        "--thinking", action="store_true", help="enable thinking in the chat template"
    )
    complete_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    complete_parser.add_argument("--record", type=Path, default=None)
    _add_resident_percent(complete_parser)
    _add_kv_cache(complete_parser)
    complete_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append the prompt and the completion to this file",
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "probe":
        return probe(as_json=args.json)
    if args.command == "matmul":
        return matmul()
    if args.command == "kernels":
        try:
            check_state_kernels()
        except Exception as exc:
            sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
            return 1
        sys.stdout.write("gdn delta and attention shaders: ok\n")
        return 0
    if args.command == "quantize":
        return quantize_main(args)
    if args.command == "serve":
        return serve(args)
    if args.command == "bf16":
        return bf16_complete(args)
    if args.command == "complete":
        return complete(args)
    return 2

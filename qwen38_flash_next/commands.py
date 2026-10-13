"""quantize, bf16, complete, and serve."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_flash_next.checks import check_state_kernels
from qwen38_flash_next.convert import quantize_checkpoint
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.probe import matmul, probe
from qwen38_flash_next.runtime import Qwen38Runtime
from qwen38_flash_next.session import _append_transcript, _log, _prompt_text


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
        quantize_checkpoint(_checkpoint(args.source), args.output)
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


def bf16_complete(args: Any) -> int:
    """Same forward as complete, with Metal dots of the checkpoint BF16 values."""
    try:
        runtime = Qwen38Runtime(_checkpoint(args.source), bf16=True)
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
        for token in runtime._new_tokens(
            logits, args.max_tokens, args.temperature, args.top_p, args.top_k
        ):
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
        runtime = Qwen38Runtime(_record(args.record))
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
        for token in runtime._new_tokens(
            logits, args.max_tokens, args.temperature, args.top_p, args.top_k
        ):
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


def serve(args: Any) -> int:
    try:
        runtime = Qwen38Runtime(_record(args.record))
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
                enable_thinking = bool(body.get("enable_thinking", False))
                max_tokens = _request_limit(body)
                temperature = float(body.get("temperature", 0.0))
                top_p = float(body.get("top_p", 1.0))
                top_k = int(body.get("top_k", 0))
                stream = bool(body.get("stream", False))
                with lock:
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
                        )
                    else:
                        tokens = runtime.generate(
                            prompt, max_tokens, temperature, top_p, top_k
                        )
                        text = runtime.decode(tokens)
                        _log(f"completion: {text}")
                        _append_transcript(args.transcript, _prompt_text(messages), text)
                        reason = "length" if runtime._hit_limit else "stop"
                        self._json(200, _completion(text, len(prompt), len(tokens), reason))
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
            produced: list[int] = []
            previous = ""
            for token in runtime._new_tokens(logits, max_tokens, temperature, top_p, top_k):
                produced.append(token)
                text = runtime.decode(produced)
                self._event(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text[len(previous) :]},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                previous = text
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

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def _request_limit(body: dict[str, Any]) -> int | None:
    """A missing cap runs until a stop id. Zero is a cap of zero, not a missing cap."""
    if "max_tokens" in body and body["max_tokens"] is not None:
        return int(body["max_tokens"])
    if "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        return int(body["max_completion_tokens"])
    return None


def _completion(
    text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-madmetal",
        "object": "chat.completion",
        "model": "qwen38-flash-next",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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
    serve_parser = sub.add_parser("serve", help="serve the quantized record")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
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
    bf16_parser.add_argument("--temperature", type=float, default=0.0)
    bf16_parser.add_argument("--top-p", type=float, default=1.0)
    bf16_parser.add_argument("--top-k", type=int, default=0)
    bf16_parser.add_argument(
        "--thinking", action="store_true", help="enable thinking in the chat template"
    )
    bf16_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    bf16_parser.add_argument("--source", type=Path, default=None)
    bf16_parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="append the prompt and the completion to this file",
    )
    complete_parser = sub.add_parser("complete", help="print one completion from the record")
    complete_parser.add_argument("--prompt", required=True)
    complete_parser.add_argument("--max-tokens", type=int, default=8)
    complete_parser.add_argument("--temperature", type=float, default=0.0)
    complete_parser.add_argument("--top-p", type=float, default=1.0)
    complete_parser.add_argument("--top-k", type=int, default=0)
    complete_parser.add_argument(
        "--thinking", action="store_true", help="enable thinking in the chat template"
    )
    complete_parser.add_argument(
        "--raw", action="store_true", help="tokenize the prompt with no chat template"
    )
    complete_parser.add_argument("--record", type=Path, default=None)
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

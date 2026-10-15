"""quantize, bf16, complete, and serve."""

from __future__ import annotations

import argparse
import json
import queue
import select
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from madmetal.registry import (
    PIPELINE,
    add_model,
    adopt_existing,
    choose_complete,
    format_listing,
    load_registry,
    model_objects,
    models_dir,
    one_model,
    record_name,
    refuse_conflicting_destination,
    register_quantized,
    resolve,
    set_alias,
)
from qwen38_flash_next.checks import check_state_kernels
from qwen38_flash_next.console import Console
from qwen38_flash_next.convert import (
    LAYER_WORKERS,
    NGRAM_SHARD_COUNT,
    importance_hot_ids,
    quantize_checkpoint,
)
from qwen38_flash_next.formats import DEFAULT_RECORD
from qwen38_flash_next.grove import write_grove
from qwen38_flash_next.probe import matmul, probe
from qwen38_flash_next.runtime import Qwen38Runtime, RequestClosed
from qwen38_flash_next.session import _append_transcript, _log, _prompt_text, bind_log
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
    ui = _console(args)
    stop = threading.Event()
    watcher: threading.Thread | None = None
    progress = None
    finished = False
    try:
        source = _checkpoint(args.source)
        output = _quantize_output(args, source)
        args.output = output
        refuse_conflicting_destination(output.name, source)
        if not ui.keeps_lines:
            progress = ui.begin("quantize", total=0, unit="layers", detail="resident")
            layers = _source_layers(source)
            watcher = threading.Thread(
                target=_watch_quantize,
                args=(output, layers, progress, ui, stop),
                daemon=True,
            )
            watcher.start()
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
            super_bits=int(args.super_bits),
            mtp_bits=int(args.mtp_bits),
        )
        write_grove(args.output, replace=packed)
        register_quantized(output.name, source=source, alias=getattr(args, "alias", None))
        finished = True
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=2)
        if progress is not None:
            summary = f"record complete: {args.output}" if finished else ""
            ui.end(progress, summary)
    return 0


def _console(args: Any, *, fold: bool = False) -> Console:
    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    bind_log(ui.log)
    ui.fold_turns = fold
    ui.live_decode = ui.mode == "pretty" and not fold
    ui.plate()
    return ui


def _quantize_output(args: Any, source: Path) -> Path:
    if args.output is not None:
        return args.output
    cold = _source_experts(source) - int(args.hot_count)
    name = record_name(source, int(args.hot_count), cold, int(args.hot_bits), int(args.cold_bits))
    return models_dir() / name


def _source_layers(source: Path) -> int:
    config = json.loads((source / "config.json").read_text())
    return len(config["text_config"]["layer_types"])


def _source_experts(source: Path) -> int:
    config = json.loads((source / "config.json").read_text())
    return int(config["text_config"]["num_experts"])


def _watch_quantize(
    output: Path,
    layers: int,
    progress: Any,
    ui: Console,
    stop: threading.Event,
) -> None:
    while not stop.wait(1):
        path = output / "quantize_progress.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        done_layers = len(data.get("layers") or [])
        if not data.get("resident"):
            ui.retarget(progress, total=0, unit="layers", detail="resident", done=0)
            continue
        if done_layers < layers or not data.get("mtp"):
            detail = "experts" if done_layers < layers else "mtp"
            ui.retarget(progress, total=layers, unit="layers", detail=detail, done=done_layers)
            progress.gauge = ("workers", LAYER_WORKERS)
            continue
        shards = len(data.get("ngram_shards") or [])
        ui.retarget(progress, total=NGRAM_SHARD_COUNT, unit="shards", detail="n-gram", done=shards)
        progress.gauge = None


def bf16_complete(args: Any) -> int:
    """Same forward as complete, with Metal dots of the checkpoint BF16 values."""
    ui = _console(args, fold=True)
    shutdown = _Shutdown()
    shutdown.arm()
    runtime: Qwen38Runtime | None = None
    try:
        source, selection = _bf16_source(args)
        runtime = Qwen38Runtime(source, bf16=True, kv_cache=_kv_choice(args, selection), console=ui)
        shutdown.runtime = runtime
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=args.thinking)
        _arm_cli(runtime, args)
        start = runtime.align(prompt)
        if start == len(prompt):
            if runtime.last_logits is None:
                message = "the cached prefix has no logits"
                raise RuntimeError(message)
            logits = runtime.last_logits
            _log("prefill 0 tok/s")
        else:
            logits = runtime.forward(prompt[start:])
        produced: list[int] = []
        shown = ""
        temperature, top_p, top_k = _cli_sampling(args)
        for token in runtime._new_tokens(logits, args.max_tokens, temperature, top_p, top_k):
            produced.append(token)
            piece, shown = _stream_delta(runtime.decode(produced), shown, final=False)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
        if produced:
            piece, shown = _stream_delta(runtime.decode(produced), shown, final=True)
            if piece:
                sys.stdout.write(piece)
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _finish_turn(ui, runtime)
        _append_transcript(args.transcript, args.prompt, shown)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    except LookupError as exc:
        _log(str(exc))
        return 1
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    finally:
        _release(runtime, shutdown)
        ui.close()
    return 0


def complete(args: Any) -> int:
    ui = _console(args, fold=True)
    shutdown = _Shutdown()
    shutdown.arm()
    runtime: Qwen38Runtime | None = None
    try:
        record, selection = choose_complete(model=getattr(args, "model", None), record=args.record)
        resident = args.resident_percent
        if resident is None and selection is not None:
            resident = selection.resident_percent
        runtime = Qwen38Runtime(
            record,
            resident_percent=resident,
            kv_cache=_kv_choice(args, selection),
            console=ui,
        )
        shutdown.runtime = runtime
        if args.raw:
            prompt = list(runtime.tokenizer.encode(args.prompt).ids)
        else:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = runtime.encode_chat(messages, None, enable_thinking=args.thinking)
        _arm_cli(runtime, args)
        start = runtime.align(prompt)
        if start == len(prompt):
            if runtime.last_logits is None:
                message = "the cached prefix has no logits"
                raise RuntimeError(message)
            logits = runtime.last_logits
            _log("prefill 0 tok/s")
        else:
            logits = runtime.forward(prompt[start:])
        produced: list[int] = []
        shown = ""
        temperature, top_p, top_k = _cli_sampling(args)
        for token in runtime._new_tokens(logits, args.max_tokens, temperature, top_p, top_k):
            produced.append(token)
            piece, shown = _stream_delta(runtime.decode(produced), shown, final=False)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
        if produced:
            piece, shown = _stream_delta(runtime.decode(produced), shown, final=True)
            if piece:
                sys.stdout.write(piece)
        sys.stdout.write("\n")
        sys.stdout.flush()
        runtime._log_decode()
        _finish_turn(ui, runtime)
        _append_transcript(args.transcript, args.prompt, shown)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    except LookupError as exc:
        _log(str(exc))
        return 1
    except Exception as exc:
        traceback.print_exc()
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    finally:
        _release(runtime, shutdown)
        ui.close()
    return 0


class _SingleRequestServer(HTTPServer):
    """Accept the next connection only after the current request returns."""

    request_queue_size = 128


class _Shutdown:
    """SIGINT and SIGTERM stop the forward and leave the process able to release."""

    def __init__(self) -> None:
        self.runtime: Qwen38Runtime | None = None
        self.closing = False
        self.signaled = False

    def arm(self) -> None:
        self._previous = (
            signal.getsignal(signal.SIGINT),
            signal.getsignal(signal.SIGTERM),
        )
        signal.signal(signal.SIGINT, self.handle)
        signal.signal(signal.SIGTERM, self.handle)

    def disarm(self) -> None:
        previous = getattr(self, "_previous", None)
        if previous is None:
            return
        signal.signal(signal.SIGINT, previous[0])
        signal.signal(signal.SIGTERM, previous[1])
        self._previous = None

    def handle(self, _signum: int, _frame: Any) -> None:
        if self.closing:
            return
        self.signaled = True
        runtime = self.runtime
        if runtime is not None:
            runtime.halt()
        raise KeyboardInterrupt


def _release(runtime: Qwen38Runtime | None, shutdown: _Shutdown, server: Any = None) -> None:
    """Close the listening socket and the runtime. A signal during this returns."""
    shutdown.closing = True
    try:
        if server is not None:
            try:
                server.server_close()
            except OSError:
                pass
        if runtime is not None:
            runtime.close()
        if shutdown.signaled and (runtime is not None or server is not None):
            _log("released")
    finally:
        shutdown.disarm()


# A client that stops reading fills the send buffer. This bounds that write.
_STREAM_WRITE_TIMEOUT = 2.0


def _peer_closed(sock: socket.socket) -> bool:
    """True when the client has closed its side or the socket is already dead."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        peeked = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return peeked == b""


def serve(args: Any) -> int:
    shutdown = _Shutdown()
    shutdown.arm()
    args.shutdown = shutdown
    state: dict[str, Any] = {
        "runtime": None,
        "server": None,
        "record": None,
        "requests": 0,
    }
    try:
        if not getattr(args, "lazy", False):
            try:
                state["runtime"] = Qwen38Runtime(
                    _record(args.record),
                    resident_percent=args.resident_percent,
                    kv_cache=args.kv_cache or "e4m3",
                    console=getattr(args, "console_ui", None),
                )
            except Exception as exc:
                traceback.print_exc()
                sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
                return 1
            shutdown.runtime = state["runtime"]
        return _run_until_stopped(args, state, shutdown)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        if not shutdown.closing:
            _release(state["runtime"], shutdown, state.get("server"))


def _run_until_stopped(args: Any, state: dict[str, Any], shutdown: _Shutdown) -> int:
    """Serve until SIGINT or SIGTERM. Then close the socket and the runtime."""
    try:
        return _serve_loop(args, state)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        _release(state["runtime"], shutdown, state.get("server"))


def _serve_loop(args: Any, state: dict[str, Any]) -> int:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *log_args: Any) -> None:
            del fmt, log_args

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.unquote(self.path.split("?", 1)[0])
            if path == "/v1/models":
                if getattr(args, "lazy", False):
                    self._json(
                        200,
                        {
                            "object": "list",
                            "data": model_objects(resident=state.get("record")),
                        },
                    )
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
                return
            prefix = "/v1/models/"
            if getattr(args, "lazy", False) and path.startswith(prefix):
                try:
                    row = one_model(path[len(prefix) :], resident=state.get("record"))
                except LookupError as exc:
                    self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
                    return
                self._json(200, row)
                return
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._started = False
            self._gone = False
            self._model_id = "qwen38-flash-next"
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                messages = body["messages"]
                tools = body.get("tools")
                selection = _bind_request(body, state, args)
                if selection is not None:
                    self._model_id = selection.requested_id
                runtime = state["runtime"]
                if runtime is None:
                    self._json(
                        404,
                        {"error": {"message": "no model is loaded", "type": "not_found"}},
                    )
                    return
                if selection is not None and selection.enable_thinking is not None:
                    enable_thinking = selection.enable_thinking
                else:
                    enable_thinking = _enable_thinking(body)
                max_tokens = _request_limit(body)
                variant = selection.sampling if selection is not None else None
                temperature, top_p, top_k = _sampling(body, enable_thinking, variant)
                presence_default = None
                if isinstance(variant, dict) and "presence_penalty" in variant:
                    presence_default = float(variant["presence_penalty"])
                stream = bool(body.get("stream", False))
                with lock:
                    runtime._stop = self._client_stopped
                    runtime.tool_names = tool_names(tools)
                    runtime.required_function = required_function(body.get("tool_choice"))
                    runtime.required_closes = call_closes(tools, runtime.required_function)
                    frequency = body.get("frequency_penalty")
                    _arm_sampler(
                        runtime,
                        thinking=enable_thinking,
                        presence=_presence(body, enable_thinking, presence_default),
                        frequency=0.0 if frequency is None else float(frequency),
                        window=_penalty_window(body),
                    )
                    try:
                        prompt = runtime.encode_chat(
                            messages,
                            tools,
                            enable_thinking,
                            reasoning_effort=_reasoning_effort(body),
                            continue_final=bool(body.get("continue_final_message")),
                        )
                        if stream:
                            self._stream(
                                runtime,
                                prompt,
                                max_tokens,
                                temperature,
                                top_p,
                                top_k,
                                _prompt_text(messages),
                                enable_thinking,
                            )
                        else:
                            tokens = runtime.generate(prompt, max_tokens, temperature, top_p, top_k)
                            text = runtime.decode(tokens)
                            reasoning, reply = split_thinking(text, enable_thinking)
                            content, calls = parse_tool_calls(reply if reply else text)
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
                                    runtime,
                                    self._model_id,
                                ),
                            )
                    finally:
                        runtime._stop = None
                        runtime.tool_names = []
                        runtime.required_function = None
                        runtime.required_closes = False
            except RequestClosed:
                self.close_connection = True
                _log("client closed")
                live = state.get("runtime")
                persist = getattr(live, "_persist_prefix", None)
                if persist is not None:
                    persist()
            except KeyError as exc:
                self.close_connection = True
                if self._started:
                    return
                self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
            except LookupError as exc:
                self.close_connection = True
                if self._started:
                    return
                self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
            except Exception as exc:
                traceback.print_exc()
                self.close_connection = True
                if self._started:
                    return
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

        def _stream(
            self,
            runtime: Any,
            prompt: list[int],
            max_tokens: int | None,
            temperature: float,
            top_p: float,
            top_k: int,
            prompt_text: str,
            enable_thinking: bool,
        ) -> None:
            self._started = True
            self.close_connection = True
            self.connection.settimeout(_STREAM_WRITE_TIMEOUT)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            except OSError:
                self._gone = True
                raise RequestClosed() from None
            self._event(
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            )
            start = runtime.align(prompt)
            if start == len(prompt):
                if runtime.last_logits is None:
                    message = "the cached prefix has no logits"
                    raise RuntimeError(message)
                logits = runtime.last_logits
                _log("prefill 0 tok/s")
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
            self._event(
                {
                    "choices": [],
                    "usage": _usage(
                        runtime,
                        len(prompt),
                        len(getattr(runtime, "_output_ids", []) or []),
                        self._model_id,
                    ),
                }
            )
            self._send(b"data: [DONE]\n\n")
            runtime._log_decode()
            _append_transcript(args.transcript, prompt_text, previous)

        def _client_stopped(self) -> bool:
            if self._gone:
                return True
            if _peer_closed(self.connection):
                self._gone = True
                return True
            return False

        def _event(self, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self._send(b"data: " + blob + b"\n\n")

        def _send(self, data: bytes) -> None:
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                self._gone = True
                raise RequestClosed() from None

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            previous = self.connection.gettimeout()
            self.connection.settimeout(_STREAM_WRITE_TIMEOUT)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                _log("client closed")
            else:
                self.connection.settimeout(previous)

    server = _SingleRequestServer((args.host, args.port), Handler)
    state["server"] = server
    _log(f"listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


# Qwen team, Qwen3.8-Flash-Next model card.
# https://huggingface.co/Qwen/Qwen3.8-Flash-Next
THINKING_TEMPERATURE = 1.0
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
INSTRUCT_TEMPERATURE = 0.7
INSTRUCT_TOP_P = 0.8
INSTRUCT_TOP_K = 20
INSTRUCT_PRESENCE = 1.5


def _stream_delta(decoded: str, shown: str, *, final: bool) -> tuple[str, str]:
    """Text that is safe to send, and the piece after ``shown``.

    The tokenizer writes U+FFFD when a token is only some of the bytes of a
    character. The next token often replaces that mark with the character, and
    the two strings have the same length, so a slice would drop the character.
    """
    if not final and decoded.endswith("\ufffd"):
        decoded = decoded[:-1]
    return decoded[len(shown) :], decoded


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
                final = token is None
                if not final:
                    produced.append(token)
                decoded = runtime.decode(produced) if produced else ""
                piece, previous = _stream_delta(decoded, previous, final=final)
                if piece:
                    emit(piece)
                if final:
                    break
        except BaseException as exc:
            failed.append(exc)
        text = previous

    writer = threading.Thread(target=write, name="madmetal-stream", daemon=True)
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


def _reasoning_effort(body: dict[str, Any]) -> str | None:
    """Template effort. ``none`` is omitted, so the template keeps xhigh."""
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str) or effort.strip().lower() == "none":
        return None
    return effort


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


def _presence(body: dict[str, Any], thinking: bool, default: float | None = None) -> float:
    """Model-card presence. Thinking omits it at 0. Instruct omits it at 1.5.

    An explicit value is kept, including 0. The card uses 1.5 in instruct mode
    to cut endless repetition, and 0 while thinking.
    """
    penalty = body.get("presence_penalty")
    if penalty is None:
        if default is not None:
            return default
        return 0.0 if thinking else INSTRUCT_PRESENCE
    return float(penalty)


def _arm_sampler(
    runtime: Any,
    *,
    thinking: bool,
    presence: float,
    frequency: float,
    window: int,
) -> None:
    """Think-tag mask and penalties for one completion."""
    close = runtime.tokenizer.encode("</think>").ids
    opened = runtime.tokenizer.encode("<think>").ids
    runtime._answer_mode = not thinking
    if thinking and len(close) == 1:
        runtime._avoid_token = int(close[0])
        runtime._avoid_until = 8
        runtime._block_tokens = set()
    elif not thinking and len(opened) == 1:
        runtime._avoid_token = None
        runtime._avoid_until = 0
        runtime._block_tokens = {int(opened[0])}
    else:
        runtime._avoid_token = None
        runtime._avoid_until = 0
        runtime._block_tokens = set()
    runtime._presence_penalty = presence
    runtime._frequency_penalty = frequency
    runtime._penalty_window = window


def _arm_cli(runtime: Any, args: Any) -> None:
    """Same draw as serve. A raw prompt keeps the runtime defaults."""
    if args.raw:
        return
    thinking = bool(args.thinking)
    _arm_sampler(
        runtime,
        thinking=thinking,
        presence=0.0 if thinking else INSTRUCT_PRESENCE,
        frequency=0.0,
        window=PENALTY_WINDOW,
    )


def _sampling(
    body: dict[str, Any],
    thinking: bool,
    variant: dict[str, Any] | None = None,
) -> tuple[float, float, int]:
    """Model-card sampling for the mode, unless the request sets a field.

    Temperature 0 stays an argmax. Thinking is 1.0 / 0.95 / 20. Instruct is 0.7 / 0.8 / 20.
    """
    temperature = body.get("temperature", None)
    top_p = body.get("top_p", None)
    top_k = body.get("top_k", None)
    if variant is not None:
        defaults = (
            float(variant["temperature"]),
            float(variant["top_p"]),
            int(variant["top_k"]),
        )
    elif thinking:
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
    runtime: Any = None,
    model: str = "qwen38-flash-next",
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
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(runtime, prompt_tokens, completion_tokens, model),
    }


def _usage(runtime: Any, prompt_tokens: int, completion_tokens: int, model: str) -> dict[str, Any]:
    del model
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    prompt_n = max(0, prompt_tokens - cached)
    prefill_s = float(getattr(runtime, "_prefill_seconds", 0.0) or 0.0)
    decode_s = float(getattr(runtime, "_decode_seconds", 0.0) or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "timings": {
            "prompt_per_second": 0.0 if prefill_s <= 0 else prompt_n / prefill_s,
            "predicted_per_second": 0.0 if decode_s <= 0 else completion_tokens / decode_s,
            "prompt_n": prompt_n,
            "predicted_n": completion_tokens,
        },
    }


def _finish_turn(ui: Console, runtime: Qwen38Runtime) -> None:
    if ui.keeps_lines:
        return
    cached = runtime._cached_tokens
    prompt = runtime._prompt_tokens
    prefill_n = runtime._prefill_tokens
    prefill_s = runtime._prefill_seconds
    decode_n = runtime._decode_tokens
    decode_s = runtime._decode_seconds
    prefill_rate = 0.0 if prefill_s <= 0 else prefill_n / prefill_s
    decode_rate = 0.0 if decode_s <= 0 else decode_n / decode_s
    ui.log(
        f"── prompt {prompt:,} tok (cached {cached:,}) · "
        f"prefill {prefill_s:.1f}s · {prefill_rate:.1f} tok/s ── "
        f"decode {decode_n:,} tok · {decode_s:.1f}s · {decode_rate:.1f} tok/s ──"
    )


def _kv_choice(args: Any, selection: Any) -> str:
    chosen = getattr(args, "kv_cache", None)
    if chosen:
        return str(chosen)
    if selection is not None and selection.kv_cache:
        return str(selection.kv_cache)
    return "e4m3"


def _bf16_source(args: Any) -> tuple[Path, Any]:
    if args.source is not None:
        return _checkpoint(args.source), None
    _record, selection = choose_complete(model=getattr(args, "model", None), record=None)
    if selection is None:
        message = "pass --source with the checkpoint directory"
        raise LookupError(message)
    entry = load_registry()["models"][selection.folder]
    source = entry.get("source")
    if not source:
        message = f"{selection.folder} has no BF16 source"
        raise LookupError(message)
    return _checkpoint(Path(source).expanduser()), selection


def _bind_request(body: dict[str, Any], state: dict[str, Any], args: Any) -> Any:
    """Load the requested folder when serve was started with an empty process."""
    if not getattr(args, "lazy", False):
        return None
    adopt_existing()
    model = body.get("model")
    if not isinstance(model, str) or not model:
        message = "model is required"
        raise LookupError(message)
    selection = resolve(model, resident=state.get("record"))
    _install_runtime(state, args, selection)
    runtime = state.get("runtime")
    state["requests"] = int(state.get("requests") or 0) + 1
    if runtime is not None:
        runtime._request_id = state["requests"]
    return selection


def _install_runtime(state: dict[str, Any], args: Any, selection: Any) -> None:
    if state.get("runtime") is not None and state.get("record") == selection.folder:
        return
    old = state.get("runtime")
    state["runtime"] = None
    state["record"] = None
    if old is not None:
        old.close()
    ui = getattr(args, "console_ui", None)
    if ui is not None:
        ui.load_detail = selection.alias or selection.folder
    runtime = Qwen38Runtime(
        selection.path,
        resident_percent=selection.resident_percent,
        kv_cache=selection.kv_cache,
        console=ui,
    )
    state["runtime"] = runtime
    state["record"] = selection.folder
    shutdown = getattr(args, "shutdown", None)
    if shutdown is not None:
        shutdown.runtime = runtime
    if ui is not None and ui.keeps_lines:
        label = selection.alias or selection.folder
        _log(f"loaded {label} {selection.folder}")


def serve_main(argv: Sequence[str] | None = None) -> int:
    """``madmetal serve``. Nothing is loaded until a request names a model."""
    parser = argparse.ArgumentParser(prog="madmetal serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    parser.add_argument("--transcript", type=Path, default=None)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    ui = _console(args)
    args.lazy = True
    args.console_ui = ui
    args.record = None
    args.resident_percent = None
    args.kv_cache = "e4m3"
    adopt_existing()
    try:
        return serve(args)
    finally:
        ui.close()


def models_main(args: Any) -> int:
    command = getattr(args, "models_command", None)
    try:
        if command == "add":
            add_model(args.folder, alias=args.alias, source=args.source)
        elif command == "alias":
            set_alias(args.folder, args.alias)
        listing = format_listing()
        if listing:
            sys.stdout.write(listing + "\n")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


def _add_kv_cache(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kv-cache",
        choices=("e4m3", "bf16"),
        default=None,
        help="K and V width. Omitted, a flavor uses its registry value, otherwise e4m3",
    )


def _add_console(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--console",
        choices=("auto", "pretty", "plain", "json"),
        default="auto",
        help="status row. auto is pretty on a terminal and plain on a pipe",
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
    quant_parser.add_argument("--output", type=Path, default=None)
    quant_parser.add_argument("--alias", default=None)
    quant_parser.add_argument("--hot", type=Path, default=None)
    quant_parser.add_argument("--importance", type=Path, default=None)
    quant_parser.add_argument("--hot-count", type=int, default=128)
    quant_parser.add_argument("--hot-bits", type=int, default=8, choices=(8, 4, 2))
    quant_parser.add_argument("--cold-bits", type=int, default=4, choices=(8, 4, 2))
    quant_parser.add_argument("--super-bits", type=int, default=8, choices=(8, 4, 2))
    quant_parser.add_argument("--mtp-bits", type=int, default=4, choices=(8, 4, 2))
    quant_parser.add_argument("--super", type=Path, default=None)
    _add_console(quant_parser)
    models_parser = sub.add_parser("models", help="list or edit the flavor registry")
    models_sub = models_parser.add_subparsers(dest="models_command")
    add_parser = models_sub.add_parser("add", help="register a finished record directory")
    add_parser.add_argument("folder")
    add_parser.add_argument("--alias", default=None)
    add_parser.add_argument("--source", default=None)
    alias_parser = models_sub.add_parser("alias", help="change the alias of a folder")
    alias_parser.add_argument("folder")
    alias_parser.add_argument("alias")
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
    bf16_parser.add_argument("--model", default=None)
    _add_kv_cache(bf16_parser)
    _add_console(bf16_parser)
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
    complete_parser.add_argument("--model", default=None)
    _add_resident_percent(complete_parser)
    _add_kv_cache(complete_parser)
    _add_console(complete_parser)
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
    if args.command == "models":
        return models_main(args)
    if args.command == "bf16":
        return bf16_complete(args)
    if args.command == "complete":
        return complete(args)
    return 2

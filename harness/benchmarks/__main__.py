"""IFEval, GSM8K, and HumanEval against ``madmetal serve``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import (
    Request,
    Server,
    apply_sampling,
    card_model_id,
    complete,
    fetch_sampling,
)

_ROOT = Path(__file__).resolve().parents[2]
_SMOKE = 30
_SUITES = ("ifeval", "gsm8k", "humaneval")
_LIMITS = {"ifeval": 2048, "gsm8k": 1024, "humaneval": 1024}
_NOTE = """# Benchmarks

As-built harness scores for this pipeline. The process contract is
[../../ARCHITECTURE.md](../../ARCHITECTURE.md).

The harness scores IFEval, GSM8K, and HumanEval against `serve`.
The `model` field names this pipeline. The checker reads `message.content`.
A dropped connection restarts the server and sends the same request again.

## Request

<!-- request -->
not run
<!-- /request -->

## Intelligence smoke

The smoke is the first 30 rows of each set, in file order.

<!-- smoke-ifeval -->
not run
<!-- /smoke-ifeval -->

<!-- smoke-gsm8k -->
not run
<!-- /smoke-gsm8k -->

<!-- smoke-humaneval -->
not run
<!-- /smoke-humaneval -->

## Sample of 100

The sample is the first 100 rows of each set, in file order.

<!-- sample-ifeval -->
not run
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
not run
<!-- /sample-gsm8k -->

<!-- sample-humaneval -->
not run
<!-- /sample-humaneval -->

## Full

<!-- full-ifeval -->
not run
<!-- /full-ifeval -->

<!-- full-gsm8k -->
not run
<!-- /full-gsm8k -->

<!-- full-humaneval -->
not run
<!-- /full-humaneval -->
"""


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[str(row["id"])] = row
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _smoke(port: int, model: str, request: Request) -> None:
    """Two short completions on the same request the suites use.

    The instruct card samples, so the digits are not a gate. An empty reply
    means the server did not return content.
    """
    prompts = (
        "What is 2+2? Answer with one number.",
        "What is 3+3? Answer with one number.",
    )
    for prompt in prompts:
        text, reason, tokens = complete(
            port,
            [{"role": "user", "content": prompt}],
            32,
            request,
            model=model,
        )
        print(f"smoke {prompt!r} -> {text!r} ({reason}, {tokens} tok)", flush=True)
        if not text.strip():
            message = "smoke completion was empty"
            raise RuntimeError(message)


def _failed(name: str, row_id: str, detail: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "response": "",
        "finish_reason": "error",
        "completion_tokens": 0,
        "seconds": 0,
        "error": detail,
    }
    if name == "ifeval":
        row.update(
            strict_prompt=False,
            strict_instructions=[],
            loose_prompt=False,
            loose_instructions=[],
        )
    else:
        row["correct"] = False
    return row


def _ask(
    server: Server,
    messages: list[dict[str, str]],
    max_tokens: int,
    request: Request,
    *,
    model: str,
):
    last = "the server closed the connection"
    for _attempt in range(3):
        try:
            return complete(server.port, messages, max_tokens, request, model=model)
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            print(f"restarting server: {last}", flush=True)
            server.restart()
    return None, "error", last


def _run_rows(
    name: str,
    server: Server,
    rows: list[dict[str, Any]],
    path: Path,
    score_row,
    request: Request,
    *,
    model: str,
) -> list[dict[str, Any]]:
    done = _load_jsonl(path)
    max_tokens = request.limit(name)
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cached = done.get(str(row["id"]))
        if cached is None:
            started = time.perf_counter()
            text, reason, tokens = _ask(
                server,
                row["messages"],
                max_tokens,
                request,
                model=model,
            )
            if reason == "error":
                outcome = _failed(name, str(row["id"]), str(tokens))
            else:
                outcome = score_row(row, text)
                outcome.update(
                    {
                        "response": text,
                        "finish_reason": reason,
                        "completion_tokens": tokens,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                )
            _append(path, outcome)
            done[str(row["id"])] = outcome
            cached = outcome
        scored.append(cached)
        mark = cached.get("correct", cached.get("passed", cached.get("strict_prompt")))
        print(f"{name} {index}/{len(rows)} {cached['id']} {mark}", flush=True)
    return scored


def _ifeval_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return ifeval.aggregate(rows)


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    count = len(rows)
    hits = sum(bool(row[key]) for row in rows)
    return {"n": count, "correct": hits, "accuracy": hits / count if count else 0.0}


def _fmt(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "not run"
    if "prompt_strict" in summary:
        return (
            f"n={summary['n']}  "
            f"prompt strict {summary['prompt_strict']:.3f}  "
            f"prompt loose {summary['prompt_loose']:.3f}  "
            f"instruction strict {summary['instruction_strict']:.3f}  "
            f"instruction loose {summary['instruction_loose']:.3f}"
        )
    return f"n={summary['n']}  {summary['correct']}/{summary['n']}  {summary['accuracy']:.3f}"


def _request_line(request: Request) -> str:
    parts = [f"thinking {'on' if request.enable_thinking else 'off'}"]
    if request.temperature is not None:
        parts.append(f"temperature {request.temperature:g}")
    if request.top_p is not None:
        parts.append(f"top-p {request.top_p:g}")
    if request.top_k is not None:
        parts.append(f"top-k {request.top_k}")
    if request.presence_penalty is not None:
        parts.append(f"presence {request.presence_penalty:g}")
    if request.reasoning_effort:
        parts.append(f"reasoning effort {request.reasoning_effort}")
    limits = request.max_tokens or {}
    caps = ", ".join(f"{name} {limits[name]}" for name in _SUITES if name in limits)
    if caps:
        parts.append(f"max tokens {caps}")
    return "- " + ", ".join(parts)


def _note_path(model: str) -> Path | None:
    name = model.split(":", 1)[0]
    directory = _ROOT / "docs" / name
    if not directory.is_dir():
        return None
    return directory / "benchmarks.md"


def _replace(text: str, name: str, body: str) -> str:
    start = f"<!-- {name} -->"
    end = f"<!-- /{name} -->"
    if start not in text or end not in text:
        message = f"benchmarks note is missing {start}"
        raise RuntimeError(message)
    before, _, rest = text.partition(start)
    _, _, after = rest.partition(end)
    return f"{before}{start}\n{body}\n{end}{after}"


def _write_block(model: str, request: Request, section: str, body: str) -> None:
    path = _note_path(model)
    if path is None:
        return
    if not path.is_file():
        path.write_text(_NOTE)
    text = _replace(path.read_text(), "request", _request_line(request))
    text = _replace(text, section, body)
    path.write_text(text)


def _ifeval_items() -> list[dict[str, Any]]:
    items = []
    for example in ifeval.load():
        items.append(
            {
                "id": str(example.key),
                "example": example,
                "messages": ifeval.messages(example),
            }
        )
    return items


def _gsm_items() -> list[dict[str, Any]]:
    items = []
    for row in gsm8k.load():
        items.append({**row, "messages": gsm8k.messages(row)})
    return items


def _he_items() -> list[dict[str, Any]]:
    items = []
    for row in humaneval.load():
        items.append({**row, "messages": humaneval.messages(row)})
    return items


def _score_ifeval(row: dict[str, Any], text: str) -> dict[str, Any]:
    return ifeval.score(row["example"], text)


def _score_gsm(row: dict[str, Any], text: str) -> dict[str, Any]:
    return {"id": row["id"], "correct": gsm8k.correct(text, row)}


def _score_he(row: dict[str, Any], text: str) -> dict[str, Any]:
    ok = humaneval.passed(row, text)
    return {"id": row["id"], "correct": ok}


def _suite(
    name: str,
    server: Server,
    items: list[dict[str, Any]],
    out: Path,
    score_row,
    summarize,
    limit: int | None,
    request: Request,
    model: str,
) -> dict[str, Any]:
    chosen = items if limit is None else items[:limit]
    rows = _run_rows(
        name, server, chosen, out / f"{name}.jsonl", score_row, request, model=model
    )
    summary = summarize(rows)
    print(f"{name} {_fmt(summary)}", flush=True)
    return summary


def _catalog() -> tuple[tuple[str, list[dict[str, Any]], Any, Any], ...]:
    return (
        ("ifeval", _ifeval_items(), _score_ifeval, _ifeval_summary),
        ("gsm8k", _gsm_items(), _score_gsm, lambda rows: _rate(rows, "correct")),
        ("humaneval", _he_items(), _score_he, lambda rows: _rate(rows, "correct")),
    )


def run(
    output: Path,
    request: Request,
    *,
    sample_only: bool,
    only: set[str] | None = None,
    limit: int | None = None,
    model: str,
) -> int:
    server = Server()
    server.start()
    try:
        request = apply_sampling(
            request,
            fetch_sampling(
                server.port, card_model_id(model, thinking=request.enable_thinking)
            ),
        )
        _smoke(server.port, model, request)
        suites = _catalog()
        if only:
            suites = tuple(item for item in suites if item[0] in only)
        if limit is not None:
            section = "smoke" if limit == _SMOKE else "sample" if limit == 100 else None
            for name, items, score_row, summarize in suites:
                summary = _suite(
                    name, server, items, output, score_row, summarize, limit, request, model
                )
                if section is not None:
                    _write_block(model, request, f"{section}-{name}", f"- {name}: {_fmt(summary)}")
            print("sample recorded", flush=True)
            return 0
        for name, items, score_row, summarize in suites:
            summary = _suite(
                name, server, items, output, score_row, summarize, _SMOKE, request, model
            )
            _write_block(model, request, f"smoke-{name}", f"- {name}: {_fmt(summary)}")
        print("smoke recorded", flush=True)
        if sample_only:
            return 0
        for name, items, score_row, summarize in suites:
            summary = _suite(
                name, server, items, output, score_row, summarize, None, request, model
            )
            _write_block(model, request, f"full-{name}", f"- {name}: {_fmt(summary)}")
        print("full recorded", flush=True)
    except Exception as exc:
        print(exc, flush=True)
        print(server.log.text()[-3000:], flush=True)
        return 1
    finally:
        server.stop()
    return 0


def _request_from_args(args: argparse.Namespace) -> Request:
    limits = dict(_LIMITS)
    if args.max_tokens is not None:
        for name in limits:
            limits[name] = args.max_tokens
    return Request(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence,
        enable_thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_tokens=limits,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IFEval, GSM8K, and HumanEval.")
    parser.add_argument("--model", required=True, help="registry model id sent to serve")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="score the first 30 of each set and stop",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score the first N rows of each set and stop",
    )
    parser.add_argument("--only", nargs="+", choices=_SUITES)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence", type=float, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="one completion cap for every suite",
    )
    args = parser.parse_args(argv)
    output = args.output or (_ROOT / "artifacts" / "benchmarks" / args.model.split(":", 1)[0])
    only = set(args.only) if args.only else None
    return run(
        output,
        _request_from_args(args),
        sample_only=args.sample_only,
        only=only,
        limit=args.limit,
        model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

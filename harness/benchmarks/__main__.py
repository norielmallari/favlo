"""Intelligence smoke: the first 30 of IFEval, GSM8K, and HumanEval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import Server, complete

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "qwen38-flash-next" / "benchmarks.md"
_SMOKE = 30
_MAX_TOKENS = {"ifeval": 2048, "gsm8k": 1536, "humaneval": 1024}


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


def _smoke(port: int, model: str) -> None:
    checks = (
        ("What is 2+2? Answer with one number.", "4"),
        ("What is 3+3? Answer with one number.", "6"),
    )
    for prompt, expect in checks:
        text, reason, tokens = complete(
            port, [{"role": "user", "content": prompt}], 32, model=model
        )
        print(f"smoke {prompt!r} -> {text!r} ({reason}, {tokens} tok)", flush=True)
        if expect not in text:
            message = f"smoke expected {expect} in {text!r}"
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
    *,
    thinking: bool = False,
    effort: str | None = None,
    presence: float | None = None,
    continue_final: bool = False,
    reply_only: bool = False,
    model: str = "qwen38-flash-next",
):
    last = "the server closed the connection"
    for _attempt in range(3):
        try:
            return complete(
                server.port,
                messages,
                max_tokens,
                thinking=thinking,
                effort=effort,
                presence=presence,
                continue_final=continue_final,
                reply_only=reply_only,
                model=model,
            )
        except (OSError, json.JSONDecodeError) as exc:
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
    *,
    model: str,
) -> list[dict[str, Any]]:
    done = _load_jsonl(path)
    max_tokens = _MAX_TOKENS[name]
    thinking = name in {"ifeval", "gsm8k", "humaneval"}
    effort = "low" if thinking else None
    # A function body repeats indentation and keywords. The instruct presence
    # penalty suppresses those repeats, so HumanEval leaves presence at 0.
    # IFEval sends that same 1.5 while thinking, so the trace does not loop,
    # and the checker reads the reply rather than the trace.
    if name == "humaneval":
        presence = 0.0
    elif name == "ifeval":
        presence = 1.5
    else:
        presence = None
    reply_only = name == "ifeval"
    continue_final = False
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cached = done.get(str(row["id"]))
        if cached is None:
            started = time.perf_counter()
            text, reason, tokens = _ask(
                server,
                row["messages"],
                max_tokens,
                thinking=thinking,
                effort=effort,
                presence=presence,
                continue_final=continue_final,
                reply_only=reply_only,
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


def _write_doc(
    smoke: dict[str, Any] | None,
    full: dict[str, Any] | None,
    sample: dict[str, Any] | None = None,
) -> None:
    if not _DOC.is_file():
        return
    text = _DOC.read_text()
    if smoke is not None:
        smoke_block = "\n".join(f"- {name}: {_fmt(summary)}" for name, summary in smoke.items())
        text = _replace(text, "smoke", smoke_block)
    if sample is not None:
        sample_block = "\n".join(f"- {name}: {_fmt(summary)}" for name, summary in sample.items())
        text = _replace(text, "sample", sample_block)
    if full is not None:
        full_block = "\n".join(f"- {name}: {_fmt(summary)}" for name, summary in full.items())
        text = _replace(text, "full", full_block)
    _DOC.write_text(text)


def _replace(text: str, name: str, body: str) -> str:
    start = f"<!-- {name} -->"
    end = f"<!-- /{name} -->"
    before, _, rest = text.partition(start)
    _, _, after = rest.partition(end)
    return f"{before}{start}\n{body}\n{end}{after}"


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
    model: str,
) -> dict[str, Any]:
    chosen = items if limit is None else items[:limit]
    rows = _run_rows(name, server, chosen, out / f"{name}.jsonl", score_row, model=model)
    summary = summarize(rows)
    print(f"{name} {_fmt(summary)}", flush=True)
    return summary


def run(
    output: Path,
    *,
    sample_only: bool,
    only: set[str] | None = None,
    limit: int | None = None,
    model: str = "qwen38-flash-next",
) -> int:
    server = Server()
    server.start()
    try:
        _smoke(server.port, model)
        suites = (
            ("ifeval", _ifeval_items(), _score_ifeval, _ifeval_summary),
            ("gsm8k", _gsm_items(), _score_gsm, lambda rows: _rate(rows, "correct")),
            ("humaneval", _he_items(), _score_he, lambda rows: _rate(rows, "correct")),
        )
        if only:
            suites = tuple(item for item in suites if item[0] in only)
        if limit is not None:
            sample: dict[str, Any] = {}
            for name, items, score_row, summarize in suites:
                sample[name] = _suite(
                    name, server, items, output, score_row, summarize, limit, model
                )
            (output / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
            if only is None:
                _write_doc(None, None, sample)
            print("sample recorded", flush=True)
            return 0
        smoke: dict[str, Any] = {}
        for name, items, score_row, summarize in suites:
            smoke[name] = _suite(
                name, server, items, output, score_row, summarize, _SMOKE, model
            )
        (output / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
        if only is None:
            _write_doc(smoke, None)
        print("smoke recorded", flush=True)
        if sample_only:
            return 0
        full: dict[str, Any] = {}
        for name, items, score_row, summarize in suites:
            full[name] = _suite(
                name, server, items, output, score_row, summarize, None, model
            )
        (output / "full.json").write_text(json.dumps(full, indent=2) + "\n")
        _write_doc(None, full)
        print("full recorded", flush=True)
    except Exception as exc:
        print(exc, flush=True)
        print(server.log.text()[-3000:], flush=True)
        return 1
    finally:
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IFEval, GSM8K, and HumanEval.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "artifacts" / "benchmarks",
    )
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
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("ifeval", "gsm8k", "humaneval"),
        help="score these sets and leave the others unread",
    )
    parser.add_argument("--model", default="qwen38-flash-next")
    args = parser.parse_args(argv)
    only = set(args.only) if args.only else None
    return run(
        args.output,
        sample_only=args.sample_only,
        only=only,
        limit=args.limit,
        model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

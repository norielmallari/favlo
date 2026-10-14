"""Run the three suites, record the first 100, then finish each set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from harness.benchmarks import gsm8k, humaneval, ifeval
from harness.benchmarks.serve import Server, complete

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "qwen38" / "benchmarks.md"
_MAX_TOKENS = {"ifeval": 2048, "gsm8k": 768, "humaneval": 512}


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


def _smoke(port: int) -> None:
    checks = (
        ("What is 2+2? Answer with one number.", "4"),
        ("What is 3+3? Answer with one number.", "6"),
    )
    for prompt, expect in checks:
        text, reason, tokens = complete(
            port, [{"role": "user", "content": prompt}], 32
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


def _ask(server: Server, messages: list[dict[str, str]], max_tokens: int):
    last = "the server closed the connection"
    for _attempt in range(3):
        try:
            return complete(server.port, messages, max_tokens)
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
) -> list[dict[str, Any]]:
    done = _load_jsonl(path)
    max_tokens = _MAX_TOKENS[name]
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cached = done.get(str(row["id"]))
        if cached is None:
            started = time.perf_counter()
            text, reason, tokens = _ask(server, row["messages"], max_tokens)
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


def _write_doc(sample: dict[str, Any], full: dict[str, Any] | None) -> None:
    text = _DOC.read_text()
    sample_block = "\n".join(f"- {name}: {_fmt(summary)}" for name, summary in sample.items())
    full_block = "not run"
    if full is not None:
        full_block = "\n".join(f"- {name}: {_fmt(summary)}" for name, summary in full.items())
    text = _replace(text, "sample", sample_block)
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
) -> dict[str, Any]:
    chosen = items if limit is None else items[:limit]
    rows = _run_rows(name, server, chosen, out / f"{name}.jsonl", score_row)
    summary = summarize(rows)
    print(f"{name} {_fmt(summary)}", flush=True)
    return summary


def run(output: Path, *, sample_only: bool) -> int:
    server = Server()
    server.start()
    try:
        _smoke(server.port)
        suites = (
            ("ifeval", _ifeval_items(), _score_ifeval, _ifeval_summary),
            ("gsm8k", _gsm_items(), _score_gsm, lambda rows: _rate(rows, "correct")),
            ("humaneval", _he_items(), _score_he, lambda rows: _rate(rows, "correct")),
        )
        sample: dict[str, Any] = {}
        for name, items, score_row, summarize in suites:
            sample[name] = _suite(name, server, items, output, score_row, summarize, 100)
        (output / "sample-100.json").write_text(json.dumps(sample, indent=2) + "\n")
        _write_doc(sample, None)
        print("sample-100 recorded", flush=True)
        if sample_only:
            return 0
        full: dict[str, Any] = {}
        for name, items, score_row, summarize in suites:
            full[name] = _suite(name, server, items, output, score_row, summarize, None)
        (output / "full.json").write_text(json.dumps(full, indent=2) + "\n")
        _write_doc(sample, full)
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
        help="score the first 100 of each set and stop",
    )
    args = parser.parse_args(argv)
    return run(args.output, sample_only=args.sample_only)


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the shared GPTQ and FOEM calibration dialogs.

Each source is a public dataset file at a pinned revision. JSON and JSONL
files are downloaded from that revision. Parquet splits are read in file order
from the dataset viewer at the same revision. The write is
``artifacts/calibration/<kind>.json``. ``manifest.json`` names those files.
A pipeline calls ``build`` and reads one kind with ``load``.
``build`` replaces only the kinds it is given. Other kind files stay.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_HOST = "https://huggingface.co/datasets"
_ROWS = "https://datasets-server.huggingface.co/rows"
_AGENT = "favlo"
_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PAGE = 100
_SCAN = 8000

# Public files. ``rows`` reads a parquet split in published file order.
SOURCES = (
    {
        "kind": "swe",
        "repo": "nvidia/Open-SWE-Traces",
        "revision": "4fa4c2ae9afc6b397bc50391d43450c79993c138",
        "path": "v1.0/sweagent/0000.parquet",
        "config": "v1.0",
        "split": "sweagent",
        "reader": "rows",
        "page": 20,
        "need": 32,
        "license": "CC-BY-4.0",
    },
    {
        "kind": "code",
        "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "revision": "b0079beaa0361d82412520b873715bee59cc7dd4",
        "path": "data-evol_instruct-decontaminated.jsonl",
        "reader": "jsonl",
        "need": 400,
        "license": "Apache-2.0",
    },
    {
        "kind": "tool",
        "repo": "NousResearch/hermes-function-calling-v1",
        "revision": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
        "path": "func-calling.json",
        "reader": "json",
        "need": 120,
        "license": "Apache-2.0",
    },
    {
        "kind": "chat",
        "repo": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "path": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        "config": "default",
        "split": "train_sft",
        "reader": "rows",
        "need": 400,
        "license": "MIT",
    },
)

QUOTAS = (("swe", 34), ("code", 34), ("tool", 34), ("chat", 26))
SEQUENCE_LENGTH = 2048


def corpus_dir() -> Path:
    """The shared calibration directory."""
    return Path(__file__).resolve().parents[2] / "artifacts" / "calibration"


def file_url(source: dict[str, str]) -> str:
    """The resolve URL of one pinned public file."""
    if source["reader"] == "rows" and source["kind"] == "swe":
        path = urllib.parse.quote(f"refs/convert/parquet/{source['path']}", safe="/")
        return f"{_HOST}/{source['repo']}/resolve/{path}"
    path = urllib.parse.quote(source["path"], safe="/")
    return f"{_HOST}/{source['repo']}/resolve/{source['revision']}/{path}"


def build(dest: Path | None = None, kinds: Any = None) -> Path:
    """Rewrite only the named kind files. Other kind files stay.

    ``kinds`` is required. Each entry is ``swe`` or ``swe.json``.
    """
    folder = Path(dest) if dest is not None else corpus_dir()
    selected = _select(kinds)
    folder.mkdir(parents=True, exist_ok=True)
    counts = _existing_counts(folder)
    groups: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCES:
        if source["kind"] not in selected:
            continue
        kept = _collect(source)
        groups[source["kind"]] = kept
        counts[source["kind"]] = len(kept)
    _publish(folder, groups, _manifest(counts))
    return folder


def _kind_name(value: Any) -> str:
    name = Path(str(value).strip()).name.lower()
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return name


def _select(kinds: Any) -> frozenset[str]:
    if isinstance(kinds, str):
        kinds = [kinds]
    known = [source["kind"] for source in SOURCES]
    if not kinds:
        message = "name a calibration file: swe.json, code.json, tool.json, or chat.json"
        raise ValueError(message)
    selected: list[str] = []
    for kind in kinds:
        name = _kind_name(kind)
        if name not in known:
            message = (
                f"calibration file {kind} is not swe.json, code.json, tool.json, or chat.json"
            )
            raise ValueError(message)
        if name not in selected:
            selected.append(name)
    return frozenset(selected)


def _existing_counts(folder: Path) -> dict[str, int]:
    path = folder / "manifest.json"
    recorded: dict[str, Any] = {}
    if path.is_file():
        body = json.loads(path.read_text(encoding="utf-8"))
        found = body.get("counts") if isinstance(body, dict) else None
        if isinstance(found, dict):
            recorded = found
    counts: dict[str, int] = {}
    for source in SOURCES:
        kind = source["kind"]
        if not (folder / f"{kind}.json").is_file():
            continue
        value = recorded.get(kind)
        if isinstance(value, int):
            counts[kind] = value
        else:
            counts[kind] = _count_file(folder / f"{kind}.json")
    return counts


def _count_file(path: Path) -> int:
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, list):
        message = f"{path.name} is not a JSON list"
        raise RuntimeError(message)
    return len(found)


def load(kind: str, dest: Path | None = None) -> list[dict[str, Any]]:
    """The dialogs for one kind, from the file named in the manifest."""
    folder = Path(dest) if dest is not None else corpus_dir()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["kind"] == kind:
            path = folder / item["file"]
            found = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(found, list):
                message = f"{path.name} is not a JSON list"
                raise RuntimeError(message)
            return found
    message = f"no calibration file for {kind}"
    raise KeyError(message)


def _collect(source: dict[str, Any]) -> list[dict[str, Any]]:
    kind = source["kind"]
    need = int(source["need"])
    found: list[dict[str, Any]] = []
    for row in _records(source):
        kept = _keep(kind, row)
        if kept is not None:
            found.append(kept)
        if len(found) == need:
            break
    if len(found) < need:
        message = f"{kind} returned {len(found)} rows, need {need}"
        raise RuntimeError(message)
    return found


def _records(source: dict[str, Any]) -> Any:
    reader = source["reader"]
    if reader == "jsonl":
        return _jsonl(file_url(source))
    if reader == "json":
        return _json_array(file_url(source))
    return _split_rows(source)


def _open(request: urllib.request.Request) -> Any:
    delay = 2.0
    last: BaseException | None = None
    for _attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            last = exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last = exc
        time.sleep(delay)
        delay *= 2
    message = f"request failed: {request.full_url}"
    raise RuntimeError(message) from last


def _jsonl(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with _open(request) as response:
        seen = 0
        for line in response:
            text = line.decode().strip()
            if not text:
                continue
            yield json.loads(text)
            seen += 1
            if seen >= _SCAN:
                break


def _json_array(url: str) -> list[Any]:
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with _open(request) as response:
        found = json.load(response)
    if not isinstance(found, list):
        message = f"{url} is not a JSON list"
        raise RuntimeError(message)
    return found


def _split_rows(source: dict[str, Any]) -> Any:
    offset = 0
    while offset < _SCAN:
        query = urllib.parse.urlencode(
            {
                "dataset": source["repo"],
                "config": source["config"],
                "split": source["split"],
                "revision": source["revision"],
                "offset": offset,
                "length": source.get("page", _PAGE),
            }
        )
        request = urllib.request.Request(f"{_ROWS}?{query}", headers={"User-Agent": _AGENT})
        with _open(request) as response:
            body = json.loads(response.read().decode())
        rows = body.get("rows") or []
        if not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
        offset += len(rows)


def _keep(kind: str, row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if kind == "swe":
        return _swe(row)
    if kind == "code":
        messages = _code(row.get("instruction"), row.get("response"))
    elif kind == "tool":
        return _tool(row.get("conversations"), row.get("tools"))
    else:
        messages = _chat(row.get("messages"))
    if messages is None:
        return None
    return {"kind": kind, "messages": messages}


def _swe(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("resolved") != 1 or row.get("license") not in _LICENSES:
        return None
    tools = _tools(row.get("tools"))
    messages = _openai(row.get("messages"))
    if messages is None or not tools:
        return None
    return {"kind": "swe", "messages": messages, "tools": tools}


def _tools(raw: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    found: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                return None
        if not isinstance(item, dict):
            return None
        found.append(item)
    return found


def _openai(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    pending: list[str] = []
    kept: list[dict[str, Any]] = []
    number = 0
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, list) and calls:
            shaped = []
            for call in calls:
                if not isinstance(call, dict):
                    return None
                function = call.get("function")
                if not isinstance(function, dict) or not function.get("name"):
                    return None
                try:
                    arguments = _arguments(function.get("arguments"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
                if _email_in(arguments):
                    return None
                number += 1
                shaped.append(
                    {
                        "id": str(call.get("id") or f"call-{number}"),
                        "type": "function",
                        "function": {"name": function["name"], "arguments": arguments},
                    }
                )
                pending.append(shaped[-1]["id"])
            item: dict[str, Any] = {"role": "assistant", "tool_calls": shaped}
            content = _text(message.get("content"))
            if content is None or _email_in(content):
                return None
            if content.strip():
                item["content"] = content
            kept.append(item)
            continue
        if role == "tool":
            if not pending:
                return None
            content = _text(message.get("content"))
            if content is None or _email_in(content):
                return None
            kept.append({"role": "tool", "tool_call_id": pending.pop(0), "content": content})
            continue
        if role not in {"system", "user", "assistant"}:
            continue
        content = _text(message.get("content"))
        if content is None or _email_in(content):
            return None
        if content.strip():
            kept.append({"role": role, "content": content})
    while pending and kept and kept[-1].get("role") == "assistant":
        calls = kept[-1].get("tool_calls") or []
        extra = len(pending)
        if len(calls) <= extra:
            pending = pending[: extra - len(calls)]
            kept.pop()
        else:
            kept[-1]["tool_calls"] = calls[: len(calls) - extra]
            pending = []
    if pending or len(kept) < 2:
        return None
    return kept


def _tool(conversations: Any, tools: Any) -> dict[str, Any] | None:
    parsed = _tools(tools)
    if parsed is None or not isinstance(conversations, list):
        return None
    pending: list[str] = []
    kept: list[dict[str, Any]] = []
    number = 0
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        who = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str) or _EMAIL.search(value):
            return None
        if who == "system":
            continue
        if who == "human":
            if pending or not value.strip():
                return None
            kept.append({"role": "user", "content": value})
            continue
        if who == "gpt":
            calls = _tagged(value, "tool_call")
            if calls is None:
                return None
            if calls:
                shaped = []
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                        return None
                    try:
                        arguments = _arguments(call.get("arguments"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        return None
                    number += 1
                    shaped.append(
                        {
                            "id": f"call-{number}",
                            "type": "function",
                            "function": {"name": call["name"], "arguments": arguments},
                        }
                    )
                    pending.append(f"call-{number}")
                kept.append({"role": "assistant", "tool_calls": shaped})
            elif value.strip():
                kept.append({"role": "assistant", "content": value.strip()})
            continue
        if who == "tool":
            responses = _tagged(value, "tool_response")
            if not responses or len(responses) > len(pending):
                return None
            for response in responses:
                if not isinstance(response, dict):
                    return None
                content = _text(response.get("content", ""))
                if content is None:
                    return None
                kept.append({"role": "tool", "tool_call_id": pending.pop(0), "content": content})
            continue
        return None
    if pending or len(kept) < 2:
        return None
    return {"kind": "tool", "messages": kept, "tools": parsed}


def _code(instruction: Any, response: Any) -> list[dict[str, str]] | None:
    if not isinstance(instruction, str) or not isinstance(response, str):
        return None
    if _EMAIL.search(instruction) or _EMAIL.search(response):
        return None
    if len(instruction) + len(response) < 800:
        return None
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def _chat(messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    kept: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            return None
        if _EMAIL.search(content):
            return None
        kept.append({"role": role, "content": content})
    if sum(len(item["content"]) for item in kept) < 800:
        return None
    return kept


def _email_in(value: Any) -> bool:
    if isinstance(value, str):
        return _EMAIL.search(value) is not None
    if isinstance(value, dict):
        return any(_email_in(item) for item in value.values())
    if isinstance(value, list):
        return any(_email_in(item) for item in value)
    return False


def _arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    message = "tool arguments are not an object"
    raise TypeError(message)


def _tagged(text: str, tag: str) -> list[Any] | None:
    needle = f"<{tag}>"
    close = f"</{tag}>"
    found: list[Any] = []
    cursor = 0
    while True:
        begin = text.find(needle, cursor)
        if begin < 0:
            return found
        stop = text.find(close, begin + len(needle))
        if stop < 0:
            return None
        try:
            found.append(json.loads(text[begin + len(needle) : stop].strip()))
        except json.JSONDecodeError:
            return None
        cursor = stop + len(close)


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value)
    return None


def _manifest(counts: dict[str, int]) -> dict[str, Any]:
    sources = []
    for source in SOURCES:
        sources.append(
            {
                "kind": source["kind"],
                "repo": source["repo"],
                "revision": source["revision"],
                "path": source["path"],
                "url": file_url(source),
                "license": source["license"],
            }
        )
    return {
        "sequences": sum(count for _kind, count in QUOTAS),
        "sequence_length": SEQUENCE_LENGTH,
        "quotas": [list(item) for item in QUOTAS],
        "counts": counts,
        "files": [
            {
                "kind": source["kind"],
                "file": f"{source['kind']}.json",
                "count": counts[source["kind"]],
            }
            for source in SOURCES
            if source["kind"] in counts
        ],
        "sources": sources,
    }


def _publish(
    folder: Path, groups: dict[str, list[dict[str, Any]]], manifest: dict[str, Any]
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for kind, rows in groups.items():
        body = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
        _replace(folder / f"{kind}.json", body)
    _replace(folder / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def _replace(path: Path, text: str) -> None:
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def main() -> None:
    raw = [arg for arg in sys.argv[1:] if arg.strip()]
    if not raw:
        message = "name a calibration file: swe.json, code.json, tool.json, or chat.json"
        raise SystemExit(message)
    folder = build(kinds=raw)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ordered: list[str] = []
    for arg in raw:
        name = _kind_name(arg)
        if name not in ordered:
            ordered.append(name)
    written = ", ".join(f"{name}.json" for name in ordered)
    print(f"wrote {written} in {folder}")
    counts = {item["kind"]: item["count"] for item in manifest["files"]}
    for name in ordered:
        print(f"{name}.json {counts[name]}")


if __name__ == "__main__":
    main()

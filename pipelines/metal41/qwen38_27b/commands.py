"""``favlo --qwen38-27b`` commands. The listener stays ``favlo serve``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.hf_cache import snapshot_dir
from favlo.prefix import checkpoint_digest, default_prefix_dir
from favlo.registry import (
    add_model,
    choose_complete,
    format_listing,
    load_registry,
    models_dir,
    register_quantized,
    save_registry,
    set_alias,
    stored_directory,
)
from qwen38_27b.chat import STOP, decode_text, encode_chat
from qwen38_27b.checks import format_report, run_checks
from qwen38_27b.convert import quantize, repack_mtp_fp8
from qwen38_27b.fetch import (
    CHECKOUT,
    REPO,
    download,
    list_files,
    remaining_download,
    shortfall,
)
from qwen38_27b.forward import Qwen38Runtime

PIPELINE = "qwen38-27b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)
_BF16_REPO = "Qwen/Qwen3.8-27B"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --qwen38-27b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the fp4 record, or a mixed feed-forward record")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument(
        "--ffn-fp8",
        nargs="?",
        const="",
        default=None,
        help="percentage of language feed-forward layers to store as fp8; empty promotes none",
    )
    fetch = sub.add_parser("download", help="download the official BF16 checkout")
    fetch.add_argument("--check", action="store_true", help="report disk space and do not write")
    fetch.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    sub.add_parser("bf16", help="run this package's shaders on stored BF16 values")
    complete = sub.add_parser("complete", help="argmax completion from the packed record")
    complete.add_argument("--kv-cache", choices=("e4m3",), default="e4m3")
    complete.add_argument("--max-memory", default=None, help="maximum bytes, such as 32GiB")
    sub.add_parser("kernels", help="compile the shader library and run the dispatch checks")
    mtp_fp8 = sub.add_parser("mtp-fp8", help="repack MTP matrices as fp8 from the BF16 cache")
    mtp_fp8.add_argument("--src", type=Path, default=None)
    mtp_fp8.add_argument("--record", type=Path, default=None)
    delete = sub.add_parser("delete", help="remove a packed model and its registry row")
    delete.add_argument("model", help="folder name or alias")
    models = sub.add_parser("models", help="list or edit the flavor registry")
    models_sub = models.add_subparsers(dest="models_command")
    add = models_sub.add_parser("add", help="register a finished record directory")
    add.add_argument("folder")
    add.add_argument("--alias", default=None)
    add.add_argument("--source", default=None)
    alias = models_sub.add_parser("alias", help="change the alias of a folder")
    alias.add_argument("folder")
    alias.add_argument("alias")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "models":
        return _models(args)
    if args.command == "kernels":
        return _kernels()
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "mtp-fp8":
        return _mtp_fp8(args)
    if args.command == "download":
        return _download(args)
    if args.command == "delete":
        return _delete(args)
    if args.command == "complete":
        return _complete(args)
    sys.stderr.write("qwen38-27b bf16 layer check is not this command\n")
    return 1


def _mtp_fp8(args: argparse.Namespace) -> int:
    """Repack the installed record's MTP matrices. The BF16 checkout stays where it is."""
    try:
        src = args.src or snapshot_dir(_BF16_REPO)
        record = args.record or installed_record()
    except (OSError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if record is None:
        sys.stderr.write("quantized record is not installed\n")
        return 1
    try:
        repack_mtp_fp8(Path(record), Path(src))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


def _download(args: argparse.Namespace) -> int:
    """Fetch Qwen/Qwen3.8-27B into ``models/Qwen3.8-27B-bf16``. Quantize is separate."""
    dest = models_dir() / CHECKOUT
    if not args.check and (dest / "fetch_complete.json").is_file():
        sys.stdout.write(f"{dest}\n")
        return 0
    try:
        files = list_files()
        message = shortfall(remaining_download(dest, files), dest)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if message is not None:
        sys.stderr.write(message + "\n")
        return 1
    if args.check:
        sys.stdout.write("ok\n")
        return 0
    if (dest / "fetch_complete.json").is_file():
        sys.stdout.write(f"{dest}\n")
        return 0
    from favlo.console import Console

    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    bar = None
    try:
        total = sum(size for _name, size in files)
        bar = ui.begin("download", total=total, unit="bytes", detail=REPO)
        download(dest, files, lambda done, whole, _name: _move(bar, done, whole))
        ui.end(bar, f"downloaded {REPO}")
        bar = None
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next download continues from the files already written")
        return 1
    finally:
        if bar is not None and not bar.closed:
            ui.end(bar, "")
        ui.close()
    sys.stdout.write(f"{dest}\n")
    return 0


def _move(progress: Any, done: int, total: int) -> None:
    progress.done = done
    progress.total = total


def _ensure_mtp_on(folder: str, root: Path | None = None) -> None:
    """Store MTP on for a 27B row that does not already say. A stored flag stays."""
    data = load_registry(root)
    entry = data["models"].get(folder)
    if not isinstance(entry, dict) or "mtp" in entry:
        return
    entry["mtp"] = True
    entry["mtp_depth"] = 3
    save_registry(data, root)


def installed_record() -> Path | None:
    """The one registered record for this pipeline. None when the registry has no row."""
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
    except LookupError:
        return None
    if not (path / "manifest.json").is_file():
        return None
    return path


def _percent(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        message = f"ffn fp8 percentage {text} is not a number"
        raise ValueError(message) from exc


def _quantize(args: argparse.Namespace) -> int:
    src = args.src or snapshot_dir(_BF16_REPO)
    try:
        output = quantize(src, args.output, ffn_fp8=_percent(args.ffn_fp8))
        register_quantized(
            output.name,
            source=src,
            alias="qwen38-27b",
            directory=output,
            pipeline=PIPELINE,
            runtime="metal41",
        )
        _ensure_mtp_on(output.name)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
        runtime = Qwen38Runtime(path, kv_cache=args.kv_cache, max_memory=args.max_memory)
        try:
            messages = [{"role": "user", "content": CARROT_PROMPT}]
            prompt = encode_chat(runtime.root, messages, None, False)
            tokens = runtime.generate(prompt, None, STOP)
            text = decode_text(runtime.root, tokens)
        finally:
            runtime.close()
        body = {"prompt": CARROT_PROMPT, "text": text}
        target = path / "carrots.json"
        target.write_text(json.dumps(body, indent=2) + "\n")
    except (OSError, RuntimeError, LookupError, ValueError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(text + "\n")
    return 0


def _delete(args: argparse.Namespace) -> int:
    """Drop one packed model, its prefix partition, a private checkout, and the row."""
    text = args.model.strip()
    base = text.split(":", 1)[0] if ":" in text else text
    if not base:
        sys.stderr.write("model id is empty\n")
        return 1
    try:
        data = load_registry()
        folder = _registered_folder(data, base)
        if folder is None:
            sys.stderr.write(f"unknown model {text}\n")
            return 1
        entry = data["models"][folder]
        if entry.get("pipeline") != PIPELINE:
            sys.stderr.write(f"{folder} is not a {PIPELINE} model\n")
            return 1
        source = entry.get("source")
        record = stored_directory(entry, folder)
        _drop_prefix(record, data, folder)
        _remove_checkout(record)
        if isinstance(source, str) and source and not _source_shared(data, folder, source):
            _remove_checkout(Path(source))
        data["models"].pop(folder, None)
        save_registry(data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"deleted {folder}\n")
    return 0


def _registered_folder(data: dict, base: str) -> str | None:
    models = data["models"]
    if base in models:
        return base
    for folder, entry in models.items():
        if entry.get("alias") == base:
            return folder
    return None


def _source_shared(data: dict, folder: str, source: str) -> bool:
    try:
        wanted = Path(source).expanduser().resolve()
    except OSError:
        return True
    for other, entry in data["models"].items():
        if other == folder:
            continue
        stored = entry.get("source")
        if not isinstance(stored, str) or not stored:
            continue
        try:
            if Path(stored).expanduser().resolve() == wanted:
                return True
        except OSError:
            continue
    return False


def _drop_prefix(record: Path, data: dict, folder: str) -> None:
    try:
        digest = checkpoint_digest(record).hex()
    except OSError:
        return
    for other, entry in data["models"].items():
        if other == folder:
            continue
        try:
            other_record = stored_directory(entry, other)
            if checkpoint_digest(other_record).hex() == digest:
                return
        except OSError:
            continue
    partition = default_prefix_dir() / PIPELINE / digest
    if partition.is_symlink() or not partition.is_dir():
        return
    root = default_prefix_dir().resolve()
    target = partition.resolve()
    if target == root or root not in target.parents:
        return
    shutil.rmtree(target)


def _remove_checkout(path: Path) -> None:
    """Remove a directory that lives under ``models/``."""
    root = models_dir().resolve()
    try:
        target = path.expanduser().resolve()
    except OSError:
        return
    if target == root or root not in target.parents:
        return
    if path.is_symlink() or not target.is_dir():
        return
    shutil.rmtree(target)


def _kernels() -> int:
    try:
        report = run_checks()
    except (OSError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(format_report(report) + "\n")
    return 0 if report["ok"] else 1


def _models(args: argparse.Namespace) -> int:
    command = getattr(args, "models_command", None)
    try:
        if command == "add":
            add_model(
                args.folder,
                alias=args.alias,
                source=args.source,
                pipeline=PIPELINE,
                runtime="metal41",
            )
            _ensure_mtp_on(args.folder)
        elif command == "alias":
            set_alias(args.folder, args.alias)
        listing = format_listing()
        if listing:
            sys.stdout.write(listing + "\n")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

"""``favlo --qwen36-35b-a3b`` commands. The listener stays ``favlo serve``."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from favlo.console import Console, Progress
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
from qwen36_35b_a3b.chat import STOP, decode_text, encode_chat, instruct_forbid
from qwen36_35b_a3b.checks import format_report, run_checks
from qwen36_35b_a3b.convert import quantize, record_name, requant_mtp
from qwen36_35b_a3b.fetch import (
    REPO,
    download,
    list_files,
    record_bytes,
    remaining_download,
    shortfall,
)
from qwen36_35b_a3b.formats import promote_count
from qwen36_35b_a3b.forward import Qwen36Runtime

PIPELINE = "qwen36-35b-a3b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)
_BF16 = Path("models/Qwen3.6-35B-A3B-bf16")
_ALIAS = "qwen36-35b-a3b"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _expert_budget(text: str) -> float:
    """Accept a routed-expert fp8 percentage from 0 to 100."""
    try:
        percent = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expert fp8 percentage is not a number") from exc
    try:
        promote_count(percent, 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return percent


def _repo_id(text: str) -> str:
    """``org/name``, or the same id taken from a huggingface.co model link."""
    raw = text.strip()
    if "://" in raw or raw.startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            message = "repository link is a huggingface.co model"
            raise argparse.ArgumentTypeError(message)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            message = "repository is org/name or a huggingface.co link"
            raise argparse.ArgumentTypeError(message)
        raw = f"{parts[0]}/{parts[1]}"
    if _REPO_ID.fullmatch(raw) is None:
        message = "repository is org/name or a huggingface.co link"
        raise argparse.ArgumentTypeError(message)
    return raw


def _folder_name(text: str) -> str:
    """One directory name under ``models/``. That name is the registry id."""
    if _FOLDER.fullmatch(text) is None:
        message = "folder name is not a single name"
        raise argparse.ArgumentTypeError(message)
    return text


def _checkout_name(repo: str) -> str:
    """BF16 directory under ``models/``. The official checkout name stays put."""
    if repo == REPO:
        return _BF16.name
    org, leaf = repo.split("/", 1)
    return f"{org}--{leaf}-bf16"


def _packed_dir(repo: str, name: str | None, percent: float) -> tuple[Path, Path, str | None]:
    """Checkout, packed folder, and alias.

    With no ``name``, the folder is the repository's model name plus the quant
    suffix. That folder is the registry id. The official default pack also
    keeps the alias ``qwen36-35b-a3b``.
    """
    src = models_dir() / _checkout_name(repo)
    if name is not None:
        return src, models_dir() / name, None
    folder = record_name(Path(repo.split("/", 1)[1]), percent)
    alias = _ALIAS if repo == REPO and percent == 20 else None
    return src, models_dir() / folder, alias


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --qwen36-35b-a3b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant.add_argument("--src", type=Path, default=None)
    place = quant.add_mutually_exclusive_group()
    place.add_argument("--output", type=Path, default=None)
    place.add_argument(
        "--name",
        type=_folder_name,
        default=None,
        help="folder under models/ and the registry id",
    )
    quant.add_argument(
        "--expert-fp8",
        type=_expert_budget,
        default=20,
        help="percentage of routed experts stored as fp8, from 0 to 100",
    )
    quant.add_argument(
        "--keep-bf16",
        action="store_true",
        help="leave the BF16 checkout under models/ after a successful quantize",
    )
    sub.add_parser("bf16", help="run this package's shaders on stored BF16 values")
    complete = sub.add_parser("complete", help="complete from the packed record")
    complete.add_argument(
        "--kv-cache",
        choices=("e4m3", "e2m1", "f32"),
        default="e4m3",
        help="key/value width: fp8, fp4, or float32",
    )
    complete.add_argument(
        "--max-memory",
        default=None,
        help="maximum bytes for this runtime, such as 32GiB",
    )
    sub.add_parser("kernels", help="compile the shader library and run the dispatch checks")
    setup = sub.add_parser("setup", help="download a checkpoint and pack fp8-e20")
    setup.add_argument(
        "--repo",
        type=_repo_id,
        default=REPO,
        help="Hugging Face repository org/name",
    )
    setup.add_argument(
        "--name",
        type=_folder_name,
        default=None,
        help="folder under models/ and the registry id",
    )
    setup.add_argument(
        "--expert-fp8",
        type=_expert_budget,
        default=20,
        help="percentage of routed experts stored as fp8, from 0 to 100",
    )
    setup.add_argument(
        "--keep-bf16",
        action="store_true",
        help="leave the BF16 checkout under models/ after a successful quantize",
    )
    setup.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    mtp = sub.add_parser("mtp-fp8", help="store every MTP matrix as fp8 from a local checkout")
    mtp.add_argument("--src", type=Path, required=True, help="local BF16 checkout")
    mtp.add_argument("--record", type=Path, required=True, help="finished packed record")
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
    if args.command == "setup":
        return _setup(args)
    if args.command == "delete":
        return _delete(args)
    if args.command == "complete":
        return _complete(args)
    sys.stderr.write("qwen36-35b-a3b bf16 layer check is not this command\n")
    return 1


def _mtp_fp8(args: argparse.Namespace) -> int:
    """Rewrite the MTP block of a finished record. Language experts stay."""
    requant_mtp(args.src, args.record)
    return 0


def _setup(args: argparse.Namespace) -> int:
    """Download a checkpoint, pack it, and remove the checkout."""
    repo = args.repo
    name = args.name
    percent = float(args.expert_fp8)
    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    src, output, alias = _packed_dir(repo, name, percent)
    download_bar = None
    quantize_bar = None
    try:
        if _manifest_complete(output):
            _register(output, src, alias)
            if not args.keep_bf16:
                _remove_checkout(src, spare=output)
            ui.log(f"record complete: {output}")
            return 0
        files = list_files(repo=repo)
        need_download = remaining_download(src, files)
        have_record = _tree_bytes(output)
        need_record = max(0, record_bytes(percent) - have_record)
        message = shortfall(need_download, need_record, src, output)
        if message is not None:
            ui.log(message)
            return 1
        total = sum(size for _name, size in files)
        download_bar = ui.begin("download", total=total, unit="bytes", detail=repo)
        download(
            src,
            files,
            lambda done, whole, _name: _move(download_bar, done, whole, repo),
            repo=repo,
        )
        ui.end(download_bar, f"downloaded {repo}")
        download_bar = None
        quantize_bar = ui.begin(
            "quantize",
            total=record_bytes(percent),
            unit="bytes",
            detail=output.name,
        )

        def _on_bytes(done: int, detail: str) -> None:
            assert quantize_bar is not None
            quantize_bar.done = done
            if done > quantize_bar.total:
                quantize_bar.total = done
            quantize_bar.detail = detail

        quantize(src, output, on_bytes=_on_bytes, expert_fp8=percent)
        quantize_bar.done = quantize_bar.total
        ui.end(quantize_bar, f"record complete: {output}")
        quantize_bar = None
        _register(output, src, alias)
        if not args.keep_bf16:
            _remove_checkout(src, spare=output)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next serve continues from the files already written")
        return 1
    finally:
        if download_bar is not None and not download_bar.closed:
            ui.end(download_bar, "")
        if quantize_bar is not None and not quantize_bar.closed:
            ui.end(quantize_bar, "")
        ui.close()


def _move(progress: Progress, done: int, total: int, detail: str) -> None:
    progress.done = done
    progress.total = total
    progress.detail = detail


def _manifest_complete(output: Path) -> bool:
    path = output / "manifest.json"
    if not path.is_file():
        return False
    body = json.loads(path.read_text())
    return bool(body.get("complete"))


def _tree_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _register(output: Path, source: Path, alias: str | None) -> None:
    register_quantized(
        output.name,
        source=source,
        alias=alias,
        directory=output,
        pipeline=PIPELINE,
        runtime="metal41",
    )


def _remove_checkout(path: Path, *, spare: Path | None = None) -> None:
    """Remove a BF16 checkout that lives under ``models/``."""
    root = models_dir().resolve()
    try:
        target = path.expanduser().resolve()
        spared = None if spare is None else spare.expanduser().resolve()
    except OSError:
        return
    if target == root or root not in target.parents:
        return
    if spared is not None and (target == spared or target in spared.parents):
        return
    if path.is_symlink() or not target.is_dir():
        return
    shutil.rmtree(target)


def _quantize(args: argparse.Namespace) -> int:
    src = args.src or _BF16
    percent = float(args.expert_fp8)
    name = args.name
    if name is not None:
        output = models_dir() / name
        alias = None
    elif args.output is not None:
        output = args.output
        alias = _ALIAS
    else:
        output = Path("models") / record_name(Path(src), percent)
        alias = _ALIAS
    try:
        quantize(src, output, expert_fp8=percent)
        register_quantized(
            output.name,
            source=src,
            alias=alias,
            directory=output,
            pipeline=PIPELINE,
            runtime="metal41",
        )
        if not args.keep_bf16:
            _remove_checkout(src, spare=output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
        runtime = Qwen36Runtime(
            path,
            kv_cache=args.kv_cache,
            max_memory=args.max_memory,
        )
        try:
            messages = [{"role": "user", "content": CARROT_PROMPT}]
            prompt = encode_chat(runtime.root, messages, None, False)
            tokens = runtime.generate(
                prompt,
                max(0, runtime.max_pos - len(prompt)),
                0.0,
                1.0,
                0,
                0.0,
                STOP,
                forbid=instruct_forbid(runtime.root),
            )
            text = decode_text(runtime.root, tokens)
        finally:
            runtime.close()
        body = {"prompt": CARROT_PROMPT, "text": text}
        target = path / "carrots.json"
        target.write_text(json.dumps(body, indent=2) + "\n")
    except (OSError, RuntimeError, LookupError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(text + "\n")
    return 0


def _kernels() -> int:
    try:
        report = run_checks()
    except (OSError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(format_report(report) + "\n")
    return 0 if report["ok"] else 1


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
        elif command == "alias":
            set_alias(args.folder, args.alias)
        listing = format_listing()
        if listing:
            sys.stdout.write(listing + "\n")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

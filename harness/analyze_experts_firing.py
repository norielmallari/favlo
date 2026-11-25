"""Summarize MoE expert firing against the packed fp8 width plane.

The same sqlite, widths file, and ``--since`` bound always print the same
capture and ideal ranks. A mid-forward stop is not in the log.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from favlo.cli import ensure_pipelines_path, repo_root
from favlo.logs import favlo_home

LAYERS = 40
EXPERTS = 256
TOP_K = 8
FP8 = 8
FP4 = 4


def plane_totals(
    path: Path,
    *,
    layers: int = LAYERS,
    experts: int = EXPERTS,
    since: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefill and decode hit sums. Shape of each is ``(layers, experts)``."""
    prefill = np.zeros((layers, experts), dtype=np.uint64)
    decode = np.zeros((layers, experts), dtype=np.uint64)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        if since is None:
            rows = db.execute("SELECT layer, prefill, decode FROM layer_hits")
        else:
            rows = db.execute(
                "SELECT layer_hits.layer, layer_hits.prefill, layer_hits.decode"
                " FROM layer_hits"
                " JOIN request ON request.id = layer_hits.request_id"
                " WHERE request.finished_at >= ?",
                (int(since),),
            )
        for layer, pre_blob, dec_blob in rows:
            if layer < 0 or layer >= layers:
                continue
            pre = np.frombuffer(pre_blob, dtype=np.uint32)
            dec = np.frombuffer(dec_blob, dtype=np.uint32)
            if pre.size != experts or dec.size != experts:
                message = f"layer {layer} hit vector length is not {experts}"
                raise ValueError(message)
            prefill[layer] += pre.astype(np.uint64)
            decode[layer] += dec.astype(np.uint64)
    return prefill, decode


def request_totals(path: Path, *, since: int | None = None) -> dict[str, float]:
    """Request count, token sums, and clocks in the window."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        if since is None:
            row = db.execute(
                "SELECT COUNT(*), MIN(finished_at), MAX(finished_at),"
                " SUM(prefill_tokens), SUM(decode_tokens), SUM(cached_tokens),"
                " SUM(prefill_seconds), SUM(decode_seconds) FROM request"
            ).fetchone()
        else:
            row = db.execute(
                "SELECT COUNT(*), MIN(finished_at), MAX(finished_at),"
                " SUM(prefill_tokens), SUM(decode_tokens), SUM(cached_tokens),"
                " SUM(prefill_seconds), SUM(decode_seconds)"
                " FROM request WHERE finished_at >= ?",
                (int(since),),
            ).fetchone()
    count = int(row[0] or 0)
    if count == 0:
        return {
            "requests": 0.0,
            "finished_min": 0.0,
            "finished_max": 0.0,
            "prefill_tokens": 0.0,
            "decode_tokens": 0.0,
            "cached_tokens": 0.0,
            "prefill_seconds": 0.0,
            "decode_seconds": 0.0,
        }
    return {
        "requests": float(count),
        "finished_min": float(row[1] or 0),
        "finished_max": float(row[2] or 0),
        "prefill_tokens": float(row[3] or 0),
        "decode_tokens": float(row[4] or 0),
        "cached_tokens": float(row[5] or 0),
        "prefill_seconds": float(row[6] or 0),
        "decode_seconds": float(row[7] or 0),
    }


def widths_layout(path: Path) -> tuple[int, int, float]:
    """Layer count, expert count, and expected hits per token for a widths file.

    Forty layers selecting 8 experts is 320. Inkling stores 24 layers and visits
    48 of them, so eight experts a visit is 384.
    """
    if not path.is_file():
        return LAYERS, EXPERTS, float(LAYERS * TOP_K)
    size = path.stat().st_size
    if size == LAYERS * EXPERTS:
        return LAYERS, EXPERTS, float(LAYERS * TOP_K)
    if size == 24 * 128:
        return 24, 128, 384.0
    message = f"widths.bin length {size} is not a known expert plane"
    raise ValueError(message)


def load_widths(path: Path, *, layers: int = LAYERS, experts: int = EXPERTS) -> np.ndarray:
    """Width plane. Missing file is all fp4."""
    if not path.is_file():
        return np.full((layers, experts), FP4, dtype=np.uint8)
    widths = np.fromfile(path, dtype=np.uint8)
    if widths.size != layers * experts:
        message = f"widths.bin length {widths.size} is not {layers * experts}"
        raise ValueError(message)
    return widths.reshape(layers, experts)


def top_mask(hits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-layer top-k mask. Equal hits keep the lower expert index."""
    layers, experts = hits.shape
    chosen = np.zeros((layers, experts), dtype=bool)
    for layer in range(layers):
        k = int(counts[layer])
        if k <= 0:
            continue
        order = np.lexsort((np.arange(experts), -hits[layer].astype(np.int64)))
        chosen[layer, order[:k]] = True
    return chosen


def global_top_mask(hits: np.ndarray, k: int) -> np.ndarray:
    """Flat top-k mask. Equal hits keep the lower flat index."""
    flat = hits.ravel()
    n = flat.size
    if k <= 0:
        return np.zeros(hits.shape, dtype=bool)
    order = np.lexsort((np.arange(n), -flat.astype(np.int64)))
    mask = np.zeros(n, dtype=bool)
    mask[order[: min(k, n)]] = True
    return mask.reshape(hits.shape)


def capture(hits: np.ndarray, mask: np.ndarray) -> float:
    total = int(hits.sum())
    if total <= 0:
        return 0.0
    return float(hits[mask].sum()) / float(total)


def scores(hits: np.ndarray) -> np.ndarray:
    """Relative score per expert. A uniform expert is 1."""
    out = np.zeros(hits.shape, dtype=np.float64)
    layer_hits = hits.sum(axis=1)
    experts = hits.shape[1]
    for layer in range(hits.shape[0]):
        if layer_hits[layer] <= 0:
            continue
        out[layer] = hits[layer] / (float(layer_hits[layer]) / experts)
    return out


def report(
    *,
    folder: str,
    db: Path,
    widths_path: Path,
    since: int | None,
    layers: int = LAYERS,
    experts: int = EXPERTS,
    expected_hits: float | None = None,
) -> dict[str, Any]:
    """Numbers for one packed folder. Empty when no request finished in the window."""
    widths = load_widths(widths_path, layers=layers, experts=experts)
    fp8 = widths == FP8
    req = request_totals(db, since=since)
    if req["requests"] <= 0:
        return {
            "folder": folder,
            "requests": 0,
            "fp8_experts": int(fp8.sum()),
            "fp4_experts": int((widths == FP4).sum()),
            "total_experts": int(layers * experts),
        }
    prefill, decode = plane_totals(db, layers=layers, experts=experts, since=since)
    both = prefill + decode
    per_layer = fp8.sum(axis=1)
    ideal_layer = top_mask(both, per_layer)
    ideal_global = global_top_mask(both, int(fp8.sum()))
    rel = scores(both)
    tokens = req["prefill_tokens"] + req["decode_tokens"]
    total_hits = float(both.sum())
    return {
        "folder": folder,
        "requests": int(req["requests"]),
        "finished_min": int(req["finished_min"]),
        "finished_max": int(req["finished_max"]),
        "span_hours": (req["finished_max"] - req["finished_min"]) / 3600.0,
        "prefill_tokens": int(req["prefill_tokens"]),
        "decode_tokens": int(req["decode_tokens"]),
        "cached_tokens": int(req["cached_tokens"]),
        "prefill_tok_s": (
            req["prefill_tokens"] / req["prefill_seconds"] if req["prefill_seconds"] > 0 else 0.0
        ),
        "decode_tok_s": (
            req["decode_tokens"] / req["decode_seconds"] if req["decode_seconds"] > 0 else 0.0
        ),
        "fp8_experts": int(fp8.sum()),
        "fp4_experts": int((widths == FP4).sum()),
        "total_experts": int(layers * experts),
        "total_hits": int(total_hits),
        "hits_per_token": total_hits / tokens if tokens > 0 else 0.0,
        "capture_pct": 100.0 * capture(both, fp8),
        "prefill_capture_pct": 100.0 * capture(prefill, fp8),
        "decode_capture_pct": 100.0 * capture(decode, fp8),
        "ideal_global_pct": 100.0 * capture(both, ideal_global),
        "ideal_layer_pct": 100.0 * capture(both, ideal_layer),
        "overlap_global": int((fp8 & ideal_global).sum()),
        "overlap_layer": int((fp8 & ideal_layer).sum()),
        "dead_experts": int((both == 0).sum()),
        "fp8_score_median": float(np.median(rel[fp8])) if fp8.any() else 0.0,
        "fp4_score_median": float(np.median(rel[~fp8])) if (~fp8).any() else 0.0,
        "score_gt_1": int((rel > 1.0).sum()),
        "score_gt_1_fp8": int(((rel > 1.0) & fp8).sum()),
        "expected_hits_per_token": (
            float(layers * TOP_K) if expected_hits is None else float(expected_hits)
        ),
    }


def format_report(summary: dict[str, Any], *, tz: timezone | None = None) -> str:
    """Stable text. Percents are two decimals. Rates are two decimals."""
    folder = summary["folder"]
    lines = [f"folder {folder}"]
    lines.append(f"fp8_experts {summary['fp8_experts']}/{summary['total_experts']}")
    if summary["requests"] == 0:
        lines.append("requests 0")
        return "\n".join(lines)
    zone = timezone.utc if tz is None else tz
    start = datetime.fromtimestamp(summary["finished_min"], zone).strftime("%Y-%m-%d %H:%M")
    end = datetime.fromtimestamp(summary["finished_max"], zone).strftime("%Y-%m-%d %H:%M")
    lines.append(f"requests {summary['requests']}")
    lines.append(f"from {start}")
    lines.append(f"to {end}")
    lines.append(f"span_hours {summary['span_hours']:.2f}")
    lines.append(f"prefill_tokens {summary['prefill_tokens']}")
    lines.append(f"decode_tokens {summary['decode_tokens']}")
    lines.append(f"prefill_tok_s {summary['prefill_tok_s']:.2f}")
    lines.append(f"decode_tok_s {summary['decode_tok_s']:.2f}")
    lines.append(f"total_hits {summary['total_hits']}")
    lines.append(f"hits_per_token {summary['hits_per_token']:.2f}")
    lines.append(f"capture_pct {summary['capture_pct']:.2f}")
    lines.append(f"prefill_capture_pct {summary['prefill_capture_pct']:.2f}")
    lines.append(f"decode_capture_pct {summary['decode_capture_pct']:.2f}")
    lines.append(f"ideal_global_pct {summary['ideal_global_pct']:.2f}")
    lines.append(f"ideal_layer_pct {summary['ideal_layer_pct']:.2f}")
    lines.append(f"overlap_global {summary['overlap_global']}/{summary['fp8_experts']}")
    lines.append(f"overlap_layer {summary['overlap_layer']}/{summary['fp8_experts']}")
    lines.append(f"dead_experts {summary['dead_experts']}")
    lines.append(f"fp8_score_median {summary['fp8_score_median']:.3f}")
    lines.append(f"fp4_score_median {summary['fp4_score_median']:.3f}")
    lines.append(f"score_gt_1 {summary['score_gt_1']}")
    lines.append(f"score_gt_1_fp8 {summary['score_gt_1_fp8']}")
    return "\n".join(lines)


def resolve_folder(model: str, *, root: Path) -> tuple[str, Path]:
    """Alias or folder name to the packed directory."""
    from favlo.registry import resolve

    selection = resolve(model, root=root)
    return selection.folder, selection.path


def list_folders(*, home: Path, root: Path) -> list[tuple[str, Path, Path]]:
    """Telemetry folders that have a sqlite log and a models/ record."""
    telemetry = home / "telemetry"
    if not telemetry.is_dir():
        return []
    found: list[tuple[str, Path, Path]] = []
    for child in sorted(telemetry.iterdir()):
        db = child / "firing.sqlite"
        if not child.is_dir() or not db.is_file():
            continue
        path = root / "models" / child.name
        if not path.is_dir():
            continue
        found.append((child.name, db, path))
    return found


def analyze_folder(
    folder: str,
    *,
    root: Path,
    home: Path,
    since: int | None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    path = model_path if model_path is not None else root / "models" / folder
    db = home / "telemetry" / folder / "firing.sqlite"
    widths = path / "experts" / "widths.bin"
    if not db.is_file():
        message = f"firing log is missing for {folder}"
        raise FileNotFoundError(message)
    layers, experts, expected = widths_layout(widths)
    return report(
        folder=folder,
        db=db,
        widths_path=widths,
        since=since,
        layers=layers,
        experts=experts,
        expected_hits=expected,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print fp8 expert firing capture.")
    parser.add_argument(
        "--model",
        default=None,
        help="registry alias or packed folder. Omit to print every telemetry pack",
    )
    parser.add_argument(
        "--since",
        default="24h",
        help="Go duration, RFC 3339, or unix seconds. Default 24h",
    )
    parser.add_argument(
        "--now",
        type=int,
        default=None,
        help="unix seconds for duration --since. Tests pin this",
    )
    parser.add_argument(
        "--utc",
        action="store_true",
        help="print finished_at in UTC. Default is the local zone",
    )
    args = parser.parse_args(argv)
    root = repo_root()
    ensure_pipelines_path(root=root)
    from qwen36_35b_a3b.metal41.firing import parse_since

    clock = time.time() if args.now is None else float(args.now)
    since = parse_since(args.since, now=clock)
    home = favlo_home()
    zone = timezone.utc if args.utc else datetime.now().astimezone().tzinfo
    blocks: list[str] = []
    if args.model:
        folder, path = resolve_folder(args.model, root=root)
        summary = analyze_folder(folder, root=root, home=home, since=since, model_path=path)
        blocks.append(format_report(summary, tz=zone))
    else:
        packs = list_folders(home=home, root=root)
        if not packs:
            print("packs 0")
            return 0
        for folder, _db, path in packs:
            summary = analyze_folder(folder, root=root, home=home, since=since, model_path=path)
            blocks.append(format_report(summary, tz=zone))
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

---
name: analyze-experts-firing
description: Summarize MoE expert firing from a packed record's SQLite log and measure how much of that traffic lands on the fp8 experts. Use when the user asks for firing rate, fp8 capture, expert hits over 24h or another window, whether a firing promote list still matches live traffic, or what capture a firing requant would get on this log. Do not use for weight-plane RMS promote. Do not run quantize or write a new pack.
---

# Analyze experts firing

One packed folder. The log is written only after a finished request. A mid-forward stop is not stored. Today only `qwen36_35b_a3b` writes this log.

Run the harness script. Do not invent hit counts. Do not requant from this skill.

```bash
uv run python -m harness.analyze_experts_firing --model <alias-or-folder>
uv run python -m harness.analyze_experts_firing --model <alias-or-folder> --since 24h
uv run python -m harness.analyze_experts_firing
```

Omitting `--model` prints every telemetry pack that has a matching `models/<folder>` record. `--since` is a Go duration (`24h`, `1h30m`), an RFC 3339 timestamp, or unix seconds. Default is `24h`. `--utc` prints finished times in UTC; omit it for the local zone. `--now` pins the clock for a duration bound (tests).

The script resolves the alias to the packed folder basename. The log is `~/.favlo/telemetry/<folder>/firing.sqlite`. The width plane is `models/<folder>/experts/widths.bin` (byte `8` is fp8, byte `4` is fp4).

## Printed fields

Read the script output. Do not recompute the ranks by hand unless the script failed.

| Field | Meaning |
|---|---|
| `capture_pct` | Share of prefill+decode hits on the current fp8 set |
| `prefill_capture_pct` / `decode_capture_pct` | Same share on each phase |
| `ideal_global_pct` | Share if the top `k` experts by hits were fp8 (`k` = current fp8 count). Same number a `--promote firing` requant would capture on this window at that `k` |
| `ideal_layer_pct` | Share if each layer kept its own top-`k_layer` set |
| `overlap_global` / `overlap_layer` | How many current fp8 experts sit in that ideal set. Slots that would move are `fp8_experts − overlap_global` |
| `hits_per_token` | Sanity check. This package selects 8 experts on 40 layers, so 320 |
| `fp8_score_median` / `fp4_score_median` | Relative score `hits / (layer_hits / 256)`. Uniform is 1 |

Equal hit counts keep the lower expert index, matching promote. A gap between capture and ideal means live traffic has drifted from the promote list.

The runtime's online relative-score window is the newest 16384 complete tokens. This script's window is `--since`. Do not mix the two without saying so.

## Current vs firing requant

When the question is the live rate, or what a firing requant would capture on this log:

1. Run the script for the pack (and `--since` when the window matters).
2. Report `capture_pct` as the current rate on that window.
3. Report `ideal_global_pct` as the rate a `--promote firing` requant would get on the **same** hits if it kept the current fp8 count (`k` = `fp8_experts`). That is the theoretical post-requant share for this log, not a forecast of later traffic.
4. Report how many slots would move: `fp8_experts − overlap_global`. Mention `ideal_layer_pct` and `overlap_layer` when they differ from the global figures.
5. Keep packs separate. Prefer the whole log for a requant answer unless the user names a window (`24h`, `--since`).

Do not run quantize from this skill. The ideal figure is the analysis; writing a new pack is out of scope.

## Reply shape

Lead with the pack name and `capture_pct`. Then the request span and token counts. Then prefill, decode, and both. Close with ideal share and overlap. When the user asks what a requant would get, lead with `ideal_global_pct` and name the window. When two packs are printed, keep them separate.

Do not quote home directories, hostnames, or private session ids. Placeholders stay `~/.favlo` and `example-model`.

## Out of scope

- `--promote error` and `expert_error.json` (weight-plane RMS).
- Writing a new pack or running quantize.
- Editing `widths.bin` or the atlas at runtime.

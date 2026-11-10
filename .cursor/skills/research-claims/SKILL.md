---
name: research-claims
description: Search public literature and write a Favlo research claim for one pipeline. Use when a needle run, a coding-agent run, or a synthetic-prompt run prints a prefill and decode rate, and when the user asks to document research claims, check an implementation against published work, or add a row to RESEARCH.md. File the printed rate when that plane, MTP setting, length, and card are not already a row.
---

# Research claims

Read [research-evidence.mdc](../../rules/research-evidence.mdc) and [RESEARCH.md](../../../RESEARCH.md) first. The rule decides where a claim lives and which words are allowed. This skill is the search and the filing.

Work on `main`. Do not create a branch unless the user asks. Land the note, the catalog row, and the README link in one change. Commit only when the user asks.

One pipeline. Do not edit another pipeline's package, shaders, tests, or docs. [RESEARCH.md](../../../RESEARCH.md) is the shared index and receives one new row.

## After a printed rate

A needle run, a coding-agent run, or `python -m harness.synthetic` that exits 0 and prints a prefill sum and a decode sum is a candidate. The synthetic prompt's end-to-end line is those two clocks added. File it on its own row for that plane, MTP setting, thinking card, and prompt length. Read [RESEARCH.md](../../../RESEARCH.md). When that pipeline, plane, MTP setting, mode, length, depth, thinking card, and vision flag are already a row, stop. Otherwise search and file in the same turn. Asking only to run the harness is enough. Page mode is a long-transcript rate and is filed. Retrieve mode is the retrieval table.

## Search

Read the serving note for the sentences that decide a behavior: a recorded difference, a refusal, or a result that is not yet what the server does. Ignore a paragraph that only restates a cited equation.

Split each candidate into the method, the model or task, the measurement, and the mechanism. Search papers, model cards, and shipped code. Then search again with the words those hits use. The closer paper often does not share the first query. Record every query, the date, and the predecessors.

A predecessor closes the part it already did. Name it with a link and one sentence. The sentence this repo may own is only the remainder. The "not found" sentence names that remainder's number or bound. It does not name a mechanism the predecessor already published. Do not write "first," "novel," or "no prior work."

A measured mapping onto one machine is a remainder even when the method is published. The table is the times, the bandwidth ceiling, or the recorded rate for that pipeline's task. Name the closest published implementation of the same task and the figure it reported. `matmul2d` tile shapes, 128-byte strides, and the vendor session that specifies them are constraints, not the table. A model card for this checkpoint closes only the score it actually used. An operand-order correction, and a gap that has not been measured, stay in the serving note. Do not invent routing hits, calibration energies, kernel times, or hidden-state errors. Compute a format bound from this pipeline's own quantizer and the public code points. Do not import another pipeline to do it.

## File it

| What you have | Where it goes |
|---|---|
| Remainder with no numbers | Outside this repository. No catalog row. No README link. |
| Format bound, a checkout table, or this implementation's dispatch or rate table | `docs/<pipeline>/research/<slug>.md` |
| Published method, no remainder | Leave it in the serving note. Do not open a research file. |
| A citation of Favlo itself | [CITATION.cff](../../../CITATION.cff). No catalog row. No research file. |

The research file says whether the table is a format bound, a checkout measurement, or a recorded run. Sections, in order: Remainder, Predecessors, Rule, Command, Table, Search, Corrections. Corrections are appended. Earlier sections stay as committed.

The command is the one that regenerates the table. If the numbers are a run already printed in the serving note or the ceilings note, and no command regenerates them, say that. A shader check on a synthetic tile is not that command. A format bound's command is the check over the legal code points. A rule that still needs the BF16 checkout is not a filled ranking. Say the checkout was not scored.

The pipeline README links the measurement in one sentence. Add a row to `docs/README.md` when that index lists pipeline notes. Change the serving behavior only when the behavior ships. Status in the catalog:

- `measured` while the serving note does not describe the behavior.
- `shipped` once that note describes it. If it already does, add the link and use `shipped`.
- `narrowed` after a correction is appended. The row stays.

Catalog columns: slug, pipeline, status, search date, shipped date, benchmark date, the one-sentence remainder, the note path. The shipped date is the date the serving note described the behavior, empty while status is `measured`. The benchmark date is the date of that pipeline's latest harness scores, empty when that pipeline has none. The index does not repeat the table, the predecessors, or the command.

The package license and readme are the rerun instructions. Do not add another license file. The pipeline id, the model, and the papers come from the note under edit. Do not reuse another pipeline's claim as this one.

## How a candidate lands

- A bound read off this pipeline's code points, when the server does not use it yet, is `measured`. If the ranking still needs a checkout that was not scored, the table is the bound and the note says so.
- A recorded count that the serving note already follows is `shipped`. The remainder is that count. A shader check on a synthetic tile does not regenerate it.
- A sort key with no table stays a plan. No catalog row.
- A cited equation, a checkpoint operand order, a tile or stride constraint, and an unmeasured gap stay in the serving note. No research file.
- A measured dispatch sum or recorded rate for this implementation is a research file. The remainder is that table.

## Key/value plane

A recorded rate, a dispatch sum, or a needle row names the key/value plane that serve loaded. Planes differ in width, scale, and layout. A number measured on one plane does not fill a row for another. The pipeline note lists the planes that server accepts. Read that note for the names, and write the loaded name after aliases resolve. On `qwen36_35b_a3b` those names are `half`, `fp8_e4m3`, `e2m1`, and `f32`. On `qwen38_27b` those names are `half` and `fp8_e4m3`; an omitted cache uses `half`. `f16` and `fp16` resolve to `half`. `e4m3` and `fp8` resolve to `fp8_e4m3`. `fp4` resolves to `e2m1`.

## Needle

`python -m harness.needle` writes one file of an exact token length and asks for the passkey in it. `--depth` is 0 at the first line and 1 at the last. `--mode page` is the default. It passes when the reply contains the number and one `read_file` returned that line. The number may sit in the newest window. That run is a long-transcript rate and is filed on its own row.

`--mode retrieve` tells the agent to follow `next_offset` until a read has none, and only then to reply. A pass requires the number, a read of its line, a last read with no `next_offset`, and a last window that starts after that line. Depth 1 has no later line, so a depth-1 retrieve run does not fill the retrieval table.

The claim table is one row per plane, MTP setting, thinking card, length, depth, and mode: pass or fail, the prefill sum, and the decode sum. The command carries `--mode`, `--tokens`, `--depth`, `--mtp`, `--thinking`, and the model id. The harness prints the loaded plane, MTP, and thinking. One row at one depth and one seed is a measured run. A grid of lengths and depths is the table set next to a published needle score. The search names that score. The "not found" sentence names this grid.

The finished transcript has to fit the served window. The file is replayed inside tool results, so a file as long as the window does not fit. `input does not fit` is a failed row, not a pass.

## Coding agent

`python -m harness.coding` is the production-readiness run. Exit 0 means `ledger.py` `collect` returns 6, 0, and -6, a later request loaded the prefix from disk, and the chat contract held. Which tools the model called is not the result. The run is a recorded rate. It is not a retrieval claim.

The claim table is one row per plane, MTP setting, thinking card, and vision flag: pass or fail, the prefill sum, and the decode sum. The command carries `--mtp`, `--thinking`, `--vision` when the pipeline embeds an image, and the model id. The harness prints the loaded plane, MTP, thinking, and vision. A pipeline whose sampling lists both `thinking` and `instruct` has two rows, thinking first. A row that omits the plane, or that was measured on another plane, does not fill this table. The "not found" sentence names this table.

## Synthetic prompt

`python -m harness.synthetic` sends one user message of an exact prompt length and stops. The default lengths are 512 and 10000. `--decode` caps the sampled tokens at 128. Exit 0 means each measured prompt length matched the server and the prefill was a cache miss. The run is a recorded rate. It is not a coding-agent row and not a needle row.

The claim table is one row per plane, MTP setting, thinking card, and prompt length: the prefill sum, the decode sum, and the end-to-end sum. The end-to-end sum is prefill tokens plus decode tokens over those two clocks. The command carries `--tokens`, `--decode`, `--mtp`, `--thinking`, and the model id. The harness prints the loaded plane, MTP, and thinking. A 512-token row is the local-engine comparison. A 10000-token row is the hosted-API comparison. The "not found" sentence names this table.

---
name: research-claims
description: Search public literature and write a Favlo research claim for one pipeline. Use when the user asks to document research claims, check an implementation against published work, or add a row to RESEARCH.md.
---

# Research claims

Read [research-evidence.mdc](../../rules/research-evidence.mdc) and [RESEARCH.md](../../../RESEARCH.md) first. The rule decides where a claim lives and which words are allowed. This skill is the search and the filing.

Work on `main`. Do not create a branch unless the user asks. Land the note, the catalog row, and the README link in one change. Commit only when the user asks.

One pipeline. Do not edit another pipeline's package, shaders, tests, or docs. [RESEARCH.md](../../../RESEARCH.md) is the shared index and receives one new row.

## Search

Read the serving note for the sentences that decide a behavior: a recorded difference, a refusal, or a result that is not yet what the server does. Ignore a paragraph that only restates a cited equation.

Split each candidate into the method, the model or task, the measurement, and the mechanism. Search papers, model cards, and shipped code. Then search again with the words those hits use. The closer paper often does not share the first query. Record every query, the date, and the predecessors.

A predecessor closes the part it already did. Name it with a link and one sentence. The sentence this repo may own is only the remainder. The "not found" sentence names that remainder's number or bound. It does not name a mechanism the predecessor already published. Do not write "first," "novel," or "no prior work."

A measured mapping onto one machine is a remainder even when the method is published. The table is the times, the bandwidth ceiling, or the recorded rate for that pipeline's task. Name the closest published implementation of the same task and the figure it reported. `matmul2d` tile shapes, 128-byte strides, and the vendor session that specifies them are constraints, not the table. A model card for this checkpoint closes only the score it actually used. An operand-order correction, and a gap that has not been measured, stay in the serving note. Do not invent routing hits, calibration energies, kernel times, or hidden-state errors. Compute a format bound from this pipeline's own quantizer and the public code points. Do not import another pipeline to do it.

## File it

| What you have | Where it goes |
|---|---|
| Remainder with no numbers | `artifacts/implementation-plan/docs/<pipeline>/<slug>.md`. Gitignored. No catalog row. No README link. |
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

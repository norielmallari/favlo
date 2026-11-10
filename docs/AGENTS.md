# Docs

The root README is the only file that lists the packages. Its Supported Models table and its checkpoint license table are the official list. A rule and an `AGENTS.md` do not name a package id, a model, a packed folder, or a checkpoint id. Do not add a glob-scoped rule. Package instructions live in that package's `AGENTS.md`. Examples use `example-model`, `Example-Model-fp4`, and `ExampleOrg/Example-Model`.

Before editing a package note, read that package's `AGENTS.md` under `pipelines/`. This folder's instructions do not replace it.

## Public wording

Write docs, commits, comments, and examples as if they will stay public. A block quote stays in the author's words. Update it only when the operator asks for that quote to be changed.

Do not commit personal data or secrets. Placeholders are `~/.favlo`, `127.0.0.1`, and `example-model`.

A pipeline note names the official base model and its model card. A finetune or other derived checkpoint stays out of `docs/<pipeline>/`. The directory stem in the note is the official repository name plus the weight quants.

Use calm sentences. A measured time names the machine as an M5 Pro with 48 GB. Outside a block quote, a claim by the author is in the third person.

A public README says what the project is, how to install it, one minimal command, the license, and where to report a problem. Keep the as-built detail in linked docs. Prefer relative links.

## Research claims

Do not write "first", "novel", or "no prior work". Write "not found in these sources on this date."

The filing procedure is the research-claims skill. Load it when a run prints a prefill and a decode rate, or when the task is a measured claim. The catalog is `RESEARCH.md`. The software citation stays in `CITATION.cff`.

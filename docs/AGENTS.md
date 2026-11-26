# Docs

The root README is the only file that lists the packages. Its Supported Models table and its checkpoint license table are the official list. A rule and an `AGENTS.md` do not name a package id, a model, a packed folder, or a checkpoint id. Do not add a glob-scoped rule. Package instructions live in that package's `AGENTS.md`. Examples use `example-model`, `Example-Model-fp4`, and `ExampleOrg/Example-Model`.

Before editing a package note, read that package's `AGENTS.md` under `pipelines/`. This folder's instructions do not replace it.

## Public wording

Write docs, commits, comments, and examples as if they will stay public. A block quote stays in the author's words. Update it only when the operator asks for that quote to be changed.

Do not commit personal data or secrets. Placeholders are `~/.favlo`, `127.0.0.1`, and `example-model`.

A pipeline note names the official base model and its model card. A finetune or other derived checkpoint stays out of `docs/<pipeline>/`. The directory stem in the note is the official repository name plus the weight quants.

Use calm, short sentences. Related facts share one sentence, and a bullet starts the next topic. One fact per bullet is too fine, and a paragraph that stacks unrelated facts is too coarse. A measured time names the machine as an M5 Pro with 48 GB. Outside a block quote, a claim by the author is in the third person.

The root README says what the project is, how to install it, one minimal command, the license, and where to report a problem. Prefer relative links.

A pipeline README under `docs/<pipeline>/` is always a summary. It names the checkpoint and the license, how to download, pack, and run the package, and links to `ARCHITECTURE.md` in that same folder. Do not add a `SETUP.md`. That `ARCHITECTURE.md` holds the as-built detail: kernels, the packed record, the forward, and registry fields. Do not restate a shader, a byte layout, a dispatch order, or a registry field table in the README. A caller-visible default changes the README. A dispatch or layout change changes `ARCHITECTURE.md`.

## Research claims

Do not write "first", "novel", or "no prior work". Write "not found in these sources on this date."

The filing procedure is the research-claims skill. Load it when a run prints a prefill and a decode rate, or when the task is a measured claim. The index is `RESEARCH.md`. Active models holds the protocol-complete synthetic rates. Historical holds the other language claims, the shared kernel notes, and the embeddings encode claim. Media holds image, video, music, and voice claims. A needle run and a coding-agent run file Historical. Only a protocol-complete synthetic run writes Active models. The software citation stays in `CITATION.cff`.

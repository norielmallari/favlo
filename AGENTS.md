# AGENTS.md

Rules for coding agents in **Favlo**.

Read this file first. Then read [ARCHITECTURE.md](ARCHITECTURE.md) for the map and the write set. The package being edited has its own note. Papers and specifications are [REFERENCES.md](REFERENCES.md). Measured research claims are [RESEARCH.md](RESEARCH.md). Cite this software from [CITATION.cff](CITATION.cff). A package's own citations live in its note. The product name, the one-pipeline rule, and the official package list are [README.md](README.md). AI-assisted features are [docs/ai-assistance](docs/ai-assistance/README.md).

[ARCHITECTURE.md](ARCHITECTURE.md) wins for the listener, the registry, the console, and which files an agent may edit. A package note wins for that package's kernels.

## Where instructions live

Cursor loads this file for every chat. A nested `AGENTS.md` applies when the work is in that folder. The closer file wins.

| Scope | File |
|---|---|
| Personal data and disallowed content before a commit | [commit-precheck.mdc](.cursor/rules/commit-precheck.mdc) |
| Metal tensor layout | [metal-mappings.mdc](.cursor/rules/metal-mappings.mdc) |
| Save generated media, then stream that file | [serving-media.mdc](.cursor/rules/serving-media.mdc) |
| Any package | [pipelines/AGENTS.md](pipelines/AGENTS.md) |
| One package | `pipelines/<model>/AGENTS.md` |
| Listener and MCP catalog | [src/favlo/AGENTS.md](src/favlo/AGENTS.md) |
| Public wording and research claims | [docs/AGENTS.md](docs/AGENTS.md) |
| Web UI | [web-spa/AGENTS.md](web-spa/AGENTS.md) |
| Web API | [web-api/AGENTS.md](web-api/AGENTS.md) |
| Playwright screens | [web-spa/e2e/AGENTS.md](web-spa/e2e/AGENTS.md) |
| Harness | [harness/AGENTS.md](harness/AGENTS.md) |

The official list is the Supported Models table and the checkpoint license table in the root README. This file does not name a package.

Skills in [`.cursor/skills`](.cursor/skills) hold the procedures. Load the skill whose description matches the task. A GPU smoke is smoke-testing. The listener is favlo-server. A language or embeddings shell check is seam-checks. Media preemption is media-preemption. A web happy path is web-happy-path. A measured claim is research-claims. A new package is bootstrap-new-model. A prefix store is prefix-caching. Language-model rate work is optimization-loop. Image generation time is optimization-image.

## Changes

Do not edit, create, or delete a file until the operator gives a go signal. `proceed` is that signal. A plan and a design are not. `continue` keeps the current activity: planning stays planning, and editing stays editing. Reading the tree and answering a question do not need a go signal. A request to undo a change is the go signal for that undo.

Work on `main`. Do not create a branch unless the user asks for one.

Run Python through `uv run` so the command uses this checkout's environment. A module is `uv run python -m <module>`. A script is `uv run python <path>`. Pytest is `uv run pytest`. Do not call `python` or `python3` from `PATH`.

Changes stay in this repository. Edit another repository only when the user asks for that repository.

The open product in this repository is the listener, the web app, and the web API. A hosted build of the same app calls another process that implements the same HTTP contract. That process is not part of this repository, and this repository does not name it. When a sibling checkout of the hosted product is present, read it only for a hosted task. When it is absent, stay in this repository.

Edit one runtime. Leave every other runtime's files untouched. The write set and the runtime table are in [ARCHITECTURE.md](ARCHITECTURE.md). Do not edit `src/favlo/cli.py` to register a runtime. Do not edit another package's sources, shaders, tests, or docs. A served media package subclasses `MediaPipeline` and sets `kind`, `id_prefix`, and `object_name`. A language-model package does not. Do not add a kernel that branches on the package. A new abstract method on `MediaPipeline` updates every media subclass in that same change.

A seam task may edit `language_pipeline.py`, `embeddings_pipeline.py`, `media_pipeline.py`, `media_memory.py`, `memory.py`, `mcp_server.py`, `response_chain.py`, `schedule.py`, and `pipelines.py`, along with the listener, the registry, and the console.

Adding a model adds `pipelines/<model>/metal41/`, the runtime package, a `docs/<model>/` note, `pipelines/<model>/AGENTS.md`, and one row in the runtime table. Load bootstrap-new-model before creating one. Load prefix-caching before wiring a package to the shared prefix store.

The listener runs one request. A language-model package does not fuse a whole layer. An image, video, music, or voice package may place several kernels of one generation in one command buffer and may fuse those kernels. The Neural Engine may run a module. A missing Metal kernel stays missing. It is not replaced by a host GEMV, a CPU matmul, or any other stand-in forward.

## Open items

When listing decisions still open for the operator, number each item uniquely in that reply (`1.`, `2.`, …). Do not reuse a number for a different item in the same list. For every numbered item, list the **options** first, then mark which of those options is the **recommendation**. An item with a single viable path still lists that path as the only option and marks it as recommended.

## Memory

Every package has a degraded mode. It runs under memory pressure and on a machine with 16 GB of memory. The listener records an opening's owners and sweeps them after `close`.

A package reads `memory.budget()` when it opens and shrinks the one knob it owns until its minimum working set fits. It refuses to open when that minimum does not fit. A media request keeps the resolution, the frame count, and the duration it named. The next read of the ledger is at `complete`. `open_runtime` accepts `owner` and uses it for every allocate in that opening. When the argument is absent, the package keeps minting its own owner string.

## Harness

Harness clients, flags, and the docs update live in [harness/AGENTS.md](harness/AGENTS.md). A coding-agent check takes `--model <id>`. Pass `--vision` when that package embeds an image. When `sampling.py` lists both `thinking` and `instruct`, run it twice, thinking first. `--thinking on` uses the thinking card. `--thinking off` uses the instruct card. One variant runs once. `--store off` is the default. `--store on` sends `store` true. Exit 0 on each run is the pass. The still is `tests/assets/favlo-horse.png`. The commands are the smoke-testing skill.

A needle run, a coding-agent run, or a synthetic-prompt run that prints a prefill sum and a decode sum is a measured rate. The synthetic prompt also prints an end-to-end rate, the sum of those two clocks. Load research-claims in that same turn and file the row when [RESEARCH.md](RESEARCH.md) does not already hold that plane, MTP setting, length, depth, mode, and card. Asking to run the harness is enough.

## Agents

Every agent is a Microsoft Agent Framework agent. Chat goes through the shared client in `src/favlo/chat_client.py`.

A unit test and a smoke test may call the OpenAI HTTP API. An agent harness and production code use the framework and that shared client. They keep the client's response parser and message preparer. A field the next turn must send is set where chat messages are built.

## Chat

A chat debug uses the Favlo horse only. That is the seeded agent named Favlo, on its channel session. Do not read, quote, or score a chat that belongs to any other agent.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders. Do not launch another application's environment.

A package's calibration strings are data, not an algorithm. The registry default is the language model a bare request loads. Its as-built note is the README under `docs/` whose folder matches that registry row.

Do not invent routing hits, calibration energies, or hidden-state errors. A measurement belongs in that package's note.

## Docs

A package change updates that package's `docs/<model>/` note in the same change. A process change (listener, registry, console, or the write set) updates [ARCHITECTURE.md](ARCHITECTURE.md). An AI-assisted feature change updates [docs/ai-assistance](docs/ai-assistance/README.md) in the same change. Add a page under that folder when the feature needs its own note, and list the page from that folder's README. Do not edit another package's note to describe this one.

`docs/` is as-built only, plus a measured research remainder at `docs/<model>/research/<slug>.md`. That file does not change the serving note until the behavior ships. Do not commit plan or option-comparison notes under `docs/`. Load research-claims before documenting a research claim. A needle or coding-agent run that printed a prefill sum and a decode sum is that claim.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary` on the subject line. Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`. Scope is optional. Prefer a short package id such as `example-model`, or `shared`, when the change is localized.

```text
feat(example-model): wire Responses store to the shared prefix session
perf(example-model): fuse the scan prefix into the prefill shader
docs: describe parallel quantize worker layout
```

Optional hook (local only; do not commit git config):

```text
git config core.hooksPath .githooks
```

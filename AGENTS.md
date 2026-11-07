# AGENTS.md

Rules for coding agents in **Favlo**.

Read this file first. Then read [ARCHITECTURE.md](ARCHITECTURE.md) for the map and the write set. The pipeline being edited has its own note. Papers and specifications are [REFERENCES.md](REFERENCES.md). Measured research claims are [RESEARCH.md](RESEARCH.md). Cite this software from [CITATION.cff](CITATION.cff). A pipeline's own citations live in its note. The product name and the one-pipeline rule are [README.md](README.md). AI-assisted features are [docs/ai-assistance](docs/ai-assistance/README.md).

[ARCHITECTURE.md](ARCHITECTURE.md) wins for the listener, the registry, the console, and which files an agent may edit. A pipeline note wins for that pipeline's kernels.

## Repo rules

The files in [`.cursor/rules`](.cursor/rules) are repo rules. Cursor loads them on its own. Every other agent reads them before editing.

| Rule | Applies |
|---|---|
| [documentation.mdc](.cursor/rules/documentation.mdc) | Always. Public wording, no personal data, official model cards only. |
| [commit-precheck.mdc](.cursor/rules/commit-precheck.mdc) | Always. Before a commit, check every file for personal data and for NSFW, violent, or racist content. |
| [pipeline-isolation.mdc](.cursor/rules/pipeline-isolation.mdc) | Always. One write set, a clean close on model switch, a host checkpoint when a chat preempts media, and the smoke lock before a pipeline test. |
| [public-algorithms.mdc](.cursor/rules/public-algorithms.mdc) | Always. Equations come from public papers and map onto Metal. Prefill, the key/value cache, and decode stay allocated. |
| [python-file-split.mdc](.cursor/rules/python-file-split.mdc) | When editing Python. One module, one responsibility. Imports point one way. |
| [optimization-llm.mdc](.cursor/rules/optimization-llm.mdc) | When optimizing a language model's prefill or decode. |
| [optimization-image.mdc](.cursor/rules/optimization-image.mdc) | When optimizing image generation time. |
| [research-evidence.mdc](.cursor/rules/research-evidence.mdc) | When writing a measured claim. The note records the remainder a search on that date did not find. The software citation stays in CITATION.cff. |
| [serving-media.mdc](.cursor/rules/serving-media.mdc) | When an API returns generated media. Save the file, then stream that file. |
| [mcp-server.mdc](.cursor/rules/mcp-server.mdc) | When editing the media MCP server. Update [MCP.md](MCP.md) in the same change. |
| [media-pipeline.mdc](.cursor/rules/media-pipeline.mdc) | When editing a served media pipeline or `MediaPipeline`. Inherit the listener contract, and run the one preemption harness after that contract changes. |
| [language-pipeline.mdc](.cursor/rules/language-pipeline.mdc) | When editing a served language model or `LanguagePipeline`. Keep the completion in the pipeline, and run the language harness after the disconnect shell changes. |
| [embeddings-pipeline.mdc](.cursor/rules/embeddings-pipeline.mdc) | When editing a served embedding model or `EmbeddingsPipeline`. Return the vector and the token count, and let the class write the list object. |
| [web-app.mdc](.cursor/rules/web-app.mdc) | When editing `web-spa/` or `web-api/`, and when a web feature is finished. The same change adds a deterministic Playwright happy path that performs each new action a person can start. An existing script counts only when it performs that action. The feature is finished when that script passes against a live `favlo serve`, with the completing control already inside the viewport. |
| [web-spa.mdc](.cursor/rules/web-spa.mdc) | When editing `web-spa/`. Best UI/UX practice, and a result that is a pleasure to use. Palette tokens, `textStyle` colors, and `IconButton` tones (`accent` / `danger` / `surface` / `ghost`) serve that. No hardcoded control colors. |
| [playwright-e2e.mdc](.cursor/rules/playwright-e2e.mdc) | When editing `web-spa/e2e/`. Natural-language prompts, teardown of agents and workspaces, Home/expand races, AG-UI durability before Stop, WCAG AA, fail-fast suite runs. |

## Changes

Do not edit, create, or delete a file until the operator gives a go signal. `proceed` is that signal. A plan and a design are not. `continue` keeps the current activity: planning stays planning, and editing stays editing. Reading the tree and answering a question do not need a go signal. A request to undo a change is the go signal for that undo.

Work on `main`. Do not create a branch unless the user asks for one.

Run Python through `uv run` so the command uses this checkout's environment. A module is `uv run python -m <module>`. A script is `uv run python <path>`. Pytest is `uv run pytest`. Do not call `python` or `python3` from `PATH`.

Changes stay in this repository. Edit another repository only when the user asks for that repository.

The open product in this repository is the listener, the web app, and the web API. A hosted build of the same app calls another process that implements the same HTTP contract. That process is not part of this repository, and this repository does not name it. When a sibling checkout of the hosted product is present, read it only for a hosted task. When it is absent, stay in this repository.

Edit one runtime. Leave every other runtime's files untouched. The write set and the runtime table are in [ARCHITECTURE.md](ARCHITECTURE.md). Do not edit `src/favlo/cli.py` to register a runtime. Do not edit another pipeline's package, shaders, tests, or docs. A served media pipeline inherits `ImagePipeline`, `VideoPipeline`, `MusicPipeline`, or `VoicePipeline` under `MediaPipeline`. A language-model pipeline does not. Do not add a kernel that branches on the pipeline. A new abstract method on `MediaPipeline` updates every media subclass in that same change.

Adding a model adds `pipelines/metal41/<pipeline>/`, the entry file and package inside it, a `docs/<pipeline>/` note, and one row in the runtime table. Load [bootstrap-new-model](.cursor/skills/bootstrap-new-model/SKILL.md) before creating one. Load [prefix-caching](.cursor/skills/prefix-caching/SKILL.md) before wiring a pipeline to the shared prefix store.

The listener runs one request. A language-model pipeline does not fuse a whole layer. An image, video, music, or voice pipeline may place several kernels of one generation in one command buffer and may fuse those kernels. The Neural Engine may run a module. A missing Metal kernel stays missing. It is not replaced by a host GEMV, a CPU matmul, or any other stand-in forward.

## Open items

When listing decisions still open for the operator, number each item uniquely in that reply (`1.`, `2.`, …). Do not reuse a number for a different item in the same list. For every numbered item, list the **options** first, then mark which of those options is the **recommendation**. An item with a single viable path still lists that path as the only option and marks it as recommended.

## Memory

Every pipeline has a degraded mode. It runs under memory pressure and on a machine with 16 GB of memory. The listener records an opening's owners and sweeps them after `close`.

A pipeline reads `memory.budget()` when it opens and shrinks the one knob it owns until its minimum working set fits. It refuses to open when that minimum does not fit. A media request keeps the resolution, the frame count, and the duration it named. The next read of the ledger is at `complete`. `open_runtime` accepts `owner` and uses it for every allocate in that opening. When the argument is absent, the pipeline keeps minting its own owner string.

## Harness

A media harness does not take the machine lock. It calls the listener. A media harness uses one `favlo serve` on `127.0.0.1:8129`. That is `harness.image`, `harness.image_smoke_agent`, `harness.image_persona`, `harness.image_attachment`, `harness.video_smoke_agent`, `harness.video_ti2v_agent`, `harness.video_quality_agent`, `harness.media_pipeline`, `harness.language_pipeline`, `harness.music_smoke_agent`, `harness.music_long_agent`, `harness.voice_smoke_agent`, `harness.embeddings_agent`, and `harness.memory_agent`. `uv run python -m harness.server run -- <command>` opens that listener, runs the command, and closes the session. `uv run python -m harness.server status` prints `listening` or `down`. An existing listener is reused and left running. A listener this session started is stopped with SIGTERM when the command exits. The daemon forwards that signal to each pipeline process. These commands do not bind a second port. The same session is what those modules call, so running one of them directly follows this rule.

`uv run python -m harness`, `uv run python -m harness.coding`, `uv run python -m harness.needle`, `uv run python -m harness.synthetic`, `uv run python -m harness.worktree`, `uv run python -m harness.coherence`, and `uv run python -m harness.benchmarks` use that same listener through `uv run python -m favlo.smoke_lock attach`. They do not take the machine lock. When the pipeline they name is resident and its sources are newer than that worker, the model is unloaded before the run. A prefix check unloads the resident model after the first turn so the next turn loads the prefix from disk. The listener stays up.

`harness.chat_agent` uses that same listener and does not take the machine lock.

A harness change updates [docs/harness.md](docs/harness.md) in the same change. Each client keeps one subsection there: a short title, the command in a bash block, then what it does.

The coding agent is `uv run python -m harness.coding --model <id>`. Pass `--vision` when that pipeline embeds an image (`embed_image` in its `vision.py`). The still is `tests/assets/favlo-horse.png`. When `sampling.py` lists both `thinking` and `instruct`, run it twice, thinking first. `--thinking on` uses the thinking card. `--thinking off` uses the instruct card. One variant runs once. `--store off` is the default. `--store on` sends `store` true. That one value is the ledger and the contract. Exit 0 on each run is the pass.

```text
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking on
uv run python -m harness.coding --model qwen36-35b-a3b --vision --thinking off
```

A needle run, a coding-agent run, or a synthetic-prompt run that prints a prefill sum and a decode sum is a measured rate. The synthetic prompt also prints an end-to-end rate, the sum of those two clocks. Load [research-claims](.cursor/skills/research-claims/SKILL.md) in that same turn and file the row when [RESEARCH.md](RESEARCH.md) does not already hold that plane, MTP setting, length, depth, mode, and card. Asking to run the harness is enough.

## Agents

Every agent is a Microsoft Agent Framework agent. Chat goes through the shared client in `src/favlo/chat_client.py`.

A unit test and a smoke test may call the OpenAI HTTP API. An agent harness and production code use the framework and that shared client. They keep the client's response parser and message preparer. A field the next turn must send is set where chat messages are built.

## Chat

A chat debug uses the Favlo horse only. That is the seeded agent named Favlo, on its channel session. Do not read, quote, or score a chat that belongs to any other agent.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders. Do not launch another application's environment.

A pipeline's calibration strings are data, not an algorithm. Treat `qwen36-35b-a3b` as the default language model; its as-built note is [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md).

Do not invent routing hits, calibration energies, or hidden-state errors. A measurement belongs in that pipeline's note.

## Docs

A pipeline change updates that pipeline's `docs/<pipeline>/` note in the same change. A process change (listener, registry, console, or the write set) updates [ARCHITECTURE.md](ARCHITECTURE.md). An AI-assisted feature change updates [docs/ai-assistance](docs/ai-assistance/README.md) in the same change. Add a page under that folder when the feature needs its own note, and list the page from that folder's README. Do not edit another pipeline's note to describe this one.

`docs/` is as-built only, plus a measured research remainder at `docs/<pipeline>/research/<slug>.md`. That file does not change the serving note until the behavior ships. Do not commit plan or option-comparison notes under `docs/`. Load [research-claims](.cursor/skills/research-claims/SKILL.md) before documenting a research claim. A needle or coding-agent run that printed a prefill sum and a decode sum is that claim.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary` on the subject line. Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`. Scope is optional; prefer a short pipeline id such as `qwen36-35b-a3b` or `shared` when the change is localized.

```text
feat(qwen36-35b-a3b): wire Responses store to the shared prefix session
perf(qwen36-35b-a3b): fuse MoE scan prefix into prefill shader
docs: describe parallel quantize worker layout
```

Optional hook (local only; do not commit git config):

```text
git config core.hooksPath .githooks
```

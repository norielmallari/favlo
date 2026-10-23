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
| [pipeline-isolation.mdc](.cursor/rules/pipeline-isolation.mdc) | Always. One write set, a clean close on model switch, and the smoke lock before a pipeline test. |
| [public-algorithms.mdc](.cursor/rules/public-algorithms.mdc) | Always. Equations come from public papers and map onto Metal. Prefill, the key/value cache, and decode stay allocated. |
| [python-file-split.mdc](.cursor/rules/python-file-split.mdc) | When editing Python. One module, one responsibility. Imports point one way. |
| [optimization-llm.mdc](.cursor/rules/optimization-llm.mdc) | When optimizing a language model's prefill or decode. |
| [optimization-image.mdc](.cursor/rules/optimization-image.mdc) | When optimizing image generation time. |
| [research-evidence.mdc](.cursor/rules/research-evidence.mdc) | When writing a measured claim. The note records the remainder a search on that date did not find. The software citation stays in CITATION.cff. |

## Changes

Work on `main`. Do not create a branch unless the user asks for one.

Changes stay in this repository. Edit another repository only when the user asks for that repository.

Edit one runtime. Leave every other runtime's files untouched. The write set and the runtime table are in [ARCHITECTURE.md](ARCHITECTURE.md). Do not edit `src/favlo/cli.py` to register a runtime. Do not edit another pipeline's package, shaders, tests, or docs. Do not add a base class or a kernel that branches on the pipeline.

Adding a model adds an entry file, a package, a `docs/<pipeline>/` note, and one row in the runtime table. Load [bootstrap-new-model](.cursor/skills/bootstrap-new-model/SKILL.md) before creating one. Load [prefix-caching](.cursor/skills/prefix-caching/SKILL.md) before wiring a pipeline to the shared prefix store.

Do not add batching, the Neural Engine, or a kernel that fuses a whole layer. A missing Metal kernel stays missing. It is not replaced by a host GEMV, a CPU matmul, or any other stand-in forward.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders. Do not launch another application's environment.

A pipeline's calibration strings are data, not an algorithm. The bucket for Flash Next is in its references note.

Do not invent routing hits, calibration energies, or hidden-state errors. A measurement belongs in that pipeline's note.

## Docs

A pipeline change updates that pipeline's `docs/<pipeline>/` note in the same change. A process change (listener, registry, console, or the write set) updates [ARCHITECTURE.md](ARCHITECTURE.md). An AI-assisted feature change updates [docs/ai-assistance](docs/ai-assistance/README.md) in the same change. Add a page under that folder when the feature needs its own note, and list the page from that folder's README. Do not edit another pipeline's note to describe this one.

`docs/` is as-built only, plus a measured research remainder at `docs/<pipeline>/research/<slug>.md`. That file does not change the serving note until the behavior ships. Implementation plans live under `artifacts/implementation-plan/` (gitignored). Do not commit plan or option-comparison notes under `docs/`. Load [research-claims](.cursor/skills/research-claims/SKILL.md) before documenting a research claim.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary` on the subject line. Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`. Scope is optional; prefer a short pipeline id such as `qwen38-flash-next` or `shared` when the change is localized.

```text
feat(qwen38): stream experts from SSD into the resident atlas
perf(qwen38): fuse scan prefix into MoE prefill shader
docs: describe parallel quantize worker layout
```

Optional hook (local only; do not commit git config):

```text
git config core.hooksPath .githooks
```

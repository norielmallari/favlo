---
name: seam-checks
description: Run the language and embeddings shell checks after a change to LanguagePipeline, EmbeddingsPipeline, handle_responses, embed, or the list object. Do not use for a kernel or quantize edit that leaves the reply alone, and do not use for image generation.
---

# Seam checks

Load this skill when the disconnect shell, `handle_responses`, `embed`, or the embeddings list object changes. A kernel or quantize edit that leaves the reply alone does not run these commands.

Both checks use one `favlo serve` on `127.0.0.1:8129` and do not take the machine lock. `harness.server` waits until no live pid holds `artifacts/pipeline-test.lock`. A GPU command in its own process uses the smoke-testing skill.

## Language shell

```text
uv run pytest tests/test_language_pipeline.py
uv run python -m harness.language_pipeline
```

Exit 0 means `check_embeddings` ran and every paraphrase cosine beat every cross-topic cosine. The harness is the one agent for the registry default chat model and the registered embeddings model.

`handle_responses` raises `RequestClosed`. The shell clears `runtime._stop`. Do not add a second harness per language model.

## Embeddings

`python -m harness.embeddings_agent` is the direct check of the same six sentences. It posts `POST /v1/embeddings`. It does not take the machine lock.

Do not kill serve to free the GPU. Close during a live hold is 409.

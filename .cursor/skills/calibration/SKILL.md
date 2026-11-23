---
name: calibration
description: >-
  Build the shared GPTQ and FOEM calibration corpus from the pinned public
  files. Use when creating, rewriting, or checking artifacts/calibration, a
  calibration kind (swe, code, tool, chat), favlo.calibration_corpus, or the
  dialogs a pipeline quantizes with GPTQ or FOEM.
---

# Calibration corpus

The shared set is `artifacts/calibration/`. That directory is gitignored. Do not commit it. The contract is [docs/calibration/calibration.md](../../../docs/calibration/calibration.md). The writer is `favlo.calibration_corpus`.

A kind file is `swe.json`, `code.json`, `tool.json`, or `chat.json`. `manifest.json` names those files. A pipeline reads one kind with `favlo.calibration_corpus.load`. It does not shuffle the file.

## Build

Run Python through `uv run`. Name each kind file to rewrite. The other kind files stay. A run with no file name does not write.

```bash
uv run python -m favlo.calibration_corpus swe.json
```

More than one file rewrites each named file and leaves the rest:

```bash
uv run python -m favlo.calibration_corpus code.json tool.json
```

`swe` is accepted as the same file as `swe.json`. Do not delete the directory. Do not delete a kind file in order to rebuild another kind. Do not hand-edit a kind file or the manifest. A change to the pinned sources, the row shape, or the quotas is a change to `src/favlo/calibration_corpus.py` and `docs/calibration/calibration.md` in the same edit.

From Python, `build(kinds=["swe.json"])` rewrites that file. `build()` with `kinds` omitted raises.

## Check

After a write, `manifest.json` `files` lists each kind that is on disk, and `counts` matches that file's list length. A row has `kind` and `messages`. `function.arguments` is an object. `swe` and `tool` rows also have `tools`.

The fill is sequences of 2048 tokens, not rows. A capture needs 34 `swe`, 34 `code`, 34 `tool`, and 26 `chat` (128 sequences, 262144 tokens). A kind that renders fewer sequences than that stops the capture. The writer stores a larger pool (32, 400, 120, and 400 rows) so a tokenizer still has margin. All four kinds fill that quota. `chat.json` has 26 rows, and its manifest entry still says 400.

The Gram written beside a checkpoint is not this corpus. Do not delete it as part of a corpus rebuild.

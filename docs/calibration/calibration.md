# GPTQ and FOEM calibration

A pipeline that rounds with GPTQ or FOEM reads one shared corpus from `artifacts/calibration/`. That directory is gitignored. The derived input Gram stays beside the checkpoint. It is not the shared set.

The sequence count and the Hessian are the ones published for GPTQ and reused by FOEM. The text is an agentic mix. No published corpus is 128 sequences of 2048 tokens with this mix. `favlo.calibration_corpus.build` writes the local set from the pinned public files:

```bash
uv run python -m favlo.calibration_corpus swe.json
```

The command requires one or more kind files. It rewrites each named file and leaves the other kind files in place. A run with no file name does not write.

Sources for the method:

- Frantar, Ashkboos, Hoefler, and Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", arXiv:2210.17323. Algorithm 1 takes the inverse Hessian `H^{-1} = (2XX^T + λI)^{-1}`. The published size is 128 sequences of 2048 tokens.
- Zheng et al., "First-Order Error Matters: Accurate Compensation for Quantized Large Language Models", AAAI 2026, arXiv:2507.11017. Equation 19 is the column update. `β` is 0.1. The gradient approximation does not add sequences.

## Corpus

`artifacts/calibration/manifest.json` names one JSON file per kind: `swe.json`, `code.json`, `tool.json`, and `chat.json`. A pipeline opens the file for the kind it is packing and does not shuffle that file.

Each row is one dialog:

- `kind` is `swe`, `code`, `tool`, or `chat`.
- `messages` is a list of `{role, content}` turns. An assistant turn may include `tool_calls`. A tool result is `{role: "tool", tool_call_id, content}`.
- `tools`, when the kind is `swe` or `tool`, is a list of `{type: "function", function: {name, description, parameters}}`.
- `function.arguments` is an object.

The rows come from these files, in file order within each kind. A parquet split is read in that same order at the pinned revision.

| Kind | File | License | Pool |
|---|---|---|---|
| `swe` | `nvidia/Open-SWE-Traces` `v1.0/sweagent/0000.parquet` at `4fa4c2ae9afc6b397bc50391d43450c79993c138`, resolved traces whose repository license is MIT, Apache-2.0, BSD-2-Clause, or BSD-3-Clause | CC-BY-4.0 | 32 |
| `code` | `ise-uiuc/Magicoder-Evol-Instruct-110K` `data-evol_instruct-decontaminated.jsonl` at `b0079beaa0361d82412520b873715bee59cc7dd4` | Apache-2.0 | 400 |
| `tool` | `NousResearch/hermes-function-calling-v1` `func-calling.json` at `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` | Apache-2.0 | 120 |
| `chat` | `HuggingFaceH4/ultrachat_200k` `data/train_sft-00000-of-00003-a3ecf92756993583.parquet` at `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT | 400 |

The pool is how many rows the writer stores. It is not the fill. A pipeline renders rows until the sequence quota below is full, then stops. `manifest.json` `counts` is the number of rows in that file. It must match the file.

A dialog that contains an email address is left out, including a tool result or function arguments. Token ids are not stored. Each pipeline renders a row with its own chat template, thinking off, and includes `tools` when the row has them.

## Sequences

The capture fills 128 sequences. Each sequence is 2048 tokens. Nothing is generated. A kind that renders fewer sequences than its quota stops the capture. The quota is sequences, not rows.

| Kind | Sequences | Tokens |
|---|---:|---:|
| `swe` | 34 | 69632 |
| `code` | 34 | 69632 |
| `tool` | 34 | 69632 |
| `chat` | 26 | 53248 |
| Total | 128 | 262144 |

The reader takes rows of one kind in file order, then the next kind. A SWE row contributes the tokens that start 128 tokens before its first tool call, up to 8192 tokens. A row with no tool call contributes nothing. A coding, tool, or chat row contributes tokens from the start, up to 4096 tokens. Those tokens are packed into sequences of 2048. A remainder that does not fill a sequence is left unused.

`swe`, `code`, `tool`, and `chat` each fill their sequence quota. `chat.json` has 26 rows. Its manifest entry still says 400.

A pipeline prefills these sequences in that order and accumulates one input Gram per matrix. The Gram stays beside that checkpoint. How many blocks share one prefill, and when a rounded block becomes visible to the next block, is that pipeline's note.

## Hessian

For one matrix, the calibration inputs are `X` with shape `[262144, d]`. One row is one token. `d` is that matrix's input width. `262144` is `128 × 2048`.

The input Gram is `G = X^T X`, shape `[d, d]`. Every row of the matrix shares that Gram. GPTQ's Hessian is `H = 2G`. The column update uses the upper Cholesky factor of `(H + λI)^{-1}`. A nonpositive diagonal entry is a dead column. `λ` is `0.01` times the mean of the corrected diagonal. The lazy-update block is 128 columns.

FOEM uses that same factor. `β` is 0.1. There is no second corpus and no second Gram.

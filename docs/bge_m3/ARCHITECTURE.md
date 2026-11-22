# BGE-M3 — architecture

As-built detail for the `bge_m3` embeddings pipeline. [README.md](README.md) is the summary.

- The listener and the write set are the [process contract](../../ARCHITECTURE.md).
- Papers for this pipeline are in [references.md](references.md).
- `handle_embed` is an `EmbeddingsPipeline`.
- `bge_m3.portable` is the serve entry. `serve` imports that package.
- `metal41` is the device. The GPU session, shaders, quantize, and pack stay there.
- `metal41` re-exports the pipeline, so an import of the device package still returns the embeddings pipeline.
- The device opens from `portable` `open_runtime`. The sampling card is `portable/sampling.py`.
- Clients reach this pipeline through `favlo serve` once the record is registered.
- The packed directory is `models/bge-m3-int8`.

```text
uv run favlo --bge_m3 quantize --src <checkout>
uv run favlo --bge_m3 embed --text <sentence>
uv run favlo --bge_m3 kernels
```

- `quantize` reads the publisher checkout and writes `models/bge-m3-int8`.
- Quantize is the only command that opens the checkout.
- The checkout is the publisher repository: `pytorch_model.bin`, `colbert_linear.pt`, `sparse_linear.pt`, and `tokenizer.json`.
- The row stores that directory.
- `quantize` is the command that imports torch.
- That library is not a Favlo dependency.
- `embed` loads the record and prints the 1024 floats.
- A missing record is an error.
- The command does not quantize.
- `kernels` dispatches the int8 matmul and one attention head against the host formula.
- A warm encode of that record is [research/recorded-encode.md](research/recorded-encode.md).
- The Neural Engine window above 512 is [failed/failed.md](failed/failed.md).

## Packed record

- One int8 weight image on the GPU, in one shared span when `weights.bin` is present.
- Each projection is symmetric int8 with one float32 scale per output channel (peak divided by 127).
- The activation row that enters a projection is quantized the same way.
- The product is int8 times int8 with an int32 accumulator, then the two scales and the float16 bias.
- Word and position rows, the type row, LayerNorm, and biases stay float16.
- One sequence argument covers every length up to 4096, padded to a multiple of 64.
- The same tokens in a longer pad must match with a minimum cosine of 0.999.
- Quantize passes when cosine against the float32 encoder on the 128 passages has a minimum of at least 0.96.
- An input longer than the served window is rejected.
- The tokenizer and the pad run on the host.
- The dense vector is the L2 norm of hidden row 0, the `<s>` token.
- Position ids start at 2.

## Weight file

- `quantize` writes `weights.npz` and `weights.bin`.
- `weights.bin` is those same arrays, raw, in the order the loader uploads them.
- That order is the word table, the position table, the type row, the embedding LayerNorm, then each layer.
- A layer is q, k, v, o, up, and down, each as int8 codes, a float32 scale, and a float16 bias, then the attention LayerNorm and the feed-forward LayerNorm.
- Codes stay `[N, K]` row-major.
- `weights.json` lists each array's name, dtype, shape, byte offset, and byte size.
- An offset is the sum of the real array sizes before it.
- The file does not store the 16-byte floor `Gpu.upload` uses, so a later array does not move.
- When `weights.bin` is present, open reads it into one shared span in 8 MiB steps and binds each tensor at its offset.
- That open does not call `np.load`.
- When `weights.bin` is absent, open still reads `weights.npz` and uploads each array.
- A finished tree may contain `weights.npz`, or `weights.bin` together with `weights.json`.
- The tokenizer, `cosine.json`, and `manifest.json` checks are unchanged, so an existing `weights.npz` tree is not packed again.

## Memory

- Device spans come from `favlo.memory`.
- The serve process keeps the ledger.
- This runtime asks for the budget at open and again before a longer passage.
- Spans go through `device_buffer`.
- The one knob is the passage length.
- Weights plus one passage at the named length must fit in `room()` before the first commit.
- A length that does not fit raises `RuntimeError` and names that length.
- `open_runtime` takes `owner`; every allocate in that opening uses it; `close` sweeps that string.

## Registry

- The row key in `models/registry.json` is the packed folder, `bge-m3-int8`.
- Registration leaves `alias` null unless one is passed, so the folder is the public id.
- `directory` is copied at open, so close the resident model after changing it.
- The listener stays up.
- An embeddings row is omitted from `GET /v1/models`.

```python
from favlo.registry import load_registry, save_registry

data = load_registry()
data["embeddings"] = "bge-m3-int8"
save_registry(data)
```

| Field | When | Values |
|---|---|---|
| `alias` | resolve | One name, with no colon and no slash, or `null`. |
| `directory` | open | The packed record `POST /v1/embeddings` opens. |
| `pipeline_default` | omitted model | `true` on one row of this pipeline. Enabling it clears the mark on the other rows of this pipeline. |

- `context_length` and `max_prompt_tokens` are written at registration (4096, the longest function quantize builds).
- The encode does not read them.
- The served length is the constant 4096 in this package, and the longest function the open record contains.
- `GET /v1/models` does not list this row.
- The file key `embeddings` is the folder `POST /v1/embeddings` uses when the body omits `model`.
- The first registration whose pipeline defines `handle_embed` fills an empty key with the folder.
- A stored key stays.
- The sampling card (`embed`, temperature 0) is copied because registration requires one variant.
- An embedding request does not read it.
- `kv_cache` and `resident_percent` are discarded at open.
- `mtp`, `mtp_depth`, `enable_thinking`, and `reasoning_effort` may sit on the row and are not read.
- `pipeline`, `runtime`, `source`, and `pipeline_root` are the row's identity.

## Request

- `embed` returns the vector and the token count.
- The list object is the class.
- `POST /v1/embeddings` takes `model` and `input` as one string or an array of strings.
- The body is one embedding object per input and `usage.prompt_tokens`.
- A pipeline that defines `handle_embed` is omitted from `GET /v1/models`.

An embeddings request:

- Waits until the open chat completion returns.
- Sits ahead of a chat that has not started.
- Does not preempt media.
- Does not close the resident model.
- Does not wait on a memory hold or on `wait_released`.
- Does not take that hold in its process while it loads or unloads.

- The embeddings process closes when the request returns.
- A chat that has not started waits while an embeddings request is queued or running.

## Checks

```text
uv run pytest tests/test_bge_m3.py tests/test_bge_m3_smoke.py
uv run python -m harness.embeddings_agent
```

- The host-math tests run always.
- The smoke skips when `models/bge-m3-int8` is absent.
- The harness posts several sentences to the route on `127.0.0.1:8129` and checks that paraphrase cosines exceed cross-topic cosines.

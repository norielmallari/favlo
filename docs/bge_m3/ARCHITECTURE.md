# BGE-M3 — architecture

This page is the as-built detail for the `bge_m3` package.

[README.md](README.md) is the summary.

The listener and the write set are the [process contract](../../ARCHITECTURE.md).

Papers for this pipeline are in [references.md](references.md).

`favlo --bge_m3 quantize --src <checkout>` reads that directory and writes `models/bge-m3-int8`.

Quantize is the only command that opens the checkout.

The checkout is the publisher repository:

- `pytorch_model.bin`
- `colbert_linear.pt`
- `sparse_linear.pt`
- `tokenizer.json`

The row stores that directory.

`embed` loads the record.

The packed record:

- Is one int8 weight image on the GPU.
- Stores each projection as symmetric int8 with one float32 scale per output channel, the peak divided by 127.
- Quantizes the activation row that enters a projection the same way.
- Forms the product as int8 times int8 with an int32 accumulator, then applies the two scales and the float16 bias.
- Keeps word and position rows, the type row, LayerNorm, and biases in float16.
- Uses one sequence argument for every length up to 4096, padded to a multiple of 64.
- Requires the same tokens in a longer pad to match with a minimum cosine of 0.999.
- Passes quantize when cosine against the float32 encoder on the 128 passages has a minimum of at least 0.96.
- Rejects an input longer than the served window.
- Runs the tokenizer and the pad on the host.
- Takes the dense vector as the L2 norm of hidden row 0, the `<s>` token.
- Starts position ids at 2.

`quantize` is the command that imports torch.

That library is not a Favlo dependency.

The Neural Engine window above 512 is [failed/failed.md](failed/failed.md).

`favlo --bge_m3 embed --text <sentence>` prints the 1024 floats.

A missing record is an error.

The command does not quantize.

A warm encode of that record is [research/recorded-encode.md](research/recorded-encode.md).

## Registry

The row key in `models/registry.json` is the packed folder, `bge-m3-int8`.

Registration leaves `alias` null unless one is passed, so the folder is the public id.

`directory` is copied at open, so close the resident model after changing it.

The listener stays up.

An embeddings row is omitted from `GET /v1/models`.

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

`context_length` and `max_prompt_tokens` are written at registration (4096, the longest function quantize builds).

The encode does not read them.

The served length is the constant 4096 in this package, and the longest function the open record contains.

`GET /v1/models` does not list this row.

The file key `embeddings` is the folder `POST /v1/embeddings` uses when the body omits `model`.

The first registration whose pipeline defines `handle_embed` fills an empty key with the folder.

A stored key stays.

The sampling card (`embed`, temperature 0) is copied because registration requires one variant.

An embedding request does not read it.

`kv_cache` and `resident_percent` are discarded at open.

`mtp`, `mtp_depth`, `enable_thinking`, and `reasoning_effort` may sit on the row and are not read.

`pipeline`, `runtime`, `source`, and `pipeline_root` are the row's identity.

`handle_embed` is an `EmbeddingsPipeline`.

`embed` returns the vector and the token count.

The list object is the class.

`POST /v1/embeddings` takes `model` and `input` as one string or an array of strings.

The body is one embedding object per input and `usage.prompt_tokens`.

A pipeline that defines `handle_embed` is omitted from `GET /v1/models`.

An embeddings request:

- Waits until the open chat completion returns.
- Sits ahead of a chat that has not started.
- Does not preempt media.
- Does not close the resident model.
- Does not wait on a memory hold or on `wait_released`.
- Does not take that hold in its process while it loads or unloads.

The embeddings process closes when the request returns.

A chat that has not started waits while an embeddings request is queued or running.

`uv run pytest tests/test_bge_m3.py tests/test_bge_m3_smoke.py` runs the host math.

The smoke skips when `models/bge-m3-int8` is absent.

`python -m harness.embeddings_agent` posts several sentences to the route on `127.0.0.1:8129` and checks that paraphrase cosines exceed cross-topic cosines.
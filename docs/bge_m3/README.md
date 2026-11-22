# BGE-M3

Embeddings pipeline `bge_m3` serves [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT License) through `favlo serve` after `quantize` registers the packed record at `models/bge-m3-int8`.

Chen et al., [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).

- `handle_embed` is an `EmbeddingsPipeline`.
- `bge_m3.portable` is the serve entry. `favlo serve` and `favlo --bge_m3` load it.
- `metal41` is the device that entry opens.
- Clients call `POST /v1/embeddings` with `model` set to that folder.
- Omit `model` when the registry `embeddings` key points at it.
- `quantize`, `embed`, and `kernels` use `favlo --bge_m3`.

```text
uv run favlo --bge_m3 --help
uv run favlo --bge_m3 kernels
uv run favlo --bge_m3 quantize --src <checkout>
uv run favlo --bge_m3 embed --text <sentence>
```

## Setup

Official weights are [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).

- `quantize` reads the publisher checkout and writes `models/bge-m3-int8`.
- It is the only command that opens that checkout.
- `embed` prints the 1024-dimensional dense vector from an existing record.
- It does not quantize.
- A missing record is an error.
- `kernels` checks a synthetic int8 tile and one attention head.

```text
uv run favlo --bge_m3 quantize --src <checkout>
uv run favlo --bge_m3 embed --text <sentence>
uv run favlo --bge_m3 kernels
```

Packed record, forward, registry, and request order: [ARCHITECTURE.md](ARCHITECTURE.md).

- [references.md](references.md) — papers and model card
- [research/recorded-encode.md](research/recorded-encode.md) — warm encode timings
- [failed/failed.md](failed/failed.md) — Neural Engine window above 512 tokens

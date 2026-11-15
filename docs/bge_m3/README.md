# BGE-M3

Embeddings for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), MIT.

Chen et al., [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).

The packed folder is `models/bge-m3-int8`.

A client calls `POST /v1/embeddings` with `model` set to that folder.

It may omit `model` when the registry `embeddings` key points at it.

It may run the CLI below.

```bash
favlo --bge_m3 quantize --src <checkout>
favlo --bge_m3 embed --text <sentence>
```

`quantize` writes the int8 record from the publisher checkout.

`embed` prints the 1024-dimensional dense vector from an existing record.

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

- [references.md](references.md) — papers and model card
- [research/recorded-encode.md](research/recorded-encode.md) — warm encode timings
- [failed/failed.md](failed/failed.md) — Neural Engine window above 512 tokens
# VoxCPM2

Tokenizer-free text-to-speech from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

Those weights are under the Apache License, Version 2.0.

`setup` downloads the checkout into `models/openbmb/VoxCPM2`.

`models add` registers a finished checkout.

Chat completions are refused.

Speech is generated only through `favlo media --pipeline voxcpm2` (or `voxcpm2__generate_voice` on `POST /mcp`).

`favlo --voxcpm2` does not speak a line.

As-built detail: [ARCHITECTURE.md](ARCHITECTURE.md).

```text
uv run favlo --voxcpm2 setup
uv run favlo --voxcpm2 inspect
uv run favlo --voxcpm2 kernels
uv run favlo --voxcpm2 models add VoxCPM2
```

Start `favlo serve`, then:

```text
uv run favlo media --pipeline voxcpm2 input="Hello." persona="Clear adult voice"
uv run favlo media --pipeline voxcpm2 --model VoxCPM2 --out pack.json input="Hello." persona="Clear adult voice"
```

- `input` and `persona` are required.
- `voice` must be empty or omitted.
- Stdout is one `file://` URI.

- [ARCHITECTURE.md](ARCHITECTURE.md) — kernels, registry, listener, forward
- [references.md](references.md) — papers and specifications
- [smoke.json](smoke.json) — voice smoke-agent line and persona

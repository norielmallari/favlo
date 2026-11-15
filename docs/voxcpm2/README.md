# VoxCPM2

Favlo serves tokenizer-free text-to-speech from the OpenBMB checkpoint [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0).

`setup` places weights under `models/openbmb/VoxCPM2`.

Chat completions are refused.

Speech goes through the voice media path.

With `favlo serve` running, `favlo media --pipeline voxcpm2` posts `input` and `persona` to the listener.

It returns a stored audio URI.

MCP clients call `voxcpm2__generate_voice` on `POST /mcp`.

```text
uv run favlo --voxcpm2 setup
uv run favlo --voxcpm2 inspect
uv run favlo --voxcpm2 kernels
```

```text
uv run favlo media --pipeline voxcpm2 input="Hello." persona="Clear adult voice"
uv run favlo media --pipeline voxcpm2 --model VoxCPM2 --out pack.json input="Hello." persona="Clear adult voice"
```

As-built detail is in [ARCHITECTURE.md](ARCHITECTURE.md).

It covers kernels, registry, listener, and inference behavior.

[references.md](references.md) lists papers and specifications.
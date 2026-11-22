# VoxCPM2 — architecture

This file is the as-built detail for the `voxcpm2` package.

[README.md](README.md) is the summary.

The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

This directory wins for this pipeline's records and kernels.

Another pipeline does not edit it.

- `favlo serve` and `favlo --voxcpm2` load `voxcpm2.portable`. That package is the serve entry.
- `metal41` is the Metal device that entry opens.
- The device package re-exports the pipeline object. The device opens from `portable` `open_runtime`.

## Checkpoint

- Model card: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
- Report: [arXiv:2606.06928](https://arxiv.org/html/2606.06928).
- VoxCPM formulation: Zhou et al., [arXiv:2509.24650](https://arxiv.org/abs/2509.24650).
- Text-semantic stack: MiniCPM-4, [arXiv:2506.07900](https://arxiv.org/abs/2506.07900).
- Kernel papers: [references.md](references.md).

`setup` downloads the checkout into `models/openbmb/VoxCPM2`.

- `model.safetensors` is 4,580,080,592 bytes, bfloat16.
- `audiovae.pth` is 376,951,122 bytes.
- A file that already has that size is left in place.
- `setup` does not register a row.
- `handle_chat` refuses a chat completion.

```text
uv run favlo --voxcpm2 setup
uv run favlo --voxcpm2 pack
uv run favlo --voxcpm2 inspect
uv run favlo --voxcpm2 kernels
```

`inspect` reads the safetensors header and prints the dtype counts.

`kernels` compiles `voxcpm2/shaders/kernels.metal` at Metal 4.1.

It checks the following against the host formulas:

- RMSNorm
- the half matmul
- rotate-half RoPE
- grouped-query attention
- the causal convolutions
- the float bias, SiLU, SwiGLU, residual, and row copy

One stack step is checked as a single command buffer.

## Packed record

`pack` reads `model.safetensors` and `audiovae.pth`.

It writes `weights.bin` and `pack.json` in the checkout.

It does not delete either source file.

```text
uv run favlo --voxcpm2 pack
uv run favlo --voxcpm2 pack --src <checkout> --dest <directory>
```

`weights.bin` is one contiguous blob.

`pack.json` is version 1.

It names the blob and each tensor's kind, dtype, shape, offset, and byte length.

A linear also stores its padded shape.

Each tensor starts on a 128-byte boundary.

The bytes between tensors are zero.

The groups are written in this order.

Each group is sorted by name:

- Padded float16 linears. The values are the bfloat16-to-float32 promotion, then the float16 cast in `Session.resident`. N is padded to 32 and K to 256. Pad cells are zero. The stored payload is that padded matrix.
- Float16 RMSNorm gains, with no pad.
- Float32 biases, with no pad.
- The token embedding, float32 and row-major, shape `[vocab, dim]`.
- Other host vectors, including `feat_encoder.special_token`, float32, in checkpoint shape.
- AudioVAE tensors after `weight_v` / `weight_g` fusion, float32 and C-order.

Open allocates one shared buffer for `weights.bin` and fills it with an 8 MiB sequential read.

`Store.weight`, `Store.gain`, and `Store.bias_handle` are spans at those offsets.

Convolution weights and biases are spans in the same buffer.

`conv1d` and `conv_transpose1d` release a weight they uploaded for that call.

A packed span is not that handle, so the buffer stays for the next call.

`matmul_f16`, `rmsnorm_row`, `bias_pack`, `causal_conv1d`, and `causal_transpose1d` are unchanged.

The linear payload is larger than the safetensors tensor by the pad.

`working_set_bytes` follows the file open reads, plus the scratch reservation.

When `weights.bin` and `pack.json` are present, that file is `weights.bin`.

When they are absent, it is `model.safetensors` plus `audiovae.pth`.

Open then memory-maps the safetensors file and unpickles `audiovae.pth`.

`models add` still accepts a tree that has `model.safetensors` and `config.json`.

A tree with `weights.bin` and `config.json`, and no safetensors, is complete as well.

`kernels` builds a temporary safetensors file for the shader check.

That check is not the serve path.

`favlo --voxcpm2` does not generate audio.

Speech goes through `favlo media` only.

## Generation

`favlo media --pipeline voxcpm2` writes one 48 kHz WAV through the listener.

- `matmul_f16` binds float16 linears padded to N 32 and K 256.
- `matmul2d` accumulates into float.
- A language-model step runs as one command buffer.
- One diffusion estimate, including its conditioning projections and the diffusion stack, runs as one command buffer.
- Values between those kernels stay float32; the matmul casts its activation to half.
- The key and value cache stays float32.
- The stop head reads two logits on the host.
- The AudioVAE and the token embedding stay on their current path.

An empty `voice` is basic TTS.

- The line is tokenized the way published inference calls `tokenize`.
- There is no BOS id.
- The audio-start id is appended.
- No reference clip is encoded.

A named `voice` is refused.

`persona` is required.

- Parentheses and sentence punctuation are removed so the phrase is not its own utterance.
- The remaining words are wrapped in one pair of parentheses and placed in front of `input`.
- A space is inserted when `input` starts with an ASCII letter or digit, so that word keeps the token it has on its own.
- The model receives that one string.
- The opening parenthesis is its initial token.

The checkpoint sets `use_mup` false, so a token embedding is unscaled and a residual add is unscaled.

- `scalar_quantization_scale` is 9.
- `scalar_quantization_latent_dim` is 512.
- The bound is `round(tanh(y) * 9) / 9` after the input projection.
- Audio positions pass through that bound.
- Text positions keep the raw hidden state.

Each patch is 4 latent frames.

The diffusion transformer takes 10 Euler steps, guidance 2, sway sampling, and CFG-Zero*.

Early Euler steps counted by `zero_init_steps` have zero velocity.

The stop head ends the line after the minimum length of 2 patches.

The patch limit is the smaller of:

- 4096
- `text tokens × 6 + 10`
- the positions left in `max_length` after the text and the audio-start id

A sample row counts each diffusion step and the decoder.

- Its detail names the patch and the Euler step before that step's estimates run.
- The row advances when those command buffers have finished.
- Each finished patch logs its seconds.
- The job records `progress`, `stage`, and `eta` from that count.

The AudioVAE decoder is float32, weight-normalized, and conditioned on the 48 kHz bin.

`MediaPipeline` calls `ensure_open` before the line and always calls `close` when the request returns.

`ensure_open` runs the shared open gate against the weight files plus scratch, and reloads when closed.

This package does not keep a private memory cap.

`models add <folder>` writes `manifest.json` with `complete` true when `config.json` is present and the directory holds `model.safetensors` or `weights.bin`, then registers the folder.

The served record is `models/VoxCPM2`, a link to the checkout.

`setup` still does not register.

## Registry

The row key in `models/registry.json` is the folder passed to `models add`, `VoxCPM2` for the served link.

`setup` does not register.

`models add` writes `manifest.json` with `complete` true when `config.json` is present and the directory holds `model.safetensors` or `weights.bin`, then registers that folder.

`alias` stays null unless `models add` is given one, so the folder is the public id.

`directory` is copied at open, so close the resident model after changing it.

The listener stays up.

```python
from favlo.registry import load_registry, save_registry

data = load_registry()
data["media"]["voice"] = "VoxCPM2"
save_registry(data)
```

| Field | When | Values |
|---|---|---|
| `alias` | resolve | One name, with no colon and no slash, or `null`. |
| `pipeline_default` | omitted model | `true` on one row of this pipeline. Enabling it clears the mark on the other rows of this pipeline. |
| `directory` | open | The checkout `serve` opens. It holds `config.json` and either `model.safetensors` or `weights.bin`. |

The file key `media.voice` is the folder `favlo media` uses when the command omits `--model` and that folder is this pipeline.

When no key is stored yet, voice registration fills it.

A stored key stays.

The sampling card (`voice`, temperature 0) is copied at registration.

Speech reads the checkpoint flow-matching card, not that row.

`kv_cache` and `resident_percent` are discarded at open.

The live cache stays float32.

`mtp`, `mtp_depth`, `enable_thinking`, `reasoning_effort`, `context_length`, and `max_prompt_tokens` may sit on the row and are not read.

`context_length` is 8192 at registration.

`pipeline`, `runtime`, `source`, and `pipeline_root` are the row's identity.

## Listener

`favlo serve` must already be listening.

`favlo media` does not start it, and it exits when that port is closed.

The command posts the body to the listener.

Stdout is one `file://` URI.

`--out` writes the pack JSON, `metadata` and `mediaUri`, and does not copy the audio file.

`--model` is optional.

- An omitted `--model` uses the voice default in the registry when that folder is this pipeline.
- Otherwise it uses the earliest folder registered for this pipeline.

`--kind` is unnecessary.

This pipeline serves one kind.

```text
uv run favlo media --pipeline voxcpm2 input="Hello." persona="Clear adult voice"
uv run favlo media --pipeline voxcpm2 --model VoxCPM2 --out pack.json input="Hello." persona="Clear adult voice"
```

- `input` and `persona` are required.
- `voice` must be empty or omitted.
- `response_format` defaults to `mp3`.

`media` is the only generation command.

There is no `speak` CLI.

`handle_voice` is a `MediaPipeline`.

A chat that arrives between patches pauses the job.

The same job speaks the line again after the chat releases the GPU.

Patches drawn before the pause are not kept.

`voxcpm2__generate_voice` on `POST /mcp` speaks `input` with the required `persona` and an empty `voice`.

The row does not list speakers or emotions.

`python -m harness.voice_smoke_agent --pipeline voxcpm2 --tool generate_voice` calls that tool for the model, line, and persona in [smoke.json](smoke.json).

The file is `artifacts/voice-smoke-agent.wav`, 48 kHz mono, with RMS above the silence floor and a duration inside `max_seconds`.

That run is not part of `pytest`.

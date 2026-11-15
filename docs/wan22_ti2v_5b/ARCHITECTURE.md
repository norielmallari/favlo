# Wan2.2 TI2V-5B architecture

This page is the as-built detail for the `wan22_ti2v_5b` package. [README.md](README.md) is the summary.

- Text-to-video and image-to-video on the Metal 4.1 GPU.
- The weights are the 5B transformer from the [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) card.
- `favlo --wan22_ti2v_5b` loads `pipelines/wan22_ti2v_5b/metal41/`.
- The record is `models/Wan2.2-TI2V-5B-int8`.
- The umT5 encoder is the int8 pack named by `parts.umt5` on the registry row, `models/google/umt5-xxl-int8`.
- A clip reads that pack and does not open the BF16 encoder file.
- The tokenizer is `tokenizer.json` inside that pack.
- `favlo --wan22_ti2v_5b umt5` writes the pack.
- This pipeline does not pack the encoder.
- The VAE is the float32 `Wan2.2_VAE.pth` from that card.

- `handle_video` is a `MediaPipeline`.
- `wan22_ti2v_5b__generate_video_t2v` samples from text.
- `wan22_ti2v_5b__generate_video_i2v` conditions the first latent frame on a PNG.
- The clip is 24 fps.
- Both tools require `size`.
- Edges are multiples of 32.
- The catalog lists `1280x704` and `704x1280`.
- MCP `duration` is clip length in seconds.
- An omitted `duration` is 121 frames.
- `steps` is 0 or 4.
- `guidance_scale` is omitted or 1.
- `negative_prompt` is accepted and not encoded.

The sampler uses four knots, 1000, 960, 888.888889, and 727.272727. Each knot is one forward. The update reads `x0 = x_t - sigma * v`. A remaining knot mixes that prediction with fresh noise, `x = (1 - sigma_next) * x0 + sigma_next * eps`. The last knot decodes `x0`.

- Self-attention, cross-attention, and the feed-forward are symmetric int8, one float scale per output row.
- `quantize` prints the relative RMS of each matrix.
- The worst matrix was `blocks.11.ffn.2` at 0.014504 ([research/int8-weight-error.md](research/int8-weight-error.md)).
- The feed-forward activation is tanh-GELU (tanh versus erf max abs 0.000473).
- Time, text, the patch, the head, norms, and modulation stay float16.
- The VAE stays float32.
- It is the 48-channel Wan2.2 autoencoder.
- Pixels are grouped 2×2 into 12 channels with the width offset outside the height offset, then restored the same way.
- The VAE walks one latent frame at a time and keeps the causal cache.
- A wide convolution is float16 im2col plus a 64-wide `matmul2d`, 4096 output rows at a time.
- Every tile of that convolution, the bias, and the float16 rounding share one submission.
- The convolution scratch stays allocated for the next one.
- A residual's channel RMS, SiLU, and convolution share one submission too.
- A narrow one is float `conv3d`.
- An int8 product and its scale share one command buffer.
- Attention submits every head before each wait.
- A knot keeps the residual stream on the device and submits its 30 blocks before it waits.
- The prompt's cross-attention key and value stay for the four knots.
- Weights already on the device stay until close.
- On an M5 Pro with 48 GB the isolated DiT sum at 576×320 and 9 frames is 4.267 s.
- A printed 832×480 9-frame text clip decoded in 18.942 s and the still clip decoded in 15.315 s.
- That measurement is [research/clip-ceiling.md](research/clip-ceiling.md).

- `MediaPipeline` calls `ensure_open` before the clip and always calls `close` when the request returns.
- `ensure_open` runs the shared open gate and reloads the DiT, text encoder, and VAE when closed.
- The serve ledger answers `memory.budget()`.
- `--max-memory` is that ledger's cap through `memory.set_max_memory`.
- Device spans come from `device_buffer`.
- The decoder frame tile and the clip fit use the largest answer this opening has seen, and a later dip does not shrink them.
- A request keeps the resolution, frame count, and duration it named when that clip fits.
- Open refuses when three decoder frames do not fit.
- The 30 DiT blocks and the 24 text blocks stay resident until that request's `close`.

- The VAE file stays.
- `still.png` is the horse still.
- A still that is not the clip size is resized to that size, and that frame is the first latent.
- A canvas smaller than 832×480 is a routing check.
- [MLX-Gen](https://github.com/lpalbou/mlx-gen/blob/main/docs/wan-video.md) already treats that smaller canvas as a routing check.
- This note does not add a score for the cutoff.

Download, pack, and the first clip are [README.md](README.md). The shared umT5 pack is [../umt5_xxl/README.md](../umt5_xxl/README.md).

```text
uv run favlo --wan22_ti2v_5b umt5 setup
uv run favlo --wan22_ti2v_5b umt5 quantize --format wan
uv run favlo --wan22_ti2v_5b setup
uv run favlo --wan22_ti2v_5b quantize --weights <checkpoint>
uv run favlo --wan22_ti2v_5b kernels
uv run favlo --wan22_ti2v_5b ceilings
```

- `setup` writes `Wan2.2_VAE.pth` under `models/Wan-AI/Wan2.2-TI2V-5B/`.
- `--student-dir` optionally downloads the Turbo student `model.pt`.
- `quantize --weights` packs `models/Wan2.2-TI2V-5B-int8` and registers that folder.
- The VAE is not packed.
- `kernels` compiles and checks `matmul2d`.
- `ceilings` times the isolated kernels.

## Registry

The row key in `models/registry.json` is the packed folder, `Wan2.2-TI2V-5B-int8`. Registration leaves `alias` null, so that folder is the public id. `directory` and `parts` are copied at open, so close the resident model after changing them. The listener stays up.

```python
from favlo.registry import load_registry, save_registry

data = load_registry()
data["models"]["Wan2.2-TI2V-5B-int8"]["parts"]["umt5"] = "models/google/umt5-xxl-int8"
save_registry(data)
```

| Field | When | Values |
|---|---|---|
| `parts.umt5` | open | The int8 umT5 pack. A missing name opens `<directory>/umt5` when that directory has `manifest.json`, and otherwise `models/google/umt5-xxl-int8`. The pack must be int8. A BF16 path is refused. |
| `alias` | resolve | One name, with no colon and no slash, or `null`. |
| `pipeline_default` | omitted model | `true` on one row of this pipeline. Enabling it clears the mark on the other rows of this pipeline. |
| `directory` | open | The packed record `serve` opens. |

- Registration writes `media.t2v` and `media.i2v` to this folder when those keys are empty, because the tools are `generate_video_t2v` and `generate_video_i2v`.
- `favlo media` looks up `media.video`, because the handler kind is `video`, so those stored keys are not that lookup.
- An omitted `--model` uses `pipeline_default`, then the first folder registered for this pipeline.

- The sampling card (`default`, temperature 0) is copied at registration.
- A video request does not read it.
- `kv_cache` and `resident_percent` are discarded at open.
- `mtp`, `mtp_depth`, `enable_thinking`, `reasoning_effort`, `context_length`, and `max_prompt_tokens` may sit on the row and are not read.
- `pipeline`, `runtime`, `source`, and `pipeline_root` are the row's identity.
- `steps` and `guidance_scale` are request fields.
- They are not registry keys.

## Listener

- `favlo serve` must already be listening.
- `favlo media` does not start it, and it exits when that port is closed.
- The command posts the body to the listener.
- Stdout is one `file://` URI.
- `--out` writes the pack JSON, `metadata` and `mediaUri`, and does not copy the MP4.
- `--model` is optional.
- An omitted `--model` uses the row marked `pipeline_default`, then the first folder registered for this pipeline.
- The lookup is [Registry](#registry).
- `--kind` is unnecessary.
- This pipeline serves one kind.

```text
uv run favlo media --pipeline wan22_ti2v_5b prompt="A horse gallops at speed. Fixed camera." size=832x480 frames=9
uv run favlo media --pipeline wan22_ti2v_5b --model Wan2.2-TI2V-5B-int8 --out pack.json prompt="A horse gallops at speed. Fixed camera." image=file:///absolute/still.png size=832x480 frames=9
```

- `prompt` and `size` are required.
- An omitted `image` is text-to-video.
- A provided `image` is a `file://` URI, a base64 PNG, or a filename in `images`, and is image-to-video.
- `frames` defaults from the tool duration when omitted.
- This is the command that writes the clip.
- The pipeline command does not.

`favlo serve` publishes `wan22_ti2v_5b__generate_video_t2v` and `wan22_ti2v_5b__generate_video_i2v` on `POST /mcp`. A path under `/v1/videos` is 404.

## Smoke

`uv run favlo --wan22_ti2v_5b kernels` compiles and checks `matmul2d`.

`uv run pytest tests/test_wan22_ti2v_5b.py -s` is the package check.

There is no `favlo --wan22_ti2v_5b smoke` subcommand. Clips use `favlo media` only.

Kernel ceilings for one 9-frame 576×320 grid are in [ceilings.md](ceilings.md). The isolated sum is a measurement in [research/clip-ceiling.md](research/clip-ceiling.md).

## Papers

The transformer and the causal VAE are Wan, [arXiv:2503.20314](https://arxiv.org/abs/2503.20314). Channel mean and standard deviation are the Diffusers VAE config on [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).

Self-forcing, [arXiv:2506.08009](https://arxiv.org/abs/2506.08009), and distribution matching, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828), describe how a four-step checkpoint is trained. The served loop stays the bidirectional update above. `matmul2d` is Metal Performance Primitives, described in WWDC26 session 330.

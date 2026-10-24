# Z-Image-Turbo

Eight-step text-to-image pipeline for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). `favlo --z-image-turbo` loads `pipelines/metal41/z-image-turbo.py`. The DiT, the Qwen3 caption encoder, and the VAE decoder run as Metal shaders. The latent stays on the device across the eight steps. Patchify, the Euler update, the time sinusoid, the pad-token fill, the image and caption join, the RoPE index tables, the VAE scale, and the uint8 clip are shaders. The tokenizer and the PNG container stay on the host. The fp4 pack is offline. `matmul2d` consumes fp4 E2M1 weights with an E8M0 scale. Biases, norms, and the two pad tokens stay float16. Activations stay float32. The fp4 matmul reads float16, so a row whose peak exceeds 60000 is scaled into that range and the product is scaled back. A noise-refiner output projection reached 81884. The VAE encoder is not packed. Those weights are under the Apache License, Version 2.0.

Download, pack, and the first image are [SETUP.md](SETUP.md). `setup` downloads the repository into `models/Z-Image-Turbo` from the Hugging Face resolve URL. `quantize` packs fp4 weights into `models/Z-Image-Turbo-fp4`, registers that folder, and stores that directory on the row. `--weights` is one `.safetensors` file or a directory of shards; with no flag, quantize uses the setup checkout. With no `--name` and no `--out`, that file or directory is packed to `models/<name>-fp4`. `--name` is the folder under `models/` and the registry id. A weights-only path uses `models/Z-Image-Turbo` for the tokenizer when that checkout is present. Otherwise pass `--tokenizer-source`. `generate` writes one PNG. `kernels` dispatches the fp4 matmul on a synthetic tile of 64 by 32 and checks it against the dequantized matrix.

```text
uv run favlo --z-image-turbo setup
uv run favlo --z-image-turbo quantize
uv run favlo --z-image-turbo quantize --weights models/example-model.safetensors
uv run favlo --z-image-turbo quantize --weights models/example-model.safetensors --name example-model
uv run favlo --z-image-turbo kernels
uv run favlo --z-image-turbo generate --prompt "a red cube on a white table" --out example.png
```

The default image is 1024×1024. Width and height are multiples of 16 and at most 1280. `--steps` must be 8. The seed selects the initial noise. The publisher's last training stage keeps each edge within 256 pixels of the 1024 grid, from 768 to 1280. 1280 is the largest square in that stage.

`favlo serve` answers `generate_image` on `POST /mcp`. The arguments follow the OpenAI image generation request. A path under `/v1/images` is 404. `prompt` is required. `model` selects the row. `n` is 1 to 10. `size` is `auto` or `WIDTHxHEIGHT`, and `auto` is 1280×720, a 16:9 frame. An omitted `generate_image` size is that same frame. Each edge is a multiple of 16, from 16 to 1280, and the aspect ratio is between 1:3 and 3:1. `quality` accepts `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `standard`, and `hd`. Every accepted value runs the same eight steps. The response reports `low`, `medium`, `high`, `xhigh`, or `max`. `auto`, `standard`, and `hd` are reported as `high`. That reported value selects the PNG zlib level: `low` is 9, `medium` is 7, `high` is 5, `xhigh` is 3, and `max` is 1. A lower quality uses higher compression. The samples stay the uint8 decode. `background` accepts `auto` and `opaque`, and the response is `opaque`. `transparent` is rejected. `moderation` accepts `low` and `auto`. `style`, `user`, and `safety_identifier` are accepted. A non-empty `safety_identifier` is echoed, and otherwise a non-empty `user` is echoed. `output_format` must be `png`. `response_format` must be omitted or `b64_json`. `stream` must be omitted or false, and `partial_images` must be omitted or 0. An optional integer `seed` selects the first image. Each later image uses the next integer. The uint8 decode is written to `~/.favlo/images/<id>.rgb`. The PNG is encoded from that file and written to `<id>.png` before the response is built, and `data[].b64_json` is that PNG. `data[].id` is the file stem. `resources/read` of `favlo://images/{id}.png` returns the PNG and does not take the request lock. Both files stay until `image_ttl_seconds` in `~/.favlo/settings.json`. The default is one day. A sweep when the listener starts, and again after each image is stored, deletes a pair that has reached that age. The response is `created`, `background`, `data[].id`, `data[].b64_json`, `output_format`, `quality`, `size`, `safety_identifier` when one was sent, and `generation_seconds`. One image returns `generation_seconds` as `total`, `caption`, `vae`, and `steps`. Several images return a list of those objects. Opening the runtime reads the packed record under a `load` row counted in bytes. Each image opens a `sample` row counted in finished layers: the 35 caption layers and the caption projection, then each Euler step's two noise refiners, two context refiners, 30 main layers, and the final readout, then 25 decoder pieces. The size is logged once when that row opens. The row names `caption`, `step i/n`, or `decode`. Each step line is logged as the velocity returns. The summary line is `generate`, then the caption seconds and the decoder seconds, and the row closes as `sample done/total layers`. A pause closes that row as `sample paused` or `sample cancelled`. The next call opens a new row for the caption it runs again, the steps still left, and the decoder. Edits and variations are not served. A chat completion on this row is rejected. The message still names `POST /v1/images/generations`.

A running image returns at a stage boundary when `cancel` or `preempt` is set. Cancel is read first. The boundaries are the return from the caption encoder, the return from each Euler step, the return from the VAE decode, and each stored PNG before the next image of `n`. The Metal dispatch already submitted finishes before that return. Cancel leaves `status` `in_progress`, keeps the same `id`, writes no further file, and drops the checkpoint. Preempt keeps finished host outputs on `image_hold` and returns the same `status` and `id`. The caption embedding stays on the device, so a resume runs the caption again. A finished Euler step keeps its latent, and the resume uploads that latent and continues at the next step. A finished decode keeps the uint8 image, and the resume encodes the PNG from those samples. Stored PNGs stay in `data` and are not drawn again. The finished response has no `status`. While the job is in progress, `stage` is `caption`, `step`, `decode`, or `store`, `progress` counts finished steps, decodes, and stored images and stays below 100, and `eta` is the remaining Euler steps times the mean finished step.

Device spans come from the process memory manager. `open_runtime` takes `owner` and `max_memory`. Every span from that opening uses the owner, and `close` sweeps it. An omitted owner is minted from this runtime's id. Scratch for the 1280 square is reserved when the ledger covers it. Otherwise the reservation is the requested width and height, and that image stays the request. A ledger that cannot hold the weights, or the scratch for that image, raises before the draw. `close` finishes the command buffer and sweeps this runtime's spans. `close` then waits, at most two seconds, until those bytes are reclaimable. The image request calls `close` when it returns, including a pause or a rejection. The next image loads the record again. Caption encoding, each Euler step, and the VAE decode release the temporary spans from that stage when it returns.

## Forward

The caption is the checkpoint chat template with one user turn and a generation prompt. The template's `enable_thinking` flag is left unset, which leaves the think block out. The caption vector is the penultimate Qwen3 layer, then the DiT caption norm and projection.

Image tokens precede the caption tokens. A 2×2 patch is stored with the channel fastest. Sequences are padded to a multiple of 32 with the learned pad token at position 0, and those pad tokens attend. Caption indices on the temporal axis start at 1. Every image token uses temporal index `padded caption length + 1`, and the patch coordinate is the two spatial axes. The base is 256 and the split is 32, 48, 48. DiT rotation multiplies adjacent pairs. The caption encoder keeps the half-split rotation. Context refiners have no AdaLN. Noise refiners and the 30 main layers do. The four AdaLN slices are attention scale, attention gate, feed-forward scale, and feed-forward gate. The scale is applied as `x · (1 + scale)`. The gate is `tanh` of its slice and multiplies the branch after the second RMSNorm. `w1` is the SwiGLU gate and `w3` is the up projection. The final readout is a parameter-free LayerNorm with epsilon 1e-6, then `(1 + scale)`, where that scale is a SiLU followed by the 3840-wide projection.

The embedded time is `(1 − sigma) · 1000`, evaluated in float32 on device. Sigma 1 is noise and sigma 0 is the image. The shift is 3. The update adds `(sigma − sigma_next)` times the predicted velocity, on the latent buffer. Kernel and bandwidth ceilings for a 1280 generation are in [ceilings.md](ceilings.md). The isolated sum is a measurement in [research/generation-ceiling.md](research/generation-ceiling.md). On this machine that generation printed 134.36 s, with the eight steps at 15.48–15.72 s.

The VAE latent is `z / 0.3611 + 0.1159`. GroupNorm uses 32 groups and epsilon 1e-5. The mid-block attention is one head of 512 channels. PNG bytes use `(clip(x, −1, 1) + 1) / 2`.

## Smoke

`uv run pytest tests/test_z_image_turbo_smoke.py -s` dispatches the shader check. That check does not need the checkpoint. `uv run favlo --z-image-turbo kernels` is the same check. On this machine the 64×32 fp4 matmul, the K-144 row, and the causal attention case matched with max absolute error 0. `ZIMAGE_SMOKE=1 uv run pytest tests/test_z_image_turbo_smoke.py -s` also generates a 512×512 image with seed 0. The first passing run writes `smoke.json` in this directory. A later run compares the latent hash. On this machine each of those eight steps took 1.41–1.43s, and the decoded image was a red cube on a white table.

`python -m harness.image --model Z-Image-Turbo-fp4` calls `generate_image` on a serve that is already listening and reads the PNG. It does not score the picture.

## Citations

Cai, Zheng, et al. *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer.* arXiv:2511.22699.

Qwen Team. *Qwen3 Technical Report.* The caption encoder is the 36-layer Qwen3 configuration shipped in the Turbo repository.

Peebles and Xie. *Scalable Diffusion Models with Transformers.* AdaLN-Zero.

Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*

Lipman et al. *Flow Matching for Generative Modeling.*

Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* The decoder byte map.

Open Compute Project. *OCP Microscaling Formats (MX) Specification.* fp4 E2M1 and E8M0.

Apple. *Metal 4 tensor operations,* WWDC26 session 330. `matmul2d`.

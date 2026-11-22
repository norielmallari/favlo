# Wan 2.2 I2V — architecture

This page is the as-built detail for the `wan22_i2v` package.

[README.md](README.md) is the summary.

- Four-step image-to-video for [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B).
- `favlo --wan22_i2v` loads `pipelines/wan22_i2v/metal41/`.
- The high-noise expert runs while the timestep is at least 900.
- The low-noise expert runs after that.
- One expert is on device at a time.
- The outgoing expert is released.
- The shared mid-run confirm waits until those pages are reclaimable before the next expert is required.
- Device spans come from `device_buffer`.

- `MediaPipeline` calls `ensure_open` before the clip.
- It always calls `close` when the request returns.
- `ensure_open` runs the shared open gate against the clip's working set.
- A larger ledger keeps every decoded latent frame on device when that volume fits.
- `--max-memory` caps that ledger through `memory.set_max_memory`.
- This package does not keep a second cap.
- `close` sweeps the owner through the shared close gate.

One diffusion step keeps every requested frame resident with one expert.

The served int8 expert is 13.332 GiB on disk.

It stays resident for that step.

The fit check is `expert_bytes` plus `forward_bytes`.

On the packed int8 record that sum is:

- 13.54 GiB for 5 frames at 576×320
- 15.21 GiB for 81 frames at 576×320
- 17.36 GiB for 81 frames at 832×480
- 22.60 GiB for 81 frames at 1280×720

- A machine with 16 GB of unified memory cannot hold that resident expert.
- A machine with 24 GB can hold short or 320p clips when the ledger still covers the sum.
- It misses an 81-frame 480p or 720p step.
- A machine with 48 GB covers the listed resolutions at 81 frames.
- This package does not stream DiT blocks.
- When the sum does not fit, the clip raises.

- The step scratch is the residual, the norm, the half workspace, and one slab.
- Query, key, and the attention output are lanes of that slab.
- Self-attention and cross-attention cast those lanes to half and run `attend_flash`. Sixteen queries share a key tile of 48. The destination stays float.
- The value projection reuses the norm.
- The FFN hidden reuses the slab.
- The decode keeps every frame on device when the remainder covers that volume.
- Otherwise one latent frame.
- The clip length and the resolution stay the request.
- A shortfall raises before the first commit.
- It raises again at each stage boundary when the ledger no longer covers that stage.

- The VAE drops its device weights after encode and after decode.
- Host checkpoint bytes load only for encode and decode when the raw VAE pack is absent.
- When `high/weights.bin` and `high/planes.json` are both present, serve reads that file into one shared span in 8 MiB steps and binds each plane at its offset. The low expert uses the same pair in `low/`.
- Codes are the packed bytes. Scales, the bias, and the permutation are separate planes. Plane starts are 128-byte aligned. `nbytes` is the plane and does not include the pad.
- One expert stays resident until `drop_expert`. The other expert is not loaded to build or read that file.
- When either file is missing, each matrix is still read from its `.npz` or `.npy`, copied to the device, and the host array is dropped before the next file.
- The device spans of that expert stay resident.
- The ledger room for decode is read after the device finishes in-flight work.
- An inflated `pending` count does not force one latent frame when the full volume fits.
- The expert and the DiT scratch are released before decode.
- The expert and the scratch spans are released when the clip finishes and when a step fails.

- The DiT linears are symmetric int8.
- One float32 scale sits per output channel.
- `matmul2d` multiplies int8 activations by those codes and writes int32.
- The product is that value times the row scale and the channel scale, plus the float16 bias.
- The umT5 linears use that same int8 plane.
- The worst relative RMS of those packs is [research/int8-weight-error.md](research/int8-weight-error.md).
- The token embedding, the RMSNorm weights, and the relative buckets stay float16.
- `kernels` still dispatches the fp4 and fp8 matmul.
- A row on that path whose peak exceeds 60000 is divided by that peak over 60000 before the half store.
- The product is multiplied back.
- Biases, QK RMSNorm, `norm3`, modulation, and the patch convolution stay float16.
- Those weights are under the Apache License, Version 2.0.

The text encoder is [google/umt5-xxl](https://huggingface.co/google/umt5-xxl), also Apache License 2.0.

Download, pack, and the first clip are [README.md](README.md).

The shared umT5 pack is [../umt5_xxl/README.md](../umt5_xxl/README.md).

- `setup` writes each download under `models/<org>/<repo>/`.
- The VAE is `models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth`.
- The two expert safetensors are a separate repository directory.
- `setup` prints their paths.
- Sixteen connections write one file.
- A short file continues from the ranges already on disk.

`quantize` reads those two expert files.

- `--int8` writes `models/Wan2.2-I2V-A14B-int8` when `--name` and `--out` are omitted.
- That folder is the served record.
- `--fp8-percent 50` writes `models/Wan2.2-I2V-A14B-fp8-e50`.
- The default percent is 20, which writes `models/Wan2.2-I2V-A14B-fp4`.
- `--name` writes `models/<name>` instead.
- That name is the registry id.
- The row stores that directory and points `parts.umt5` at `models/google/umt5-xxl-int8`.
- `--high-dir` and `--low-dir` store the quantized expert directories.
- Each expert directory also receives `weights.bin` and `planes.json`, copied from those per-file planes in bind order.

It scores every block by the maximum fp4 relative error of its eight attention matrices and its two feed-forward matrices, after one K permutation per matrix.

The served pack stores twenty of the forty blocks as fp8.

An equal score keeps the lower block index.

The high-noise fp8 blocks are 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, and 27.

The low-noise fp8 blocks are 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 24, and 26.

The default percent is 20, eight of forty.

Both rankings are [research/block-fp4-error.md](research/block-fp4-error.md).

- Text and time stay fp4.
- The head weight and its bias are float16.
- `kernels` dispatches a 64 by 32 fp4 matmul, the same tile as fp8, and a 64 by 128 int8 matmul.
- The stored clip is an MP4 encoded from an uncompressed 24-bit AVI.
- Rows are bottom-up BGR.

```text
uv run favlo --wan22_i2v umt5 setup
uv run favlo --wan22_i2v umt5 quantize --format wan
uv run favlo --wan22_i2v setup
uv run favlo --wan22_i2v quantize --int8 --high <high.safetensors> --low <low.safetensors>
uv run favlo --wan22_i2v kernels
```

- The solver is Euler steps with shift 5.
- The default is four steps.
- The knots are 1000, 937.5, 833.3333, 625, and 0.
- `steps` from 1 to 40 rebuilds that shifted grid.
- The update adds `(sigma_next - sigma)` times the velocity.
- `guidance_scale` defaults to 1, so the step is one forward.
- A number sets both experts.
- A pair is the low-noise scale, then the high-noise scale.
- The high-noise scale is used while the timestep is at least 900.
- A scale other than 1 mixes `v_uncond + s * (v_cond - v_uncond)` on the velocity before that update.
- A blank `negative_prompt` is Wan's published sample negative.
- That string is encoded only when a scale is not 1.
- Both prompts share one packed umT5.

- Frame count is `4n+1`.
- The shortest moving clip is 5 frames.
- At 16 fps, 81 frames is the 5 second clip.

Accepted sizes are:

- 240p as 416×240 and 240×416
- 320p as 576×320 and 320×576
- 480p as 832×480 and 480×832
- 720p as 1280×720 and 720×1280

Both edges are multiples of 16.

- The image is the first pixel frame.
- `last_image` is the last pixel frame. Both stills are resized to the clip size.
- The frames between them are zeros. The mask marks that last pixel only, so the last latent group is `[0, 0, 0, 1]`.
- Five frames leaves one latent step between the stills. The 4-step record is built for 81 frames.
- A clip with no last still leaves those frames, including the last one, at zero.

`Wan2.1_VAE.pth` is the causal encoder and decoder.

- `vae.weights.bin` and `vae.planes.json` beside that file are the float tensors in the order encode and decode upload them, after the reshape the convolution uses.
- `conv1` and `conv2` are in that span. Encode applies `conv1` after the encoder head. Decode applies `conv2` before the decoder.
- Serve reads that file into one shared span when both are present and does not unpickle the checkpoint.
- The span is released with the other VAE device weights.
- When either file is missing, the loader unpickles `Wan2.1_VAE.pth`.
- `quantize` writes the pair when the checkpoint is already on disk.

- Base width 96, multipliers 1, 2, 4, 4, two residual blocks, temporal downsample on the last two stages, latent width 16.
- Spatial stride is 8 and temporal stride is 4.
- The first frame changes in space only.
- The first encoded chunk skips the temporal convolution.
- The next chunk's first temporal upsample does not read that frame.

- The latent the DiT sees is `(μ − mean) / std` with the published 16-channel vectors.
- The decoder inverts that affine and clamps to `[−1, 1]`.
- The four mask channels are 1 on the first latent time and 0 after that.
- A last still marks the last pixel. The first pixel is repeated four times, then the remaining pixels are grouped by four, so the last latent time is `[0, 0, 0, 1]`.
- That still is written into the last pixel of the causal chunk that contains it. It is not encoded as its own latent and pasted over the last time.
- The DiT input is 16 noise channels, the four mask channels, then the 16-channel latent.
- `split` omitted, or `split` 1, denoises the whole clip once.
- A larger `split` shortens the clip until that many chunks tile it. `overlap` is how many latent frames the chunks share. The default is 1. `0` places the chunks end to end.
- Each chunk runs every Euler step before the next chunk starts.
- The first chunk is conditioned on the still. Its sample is noise on every frame, including the conditioned one.
- A later chunk is its own full run. Overlap 0 opens it on the previous slice's last frame. A larger overlap opens it on the first shared frame. That frame is encoded with the same still encode as a fresh clip. Each run is decoded on its own.
- Overlap 0 averages the two copies of that boundary frame, then keeps the rest of the next run. A larger overlap crossfades the shared frames from the previous run to the next one.
- The noise buffer still records the latents each chunk owns. The delivered frames come from those separate decodes.
- DiT scratch is one chunk. The noise, the mask, and the condition latent of the whole clip stay allocated.
- The head's 64 outputs are one time patch, a 2×2 spatial patch, then the 16 latent channels.
- Without that file, each 4×8×8 cell is averaged into the latent and the decoder repeats each cell.

- The text context is the `umt5` directory on the row, `models/google/umt5-xxl-int8`.
- A clip encodes with that pack and reads `tokenizer.json` from it.
- `favlo --wan22_i2v umt5 quantize` writes that directory.
- That BF16 file is not opened during a clip.
- When that directory also has `weights.bin` and `planes.json`, the encoder reads the file into one shared span and binds each plane at its offset.
- The span stays for the encode and is released at the end of the encode. A view is not a separate allocation, so dropping it leaves the span in place.
- When either file is missing, each `.npz` and `.npy` is loaded and uploaded for that linear, then dropped.
- Every linear is symmetric int8.
- The token embedding, the RMSNorm weights, and the relative buckets stay float16.
- `matmul2d` consumes the linears.
- The token rows are gathered on the host from `embed`, whether that plane is `embed.npy` or a slice of the span.
- The real tokens, including the end token, are padded with zeros to 512.
- Without that pack or the tokenizer, the prompt bytes index the embedding table.
- The DiT text MLP still runs.

- The VAE uploads weights only while encode or decode runs.
- Every convolution is an im2col tile and a half `matmul2d`. The kernel width and the channel count stay the checkpoint's. A narrow layer is padded with zeros out to the matrix tile.
- The 384-channel mid-block attention packs each frame into a token-major matrix and runs `attend_flash_wide`. Three `matmul2d` chunks of 128 form one dot product. The softmax stays between that product and the value mix.
- The encoder and decoder commit once per block.
- Each DiT block commits before the next one is recorded.
- The 1×1 projections, the channel RMS, and the clipped SiLU are kernels. Both mid-block attentions are `attend_flash_wide`.
- The causal cache is a device buffer.
- A frame larger than 832×480 is encoded and decoded in spatial tiles of 512 pixels, stepping by 384. Each tile keeps its own causal cache. A linear ramp blends the 128-pixel overlap. A 320p or 480p frame stays one tile.
- The Euler step adds the velocity on device.
- The host sees the latent again at the decoder, then reads the uint8 frames for the AVI.

Kernel and bandwidth ceilings for one 9-frame 576×320 clip are in [ceilings.md](ceilings.md).

The isolated sum is a measurement in [research/clip-ceiling.md](research/clip-ceiling.md).

A 9-frame int8 smoke printed 23.11 s.

## Registry

The row key in `models/registry.json` is the packed folder, such as `Wan2.2-I2V-A14B-int8`.

Registration leaves `alias` null, so that folder is the public id.

`directory` and `parts` are copied at open, so close the resident model after changing them.

The listener stays up.

```python
from favlo.registry import load_registry, save_registry

data = load_registry()
row = data["models"]["Wan2.2-I2V-A14B-int8"]
row.setdefault("parts", {})["umt5"] = "models/google/umt5-xxl-int8"
save_registry(data)
```

| Field | When | Values |
|---|---|---|
| `parts.high` | open | The high-noise expert directory. A missing name opens `<directory>/high`. |
| `parts.low` | open | The low-noise expert directory. A missing name opens `<directory>/low`. |
| `parts.umt5` | open | The int8 umT5 pack. A missing name opens `<directory>/umt5` when that directory has `manifest.json`, and otherwise `models/google/umt5-xxl-int8`. |
| `alias` | resolve | One name, with no colon and no slash, or `null`. |
| `pipeline_default` | omitted model | `true` on one row of this pipeline. Enabling it clears the mark on the other rows of this pipeline. |
| `directory` | open | The packed record `serve` opens. |

- Registration writes `media.i2v` to this folder when that key is empty.
- `favlo media` looks up `media.video`, because the handler kind is `video`.
- The stored `i2v` key is not that lookup.
- An omitted `--model` uses `pipeline_default`, then the first folder registered for this pipeline.

- The sampling card (`generate`, temperature 0) is copied at registration.
- This pipeline does not read it.
- `kv_cache` and `resident_percent` are discarded at open.
- `mtp`, `mtp_depth`, `enable_thinking`, `reasoning_effort`, `context_length`, and `max_prompt_tokens` may sit on the row and are not read.
- `context_length` is 512 at registration, the text length from the Wan config.
- `pipeline`, `runtime`, `source`, and `pipeline_root` are the row's identity.

## Listener

- `favlo serve` must already be listening.
- `favlo media` does not start it.
- It exits when that port is closed.
- The command posts the body to the listener.
- Stdout is one `file://` URI.
- `--out` writes the pack JSON, `metadata` and `mediaUri`, and does not copy the MP4.
- `--model` is optional.
- An omitted `--model` uses the row marked `pipeline_default`, then the first folder registered for this pipeline.
- The lookup is [Registry](#registry).
- `--kind` is unnecessary.
- This pipeline serves one kind.

```text
uv run favlo media --pipeline wan22_i2v prompt="The cube slides to the right. Fixed camera." image=file:///absolute/still.png size=576x320 frames=5
uv run favlo media --pipeline wan22_i2v --model Wan2.2-I2V-A14B-int8 --out pack.json prompt="The cube slides to the right. Fixed camera." image=file:///absolute/still.png size=576x320 frames=5
uv run favlo media --pipeline wan22_i2v prompt="The cube slides to the right. Fixed camera." image=file:///absolute/still.png size=720x1280 frames=81 split=2 overlap=0 social=true
uv run favlo media --pipeline wan22_i2v prompt="The cube slides to the right. Fixed camera." image=file:///absolute/still.png size=576x320 frames=5 mp4=false
```

- `prompt`, `image` (or `input_reference`), and `size` are required.
- `image` is a `file://` URI, a base64 PNG, an image id, or a filename in `images`.
- `last_image` is optional and uses the same forms.
- `size` is one of the accepted resolutions.
- `frames` defaults to 81.
- `split` and `overlap` are optional. `overlap` requires `split`. A field named `context` is rejected.
- `social` is optional. `true` pads to 1080p, interpolates to 30 fps, and encodes H.264 High for short-form upload.
- `mp4` is optional and defaults to true. `false` stores the AVI and skips the MP4. The result URI is that AVI.
- This is the command that writes the clip.
- The pipeline command does not.

`favlo serve` publishes `wan22_i2v__generate_video` on `POST /mcp`.

`handle_video` is a `BulkMediaPipeline`.

The DiT, the checkpoint fields, and the stage list stay in this package.

The response is a video job.

A path under `/v1/videos` is 404.

- `progress` is a percent of the clip.
- Text, each expert load, the VAE encode, each DiT walk, the VAE decode, and the pack have fixed shares.
- The DiT walks take what remains.
- Inside a share the finished pieces move the percent.
- A guidance scale other than 1 adds a text pass and a DiT walk for each step that uses that scale.
- The percent stays below 100 while `status` is `in_progress`.
- It is 100 when the MP4 is stored.
- `stage` names the part in progress.
- `eta` is the remaining seconds from that percent and the time since the clip started.
- It is 0 when the job is completed.

`serve` opens one row per stage and closes it when the stage ends.

The open row is labeled with the stage: `compile`, `text`, `encode`, `load`, `high`, `low`, or `decode`.

Compile and text share one row of layers.

The expert load counts files.

Each DiT walk counts layers.

Encode and decode count blocks.

The size, frame count, and duration are logged once when the first row opens.

Each Euler step also logs its seconds.

`resources/read` of `favlo://videos/{id}.mp4` reads `~/.favlo/videos/<id>.mp4`.

That MP4 is encoded from the uncompressed `<id>.avi` beside it.

Both stay until `video_ttl_seconds` in `~/.favlo/settings.json`.

`<id>.mp4.json` beside the MP4 stores `generation.startdatetime` and `generation.enddatetime` in UTC. The start is when the job opened. The end is when the MP4 was stored. A resumed job keeps the original start.

The default is one day.

`prompt` is required.

`generate_video` passes the still as `image`:

- An id starting with `image_` from `generate_image`, resolved to PNG bytes before the handler runs
- A PNG block `{"type": "image", "data": "<base64>"}`

A filesystem path and a `favlo://` URI are refused.

The handler also accepts `input_reference`.

`size` is required and is one of those resolutions.

`frames` defaults to 81.

`seconds` snaps to the nearest `4n+1` frame count at 16 fps when `frames` is omitted.

`quality` accepts `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `standard`, and `hd`.

The job reports `low`, `medium`, `high`, `xhigh`, or `max`.

`auto`, `standard`, and `hd` are reported as `high`.

An omission is `high`.

The MP4 is `libx264` at CRF 32, 28, 23, 18, or 14 for those five reported levels.

The MP4 stored for the job uses that reported level.

`resources/read` returns that file.

There is no later call that re-encodes another level.

`social` defaults to false. When true, the pack pads the AVI into the matching 1080p canvas (1080×1920, 1920×1080, or 1080×1080), motion-interpolates to 30 fps (`minterpolate` mci/aobmc), and encodes H.264 High `yuv420p` at about 8 Mbps for `high` (4/6/8/10/12 Mbps across the five reported levels), closed GOP, `+faststart`. That path is for TikTok, Instagram Reels, Facebook Reels, and YouTube Shorts. The AVI size and rate stay the CRF path when `social` is omitted or false.

`mp4` defaults to true. `false` stores the AVI and does not encode. The result URI is that AVI. `social` with `mp4` false is rejected.

- `seed` defaults to 0.
- `negative_prompt` defaults to empty.
- That is Wan's published sample negative.
- It is encoded only when a guidance scale is not 1.
- `guidance_scale` is one number or a low, high pair.
- The default is 1.
- `steps` defaults to 4 and accepts 1 to 40.
- A chat completion on this row is rejected.
- The message still names `POST /v1/videos`.

A chat that arrives while this clip holds the GPU sets `state["preempt"]`.

Cancel of the running task sets `state["cancel"]`.

Both are `threading.Event`s.

`generate` finishes the Metal dispatch already submitted, then stops.

The boundaries are after compile, between text layers, after the text context, between VAE blocks, after the VAE encode, between expert weight files, between DiT layers, after each Euler step, between decoder blocks, after the VAE decode, and after the AVI is packed.

- A stop during the expert load keeps that step index. The resume reloads the expert and continues the same step.
- A stop between DiT layers keeps that step index and runs the step again.
- A stop between text layers or VAE blocks keeps the stage before that pass and runs the pass again.

The device is closed, and that sweep is confirmed, before the waiting chat loads its model.

The text table, the negative table when it was encoded, the noise, the condition latent, the mask, the decoded frames, and the AVI bytes stay on `state["video_hold"]` with the same job id.

The listener runs the waiting chats and calls `handle_video` again with the same body.

That call keeps the id and skips a stage whose host outputs are kept.

Compile and the resident expert are opened again.

Layer counts already finished are not added a second time.

Cancel leaves the job `in_progress` and does not keep the hold.

The listener deletes a stored AVI and MP4 for that id.

## Smoke

`uv run pytest tests/test_wan22_i2v_smoke.py -s` dispatches the fp4, fp8, and int8 matmul tiles.

`uv run favlo --wan22_i2v kernels` is the same check.

`WAN22_SMOKE=1 uv run pytest tests/test_wan22_i2v_smoke.py -s` also generates 9 frames at 576×320 from `models/Wan2.2-I2V-A14B-int8`.

The step lines are the seconds printed by `generate`.

ffmpeg writes `artifacts/wan22-i2v-smoke.mp4`.

`python -m harness.video_smoke_agent --pipeline wan22_i2v --tool generate_video` uses the registry default chat model.

The still is `tests/assets/favlo-horse.png`.

The pipeline resizes it to the clip size.

The clip sentence is `video_prompt`: the motion and a fixed camera.

It asks for the 9-frame 320p clip.

The published `wan22_i2v__generate_video` tool reads into `artifacts/video-smoke-agent.mp4`.

The same model then reads frame 0, the middle frame, and the last frame of that file and must affirm `prompt` in `smoke.json`.

`python -m harness.video_quality_agent --pipeline wan22_i2v --tool generate_video` uses that same still, resized to 416×240.

It calls that tool twice, at `quality` `low` and then `max`, with no chat between those tasks.

Those clips are `artifacts/video-quality-low.mp4` and `artifacts/video-quality-max.mp4`.

Exit 0 means both files are 5 frames at 416×240 and the max file is larger.

`python -m harness.media_pipeline` is the preemption harness for every served media pipeline.

`python -m harness.media_pipeline i2v` runs this pipeline's case.

That case uses `tests/assets/favlo-horse.png`.

It calls `wan22_i2v__generate_video` for 5 frames at 576×320, then a chat completion preempts that task.

The file is `artifacts/media-preempt-i2v.mp4`.

Those commands are not part of pytest.

## Citations

Wan, Team, et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314.

Apple. *Metal tensor ops and `matmul2d`.* WWDC26 session 330.

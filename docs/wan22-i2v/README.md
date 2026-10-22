# Wan 2.2 I2V

Four-step image-to-video pipeline for [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B). The high-noise expert runs while the timestep is at least 900. The low-noise expert runs after that. One expert is on device at a time. `matmul2d` consumes fp4 E2M1 or fp8 E4M3 weights with an E8M0 scale. The left operand is float16 and the destination is float32. A row whose peak exceeds 60000 is divided by that peak over 60000 before the half store, and the product is multiplied back. Biases, QK RMSNorm, `norm3`, modulation, and the patch convolution stay float16.

`setup` downloads the two expert safetensors, the VAE, the umT5 encoder, and the umT5 tokenizer into `models/Wan2.2-I2V-A14B`. Sixteen connections write one file. A short file continues from the ranges already on disk. `quantize` reads one local safetensors file for each expert and writes `models/Wan2.2-I2V-A14B-fp4`. It scores every block by the maximum fp4 relative error of its eight attention matrices and its two feed-forward matrices, after one K permutation per matrix. Twenty percent of the blocks, eight of forty, are stored as fp8. An equal score keeps the lower block index. Text, time, and the head stay fp4. `kernels` dispatches a 64 by 32 fp4 matmul and the same tile as fp8. `generate` writes an uncompressed 24-bit AVI. Rows are bottom-up BGR.

```text
uv run favlo --wan22-i2v setup
uv run favlo --wan22-i2v quantize --high models/Wan2.2-I2V-A14B/high.safetensors --low models/Wan2.2-I2V-A14B/low.safetensors
uv run favlo --wan22-i2v kernels
uv run favlo --wan22-i2v generate --record models/Wan2.2-I2V-A14B-fp4 --image still.png --prompt "The horse gallops at speed, its raised foreleg driving down and the trailing legs swinging forward into the next stride, mane and tail whipping. Fixed camera." --frames 5 --width 576 --height 320 --out clip.avi
```

The solver is four Euler steps with shift 5. The knots are 1000, 937.5, 833.3333, 625, and 0. The update adds `(sigma_next - sigma)` times the velocity. Frame count is `4n+1`. The shortest moving clip is 5 frames. At 16 fps, 81 frames is the 5 second clip. 320p is 576×320. Both edges are multiples of 16.

The image is the first pixel frame. The other frames are zeros. `Wan2.1_VAE.pth` is the causal encoder and decoder: base width 96, multipliers 1, 2, 4, 4, two residual blocks, temporal downsample on the last two stages, latent width 16. Spatial stride is 8 and temporal stride is 4. The first frame changes in space only. The first encoded chunk skips the temporal convolution, and the next chunk's first temporal upsample does not read that frame. The latent the DiT sees is `(μ − mean) / std` with the published 16-channel vectors. The decoder inverts that affine and clamps to `[−1, 1]`. The four mask channels are 1 on the first latent time and 0 after that. The DiT input is 16 noise channels, the four mask channels, then the 16-channel latent. The head's 64 outputs are one time patch, a 2×2 spatial patch, then the 16 latent channels. Without that file, each 4×8×8 cell is averaged into the latent and the decoder repeats each cell. The text context is the umT5 encoder in `models_t5_umt5-xxl-enc-bf16.pth`, tokenized with the umT5 vocabulary. The real tokens, including the end token, are padded with zeros to 512. Without those files the context is zeros. The DiT text MLP still runs.

`favlo serve` answers `POST /v1/videos` when the row defines `handle_video`. The response is a video job. `GET /v1/videos/{id}/content` is the MP4. `prompt` is required. The still is `input_reference` or a base64 PNG `image`. `size` defaults to `576x320`. `frames` defaults to 81. `seconds` snaps to the nearest `4n+1` frame count at 16 fps when `frames` is omitted. `seed` defaults to 0. A chat completion on this row returns an error that names `POST /v1/videos`.

## Smoke

`WAN22_SMOKE=1 uv run pytest tests/test_wan22_i2v_smoke.py -s` dispatches both matmul tiles, then generates 9 frames at 576×320 from `models/Wan2.2-I2V-A14B-fp4`. The step lines are the seconds printed by `generate`. ffmpeg writes `artifacts/wan22-i2v-smoke.mp4`. `python -m harness.video_agent` holds the machine lock and uses `qwen36-35b-a3b:instruct`. It draws the still with `Z-Image-Turbo-fp4` when `artifacts/video-agent.png` is missing or is not the clip size. The still sentence is the `prompt` in `smoke.json`, at that file's width and height. The clip sentence is `video_prompt`: the motion and a fixed camera. The still already defines the horse. It asks for the 9-frame 320p clip. The content route writes `artifacts/video-agent.mp4`. The same model then reads frame 0 of that file and must affirm the still sentence. That command is not part of pytest.

## Citations

Wan, Team, et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314.

Apple. *Metal tensor ops and `matmul2d`.* WWDC26 session 330.

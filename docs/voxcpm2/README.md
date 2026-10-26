# VoxCPM2

As-built note for the `voxcpm2` pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md). This directory wins for this pipeline's records and kernels. Another pipeline does not edit it.

The model card is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). The report is [arXiv:2606.06928](https://arxiv.org/html/2606.06928). The VoxCPM formulation is Zhou et al., [arXiv:2509.24650](https://arxiv.org/abs/2509.24650). The text-semantic stack is MiniCPM-4, [arXiv:2506.07900](https://arxiv.org/abs/2506.07900). Papers for the kernels are [references.md](references.md).

`setup` downloads the checkout into `models/openbmb/VoxCPM2`. The weight file is `model.safetensors`, 4,580,080,592 bytes, bfloat16. The AudioVAE file is `audiovae.pth`, 376,951,122 bytes. A file that already has that size is left in place. `setup` does not register a row. `handle_chat` refuses a chat completion.

```text
uv run favlo --voxcpm2 setup
uv run favlo --voxcpm2 inspect
uv run favlo --voxcpm2 kernels
uv run favlo --voxcpm2 speak --text "Hello." --out clip.wav
```

`inspect` reads the safetensors header and prints the dtype counts. `kernels` compiles `voxcpm2/shaders/kernels.metal` at Metal 4.1 and checks RMSNorm, the half matmul, rotate-half RoPE, grouped-query attention, and the causal convolutions against the host formulas.

`speak` writes one 48 kHz WAV. The checkpoint bfloat16 linears are uploaded as `half`. `matmul2d` accumulates into float. An empty `voice` is basic TTS: the line is tokenized, the audio-start id is appended, and no reference clip is encoded. A named `voice` is refused. A style string in `instructions` is wrapped in parentheses and placed in front of `input` with no added space. An empty string leaves the line unchanged.

The checkpoint sets `use_mup` false, so a token embedding is unscaled and a residual add is unscaled. `scalar_quantization_scale` is 9 and `scalar_quantization_latent_dim` is 512. The bound is `round(tanh(y) * 9) / 9` after the input projection. Audio positions pass through that bound. Text positions keep the raw hidden state.

Each patch is 4 latent frames. The diffusion transformer takes 10 Euler steps, guidance 2, sway sampling, and CFG-Zero*. The first of those steps has zero velocity. The stop head can end the line after the minimum length. The AudioVAE decoder is float32, weight-normalized, and conditioned on the 48 kHz bin. The opening refuses the line when the weight files plus scratch do not fit.

`models add <folder>` writes `manifest.json` with `complete` true when that directory holds `model.safetensors` and `config.json`, then registers the folder. The served record is `models/VoxCPM2`, a link to the checkout. `setup` still does not register.

`handle_voice` is a `MediaPipeline`. A chat that arrives between patches pauses the job. The same job speaks the line again after the chat releases the GPU. Patches drawn before the pause are not kept. `generate_voice` on `POST /mcp` speaks `input` with an empty `voice`. `instructions` is the style prefix. The row does not list speakers or emotions. `python -m harness.voice_smoke_agent` calls that tool for the model, line, and instructions in [smoke.json](smoke.json). The file is `artifacts/voice-smoke-agent.wav`, 48 kHz mono, with RMS above the silence floor and a duration inside `max_seconds`. That run is not part of `pytest`.

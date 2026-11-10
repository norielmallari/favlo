# VoxCPM2

As-built note for the `voxcpm2` pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md). This directory wins for this pipeline's records and kernels. Another pipeline does not edit it.

The model card is [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). The report is [arXiv:2606.06928](https://arxiv.org/html/2606.06928). The VoxCPM formulation is Zhou et al., [arXiv:2509.24650](https://arxiv.org/abs/2509.24650). The text-semantic stack is MiniCPM-4, [arXiv:2506.07900](https://arxiv.org/abs/2506.07900). Papers for the kernels are [references.md](references.md).

`setup` downloads the checkout into `models/openbmb/VoxCPM2`. The weight file is `model.safetensors`, 4,580,080,592 bytes, bfloat16. The AudioVAE file is `audiovae.pth`, 376,951,122 bytes. A file that already has that size is left in place. `setup` does not register a row. `handle_chat` refuses a chat completion.

```text
uv run favlo --voxcpm2 setup
uv run favlo --voxcpm2 inspect
uv run favlo --voxcpm2 kernels
uv run favlo --voxcpm2 speak --text "Hello." --persona "Clear adult voice" --out clip.wav
```

`inspect` reads the safetensors header and prints the dtype counts. `kernels` compiles `voxcpm2/shaders/kernels.metal` at Metal 4.1 and checks RMSNorm, the half matmul, rotate-half RoPE, grouped-query attention, the causal convolutions, and the float bias, SiLU, SwiGLU, residual, and row copy against the host formulas. One stack step is checked as a single command buffer.

`speak` writes one 48 kHz WAV. The checkpoint bfloat16 linears are uploaded as `half`. `matmul2d` accumulates into float. A language-model step runs as one command buffer. One diffusion estimate, including its conditioning projections and the diffusion stack, runs as one command buffer. Values between those kernels stay float32, and the matmul casts its activation to half. The key and value cache stays float32. The stop head reads two logits on the host. The AudioVAE and the token embedding stay on their current path. An empty `voice` is basic TTS: the line is tokenized the way published inference calls `tokenize`, with no BOS id, the audio-start id is appended, and no reference clip is encoded. A named `voice` is refused. `persona` is required. Parentheses and sentence punctuation are removed so the phrase is not its own utterance. The remaining words are wrapped in one pair of parentheses and placed in front of `input`. A space is inserted when `input` starts with an ASCII letter or digit, so that word keeps the token it has on its own. The model receives that one string, and its first token is the opening parenthesis.

The checkpoint sets `use_mup` false, so a token embedding is unscaled and a residual add is unscaled. `scalar_quantization_scale` is 9 and `scalar_quantization_latent_dim` is 512. The bound is `round(tanh(y) * 9) / 9` after the input projection. Audio positions pass through that bound. Text positions keep the raw hidden state.

Each patch is 4 latent frames, 160 ms. The diffusion transformer takes 10 Euler steps, guidance 2, sway sampling, and CFG-Zero*. The first of those steps has zero velocity. The stop head ends the line after the minimum length of 2 patches. The patch ceiling is 4096, then `text tokens × 6 + 10`, then the positions left in `max_length` after the text and the audio-start id. A sample row counts each diffusion step and the decoder. Its detail names the patch and the Euler step before that step's estimates run, and the row advances when those command buffers have finished. Each finished patch logs its seconds. The job records `progress`, `stage`, and `eta` from that count. The AudioVAE decoder is float32, weight-normalized, and conditioned on the 48 kHz bin. `MediaPipeline` calls `ensure_open` before the line and always calls `close` when the request returns. `ensure_open` runs the shared open gate against the weight files plus scratch, and reloads when closed. This package does not keep a private memory cap.

`models add <folder>` writes `manifest.json` with `complete` true when that directory holds `model.safetensors` and `config.json`, then registers the folder. The served record is `models/VoxCPM2`, a link to the checkout. `setup` still does not register.

## Listener

`favlo serve` must already be listening. `favlo media` does not start it, and it exits when that port is closed. The command posts the body to the listener. Stdout is one `file://` URI. `--out` writes the pack JSON, `metadata` and `mediaUri`, and does not copy the audio file. `--model` is optional. An omitted `--model` uses the voice default in the registry when that folder is this pipeline. Otherwise it uses the first folder registered for this pipeline. `--kind` is unnecessary. This pipeline serves one kind.

```text
uv run favlo media --pipeline voxcpm2 input="Hello." persona="Clear adult voice"
uv run favlo media --pipeline voxcpm2 --model VoxCPM2 --out pack.json input="Hello." persona="Clear adult voice"
```

`input` and `persona` are required. `voice` must be empty or omitted. `response_format` defaults to `mp3`. Offline `speak` writes a WAV without the listener. `media` is the command that goes through serve.

`handle_voice` is a `MediaPipeline`. A chat that arrives between patches pauses the job. The same job speaks the line again after the chat releases the GPU. Patches drawn before the pause are not kept. `voxcpm2__generate_voice` on `POST /mcp` speaks `input` with the required `persona` and an empty `voice`. The row does not list speakers or emotions. `python -m harness.voice_smoke_agent --pipeline voxcpm2 --tool generate_voice` calls that tool for the model, line, and persona in [smoke.json](smoke.json). The file is `artifacts/voice-smoke-agent.wav`, 48 kHz mono, with RMS above the silence floor and a duration inside `max_seconds`. That run is not part of `pytest`.

# Harness

As-built clients that score a running `favlo serve`. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). A pipeline's scores live in that pipeline's `benchmarks.md`.

## Shared rules

### Listener session

```bash
python -m harness.server status
python -m harness.server run -- <command>
```

`status` prints `listening` or `down` for `127.0.0.1:8129`. `run` opens the listener session. A process already listening there is reused and left running. A closed port starts the daemon, and the session stops that process when the command exits. Media harnesses call that same session and do not bind a second port.

### Machine lock

```bash
python -m harness --model <id>
python -m harness.coding --model <id>
python -m harness.worktree --model <id>
```

These take the machine lock with `python -m favlo.smoke_lock`. The recorded pid is the shell that invoked it. A lock that shell already holds stays in place. They restart the process they started so a later turn reloads a prefix. They require port 8129 to be free. They do not attach to a listener someone else left running.

A media harness does not take that lock. It calls the listener on `127.0.0.1:8129` (directly or through `harness.server`).

### Default chat model

Agent harnesses that call `POST /v1/chat/completions` take their chat model from the top-level `"default"` field in `models/registry.json` when `--model` or `--judge` is omitted. The first registered language model sets that field to its instruct variant when it is still unset. See [guides/models.md](guides/models.md).

## Listener wrapper

Wraps the shared listener on port 8129. Other harness modules use the same attach-or-start behavior when they call into `harness.server`.

```bash
python -m harness.server status
python -m harness.server run -- <command>
```

## Language and chat

### Prefix reload rate

```bash
python -m harness --model <id>
```

Starts `serve`, reads repository files one turn at a time until the prompt reaches the token target (default 65536), and prints the server's prefill and decode rates. The model field is the pipeline name. The request sets `enable_thinking` false and leaves every other sampling field unset, so the listener fills the instruct card. The server stops after the first turn and starts another, so the second turn loads the prefix from disk. Later turns stay on that server. Takes the machine lock. Not part of `pytest`.

### File agent

```bash
python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace <dir>
```

File tools driven by the chat endpoint. The agent creates a note, replaces its text, and confirms the file. `pytest` includes this check: it starts `serve` and checks that create, edit, and confirm land in the workspace. The command above is the same client against an already running server. Not a prefix-rate run (`python -m harness`).

### Thinking persona coherence

```bash
python -m harness.coherence [--model <id>] [--port <port>]
```

Starts `serve` on a free port (or `--port`), runs two thinking-mode turns against a long persona system prompt (~24k characters), and judges checkable facts, reasoning-before-reply ordering, and server log errors. Sampling follows the Qwen3.8-Flash-Next thinking card when used with that family. Optional `--model` overrides the registry default. Stops the daemon when finished. Not part of `pytest`. `tests/test_coherence.py` checks persona and judge helpers without a listener.

### Coding ledger

```bash
python -m harness.coding --model <id>
```

Starts `serve` and sets `enable_thinking` true on the bare model id. The model writes `ledger.py` through tools. A pytest file outside the workspace is the result. Exit 0 is a pass. Prints each request's prefill and decode line. Takes the machine lock. Not part of `pytest`.

### Worktree summarization

```bash
python -m harness.worktree --model <id>
```

Starts `serve`, checks out a detached worktree of `main` under `artifacts/`, and drives tool writes to `notes/span.py`, `notes/check.py`, and `notes/README.md`. Write turns use thinking; read turns do not until a 131072-token prompt triggers summarization. A grader outside the worktree validates the files. Exit 0 requires the grader, a prompt of at least 131072 tokens, and a later prompt shorter than that peak. Prints each request's prefill and decode line. Takes the machine lock. Not part of `pytest`.

### Public benchmarks

```bash
python -m harness.benchmarks --model <id> [--limit 30|100]
```

Scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field is that id. Thinking stays off. Omitted sampling fields are filled from `GET /v1/models` row `<id>:instruct`. A CLI flag overrides that field. The checker reads `message.content`. `--limit 30` scores the first 30 rows and writes the smoke section. `--limit 100` writes the sample of 100. A run with no limit scores the smoke, then the full sets. Not part of `pytest`. Each finished suite is written into `docs/<pipeline>/benchmarks.md`.

### Language pipeline agent

```bash
python -m harness.language_pipeline
```

One Microsoft Agent Framework agent on the registry default chat model calls `check_embeddings`. That tool posts the six sentences from `harness.embeddings_agent` to `POST /v1/embeddings` for `bge-m3-int8`. Exit 0 means the tool ran and every paraphrase cosine is higher than every cross-topic cosine. Uses the listener session. Does not take the machine lock. Not part of `pytest`. `tests/test_language_pipeline.py` checks the disconnect shell and the embedding reply without a listener.

### Embeddings cosine check

```bash
python -m harness.embeddings_agent
```

Posts six sentences to `POST /v1/embeddings` for `bge-m3-int8`. Each response is one unit vector of length 1024 with a token count. Two pairs are paraphrases; the other sentences are different topics. Exit 0 means every paraphrase cosine is higher than every cross-topic cosine. Prints each score. Uses the listener session. Does not take the machine lock. Not part of `pytest`. `tests/test_embeddings_agent.py` checks the response shape and ordering without a listener.

### MCP memory agent

```bash
python -m harness.memory_agent
```

A Microsoft Agent Framework agent on the registry default chat model calls `remember`, `recall_memories`, and `forget_memory`, which forward to `remember`, `recall`, and `forget` on `POST /mcp`. When this session starts the daemon, it sets `FAVLO_MEMORY_HOME` to a temporary directory so the run does not touch `~/.favlo/memory.db`. A reused listener uses that process's database; each sentence is tagged with a run id and removed before exit. Exit 0 means a duplicate remember returns the same id, a recall ranks the aurora sentence above the wheat sentence, and forget drops that row from a later recall. Embeddings use `bge-m3-int8`. Uses the listener session. Not part of `pytest`. `tests/test_memory_agent.py` checks result parsing without a listener.

## Image

### MCP image read

```bash
python -m harness.image --model <id>
```

Calls `generate_image` on `POST /mcp`, then reads the PNG with `resources/read`. The prompt, size, and seed come from that pipeline's `smoke.json`. Uses the listener session. Exit 0 means those bytes were read. Does not score the picture. Not part of `pytest`.

### Image smoke agent

```bash
python -m harness.image_smoke_agent
```

A Microsoft Agent Framework agent on the registry default chat model calls `generate_image` twice for `Z-Image-Turbo-fp4` with the prompt, size, and seed in `docs/z-image-turbo/smoke.json`, at `quality` `low` and then `max`. Each PNG is read with `resources/read` and written to `artifacts/image-smoke-agent-low.png` or `artifacts/image-smoke-agent-max.png`. The next turns send each PNG to the same model and ask whether the picture matches the prompt. Exit 0 means both replies affirm it and the max file is larger. Uses the listener session. Not part of `pytest`.

### Image attachment

```bash
python -m harness.image_attachment
```

Posts one chat completion to `qwen38-27b` with `artifacts/favlo-horse.png`. The user content is a text part, then an `image_url` part whose URL is `data:image/png;base64,` and whose `detail` is `auto`. Thinking is off and temperature is 0. The question does not name the animal. Exit 0 means `prompt_tokens` is at least the merged patch count and the reply names a horse plus two marks that are only in the picture: blue, desk, blaze, bandana, chestnut, mane, or hoof. A refusal, including a problem with analysis, fails. Uses the listener session. Does not take the machine lock. Not part of `pytest`. `tests/test_image_attachment.py` checks the request and the reply rule without a listener.

### Image persona

```bash
python -m harness.image_persona
```

The agent instructions are the terminal persona. The user asks that horse to draw a picture of himself. The image prompt is a cute 2.5D horse with a white blaze and large blue eyes, sitting at a desk. The image call is `Z-Image-Turbo-fp4` at the pipeline's maximum edge, seed 0. Prints the PNG byte count. The next turn asks whether the picture shows that stylized horse rather than a photograph. Exit 0 means the reply affirms it. The PNG is `artifacts/favlo-horse.png`. Uses the listener session. Not part of `pytest`.

## Video

### I2V smoke agent

```bash
python -m harness.video_smoke_agent
```

Draws `artifacts/video-smoke-agent.png` with `Z-Image-Turbo-fp4` when that file is missing or is not the clip size. The still sentence is `prompt` in `docs/wan22-i2v/smoke.json`. The width, height, and seed are the clip's. The next turn calls `generate_video_i2v` for the model in that file, using `video_prompt`, the smoke frame count, and the still as a PNG image block. It reads `favlo://videos/{id}.mp4` into `artifacts/video-smoke-agent.mp4`. The same model reads frame 0, the middle frame, and the last frame and must affirm the still sentence. Exit 0 means each frame was affirmed and the file has that geometry. Uses the listener session. Not part of `pytest`.

### TI2V smoke agent

```bash
python -m harness.video_ti2v_agent
```

Draws `artifacts/video-ti2v-agent.png` from `prompt` in `docs/wan22-ti2v-5b/smoke.json` at `i2v_width` and `i2v_height`. Then calls `generate_video_t2v` with no still at `width` and `height`, then `generate_video_i2v` with the PNG at the still size. Outputs are `artifacts/video-ti2v-t2v.mp4` and `artifacts/video-ti2v-agent.mp4`. Vision checks frame 0, the middle frame, and the last frame of the still clip against the prompt. Exit 0 means both files match smoke sizes and frame count and each frame was affirmed. Uses the listener session. Not part of `pytest`.

### Video quality comparison

```bash
python -m harness.video_quality_agent
```

Draws `artifacts/video-quality.png` at 416×240 with `generate_image` when that file is missing or is not that size. The still sentence is the `prompt` in `docs/wan22-i2v/smoke.json`. Calls `generate_video_i2v` twice at `quality` `low` and then `max`, with no chat between tasks, and reads each MP4 into `artifacts/video-quality-low.mp4` and `artifacts/video-quality-max.mp4`. Exit 0 means both files are 5 frames at 416×240 and the max file is larger. Uses the listener session. Not part of `pytest`.

## Music

### Music smoke agent

```bash
python -m harness.music_smoke_agent
```

Calls `generate_music` for the model, prompt, lyrics, seconds, seed, bpm, keyscale, and timesignature in `docs/ace-step-15-turbo/smoke.json`. Reads `favlo://music/{id}.wav` into `artifacts/music-smoke-agent.wav`. `python -m harness.inspect_wav` writes the report and `artifacts/music-smoke-agent.inspect.png`. The report adds key, pitch, harmony, quiet slices, and tempo; optional flags compare to the request. Exit 0 means template seconds, key, and bpm match the smoke request and the file is 10 seconds, 48 kHz, stereo, with RMS above the silence floor. Uses the listener session. Not part of `pytest`.

### Music long agent

```bash
python -m harness.music_long_agent
```

Calls `generate_music` for the 3 minute request in `docs/ace-step-15-turbo/long.json` and reads the WAV into `artifacts/music-long-agent.wav`. The inspect report is checked like the 10 second smoke. Exit 0 also requires spectral flatness at or below 0.35, quiet slices at or below 15% of the half-second grid, a key margin of at least 0.02, and a one-second join no more than 15 times the median step beside it. Prints `seam_ratio`. Uses the listener session. Not part of `pytest`. `tests/test_music_long_agent.py` checks those gates without a listener.

## Voice

### Voice smoke agent

```bash
python -m harness.voice_smoke_agent
```

Calls `generate_voice` for the model, line, and instructions in `docs/voxcpm2/smoke.json`. Reads `favlo://voice/{id}.wav` into `artifacts/voice-smoke-agent.wav`. Exit 0 means mono at the note's sample rate, RMS above the silence floor, and duration inside `max_seconds`. Uses the listener session. Not part of `pytest`. `tests/test_voice_smoke_agent.py` checks the same format rule without a listener.

## Media preemption

### Media pipeline preemption

```bash
python -m harness.media_pipeline
python -m harness.media_pipeline i2v
```

One Microsoft Agent Framework agent preempts served media pipelines: image (`Z-Image-Turbo-fp4`), I2V (5 frames at 576×320), T2V (5 frames at 1280×704), music (10 second smoke), and voice (voxcpm2 smoke). Names after the command run that subset, in that order. An omitted list runs every one. Each case starts the MCP task; the next turn is a chat completion that preempts it. The tool polls `media_status` and `tasks/get` and reads the file into `artifacts/media-preempt-*`. Exit 0 means every selected task was paused with `task is ongoing` and the same task finished at the expected shape. Uses the listener session. Not part of `pytest`. `tests/test_media_pipeline_harness.py` checks the pause-then-complete rule and the name filter without a listener.

## Helpers

These modules support harnesses; they are not full end-to-end scores on their own.

### WAV inspect report

```bash
python -m harness.inspect_wav <path.wav> [options]
```

Turns a WAV into a spectrogram PNG and a deterministic text report for agents and music harnesses. See the inspect-sound skill for interpretation.

### Voice fundamental

```bash
python -m harness.inspect_voice <path.wav> [options]
```

Measures the speech fundamental of a WAV and optional gap to a reference clip. Used for voice diagnosis, not music key or tempo.

### MCP media client

Import `harness.mcp_media` (no `python -m` entry). Shared MCP client helpers (`start_task`, `task_view`, `read_link`, and related) used by media agent harnesses.

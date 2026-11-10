# Harness

As-built clients that score a running `favlo serve`. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). A pipeline's scores live in that pipeline's `benchmarks.md`.

## Shared rules

### Listener session

```bash
python -m harness.server status
python -m harness.server run -- <command>
```

`status` prints `listening` or `down` for `127.0.0.1:8129`. `run` opens the listener session. A process already listening there is reused and left running. A closed port starts `favlo serve`. When the session stops that process, it sends SIGTERM. The daemon forwards that signal to each pipeline process. Media harnesses call that same session and do not bind a second port.

### Machine lock

```bash
python -m harness --model <id>
python -m harness.coding --model <id>
python -m harness.synthetic --model <id>
python -m harness.worktree --model <id>
```

These reuse `127.0.0.1:8129` through `python -m favlo.smoke_lock attach`. That command waits until no live pid holds the machine lock, then uses the listener already on that port. The followed log is `~/.favlo/logs/favlo.log`. A closed port starts `favlo serve`, and that process is stopped when the command exits. A listener already running is left running. When `--pipeline` is the resident model and that pipeline's sources are newer than its worker, `attach` closes the model before the command. The listener does the same check on the next request for that pipeline. A harness that checks a prefix reload unloads the resident model after the first turn, on that same listener, so the next turn loads the prefix from disk. Media harnesses do not take the lock. `harness.server` is the same attach path with no pipeline name.

A media harness does not take that lock. It calls the listener on `127.0.0.1:8129` (directly or through `harness.server`).

### Default chat model

Agent harnesses that call `POST /v1/chat/completions` take their chat model from the top-level `"default"` field in `models/registry.json` when `--model` or `--judge` is omitted. The first registered language model sets that field to its instruct variant when it is still unset. See [guides/models.md](guides/models.md).

An Agent Framework harness opens the chat client in `src/favlo/chat_client.py`. That client sends `reasoning_content` on the next request. A harness does not supply its own response parser or message preparer.

## Listener wrapper

Wraps the shared listener on port 8129. Other harness modules use the same attach-or-start behavior when they call into `harness.server`.

```bash
python -m harness.server status
python -m harness.server run -- <command>
```

## Language and chat

### Prefix reload rate

```bash
python -m harness --model <id> [--mtp on|off] [--mtp-depth <n>]
```

Reuses the listener on `127.0.0.1:8129`, then a Microsoft Agent Framework agent reads `src/favlo` files one turn at a time until the prompt reaches the token target (default 65536). The corpus is the shared package only, so the ladder does not depend on any pipeline tree. The client opens the `<id>:instruct` card and sets `reasoning.effort` to `none` (thinking off). `--mtp on` sends `favlo:mtp` true; `--mtp off` sends false. Omitted, the request leaves `mtp` unset and the registry row decides. `--mtp-depth` sends `favlo:mtp_depth` when `--mtp` is set; omitted depth is 3. Each model call streams its reply and tool calls. The command also prints the server's prefill and decode rates. After the first turn the harness unloads the resident model, so the second turn loads the prefix from disk. Later turns stay on that listener. Does not take the machine lock. Not part of `pytest`.

### File agent

```bash
python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace <dir>
```

File tools driven by the chat endpoint. The agent creates a note, replaces its text, and confirms the file. `pytest` includes this check: it starts `serve` and checks that create, edit, and confirm land in the workspace. The command above is the same client against an already running server. Not a prefix-rate run (`python -m harness`).

### Chat scenarios

```bash
python -m harness.chat_agent [--model <id>] [--scenario <name>]
```

Reuses the listener on `127.0.0.1:8129` and does not take the machine lock. Each scenario is a short conversation through the shared chat client. The scenario keeps one Agent Framework session, so the next request is the messages that session stored. The request does not send `max_tokens`. `thinking-follow-up` asks two questions with thinking on. The second request must extend the first request's stored prefix, and both turns must return a reply. `--scenario` runs one named conversation. The default runs every scenario. Exit 0 means every selected scenario held. Not part of `pytest`. `tests/test_chat_agent.py` checks the prefix rule without a listener.

### Thinking persona coherence

```bash
python -m harness.coherence [--model <id>] [--port <port>]
```

Reuses the listener on `127.0.0.1:8129`, runs two thinking-mode turns against a long persona system prompt (~24k characters), and judges checkable facts, reasoning-before-reply ordering, and server log errors. Sampling follows the thinking card of the model named by `--model`. Optional `--model` overrides the registry default. Stops the daemon only when this command started it. Not part of `pytest`. `tests/test_coherence.py` checks persona and judge helpers without a listener.

### Coding ledger

```bash
python -m harness.coding --model <id> [--mtp on|off] [--mtp-depth <n>] [--vision] [--thinking on|off] [--store on|off]
```

Reuses the listener on `127.0.0.1:8129`. This command is the production-readiness run for one language pipeline. A Microsoft Agent Framework agent on the bare model id speaks `POST /v1/responses`. `--thinking on` is the default and sets `reasoning.effort` to `medium`. `--thinking off` sets `reasoning.effort` to `none`. Sampling stays on the selected model id. A boolean, integer, or number sent as text is accepted before the schema type check. A write, an edit, or a shell start or input that omits `idempotency_key` still runs. The harness assigns a key from the tool name and the other arguments, and the same arguments return the recorded outcome. A call that sends a key keeps that key. A schema rejection names the missing field. Sampling fields stay unset, so the thinking card fills them. The framework runs the tools and sends each result back. Each model call prints its `reasoning` text and its `reply` text as those tokens arrive, then its `tool_calls` when that call ends. An empty reply still prints its header. After the first model turn the harness unloads the resident model on that listener, so the next turn loads the stored prefix from disk. Later turns stay on that listener and must extend that prefix in memory. Exit 1 when no later request loads the prefix from disk, or when a request misses or hits the wrong length. Prefix, vision, and the printed rates read only this model's lines, from its `request open` until the next `request open`, so another pipeline on the same listener does not count. A log that rolls is followed from the new bytes only, so retained lines from before the run are not part of that check. A later request with a shorter prompt is another client on this model and stays out of the prefix check, the vision check, and the printed rates. The chain runs from that still's tower through the request that logs the still twice, as `vision cached` for the copy already in the prefix or `vision cache` for a copy past the hit. The banner prints the loaded key/value plane. `--mtp off` is the default and sends `favlo:mtp` false. `--mtp on` sends `favlo:mtp` true. `--mtp-depth` sends `favlo:mtp_depth`. Omitted, that length is 3. `--vision` is off unless set. When it is set, the first chat is `tests/assets/favlo-horse.png` as an `input_image` with `detail` `auto` and `tool_choice` `none`. That turn holds when the reply names the horse. The ledger prompt is the next chat on that same session, so the still stays in the prefix. Before that chat, the harness clears `~/.favlo/vision-cache/<pipeline>/` so the first tower logs `vision patches` even when a prior run left a disk hit. The harness unloads the resident model after that clear, so an in-memory vision row is not the first tower. The server log must show the vision tower once (`vision patches`) and a later request that reuses it (`vision cache` or `vision cached`) with no second tower run. After the ledger agent returns, the same session sends that still again. That request must extend the prefix, and it reuses the tower. A second tower run on that transcript is a failure. The workspace begins with `notes/brief.txt`. The user message asks for one step at a time: find the text files under notes, search and read `notes/brief.txt`, create `ledger.py`, replace the reversed return, and run a command until its output contains 6. It does not name a tool. The instructions are `You are a coding assistant`, then to carry out the next step and leave the reply empty. The model uses those calls to write `ledger.py`. A pytest file outside the workspace is the result. `--store off` is the default. `--store on` sends `store` true. That one value is set at the start and kept for the ledger and every contract check. `store` false makes the client replay the transcript. `store` true sends `previous_response_id`, and the listener chain rebuilds one transcript before the pipeline encodes. After the ledger, the same listener checks the chat contract at that same `store` value: a system instruction the user text does not repeat, with thinking off, `finish_reason` `stop`, usage of `prompt_tokens` plus `completion_tokens`, and no think or tool markup in the reply; a two-turn follow-up that repeats the secret word; tools listed with `tool_choice` `none` and no call; one `web_search` tool, which the listener runs and returns as `web_search_call`; `max_output_tokens` 16 with status `incomplete` and reason `max_output_tokens`; and a streamed reply whose text deltas assemble to that reply. One thinking turn asks what vegetable a horse eats. The reasoning stays on the session. A required `locker` call binds the tool result by id, and the reply repeats that call's label. Content filters, audio parts, file parts, logprobs, and `n` greater than one are not checks. Exit 0 means `ledger.py` `collect` returns 6, 0, and -6, the vision log held when `--vision` was set, the prefix reload held, and every contract check held. Which tools the model called is not part of that result. Prints each request's prefill and decode line. A pass that prints those rates is filed with the research-claims skill in the same turn. Does not take the machine lock. Not part of `pytest`. `tests/test_coding_harness.py`, `tests/test_coding_tools.py`, and `tests/test_coding_compliance.py` check the request, the tools, and the contract rules without a listener. The tool contract is [coding-primitives.md](coding-primitives.md).

### Needle

```bash
python -m harness.needle --model <id> [--tokens 65536] [--depth 0.5] [--seed 0] [--thinking on|off] [--mode page|retrieve]
```

Reuses the listener on `127.0.0.1:8129`. Python writes `haystack.txt` before the agent starts. The file is repeated ordinary sentences and one line, `The special magic number is <digits>.` `--tokens` is the exact length of that file under the model tokenizer. The default is 65536. `--depth` is 0 at the first line and 1 at the last. The default is the middle. `--seed` fixes the number and the filler. The same seed, length, and depth write the same file. A Microsoft Agent Framework agent has one tool, the coding `read_file` window, and is asked for the number. The needle tool drops `limit`, so each read is a full window. Page mode replies only after a read shows the magic-number line. A page-mode miss or a transcript that does not fit starts a fresh turn at the last `next_offset`. `store` is false, so a later read replays the windows already returned. `--thinking off` is the default and sets `reasoning.effort` to `none`. `--thinking on` sets it to `medium`. `--mtp off` is the default. `--mode page` is the default. Exit 0 means the reply contains the number and one read returned that line. `--mode retrieve` tells the agent to follow `next_offset` until a read has none, and exit 0 also requires that the last window starts after the needle line. An empty page is not that window. A transcript longer than the served window exits 1 with `the transcript did not fit`. The banner prints the loaded key/value plane. Prints each request's prefill and decode line. Those lines are this model's only, from its `request open` until the next `request open`. A pass that prints those rates is filed with the research-claims skill in the same turn. Does not take the machine lock. Not part of `pytest`. `tests/test_needle_harness.py` checks the file and the grade without a listener.

### Synthetic prompt

```bash
python -m harness.synthetic --model <id> [--tokens 512 --tokens 10000] [--decode 128] [--thinking on|off]
```

Reuses the listener on `127.0.0.1:8129`. A Microsoft Agent Framework agent sends one user message and stops. The message is ordinary sentences under a length label, and it encodes to exactly `--tokens` under that model's chat template. The default lengths are 512 and 10000. `--decode` is the maximum sampled tokens, 128 when omitted. `--thinking off` is the default and selects the instruct card with `reasoning.effort` `none`. `--thinking on` selects the thinking card with `reasoning.effort` `medium`. `--mtp off` is the default. A warmup prompt of 64 tokens runs first, with a 16-token cap, so the measured prompts are not the load. Each measured prompt starts with its own label, so the prefill is a cache miss. Sampling stays on the card. The end-to-end line is the server's prefill tokens plus its decode tokens, over those two clocks. Exit 0 means each measured prompt length matched and the prefill did not reuse a prefix. A pass that prints the end-to-end rate is filed with the research-claims skill in the same turn. Does not take the machine lock. Not part of `pytest`. `tests/test_synthetic_harness.py` checks the prompt and the rate line without a listener.

### Worktree summarization

```bash
python -m harness.worktree --model <id>
```

Reuses the listener on `127.0.0.1:8129`, checks out a detached worktree of `main` under `artifacts/`, and drives tool writes to `notes/span.py`, `notes/check.py`, and `notes/README.md`. Write turns use thinking; read turns do not until a 131072-token prompt triggers summarization. A grader outside the worktree validates the files. Exit 0 requires the grader, a prompt of at least 131072 tokens, and a later prompt shorter than that peak. Prints each request's prefill and decode line. Does not take the machine lock. Not part of `pytest`.

### Public benchmarks

```bash
python -m harness.benchmarks --model <id> [--limit 30|100] [--mtp on|off] [--thinking] [--reasoning-effort <name>]
```

Reuses the listener on `127.0.0.1:8129` and scores IFEval, GSM8K, and HumanEval. A dropped connection unloads the resident model and sends that request again. The `model` field is that id. Thinking stays off unless `--thinking` is set. `--reasoning-effort` names the effort, and that name wins over the thinking flag. Temperature is 0 and the presence penalty is 0, so the draw is an argmax and the instruct card is not applied. `--temperature`, `--top-p`, `--top-k`, and `--presence` replace those defaults. Two short prompts run before the suites. They use 32 tokens when thinking is off, and the smallest suite cap when thinking is on, so a trace can close before the reply is checked. A CLI flag overrides a sampling field. `--mtp on` or `--mtp off` sends that flag. Omitting it leaves the flag out of the body, so the registry row is used. A row that omits `mtp` does not draft. `--mtp-depth` sends the draft length when it is set. IFEval sends the dataset prompt and scores it with the published strict and loose checkers. GSM8K sends five shots from the training split, in file order, as `Question` / `Answer`, and counts a reply only when its first `####` number matches. HumanEval sends the instruct prompt and tests the continuation after the published signature. A new top-level `class`, `def`, `#`, `if`, or `print` ends that continuation. The checker reads `message.content`. `--limit 30` scores the first 30 rows and writes the smoke section. `--limit 100` writes the sample of 100. A run with no limit scores the smoke, then the full sets. A set flag writes `smoke-mtp-on-<suite>` or `smoke-mtp-off-<suite>`, and the same prefix on the sample and full sections. Those completions are stored apart from a run that omitted the flag. Not part of `pytest`. Each finished suite is written into `docs/<pipeline>/benchmarks.md`.

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
python -m harness.image --pipeline <id> --tool <name>
```

Calls the published `{id}__{name}` tool on `POST /mcp` and reads the `file://` URI in the result. The prompt, size, and seed come from `docs/<id>/smoke.json`. An omitted `--model` leaves the folder to the host. Uses the listener session. Exit 0 means those bytes were read. Does not score the picture. Not part of `pytest`.

### Image smoke agent

```bash
python -m harness.image_smoke_agent --pipeline <id> --tool <name>
```

A Microsoft Agent Framework agent on the registry default chat model. `--tool` may be repeated. Those local names are the tools the pipeline already publishes. The agent registers exactly those published names through `MCPStreamableHTTPTool` and calls the first one. The prompt, negative prompt, size, and seed come from `docs/<id>/smoke.json`. The call omits quality, so the pipeline default applies. The PNG must be RGB. The next turn sends that PNG to the same model and asks whether the picture matches the prompt. That check uses reasoning effort `none` and stops at 256 tokens. Exit 0 means the reply affirms it. Uses the listener session. Not part of `pytest`. `tests/test_image_smoke_agent.py` registers the named tool on the test listener and reads the file.

### Image attachment

```bash
python -m harness.image_attachment
```

Posts one chat completion to `qwen38_27b` with `tests/assets/favlo-horse.png`. The user content is a text part, then an `image_url` part whose URL is `data:image/png;base64,` and whose `detail` is `auto`. Thinking is off and temperature is 0. The question does not name the animal. Exit 0 means `prompt_tokens` is at least the merged patch count and the reply names a horse plus two marks that are only in the picture: mane, tail, hoof, chestnut, gallop, or running. A refusal, including a problem with analysis, fails. Uses the listener session. Does not take the machine lock. Not part of `pytest`. `tests/test_image_attachment.py` checks the request and the reply rule without a listener.

### Image persona

```bash
python -m harness.image_persona --pipeline <id> --tool generate_image [--size WIDTHxHEIGHT]
```

The agent instructions are the horse persona. The user asks that horse to draw a picture of himself. The image prompt is a cute 2.5D horse with a white blaze and large blue eyes, sitting at a desk. `--tool` may be repeated; the agent registers those published names through `MCPStreamableHTTPTool` and calls the first one. The image call omits `model`, so the host uses the registry image folder. An omitted `--size` uses `docs/<id>/smoke.json` width and height. Seed is 0. Prints the PNG byte count. The next turn asks whether the picture shows that stylized horse rather than a photograph. Exit 0 means the reply affirms it. The PNG is `artifacts/favlo-horse.png`. Uses the listener session. Not part of `pytest`.

## Video

### I2V smoke agent

```bash
python -m harness.video_smoke_agent --pipeline <id> --tool generate_video
```

The still is `tests/assets/favlo-horse.png`. The pipeline resizes it to the clip size. `--tool` may be repeated. Those local names are the tools the pipeline already publishes. The agent registers exactly those published names through `MCPStreamableHTTPTool` and calls the first one with `video_prompt` (or `prompt`), the smoke frame count, and that PNG as an image block from `docs/<id>/smoke.json`. It reads the result into `artifacts/video-smoke-agent.mp4`. The same chat model reads frame 0, the middle frame, and the last frame and must affirm `prompt` in that file. Exit 0 means each frame was affirmed and the file has that geometry. Uses the listener session. Not part of `pytest`.

### TI2V smoke agent

```bash
python -m harness.video_ti2v_agent --pipeline <id> --tool generate_video_t2v --tool generate_video_i2v
```

Pass the text tool first and the still tool second. The text clip omits the still. The still clip passes `tests/assets/favlo-horse.png`, and the pipeline resizes it to `i2v_width` and `i2v_height` from `docs/<id>/smoke.json`. Outputs are `artifacts/video-ti2v-t2v.mp4` and `artifacts/video-ti2v-agent.mp4`. Vision checks frame 0, the middle frame, and the last frame of the still clip against the prompt. Exit 0 means both files match smoke sizes and frame count and each frame was affirmed. Uses the listener session. Not part of `pytest`.

### Video quality comparison

```bash
python -m harness.video_quality_agent --pipeline <id> --tool generate_video
```

The still is `tests/assets/favlo-horse.png`. The pipeline resizes it to 416×240. Calls the published tool twice at `quality` `low` and then `max`, with no chat between tasks, and reads each MP4 into `artifacts/video-quality-low.mp4` and `artifacts/video-quality-max.mp4`. Exit 0 means both files are 5 frames at 416×240 and the max file is larger. Uses the listener session. Not part of `pytest`.

## Music

### Music smoke agent

```bash
python -m harness.music_smoke_agent --pipeline <id> --tool generate_music
```

`--tool` may be repeated. The agent registers those published names through `MCPStreamableHTTPTool` and calls the first one with the model, prompt, lyrics, seconds, seed, bpm, keyscale, and timesignature in `docs/<id>/smoke.json`. Reads the WAV into `artifacts/music-smoke-agent.wav`. `python -m harness.inspect_wav` writes the report and `artifacts/music-smoke-agent.inspect.png`. The report adds key, pitch, harmony, quiet slices, and tempo; optional flags compare to the request. Exit 0 means template seconds, key, and bpm match the smoke request and the file is the note's duration, 48 kHz, stereo, with RMS above the silence floor. Uses the listener session. Not part of `pytest`.

### Music long agent

```bash
python -m harness.music_long_agent --pipeline <id> --tool generate_music [--card long.json]
```

Calls the published tool for the request in `docs/<id>/<card>` (default `long.json`) and reads the WAV into `artifacts/music-long-agent.wav`. The inspect report is checked like the smoke agent. Exit 0 also requires spectral flatness at or below 0.35, quiet slices at or below 15% of the half-second grid, a key margin of at least 0.02, and a one-second join no more than 15 times the median step beside it. Prints `seam_ratio`. Uses the listener session. Not part of `pytest`. `tests/test_music_long_agent.py` checks those gates without a listener.

## Voice

### Voice smoke agent

```bash
python -m harness.voice_smoke_agent --pipeline <id> --tool generate_voice
```

`--tool` may be repeated. The agent registers those published names through `MCPStreamableHTTPTool` and calls the first one with the model, line, and instructions in `docs/<id>/smoke.json`. Reads the WAV into `artifacts/voice-smoke-agent.wav`. Exit 0 means mono at the note's sample rate, RMS above the silence floor, and duration inside `max_seconds`. Uses the listener session. Not part of `pytest`. `tests/test_voice_smoke_agent.py` checks the same format rule without a listener.

## Media preemption

### Media pipeline preemption

```bash
python -m harness.media_pipeline
python -m harness.media_pipeline i2v
```

One Microsoft Agent Framework agent preempts the registered media pipelines. It discovers them from the catalog and each pipeline's `smoke.json`. Kind names after the command run that subset, in catalog order: `image`, `i2v`, `t2v`, `music`, `voice`. An omitted list runs every kind. Each case starts the MCP task; the next turn is a chat completion that preempts it. The tool polls `media_status` and `tasks/get` and reads the file into `artifacts/media-preempt-*`. Exit 0 means every selected task was paused with `task is ongoing` and the same task finished at the expected shape. Uses the listener session. Not part of `pytest`. `tests/test_media_pipeline_harness.py` checks the pause-then-complete rule and the name filter without a listener.

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

Import `harness.mcp_media` (no `python -m` entry). Shared MCP client helpers (`start_task`, `task_view`, `read_link`, and related) used by media agent harnesses. `read_link` reads `mediaUri` from the tool text. A local tool name is published as `{pipeline}__{local}`. An omitted `model` picks the first published match. MCP does not list models; the pipeline is the model.

# Point a client at this server

The server daemon listens on `http://127.0.0.1:8129`. The OpenAI-compatible base URL is `http://127.0.0.1:8129/v1`.

`--host` and `--port` change that address. The default host is loopback, so another machine cannot connect until `--host` is set to an address that machine can reach.

The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

`GET /v1/models` lists registered models. `POST /v1/chat/completions` runs one chat. The `model` field is a registry id: the folder name, the alias, or either of those with `:thinking` or `:instruct`. A bare id uses that pipeline's default variant. `enable_thinking` set to false selects the instruct card when the id has no variant suffix.

One request runs at a time. A connection that arrives while a request is open waits. The first request that names a model loads it. Naming a different model unloads the one that is resident.

```bash
curl http://127.0.0.1:8129/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'
```

A streaming client sets `"stream": true`. Each event is a `chat.completion.chunk`. The stream ends with `data: [DONE]`.

`favlo serve` on a terminal is the CLI for that daemon. `favlo serve --non-interactive` starts the daemon and exits. Closing the terminal leaves the daemon running. Type in the CLI while no other request is open. `/quit` stops the daemon.

The first time that terminal starts, and `~/.favlo/intros.json` has no lines, Favlo asks the registered model for the opening through `save_intro`. The tool stores the lines and returns the first, which is the line shown. That request is the check that the model answers. Later starts show that same first line and do not call the model until someone types.

After 30 minutes with no open request and no typing, Favlo checks physical memory and free disk. 30 minutes is the gap, because a shorter pause is still a conversation and the check uses the only request slot. When the machine has at least 16 GB and the volume can hold the files, Favlo starts the download of `Qwen/Qwen3.8-27B` into `models/Qwen3.8-27B-bf16` and asks whether to remind them again. A machine that cannot hold it, or a checkout that is already present, stays quiet. `download_suggestions` with enabled false stores that answer in `~/.favlo/suggestions.json` and later idle checks stay quiet. Enabled true turns the reminders back on.

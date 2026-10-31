# Point a client at this server

The server daemon listens on `http://127.0.0.1:8129`. The OpenAI-compatible base URL is `http://127.0.0.1:8129/v1`.

`--host` and `--port` change that address. The default host is loopback, so another machine cannot connect until `--host` is set to an address that machine can reach.

The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

`GET /v1/models` lists registered language models. Media models are listed by the media MCP tools. Each object includes `context_length`, `mtp`, and `mtp_depth`. The context default is 131072. A row that omits `mtp` is false. A row that omits `mtp_depth` is 3. A completion that sets either field overrides the row. A completion's `usage` object includes `prompt_tokens`, `completion_tokens`, `total_tokens`, `context_length`, and `prompt_tokens_details.cached_tokens`. `POST /v1/chat/completions` runs one chat. The `model` field is a registry id: the folder name, the alias, or either of those with `:thinking` or `:instruct`. A bare id uses that pipeline's default variant. `enable_thinking` set to false selects the instruct card when the id has no variant suffix.

One request runs at a time. A connection that arrives while a request is open waits. The first request that names a model loads it. Naming a different chat or media model unloads the one that is resident. An embeddings request leaves that model loaded and closes its own process when the request returns.

```bash
curl http://127.0.0.1:8129/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'
```

A streaming client sets `"stream": true`. Each event is a `chat.completion.chunk`. The stream ends with `data: [DONE]`.

`favlo serve` on a terminal is the daemon. It paints progress on stderr. It does not read the terminal and it does not call a model. Ctrl+C closes every open pipeline and stops the process. `favlo serve --non-interactive` starts a detached daemon and exits. Closing that terminal leaves the detached daemon running. A second `favlo serve` exits when that address is already in use. `POST /v1/pipelines/reload` closes every open pipeline, drops every waiter in the foreground queue and the media queue, drops a media task that has not started, and drops the imported pipeline modules. A waiter does not run. The request that holds the GPU is cancelled.

A terminal whose registry has no rows checks that a small shader compiles at Metal 4.1, then downloads and packs Qwen3.6-35B-A3B before the daemon starts. The steps are [models.md](models.md). `--non-interactive` does not download. A pipe does not download.

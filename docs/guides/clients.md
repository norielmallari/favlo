# Point a client at this server

The server daemon listens on `http://127.0.0.1:8129`. The OpenAI-compatible base URL is `http://127.0.0.1:8129/v1`.

`--host` and `--port` change that address. The default host is loopback, so another machine cannot connect until `--host` is set to an address that machine can reach.

The server does not check an API key. A client that requires one can send any placeholder, such as `favlo`.

`GET /v1/models` lists registered language models. Media models are listed by the media MCP tools. Each object includes `context_length`, `mtp`, and `mtp_depth`. The context default is 131072. A row that omits `mtp` is false. A row that omits `mtp_depth` is 3. A completion that sets either field overrides the row. A completion's `usage` object includes `prompt_tokens`, `completion_tokens`, `total_tokens`, `context_length`, and `prompt_tokens_details.cached_tokens`. `POST /v1/responses` runs one chat. Every registered language row answers that route. The `model` field is a registry id: the folder name, the alias, or either of those with a sampling suffix such as `:thinking` or `:instruct`. A bare id uses that pipeline's default sampling card. A suffix is a sampling preset only. Thinking follows `reasoning.effort` on the request, or the row's `enable_thinking` when effort is omitted. `POST /v1/chat/completions` translates into that request and translates the Response back. On that legacy body, `enable_thinking` maps to effort when the Responses body has none.

One request runs at a time. A connection that arrives while a request is open waits. The first request that names a model loads it. Naming a different chat or media model unloads the one that is resident. An embeddings request leaves that model loaded and closes its own process when the request returns.

```bash
curl http://127.0.0.1:8129/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-4b","input":"Hello"}'
```

A streaming client sets `"stream": true`. Each Responses event is one JSON object. The stream ends with `data: [DONE]`. The Chat Completions translation streams a `chat.completion.chunk` and ends the same way.

`favlo serve` on a terminal is the daemon. It paints progress on stderr. It does not read the terminal and it does not call a model. Ctrl+C closes every open pipeline and stops the process. `favlo serve --non-interactive` starts a detached daemon and exits. Closing that terminal leaves the detached daemon running. A second `favlo serve` stops the listener already on that address and any pipeline process, then listens. An address held by another program still exits. `POST /v1/pipelines/reload` closes every open pipeline, drops every waiter in the foreground queue and the media queue, drops a media task that has not started, and drops the imported pipeline modules. A waiter does not run. The request that holds the GPU is cancelled.

A terminal whose registry has no rows checks that a small shader compiles at Metal 4.1, then downloads and packs Qwen3.5-4B before the daemon starts. The steps are [models.md](models.md). `--non-interactive` does not download. A pipe does not download.

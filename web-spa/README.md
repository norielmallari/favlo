# Web app

Expo app for the Favlo web workspace. A browser at `http://127.0.0.1:8081` talks to the Web API. This directory is part of the Favlo repository.

The API is [../web-api/README.md](../web-api/README.md). The process map is [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Getting started

`favlo web` starts this app and the API together. The listener stays a separate process. From a checkout of this repository, with at least one language model registered:

```bash
cd web-spa && npm install
uv run favlo serve --non-interactive
uv run favlo web
```

Open `http://127.0.0.1:8081`. A page opened at `http://localhost:8081` moves to that address before it calls the API. The API listens on `127.0.0.1:8130`. Stopping `favlo web` stops both children.

## Chat

Each agent has one channel. Opening a conversation, or refreshing it, shows the latest message. A message renders Markdown, including bold, italics, and tables. Raw HTML stays text. A reply that includes reasoning shows Thinking. The block starts open and closes when the reply is stored. Choosing Thinking opens or closes the reasoning. While a reply is in progress, Working shows a light band sweeping across the word, and Send becomes Stop. Reduced motion leaves the word still. When the reply finishes, Reply ready is announced. The conversation scrollbar uses the theme colors. A button shows a tint while it is pressed. A chat message that contains a `favlo://` image, video, or audio link shows that stored file once it has loaded. A missing file stays off the screen. A message that is only that missing file leaves the transcript. The tool name and the file id stay off the screen. Choosing the image opens it over the conversation, and Download saves it in the browser as `favlo-image.png`. Video and audio keep their playback controls and a download button. Closing the view returns to the same place in the chat. The API streams the file. The contract is [../web-api/README.md](../web-api/README.md).

An empty conversation shows the agent name, a one-line purpose, and three starters that fill the message field: generate an image, list registered models, and set up a model. On a compact window the greeting uses the title size and the starters stack. Return sends. Shift+Return inserts a line. You turns sit on a raised surface. Assistant turns stay full width with a Copy control. Type uses the system UI face at 16/26 for body, 20/28 for titles, and 28/36 for the empty greeting. Code uses the UI monospace face. Replies and the message field share one reading column, 72 characters wide on a wide window.

New agent asks for a name, a persona, a provider, a model, and the tool switches. The form is a dialog that keeps keyboard focus until Close. On a narrow window the header wraps onto more than one line, and the conversation stays inside the screen. The channels column is a drawer behind Channels. On a wide window that column stays open at 280 pixels. A selected row has an accent mark and a heavier label. Workspaces stay off the screen while the list is empty. Skip to message stays out of view until it has focus. Appearance is a menu of System, Light, and Dark. Its mark is a half-filled circle, a sun, or a moon. On a narrow window the menu opens to the right of its button.

## Configuration

Configuration is a sidebar of sections. A catalog such as Agents lists entries in the second column and the selected entry in the third. A section that is not a catalog shows its fields in the second column. New entries are created from a button in the third column. A selected menu row keeps space around its focus ring so the outline stays visible. A successful save shows one confirmation at the bottom of the screen. Dismiss closes it, and the rest of the screen stays usable.

| Section | Columns |
|---|---|
| Agents | Sections, agents, the selected agent |
| Providers | Sections, providers, the selected provider |
| Heartbeats | Sections, agents, that agent's heartbeats, the selected heartbeat |
| MCP | Sections, agents, that agent's servers, the selected server |
| Orchestration | Sections, the exchange budget |

New agent and the selected agent use the same fields: a name, a persona, a provider, a model, and one switch per tool group. Delete removes that agent and its channel. Favlo has no delete button. The groups and their tools are in the [API note](../web-api/README.md). Heartbeat details take a five-field cron expression, a timezone, instructions, and an enabled switch. The API starts a run when that minute arrives. Run heartbeat starts one immediately. MCP details take an HTTP URL or a stdio command. Test connection lists the live tools in the form before Save stores the server and the switches.

## Development

Unit tests use Jest. They do not start the listener.

```bash
cd web-spa && npx jest
```

End-to-end tests use headless Playwright against a live `favlo web` and `favlo serve`.

```bash
cd web-spa && npx playwright test --project=chromium
```

## Problems

Report a problem in this repository's issue tracker.

## License

The license is the Apache License, Version 2.0. The terms are in [../LICENSE.md](../LICENSE.md).

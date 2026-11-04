# Web app

Expo app for the Favlo web workspace. A browser at `http://127.0.0.1:8012` talks to the Web API. This directory is part of the Favlo repository.

The API is [../web-api/README.md](../web-api/README.md). The process map is [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Getting started

`favlo web` starts this app and the API together. The listener stays a separate process. From a checkout of this repository, with at least one language model registered:

```bash
cd web-spa && npm install
uv run favlo serve --non-interactive
uv run favlo web
```

Open `http://127.0.0.1:8012`. A page opened at `http://localhost:8012` moves to that address before it calls the API. The API listens on `127.0.0.1:8130`. Stopping `favlo web` stops both children.

## Chat

Each agent has one channel. Opening a conversation, or refreshing it, shows the latest message. A message renders Markdown, including bold, italics, and tables. A closed `*action*` in the reply gets a line break after it for display; one newline the model already sent there is dropped. Block-quoted lines and fenced code blocks are left alone. That formatting does not change the stored reply or inference. A list item keeps its text beside the marker, including a line that wraps. The composer command `/session clear` clears this chat. `/session archive` asks the agent to store worthwhile memories, and clears this chat when that turn stops. Raw HTML stays text. A reply that includes reasoning shows Thinking. Thinking, each tool call, and the reply stay in the order they arrive. Thinking closes when a tool call or the reply starts. A tool call closes when thinking or the reply starts. Both close when the turn is stored. Choosing the row opens or closes the call. While a reply is in progress, Working shows a light band sweeping across the word, and Send becomes Stop. Stop ends the Agent Framework turn and disconnects the listener. A failed reply names the reason in plain language above the composer. Send again posts that user message once more. Reduced motion leaves the word still. When the reply finishes, Reply ready is announced. Scrollbars use the theme colors. A button shows a tint while it is pressed. A chat message that contains a `favlo://` image, video, or audio link shows that stored file once it has loaded. A missing file stays off the screen. A message that is only that missing file leaves the transcript. The tool name and the file id stay off the screen. Choosing the image opens it over the conversation, and Download saves it in the browser as `favlo-image.png`. Video and audio keep their playback controls and a download button. Closing the view returns to the same place in the chat. The API streams the file. The contract is [../web-api/README.md](../web-api/README.md).

A new agent opens on a greeting: the agent name, a one-line purpose, and three starters that fill the message field: generate an image, list registered models, and set up a model. An existing channel, including one whose session was just cleared, keeps the conversation heading and an empty transcript. A transcript shorter than the screen rests on the composer, with the empty space above the first line. A longer transcript keeps the latest line on the composer. The gallery chat uses the same alignment. On a compact window the greeting uses the title size and the starters stack. Return sends. Shift+Return inserts a line. You turns sit on a raised surface. Assistant turns stay full width with a Copy control. Type uses the system UI face at 16/26 for body, 20/28 for titles, and 28/36 for the empty greeting. Code uses the UI monospace face. Replies and the message field share one reading column, 72 characters wide on a wide window.

New agent asks for a name, a persona, a provider, reasoning effort, preserve thinking, Expand thinking, Expand tool calls, and the tool switches. Provider Not set leaves the model picker and Refresh models hidden and uses the Configuration → Providers default model. Choosing a provider shows that provider's model list. Reasoning effort is `none`, `low`, `medium`, `high`, or `xhigh`. Not set uses the Providers effort, and Not set there leaves that field off the request. The form is a dialog that keeps keyboard focus until Close. On a narrow window the header wraps onto more than one line, and the conversation stays inside the screen. The channels column is a drawer behind Channels. The open channel starts expanded. Its name opens and closes that list and leaves the transcript where it is. Home, with the Feather `home` icon, opens that agent's channel. Each workspace follows Home, with the Feather icon from its shell. Delete on that row asks first, then removes the workspace. Files on disk stay. New workspace for that agent sits under the agent's menu and creates the workspace on that channel. It asks for a name and a shell from `GET /shells`. Choosing a workspace opens `/w/<id>` in a new tab and leaves the channel tab on the conversation. The workspace page titles the document with the workspace name, keeps Appearance, and links to Favlo. It does not show the channel list, Configuration, or Back to channel. Deleting the workspace leaves that tab on a centered note that the workspace is gone, with the Favlo link still in the banner. On a narrow window there is no separate Workspaces control. On a wide window the channel column stays open at 280 pixels while the transcript is showing. A selected row has an accent mark and a heavier label. Action buttons include a Feather icon and sit on one row when they fit the width. The Favlo title opens the home page. The header Configuration control is the Feather `settings` icon. Its name stays Configuration. Skip to message stays out of view until it has focus. Appearance is a menu of System, Light, and Dark. Its mark is a half-filled circle, a sun, or a moon. On a narrow window the menu opens to the right of its button.

A new workspace shows a centered Creating card on its page, and Creating on the channel row, until `provisioning_state` is `provisioned`. The shell and the composer stay off until then. The page then walks the shell's slots and mounts the components it knows. A gallery workspace is generative UI on that shell (`preview` + `filmstrip`). Bind a local folder once. Provisioning captions every image, including every visible human anatomical feature, then the filmstrip shows those images. The sidecar caption stays in the placement. The filmstrip keeps 60. Chat listings stay path-shaped; metadata and pixels load through gallery tools when needed. The greeting does not call a placement tool. Captions use that workspace agent's model, or the Providers default when the agent has no model. Gallery theme is a menu on that page: Favlo, Rosé Pine, or Catppuccin. A new gallery starts on Rosé Pine. Each scheme follows Appearance (System, Light, or Dark). The Favlo home page stays on the Favlo palette. The agent can set the same scheme with `gallery_set_theme`. Gestures (select, arrows, Download, Add to chat) do not start a model run. As-built detail is in the closed tree: `favlo-cloud/docs/maf/curated-gallery.md`. A coding workspace reuses chat chrome without that shell. Skills and Delete workspace apply to both types. Chat turns use `POST /ag-ui`. The API contract is [../web-api/README.md](../web-api/README.md).

## Configuration

Configuration is a sidebar of sections. A catalog such as Agents lists entries in the second column and the selected entry in the third. A section that is not a catalog shows its fields in the second column. New agent, New skill, New heartbeat, and New MCP are the first row of that catalog list. A selected menu row keeps space around its focus ring so the outline stays visible. A successful save shows one confirmation at the bottom of the screen. Dismiss closes it, and the rest of the screen stays usable.

| Section | Columns |
|---|---|
| Agents | Sections, agents, the selected agent |
| Providers | Sections, providers, the selected provider, its default model, and Reasoning effort |
| Heartbeats | Sections, agents, that agent's heartbeats, the selected heartbeat |
| Skills | Sections, global skills, the selected skill. Add to new agents copies that skill onto an agent created later. |
| MCP | Sections, global MCP servers, the selected server. Add to new agents copies that server onto an agent created later. |
| Orchestration | Sections, the exchange budget, and the note that gallery captions use the workspace agent's model |

New agent and the selected agent use the same fields: a name, a persona, a provider, reasoning effort, preserve thinking, Expand thinking, Expand tool calls, and one switch per tool group. Expand thinking defaults to Collapsed. Expand tool calls defaults to Not shown, which leaves that block out and the activity chip says Working. Not shown does the same for thinking. Open while running and Always open are stored separately. Provider Not set hides the model controls and uses the Providers default model at chat time; Favlo and new agents start that way. Choosing a provider shows Model and Refresh models for that provider. Reasoning effort becomes Responses `reasoning.effort`. Providers has the same control, including Not set. An agent effort wins. Not set on both leaves that field off the request. A new agent starts with each global skill and MCP server marked Add to new agents, including Favlo media. Delete removes that agent and its channel. Favlo has no delete button. The groups and their tools are in the [API note](../web-api/README.md). The selected agent opens with an Agent details tablist at the top of the detail pane: Agent, Skills, and MCP. New skill and New MCP are the first row of each list. The Skills and MCP sections are catalogs. Copy to agent writes a separate row on that agent. Those copies are what a turn uses. A catalog row is not. Heartbeat details take a five-field cron expression, a timezone, instructions, and an enabled switch. Each run uses that agent's provider and model when set, otherwise the Providers default. The API starts a run when that minute arrives. Run heartbeat starts one immediately. MCP details take an HTTP URL or a stdio command. Test connection lists the live tools in the form before Save stores the server and the switches. The tool list itself is read from the MCP server and cached in the API process for 60 seconds.

## Development

Unit tests use Jest. They do not start the listener.

```bash
cd web-spa && npx jest
```

End-to-end tests use headless Playwright against a live `favlo web` and `favlo serve`.

```bash
cd web-spa && npx playwright test --project=chromium
```

`web-new-user-e2e-setup` starts its own `favlo web` on an empty home, so ports 8130 and 8012 must be free. That run creates the database, seeds Favlo, and sends the first chat. It does not use `~/.favlo/web`.

```bash
cd web-spa && npx playwright test --config e2e/web-new-user-e2e-setup/playwright.config.ts --project=chromium
```

## Problems

Report a problem in this repository's issue tracker.

## License

The license is the Apache License, Version 2.0. The terms are in [../LICENSE.md](../LICENSE.md).

---
name: web-happy-path
description: Finish a Favlo web feature with a deterministic Playwright happy path, or run and write scripts under web-spa/e2e. Use when a person can start a new action from the app, including a web-api change the app can trigger.
---

# Web happy path

The feature is finished when the script passes against a live `favlo web` and `favlo serve`. Write the script before treating the feature as finished. An instruction to leave the suite unrun still leaves the script in the change.

The script is `web-spa/e2e/<feature>.spec.ts`, or an extension of the script that already owns that action. A fresh install uses `e2e/web-new-user-e2e-setup/`. The script performs the action and shows the outcome. Opening a screen and checking a heading is not a happy path. A script that does not perform the action does not count. Add a row to the table below when the script is new. `web-spa/playwright.config.ts` sets `screenshot` to `on`.

Stub only what the script already stubs, such as `**/folders/pick`. Do not drive the macOS folder dialog. Never use the Cursor IDE browser, Computer Use, or an interactive browser MCP.

```bash
cd web-spa && npx playwright test e2e/<feature>.spec.ts --project=chromium
```

Do not pipe the run through `tail`. Prefer `step` from `e2e/progress.ts`. Open `http://127.0.0.1:8012`. Do not stub the model. Never open a browser in Cursor.

Before the click or key that finishes the feature, call `onScreen` from `web-spa/e2e/on-screen.ts` on that control. `onScreen` requires the whole control inside the viewport. It does not scroll. Do not use `toBeVisible`, `toBeInViewport`, `boundingBox`, or `scrollIntoView` to reach the control. If `onScreen` fails, fix the layout.

Cover 1280×800. When the feature has a narrow layout, cover 390×844 in the same script. A dialog or a form also covers 1280×640. Do not use `waitForTimeout`. Assert exact wording only when that wording is not model output.

Playwright writes a screenshot under `web-spa/test-results/`. Read it before treating the run as finished. Leave the screenshot uncommitted.

Jest unit tests do not start `favlo serve` or a browser: `cd web-spa && npx jest <test>`.

## Timeouts

A hung step must fail within 15 seconds. Do not pass `timeout: 60_000`, `180_000`, or `360_000` on `expect`, `poll`, `waitForEvent`, or `test.setTimeout` for a single wait. `playwright.config.ts` sets `expect.timeout`, `actionTimeout`, and `navigationTimeout` to 15 seconds. Override only downward. A slow model load or a dead serve is a failed step, not a longer wait.

## Teardown

Every script that creates an agent or workspace deletes it in `finally`.

- Agents: `deleteAgentsNamed` from `e2e/cleanup.ts`. Never delete Favlo.
- Workspaces: `deleteWorkspacesNamed` from `e2e/cleanup.ts`.
- Temp folders: `fs.rmSync` with `recursive` and `force`.
- Do not leave Align, Place, Session, Stills, Bind, Frame, or Captions-style names on disk or in the API after the run.

## Suite

Fail fast. One Chromium project. Re-check serve `200` between specs when chaining. Gallery scripts skip coding workspaces.

## Scripts

| Feature | Script |
|---|---|
| Chat, configuration, heartbeats | `e2e/chat.spec.ts` |
| Basic home chat | `e2e/home-chat.spec.ts` |
| Gallery captions use the agent model | `e2e/gallery-vision-settings.spec.ts` |
| Short transcript on the composer | `e2e/chat-align.spec.ts` |
| New agent | `e2e/agent-create.spec.ts` |
| Delete agent | `e2e/agent-delete.spec.ts` |
| Empty conversation | `e2e/empty.spec.ts` |
| Working | `e2e/working.spec.ts` |
| Thinking | `e2e/thinking.spec.ts` |
| Thinking chat through serve | `e2e/thinking-serve.spec.ts` |
| Reasoning effort on/off | `e2e/reasoning-effort.spec.ts` |
| Tool call | `e2e/tool-call.spec.ts` |
| Markdown | `e2e/markdown.spec.ts` |
| Stored media | `e2e/media.spec.ts` |
| MCP server | `e2e/mcp.spec.ts` |
| Global skill copied onto an agent | `e2e/agent-skills.spec.ts` |
| Gallery guide | `e2e/gallery.spec.ts` |
| Gallery theme | `e2e/gallery-theme.spec.ts` |
| Coding workspace header | `e2e/coding-workspace-header.spec.ts` |
| Gallery folder bind | `e2e/gallery-bind.spec.ts` |
| Gallery chat width resize | `e2e/gallery-chat-resize.spec.ts` |
| Gallery narrow one-view chat | `e2e/gallery-narrow.spec.ts` |
| Gallery placement change | `e2e/workspace-placement.spec.ts` |
| Gallery image embedded once | `e2e/gallery-image.spec.ts` |
| Gallery selection rides the sent message | `e2e/gallery-selection.spec.ts` |
| Filmstrip highlights the selected folder file | `e2e/gallery-filmstrip-select.spec.ts` |
| Reply while an image generates, then the image | `e2e/image-generate.spec.ts` |
| Gallery reply while an image generates, filmstrip, reload | `e2e/gallery-image-generate.spec.ts` |
| Gallery default folder | `e2e/gallery-default-folder.spec.ts` |
| Session clear and archive | `e2e/session.spec.ts` |
| Workspace skill | `e2e/workspace-skills.spec.ts` |
| Workspace type and delete | `e2e/workspace-type-delete.spec.ts` |
| Delete a workspace from the channel menu | `e2e/workspace-menu-delete.spec.ts` |
| Wide workspace menu focus ring | `e2e/workspace-menu-focus.spec.ts` |
| Workspace shells from the API | `e2e/workspace-shells.spec.ts` |
| Fresh install and first chat | `e2e/web-new-user-e2e-setup/setup.spec.ts` |
| Failed reply sent again | `e2e/reply-error.spec.ts` |

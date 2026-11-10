# Playwright screens

Live `favlo web` on `127.0.0.1:8012` and `favlo serve` on `8129`. Do not stub the model. Prefer `step` from `e2e/progress.ts`. Use `exact: true` on `Send`, `Home`, and workspace names. Do not pipe runs through `tail`.

Timeouts, teardown, the suite, `onScreen`, and the script table are the web-happy-path skill.

Type what a person would type. Do not put tool names or schemas in the message box. A stalled agent gets one natural-language nudge, then fail.

Clicking an already-expanded channel collapses it. Wait for `Home` before clicking Favlo. Scope Home under `getByLabel('Channels')`. When a gallery is open, click `Back to channel` first. On a narrow viewport, open the Channels drawer before a workspace row.

After Send for ordinary chat, wait until `POST /ag-ui` returns OK before Stop. Prefer letting a reply finish before `/session archive`.

Workspace transcript uses `transcriptFrame(space[4])` like home.

Use `assertWcagAa` from `e2e/a11y.ts`. Cover light and dark on screens that own text-heavy chrome. Text and control colors come from `palette` and `textStyle`.

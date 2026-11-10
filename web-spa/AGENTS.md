# Web UI

Every screen follows best UI/UX practice and is a pleasure to use. Published guidance this repo already uses: [Material 3](https://m3.material.io/), [Apple's Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/), and [WCAG 2.2](https://www.w3.org/TR/WCAG22/) at AA. Where those sources are silent or disagree, pick the option that serves the person using the screen.

The task is the subject. Chrome stays quiet. Nothing the person did goes away when a view is hidden: scroll, draft, selection, size, and an open panel survive a resize and a trip to another view. Do not unmount a view to hide it when remounting loses that state. Context stays in sight. Prefer one view, and a non-modal surface, when the person is working with what is behind it. Every action answers, including loading, empty, error, and disabled. Motion has a cause and freezes under reduced motion. Errors keep what the person typed and say what to do next. A narrow window is a first-class layout, including 320 pixels wide, with no two-dimensional scrolling. Targets are at least 44 points. A control has a name, a role, a state, and a visible focus. A drag has a keyboard equivalent. Destructive actions are asked once, name what is affected, and stay distinct.

A screen check is a headless Playwright script. Never open a browser in Cursor, including the IDE browser, browser MCP, and Computer Use. The procedure is the web-happy-path skill. The API half of a feature lives in `web-api/AGENTS.md`.

## Tokens

Tokens live in `web-spa/src/theme/tokens.ts`. Read the palette with `useTheme()`. Paint text with `textStyle(role, palette.<token>)` from `web-spa/src/theme/type.ts`. Do not omit the color argument on visible text. Nested `Text` sets its own color. Do not hardcode hex or rgb on text, icons, or buttons.

Body and UI text meet WCAG AA against the surface behind them. Primary copy uses `palette.textPrimary`. Secondary copy uses `palette.textSecondary`. Links and ghost controls use `palette.accent`. Errors use `palette.danger`. Disabled labels use `palette.disabledText` on `palette.disabledSurface`.

`IconButton` takes `tone` or an explicit `background` and `color`. Tones are `accent` (`onAccent`), `danger` (`onDanger`), `surface` (`textPrimary`, border `borderStrong`), `ghost` (`accent`), and `dangerGhost` (`danger` on transparent). `buttonPaint` in `web-spa/src/theme/buttons.ts` is the map. Disabled controls use `disabledSurface` and `disabledText`.

Page chrome uses `palette.background`. Cards, drawers, and dialogs use `palette.surface` or `palette.surfaceRaised`. Borders use `palette.border` or `palette.borderStrong`. Focus rings use `palette.focusRing`. Overlays use `palette.overlay`.

Do not mix the `flex` shorthand with `flexGrow`, `flexShrink`, or `flexBasis` on the same style object. Do not toggle `flex` to `undefined` while those longhands stay set. Prefer the longhands when any one of them varies.

Ambient motion uses opacity, about 2 to 2.5 seconds, and `ease-in-out`. Freeze it under reduced motion. The Working chip is label-only. Do not clip a label with a shimmer that flashes to `onAccent` or white.

## Seams

A passive seam is 1 pixel, `palette.border` or `palette.borderStrong`. A resizable splitter is 1 pixel at rest and 2 when hot, with a hit target of at least `space[3]` centered on the seam. The hairline uses `palette.borderStrong` at rest and `palette.accent` when hot. The track stays transparent until hot, then `palette.accentSubtle`. A short grip pill marks that the edge moves. The cursor is `col-resize` or `row-resize`. Keyboard arrows change the size. The reference is `web-spa/src/workspace/GalleryChatResizeHandle.tsx`. A control outline is 1 pixel, `palette.border` or `palette.borderStrong`, and focus uses `palette.focusRing`.

## Chat prefix

Leave prior messages byte-stable. Host cues use a fenced `user` row with `metadata.favlo.ui=hidden`. A chat turn sets `store` true and sends `previous_response_id`. The listener prepends the stored chain. Clear and archive mint a new thread. The durable rules are in `web-api/AGENTS.md`.

Catalog names in `web_api/catalog.py` and `web-spa/src/workspace/catalog.ts` stay paired. A gallery or workspace change updates `web-spa/README.md`, `web-api/README.md`, and `docs/ai-assistance/README.md` in the same change.

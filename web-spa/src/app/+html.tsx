import { ScrollViewStyleReset } from 'expo-router/html';
import type { PropsWithChildren } from 'react';

const FAVLO_ICON =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2563EB"/><text x="16" y="22" text-anchor="middle" font-size="16" font-family="system-ui,sans-serif" font-weight="700" fill="#FFFFFF">F</text></svg>`,
  );

/**
 * Root HTML for the web export. ``interactive-widget=resizes-content`` makes
 * Chrome and Firefox shrink the layout viewport with the on-screen keyboard so
 * a bottom-docked composer stays above it.
 */
export default function Root({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, shrink-to-fit=no, interactive-widget=resizes-content"
        />
        <meta name="theme-color" content="#FFFFFF" />
        <link rel="icon" href={FAVLO_ICON} />
        <ScrollViewStyleReset />
      </head>
      <body>{children}</body>
    </html>
  );
}

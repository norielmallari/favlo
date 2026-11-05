import type { Palette } from '@/theme/tokens';

/** Form controls use a spread shadow so a rounded field keeps the ring on every side. */
export function focusRingCss(color: string): string {
  return `
    :root { --favlo-focus-ring: ${color}; }
    input:focus,
    textarea:focus,
    select:focus,
    [role="switch"]:focus {
      outline: none;
      box-shadow: 0 0 0 2px var(--favlo-focus-ring);
    }
  `;
}

/** Thin scrollbar in the theme colors. Applies to every scrollable element. */
export function scrollbarStyle(palette: Palette): string {
  const thumb = palette.borderStrong;
  const track = palette.background;
  const hover = palette.textSecondary;
  return `
    * {
      scrollbar-width: thin;
      scrollbar-color: ${thumb} ${track};
    }
    *::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    *::-webkit-scrollbar-track {
      background: ${track};
    }
    *::-webkit-scrollbar-thumb {
      background-color: ${thumb};
      border: 2px solid ${track};
      border-radius: 999px;
    }
    *::-webkit-scrollbar-thumb:hover {
      background-color: ${hover};
    }
  `;
}

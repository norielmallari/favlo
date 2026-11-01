import type { Palette } from '@/theme/tokens';

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

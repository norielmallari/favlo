import type { Palette } from '@/theme/tokens';

/** Filled and ghost button paints from the active palette. */
export type ButtonTone = 'accent' | 'danger' | 'surface' | 'ghost' | 'dangerGhost';

export type ButtonPaint = {
  background: string;
  border: string;
  color: string;
};

export function buttonPaint(palette: Palette, tone: ButtonTone): ButtonPaint {
  switch (tone) {
    case 'accent':
      return { background: palette.accent, border: palette.accent, color: palette.onAccent };
    case 'danger':
      return { background: palette.danger, border: palette.danger, color: palette.onDanger };
    case 'surface':
      return {
        background: palette.surface,
        border: palette.borderStrong,
        color: palette.textPrimary,
      };
    case 'ghost':
      return { background: 'transparent', border: 'transparent', color: palette.accent };
    case 'dangerGhost':
      return { background: 'transparent', border: 'transparent', color: palette.danger };
  }
}

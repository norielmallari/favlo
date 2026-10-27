import { contrast, dark, light } from '@/theme/tokens';

const TEXT_PAIRS: [string, string, string][] = [
  ['text on light background', light.textPrimary, light.background],
  ['secondary on light background', light.textSecondary, light.background],
  ['text on light surface', light.textPrimary, light.surface],
  ['label on light accent', light.onAccent, light.accent],
  ['accent on light background', light.accent, light.background],
  ['text on dark background', dark.textPrimary, dark.background],
  ['secondary on dark background', dark.textSecondary, dark.background],
  ['text on dark surface', dark.textPrimary, dark.surface],
  ['label on dark accent', dark.onAccent, dark.accent],
  ['accent on dark background', dark.accent, dark.background],
  ['danger on light background', light.danger, light.background],
  ['danger on dark background', dark.danger, dark.background],
];

describe('theme contrast', () => {
  test.each(TEXT_PAIRS)('%s meets WCAG AA', (_name, foreground, background) => {
    expect(contrast(foreground, background)).toBeGreaterThanOrEqual(4.5);
  });

  test('control borders meet the UI contrast minimum', () => {
    expect(contrast(light.borderStrong, light.surface)).toBeGreaterThanOrEqual(3);
    expect(contrast(dark.borderStrong, dark.surface)).toBeGreaterThanOrEqual(3);
  });
});

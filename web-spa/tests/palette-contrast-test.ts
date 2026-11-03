import { dark, light, type Palette } from '@/theme/tokens';

test('light and dark text roles meet WCAG AA on their surfaces', () => {
  assertTextContrast(light);
  assertTextContrast(dark);
});

function assertTextContrast(palette: Palette) {
  expect(contrast(palette.textPrimary, palette.background)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.textPrimary, palette.surface)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.textSecondary, palette.background)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.textSecondary, palette.surface)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.disabledText, palette.disabledSurface)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.disabledText, palette.background)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.onAccent, palette.accent)).toBeGreaterThanOrEqual(4.5);
  expect(contrast(palette.danger, palette.background)).toBeGreaterThanOrEqual(4.5);
}

function contrast(foreground: string, background: string): number {
  const left = luminance(foreground);
  const right = luminance(background);
  const lighter = Math.max(left, right);
  const darker = Math.min(left, right);
  return (lighter + 0.05) / (darker + 0.05);
}

function luminance(hex: string): number {
  const value = hex.replace('#', '');
  const channels = [0, 2, 4].map((index) => {
    const channel = Number.parseInt(value.slice(index, index + 2), 16) / 255;
    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0]! + 0.7152 * channels[1]! + 0.0722 * channels[2]!;
}

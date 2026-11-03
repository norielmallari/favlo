import { buttonPaint } from '@/theme/buttons';
import { contrast, dark, light } from '@/theme/tokens';

describe('buttonPaint', () => {
  it.each(['light', 'dark'] as const)('%s danger label meets AA on the danger fill', (mode) => {
    const palette = mode === 'light' ? light : dark;
    const paint = buttonPaint(palette, 'danger');
    expect(paint.color).toBe(palette.onDanger);
    expect(paint.background).toBe(palette.danger);
    expect(contrast(paint.color, paint.background)).toBeGreaterThanOrEqual(4.5);
  });

  it.each(['light', 'dark'] as const)('%s accent label meets AA on the accent fill', (mode) => {
    const palette = mode === 'light' ? light : dark;
    const paint = buttonPaint(palette, 'accent');
    expect(contrast(paint.color, paint.background)).toBeGreaterThanOrEqual(4.5);
  });
});

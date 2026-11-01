import { scrollbarStyle } from '@/theme/scrollbar';
import { dark, light } from '@/theme/tokens';

test('scrollbars follow the light and dark theme', () => {
  const day = scrollbarStyle(light);
  expect(day).toContain('scrollbar-width: thin');
  expect(day).toContain('*::-webkit-scrollbar');
  expect(day).not.toContain('[data-testid="transcript"]');
  expect(day).toContain(light.borderStrong);
  expect(day).toContain(light.background);
  expect(day).toContain(light.textSecondary);
  expect(day).not.toContain(dark.borderStrong);

  const night = scrollbarStyle(dark);
  expect(night).toContain(dark.borderStrong);
  expect(night).toContain(dark.background);
  expect(night).not.toContain(light.borderStrong);
});

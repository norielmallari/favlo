import { transcriptScrollbar } from '@/components/scrollbar';
import { dark, light } from '@/theme/tokens';

test('the chat scrollbar follows the light and dark theme', () => {
  const day = transcriptScrollbar(light.borderStrong, light.background, light.textSecondary);
  expect(day).toContain('scrollbar-width: thin');
  expect(day).toContain(light.borderStrong);
  expect(day).toContain(light.background);
  expect(day).toContain(light.textSecondary);
  expect(day).not.toContain(dark.borderStrong);

  const night = transcriptScrollbar(dark.borderStrong, dark.background, dark.textSecondary);
  expect(night).toContain(dark.borderStrong);
  expect(night).toContain(dark.background);
  expect(night).not.toContain(light.borderStrong);
});

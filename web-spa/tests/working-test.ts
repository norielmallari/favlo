import { workingShimmer } from '@/components/working';
import { dark, light } from '@/theme/tokens';

test('working text sweeps a light band and stays still when motion is reduced', () => {
  const day = workingShimmer(light.textSecondary);
  expect(day).toContain('@keyframes favlo-working-shimmer');
  expect(day).toContain('animation: favlo-working-shimmer 2s linear infinite');
  expect(day).toContain(light.textSecondary);
  expect(day).toContain('color-mix(in srgb,');
  expect(day).toContain('prefers-reduced-motion: reduce');
  expect(day).toContain('animation: none');
  expect(day).not.toContain(dark.textSecondary);

  const night = workingShimmer(dark.textSecondary);
  expect(night).toContain(dark.textSecondary);
  expect(night).not.toContain(light.textSecondary);
});

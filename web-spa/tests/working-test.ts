import { workingActivity } from '@/components/working';
import { dark, light } from '@/theme/tokens';

test('working activity pulses the dot with ease-in-out and stays still when motion is reduced', () => {
  const day = workingActivity(light.accent);
  expect(day).toContain('@keyframes favlo-working-pulse');
  expect(day).toContain('animation: favlo-working-pulse 2.4s ease-in-out infinite');
  expect(day).toContain('[data-testid="working-dot"]');
  expect(day).toContain(light.accent);
  expect(day).toContain('prefers-reduced-motion: reduce');
  expect(day).toContain('animation: none');
  expect(day).not.toContain('background-clip');
  expect(day).not.toContain(dark.accent);

  const night = workingActivity(dark.accent);
  expect(night).toContain(dark.accent);
  expect(night).not.toContain(light.accent);
});

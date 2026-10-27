import { render, userEvent } from '@testing-library/react-native';

import { Frame, nextAppearance } from '@/components/Frame';
import { ThemeProvider } from '@/theme/Theme';

test('the header wraps instead of growing past the screen', async () => {
  const view = await render(
    <ThemeProvider>
      <Frame link={{ label: 'Configuration', onPress: () => undefined }}>{null}</Frame>
    </ThemeProvider>,
  );
  const banner = view.getByTestId('app-banner');
  const style = banner.props.style as { flexWrap?: string; maxWidth?: string };
  expect(style.flexWrap).toBe('wrap');
  expect(style.maxWidth).toBe('100%');
  expect(view.getByRole('button', { name: 'Configuration' })).toBeTruthy();
});

test('the appearance button cycles system, light, and dark', async () => {
  expect(nextAppearance('system')).toBe('light');
  expect(nextAppearance('light')).toBe('dark');
  expect(nextAppearance('dark')).toBe('system');
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <Frame link={{ label: 'Configuration', onPress: () => undefined }}>{null}</Frame>
    </ThemeProvider>,
  );
  await user.press(view.getByRole('button', { name: 'Appearance, System' }));
  expect(view.getByRole('button', { name: 'Appearance, Light' })).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Appearance, Light' }));
  expect(view.getByRole('button', { name: 'Appearance, Dark' })).toBeTruthy();
});

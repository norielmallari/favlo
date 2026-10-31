import { render, userEvent } from '@testing-library/react-native';
import { Dimensions } from 'react-native';

import { Frame } from '@/components/Frame';
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

test('appearance is a menu of system, light, and dark', async () => {
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <Frame onSkip={() => undefined} link={{ label: 'Configuration', onPress: () => undefined }}>{null}</Frame>
    </ThemeProvider>,
  );
  expect(view.getByRole('link', { name: 'Skip to message' })).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Appearance, System' }));
  await user.press(view.getByRole('menuitem', { name: 'Light' }));
  expect(view.getByRole('button', { name: 'Appearance, Light' })).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Appearance, Light' }));
  await user.press(view.getByRole('menuitem', { name: 'Dark' }));
  expect(view.getByRole('button', { name: 'Appearance, Dark' })).toBeTruthy();
});

test('the appearance menu stays inside a narrow window', async () => {
  jest.spyOn(Dimensions, 'get').mockReturnValue({ width: 390, height: 844, scale: 1, fontScale: 1 });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <Frame link={{ label: 'Configuration', onPress: () => undefined }}>{null}</Frame>
    </ThemeProvider>,
  );
  await user.press(view.getByRole('button', { name: 'Appearance, System' }));
  expect(view.getByLabelText('Appearance').props.style).toEqual(
    expect.objectContaining({ left: 0, maxWidth: 240 }),
  );
});

import { render } from '@testing-library/react-native';

import { Composer } from '@/components/Composer';
import { ThemeProvider } from '@/theme/Theme';

test('Send stays disabled until there is a message', async () => {
  const view = await render(
    <ThemeProvider>
      <Composer value="" working={false} onChange={() => undefined} onSend={() => undefined} />
    </ThemeProvider>,
  );
  expect(view.getByRole('button', { name: 'Send' }).props.accessibilityState).toMatchObject({ disabled: true });
});

test('Send becomes Stop while a reply is in progress', async () => {
  const view = await render(
    <ThemeProvider>
      <Composer value="" working onChange={() => undefined} onSend={() => undefined} onStop={() => undefined} />
    </ThemeProvider>,
  );
  expect(view.getByRole('button', { name: 'Stop' })).toBeTruthy();
  expect(view.queryByRole('button', { name: 'Send' })).toBeNull();
});

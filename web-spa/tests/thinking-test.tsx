import { render, userEvent, waitFor } from '@testing-library/react-native';

import { showsMessage } from '@/components/ChatScreen';
import { Thinking } from '@/components/Thinking';
import { ThemeProvider } from '@/theme/Theme';

function Host({ settled }: { settled: boolean }) {
  return (
    <ThemeProvider>
      <Thinking text="Looking at the question." settled={settled} />
    </ThemeProvider>
  );
}

test('a reasoning-only message stays on screen', () => {
  expect(showsMessage(undefined, 'User handed over carrots.')).toBe(true);
  expect(showsMessage('', 'User handed over carrots.')).toBe(true);
  expect(showsMessage('', '  ')).toBe(false);
  expect(showsMessage(undefined, undefined)).toBe(false);
});

test('thinking starts open and closes when the reply is stored', async () => {
  const user = userEvent.setup();
  const view = await render(<Host settled={false} />);
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: true });
  expect(view.getByText('Looking at the question.')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.queryByText('Looking at the question.')).toBeNull();
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: false });
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.getByText('Looking at the question.')).toBeTruthy();
  view.rerender(<Host settled />);
  await waitFor(() => {
    expect(view.queryByText('Looking at the question.')).toBeNull();
  });
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: false });
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.getByText('Looking at the question.')).toBeTruthy();
});

test('a stored reply keeps thinking closed', async () => {
  const user = userEvent.setup();
  const view = await render(<Host settled />);
  expect(view.queryByText('Looking at the question.')).toBeNull();
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.getByText('Looking at the question.')).toBeTruthy();
});

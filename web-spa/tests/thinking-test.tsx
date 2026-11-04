import { render, userEvent, waitFor } from '@testing-library/react-native';

import { showsMessage } from '@/components/ChatScreen';
import { type ExpandTrace } from '@/components/expandTrace';
import { Thinking } from '@/components/Thinking';
import { ThemeProvider } from '@/theme/Theme';

function Host({ settled, expandTrace }: { settled: boolean; expandTrace?: ExpandTrace }) {
  return (
    <ThemeProvider>
      <Thinking text="Looking at the question." settled={settled} expandTrace={expandTrace} />
    </ThemeProvider>
  );
}

test('a reasoning-only message stays on screen', () => {
  expect(showsMessage(undefined, 'User handed over carrots.')).toBe(true);
  expect(showsMessage('', 'User handed over carrots.')).toBe(true);
  expect(showsMessage('', '  ')).toBe(false);
  expect(showsMessage(undefined, undefined)).toBe(false);
});

test('collapsed thinking stays closed until it is opened', async () => {
  const user = userEvent.setup();
  const view = await render(<Host settled={false} />);
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: false });
  expect(view.queryByText('Looking at the question.')).toBeNull();
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.getByText('Looking at the question.')).toBeTruthy();
});

test('thinking opens while a reply is running and closes when it is stored', async () => {
  const user = userEvent.setup();
  const view = await render(<Host settled={false} expandTrace="running" />);
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: true });
  expect(view.getByText('Looking at the question.')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.queryByText('Looking at the question.')).toBeNull();
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: false });
  await user.press(view.getByRole('button', { name: 'Thinking' }));
  expect(view.getByText('Looking at the question.')).toBeTruthy();
  view.rerender(<Host settled expandTrace="running" />);
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

import { render, userEvent, within } from '@testing-library/react-native';
import { Dimensions, StyleSheet } from 'react-native';

import { ChatScreen } from '@/components/ChatScreen';
import { EmptyState, STARTERS } from '@/components/EmptyState';
import { ThemeProvider } from '@/theme/Theme';

jest.mock('expo-router', () => ({
  useRouter: () => ({ push: jest.fn() }),
  useFocusEffect: () => undefined,
}));

test('an empty conversation offers starters and hides workspaces', async () => {
  jest.spyOn(Dimensions, 'get').mockReturnValue({ width: 1280, height: 800, scale: 1, fontScale: 1 });
  mockChat();
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <ChatScreen />
    </ThemeProvider>,
  );
  expect(await view.findByRole('button', { name: 'New agent' })).toBeTruthy();
  expect(await view.findByText('Chat, images, music, and voice on this machine.')).toBeTruthy();
  expect(view.getByRole('button', { name: 'Generate an image' })).toBeTruthy();
  expect(view.queryByLabelText('Workspaces')).toBeNull();
  expect(view.getByTestId('selected-mark')).toBeTruthy();
  expect(within(view.getByTestId('reading-column')).getByLabelText('Message')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Generate an image' }));
  expect(view.getByLabelText('Message').props.value).toBe(STARTERS[0].fill);
});

test('a compact empty conversation uses a channels drawer', async () => {
  jest.spyOn(Dimensions, 'get').mockReturnValue({ width: 390, height: 844, scale: 1, fontScale: 1 });
  mockChat();
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <ChatScreen />
    </ThemeProvider>,
  );
  expect(await view.findByRole('button', { name: 'Channels' })).toBeTruthy();
  expect(view.queryByRole('button', { name: 'New agent' })).toBeNull();
  expect(view.getByRole('button', { name: 'Generate an image' })).toBeTruthy();
  expect(pressedStyle(view, 'Generate an image')).toEqual(expect.objectContaining({ alignSelf: 'stretch' }));
  expect(within(view.getByTestId('reading-column')).getByLabelText('Message')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Channels' }));
  expect(view.getByRole('button', { name: 'New agent' })).toBeTruthy();
});

test('starters stack on a compact empty conversation', async () => {
  const view = await render(
    <ThemeProvider>
      <EmptyState name="Favlo" compact onFill={() => undefined} />
    </ThemeProvider>,
  );
  expect(view.getByText('Favlo')).toBeTruthy();
  expect(pressedStyle(view, 'List registered models')).toEqual(expect.objectContaining({ alignSelf: 'stretch' }));
});

function pressedStyle(view: ReturnType<typeof render>, name: string) {
  const style = view.getByRole('button', { name }).props.style;
  return StyleSheet.flatten(typeof style === 'function' ? style({ pressed: false }) : style);
}

function mockChat() {
  globalThis.fetch = jest.fn(async (input: RequestInfo | URL) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/channels') {
      return json({
        channels: [{ id: 'channel_favlo', name: 'Favlo', agent_id: 'agent_favlo', session_id: 'session_favlo' }],
      });
    }
    if (path === '/settings') {
      return json({ default_model_id: 'model_local', exchange_budget: 10, revision: 1 });
    }
    if (path === '/sessions/session_favlo') {
      return json({ messages: [], runs: [], cursor: 0 });
    }
    return json({});
  }) as typeof fetch;
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

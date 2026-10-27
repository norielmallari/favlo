import { render } from '@testing-library/react-native';
import { Dimensions } from 'react-native';

import { ChatScreen } from '@/components/ChatScreen';
import { ThemeProvider } from '@/theme/Theme';

jest.mock('expo-router', () => ({
  useRouter: () => ({ push: jest.fn() }),
  useFocusEffect: () => undefined,
}));

test('the channels column has no width control', async () => {
  jest.spyOn(Dimensions, 'get').mockReturnValue({ width: 1280, height: 800, scale: 1, fontScale: 1 });
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
  const view = await render(
    <ThemeProvider>
      <ChatScreen />
    </ThemeProvider>,
  );
  expect(await view.findByRole('button', { name: 'New agent' })).toBeTruthy();
  expect(view.queryByText('Width')).toBeNull();
  expect(view.queryByLabelText('Channels width')).toBeNull();
});

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

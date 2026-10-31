import { render, userEvent } from '@testing-library/react-native';
import { Dimensions } from 'react-native';

import { SettingsScreen } from '@/components/SettingsScreen';
import { ThemeProvider } from '@/theme/Theme';

jest.mock('expo-router', () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

const agent = {
  id: 'agent_favlo',
  name: 'Favlo',
  persona: '',
  default_model_id: null,
  revision: 1,
  tool_groups: [],
};

describe('heartbeats configuration', () => {
  beforeEach(() => {
    jest.spyOn(Dimensions, 'get').mockReturnValue({
      width: 1280,
      height: 800,
      scale: 1,
      fontScale: 1,
    });
    globalThis.fetch = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input).replace('http://127.0.0.1:8130', '');
      const method = init?.method ?? 'GET';
      if (path === '/providers') {
        return json({ providers: [] });
      }
      if (path === '/settings') {
        return json({ default_model_id: null, exchange_budget: 10, revision: 1 });
      }
      if (path === '/agents') {
        return json({ agents: [agent] });
      }
      if (path === '/tool-groups') {
        return json({ groups: [] });
      }
      if (path === '/heartbeats' && method === 'GET') {
        return json({ heartbeats: [] });
      }
      if (path === '/mcp-servers') {
        return json({ servers: [] });
      }
      if (path === '/agent-skills') {
        return json({ skills: [] });
      }
      if (path === '/global-skills') {
        return json({ skills: [] });
      }
      if (path === '/workspaces') {
        return json({ workspaces: [] });
      }
      return json({ message: 'The request failed.' }, 404);
    }) as typeof fetch;
  });

  test('shows agents, then heartbeats, then the heartbeat detail', async () => {
    const view = await render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    const user = userEvent.setup();
    await user.press(await view.findByRole('tab', { name: 'Heartbeats' }));
    expect(view.getByLabelText('Agents list')).toBeTruthy();
    expect(view.getByLabelText('Heartbeats list')).toBeTruthy();
    expect(view.getByRole('button', { name: 'New heartbeat' })).toBeTruthy();
    expect(view.getByText('Choose a heartbeat.')).toBeTruthy();
  });
});

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

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

describe('skills configuration', () => {
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
      if (path === '/heartbeats') {
        return json({ heartbeats: [] });
      }
      if (path === '/mcp-servers') {
        return json({ servers: [] });
      }
      if (path === '/agent-skills' && method === 'GET') {
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

  test('shows global skills, and agent skills under the agent', async () => {
    const view = await render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    const user = userEvent.setup();
    expect(await view.findByLabelText('Agent skills')).toBeTruthy();
    expect(view.getByText('No skills')).toBeTruthy();
    await user.press(view.getByRole('tab', { name: 'Skills' }));
    expect(view.getByLabelText('Skills list')).toBeTruthy();
    expect(view.queryByLabelText('Agents list')).toBeNull();
    expect(view.queryByLabelText('Agent skills')).toBeNull();
    expect(view.getByRole('button', { name: 'New skill' })).toBeTruthy();
    expect(view.getByText('Choose a skill.')).toBeTruthy();
    await user.press(view.getByRole('button', { name: 'New skill' }));
    expect(view.getByRole('switch', { name: 'Add to new agents' })).toBeTruthy();
  });
});

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

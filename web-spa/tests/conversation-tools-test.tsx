import { render } from '@testing-library/react-native';
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
  tool_groups: ['favlo-config'],
};

describe('conversation tools switch', () => {
  beforeEach(() => {
    jest.spyOn(Dimensions, 'get').mockReturnValue({
      width: 1280,
      height: 800,
      scale: 1,
      fontScale: 1,
    });
    globalThis.fetch = jest.fn(async (input: RequestInfo | URL) => {
      const path = String(input).replace('http://127.0.0.1:8130', '');
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
        return json({
          groups: [
            { id: 'favlo-config', name: 'Configuration tools' },
            { id: 'conversation', name: 'Conversation tools' },
          ],
        });
      }
      if (path === '/heartbeats') {
        return json({ heartbeats: [] });
      }
      if (path === '/mcp-servers') {
        return json({ servers: [] });
      }
      return json({ message: 'The request failed.' }, 404);
    }) as typeof fetch;
  });

  test('shows one switch for the conversation tool group', async () => {
    const view = await render(
      <ThemeProvider>
        <SettingsScreen />
      </ThemeProvider>,
    );
    const conversation = await view.findByRole('switch', { name: 'Conversation tools for Favlo' });
    const configuration = view.getByRole('switch', { name: 'Configuration tools for Favlo' });
    expect(conversation.props.accessibilityState.checked).toBe(false);
    expect(configuration.props.accessibilityState.checked).toBe(true);
  });
});

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

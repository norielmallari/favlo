import { fireEvent, render, userEvent } from '@testing-library/react-native';
import { Dimensions } from 'react-native';

import { AgentForm, SettingsScreen } from '@/components/SettingsScreen';
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

const providers = [
  {
    id: 'provider_local',
    name: 'Favlo',
    base_url: 'http://127.0.0.1:8129/v1',
    models: [
      {
        id: 'model_local',
        provider_id: 'provider_local',
        model_id: 'example-model',
        display_name: 'Example',
        enabled: true,
        revision: 1,
      },
    ],
  },
];

const groups = [
  { id: 'favlo-config', name: 'Configuration tools' },
  { id: 'conversation', name: 'Conversation tools' },
  { id: 'heartbeats', name: 'Heartbeat tools' },
];

test('a new agent shows the same options as edit', async () => {
  jest.spyOn(Dimensions, 'get').mockReturnValue({ width: 1280, height: 800, scale: 1, fontScale: 1 });
  const posted: { body?: { name?: string; default_model_id?: string | null; tool_groups?: string[] } } = {};
  globalThis.fetch = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    const method = init?.method ?? 'GET';
    if (path === '/providers') {
      return json({ providers });
    }
    if (path === '/settings') {
      return json({ default_model_id: 'model_local', exchange_budget: 10, revision: 1 });
    }
    if (path === '/agents' && method === 'POST') {
      posted.body = JSON.parse(String(init?.body)) as typeof posted.body;
      return json({
        agent: {
          id: 'agent_reed',
          name: posted.body?.name,
          persona: '',
          default_model_id: posted.body?.default_model_id ?? null,
          revision: 1,
          tool_groups: posted.body?.tool_groups ?? [],
        },
      });
    }
    if (path === '/agents') {
      return json({ agents: [agent] });
    }
    if (path === '/tool-groups') {
      return json({ groups });
    }
    if (path === '/heartbeats') {
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
  Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: () => 'key' }, configurable: true });

  const view = await render(
    <ThemeProvider>
      <SettingsScreen />
    </ThemeProvider>,
  );
  const user = userEvent.setup();
  await user.press(await view.findByRole('button', { name: 'New agent' }));
  expect(view.getByLabelText('Agent name')).toBeTruthy();
  expect(view.getByLabelText('Persona')).toBeTruthy();
  expect(view.getByLabelText('Provider')).toBeTruthy();
  expect(view.getByLabelText('Model')).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Configuration tools' })).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Conversation tools' })).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Heartbeat tools' })).toBeTruthy();
  expect(view.queryByLabelText('Provider for Favlo')).toBeNull();
  fireEvent(view.getByRole('switch', { name: 'Conversation tools' }), 'valueChange', true);
  await user.type(view.getByLabelText('Agent name'), 'Reed');
  await user.press(view.getByRole('button', { name: 'Create agent' }));
  expect(await view.findByText('Reed was created.')).toBeTruthy();
  expect(posted.body?.name).toBe('Reed');
  expect(posted.body?.default_model_id).toBe('model_local');
  expect(posted.body?.tool_groups).toEqual(['conversation']);
});

test('delete removes that agent', async () => {
  const deleted: string[] = [];
  globalThis.fetch = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if ((init?.method ?? 'GET') === 'DELETE') {
      deleted.push(String(input));
      return json({ id: 'agent_reed', name: 'Reed' });
    }
    return json({});
  }) as typeof fetch;
  const onDeleted = jest.fn();
  const reed = {
    id: 'agent_reed',
    name: 'Reed',
    persona: '',
    default_model_id: null,
    revision: 1,
    tool_groups: [] as string[],
  };
  const view = await render(
    <ThemeProvider>
      <AgentForm
        agent={reed}
        groups={groups}
        providers={providers}
        settings={{ default_model_id: 'model_local', exchange_budget: 10, revision: 1 }}
        onSuccess={jest.fn()}
        onSaved={jest.fn()}
        onDeleted={onDeleted}
      />
    </ThemeProvider>,
  );
  const user = userEvent.setup();
  await user.press(view.getByRole('button', { name: 'Delete Reed' }));
  expect(deleted).toEqual(['http://127.0.0.1:8130/agents/agent_reed']);
  expect(onDeleted).toHaveBeenCalledWith(reed);
});

test('Favlo has no delete button', async () => {
  const view = await render(
    <ThemeProvider>
      <AgentForm
        agent={agent}
        groups={groups}
        providers={providers}
        settings={{ default_model_id: 'model_local', exchange_budget: 10, revision: 1 }}
        onSuccess={jest.fn()}
        onSaved={jest.fn()}
      />
    </ThemeProvider>,
  );
  expect(view.queryByRole('button', { name: 'Delete Favlo' })).toBeNull();
});

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

import { act, fireEvent, render, userEvent } from '@testing-library/react-native';
import { Dimensions } from 'react-native';

import { AgentForm, SettingsScreen } from '@/components/SettingsScreen';
import { ThemeProvider } from '@/theme/Theme';
import { FAVLO_AGENT_ID, LOCAL_PROVIDER_ID } from '@/api/seed-ids';

jest.mock('expo-router', () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

const agent = {
  id: FAVLO_AGENT_ID,
  name: 'Favlo',
  persona: '',
  default_model_id: null,
  revision: 1,
  tool_groups: ['favlo-config'],
};

const providers = [
  {
    id: LOCAL_PROVIDER_ID,
    name: 'Favlo',
    base_url: 'http://127.0.0.1:8129/v1',
    models: [
      {
        id: 'model_local',
        provider_id: LOCAL_PROVIDER_ID,
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
  const posted: {
    body?: {
      name?: string;
      default_model_id?: string | null;
      tool_groups?: string[];
      enable_thinking?: boolean | null;
      preserve_thinking?: boolean | null;
      reasoning_effort?: string | null;
    };
  } = {};
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
  expect(view.queryByLabelText('Model')).toBeNull();
  expect(view.queryByRole('button', { name: 'Refresh models' })).toBeNull();
  expect(view.getByText('Uses the default provider and model from Configuration → Providers.')).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Configuration tools' })).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Conversation tools' })).toBeTruthy();
  expect(view.getByRole('switch', { name: 'Heartbeat tools' })).toBeTruthy();
  expect(view.queryByLabelText('Provider for Favlo')).toBeNull();
  await act(() => {
    fireEvent(view.getByRole('switch', { name: 'Conversation tools' }), 'valueChange', true);
    fireEvent(view.getByLabelText('Preserve thinking'), 'valueChange', 'false');
    fireEvent(view.getByLabelText('Reasoning effort'), 'valueChange', 'high');
  });
  await user.type(view.getByLabelText('Agent name'), 'Reed');
  await user.press(view.getByRole('button', { name: 'Create agent' }));
  expect(await view.findByText('Reed was created.')).toBeTruthy();
  expect(posted.body?.name).toBe('Reed');
  expect(posted.body?.default_model_id).toBeNull();
  expect(posted.body?.tool_groups).toEqual(['conversation']);
  expect(posted.body?.enable_thinking).toBeNull();
  expect(posted.body?.preserve_thinking).toBe(false);
  expect(posted.body?.reasoning_effort).toBe('high');
  expect(view.queryByLabelText('Thinking')).toBeNull();
});

test('unset provider hides model controls; picking a provider shows them', async () => {
  const view = await render(
    <ThemeProvider>
      <AgentForm
        agent={agent}
        groups={groups}
        providers={providers}
        onSuccess={jest.fn()}
        onSaved={jest.fn()}
      />
    </ThemeProvider>,
  );
  expect(view.getByLabelText('Provider for Favlo')).toBeTruthy();
  expect(view.queryByLabelText('Model for Favlo')).toBeNull();
  expect(view.queryByRole('button', { name: 'Refresh models' })).toBeNull();
  expect(view.getByText('Uses the default provider and model from Configuration → Providers.')).toBeTruthy();
  await act(() => {
    fireEvent(view.getByLabelText('Provider for Favlo'), 'valueChange', LOCAL_PROVIDER_ID);
  });
  expect(view.getByLabelText('Model for Favlo')).toBeTruthy();
  expect(view.getByRole('button', { name: 'Refresh models' })).toBeTruthy();
  expect(view.queryByText('Uses the default provider and model from Configuration → Providers.')).toBeNull();
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

test('refresh loads the models the server lists', async () => {
  const posted: string[] = [];
  globalThis.fetch = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if ((init?.method ?? 'GET') === 'POST') {
      posted.push(String(input));
      return json({
        message: 'Models refreshed',
        models: [
          ...providers[0].models,
          {
            id: 'model_gemma',
            provider_id: LOCAL_PROVIDER_ID,
            model_id: 'gemma4-26b-a4b:instruct',
            display_name: 'Gemma 4',
            enabled: true,
            revision: 1,
          },
        ],
      });
    }
    return json({});
  }) as typeof fetch;
  const onModels = jest.fn();
  const view = await render(
    <ThemeProvider>
      <AgentForm
        agent={null}
        groups={groups}
        providers={providers}
        onSuccess={jest.fn()}
        onSaved={jest.fn()}
        onModels={onModels}
      />
    </ThemeProvider>,
  );
  const user = userEvent.setup();
  await act(() => {
    fireEvent(view.getByLabelText('Provider'), 'valueChange', LOCAL_PROVIDER_ID);
  });
  await user.press(view.getByRole('button', { name: 'Refresh models' }));
  expect(posted).toEqual(['http://127.0.0.1:8130/providers/' + LOCAL_PROVIDER_ID + '/refresh']);
  expect(await view.findByText('Models refreshed')).toBeTruthy();
  expect(onModels).toHaveBeenCalledWith(LOCAL_PROVIDER_ID, expect.arrayContaining([
    expect.objectContaining({ model_id: 'gemma4-26b-a4b:instruct' }),
  ]));
});

test('Favlo has no delete button', async () => {
  const view = await render(
    <ThemeProvider>
      <AgentForm
        agent={agent}
        groups={groups}
        providers={providers}
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

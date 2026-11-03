import { render, userEvent, waitFor } from '@testing-library/react-native';

import { ThemeProvider } from '@/theme/Theme';
import { WorkspaceScreen } from '@/workspace/WorkspaceScreen';
import { filmstripPaths, initialGallery, selectPath } from '@/workspace/placement';

test('initial gallery has preview and an empty filmstrip', () => {
  const state = initialGallery();
  expect(state.favlo_workspace.elements.map((item) => item.element)).toEqual(['preview', 'filmstrip']);
  expect(filmstripPaths(state)).toEqual([]);
  expect(state.favlo_workspace.selection).toBeNull();
});

test('selectPath updates selection only', () => {
  const next = selectPath(initialGallery(), 'Albums/zebra.png');
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'Albums/zebra.png' });
  expect(filmstripPaths(next)).toEqual([]);
});

test('folder setup binds a local folder then shows the empty strip', async () => {
  const fetchMock = mockWorkspace();
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <WorkspaceScreen workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  expect(await view.findByText('Choose a local folder')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Choose folder' }));
  await waitFor(() => expect(view.getByLabelText('Folder location').props.value).toBe('/tmp/gallery'));
  await user.press(view.getByRole('button', { name: 'Save folder' }));
  await waitFor(() => expect(view.getByText('Ask the agent to fill the filmstrip')).toBeTruthy());
  expect(view.getByText('No media selected')).toBeTruthy();
  expect(fetchMock).toHaveBeenCalled();
});

function mockWorkspace() {
  let bound = false;
  const fetchMock = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    const method = (init?.method ?? 'GET').toUpperCase();
    if (path === '/workspaces/workspace_1') {
      return json({
        profile: 'local',
        folder_kinds: [],
        workspace: {
          id: 'workspace_1',
          name: 'Stills',
          type: 'gallery',
          session_id: 'session_1',
          channel_id: 'channel_1',
          skills: [],
        },
      });
    }
    if (path === '/workspaces/workspace_1/folders/pick' && method === 'POST') {
      return json({ cancelled: false, location: '/tmp/gallery' });
    }
    if (path === '/workspaces/workspace_1/folders' && method === 'POST') {
      bound = true;
      return json({ id: 'folder_1', kind: 'local', location: '/tmp/gallery', name: 'Library' }, 201);
    }
    if (path.startsWith('/workspaces/workspace_1/folders')) {
      return json({ folders: bound ? [{ id: 'folder_1', kind: 'local' }] : [] });
    }
    if (path === '/sessions/session_1') {
      return json({ messages: [], runs: [], cursor: 0, state: null });
    }
    return json({});
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

function json(body: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response);
}

import { render, userEvent, waitFor } from '@testing-library/react-native';

import { ThemeProvider } from '@/theme/Theme';
import { Preview, previewActionsInPane } from '@/workspace/elements/Preview';
import { WorkspaceScreen } from '@/workspace/WorkspaceScreen';
import { documentFromShell, filmstripPaths, selectPath, type ShellDocument } from '@/workspace/placement';

const galleryShell: ShellDocument = {
  id: 'gallery',
  slots: ['stage', 'rail'],
  elements: [
    { name: 'preview', slot: 'stage', initial_props: { fit: 'contain' } },
    { name: 'filmstrip', slot: 'rail', initial_props: { title: '', items: [] } },
  ],
};

test('the shell document starts with preview and an empty filmstrip', () => {
  const state = documentFromShell(galleryShell);
  expect(state.favlo_workspace.elements.map((item) => item.element)).toEqual(['preview', 'filmstrip']);
  expect(filmstripPaths(state)).toEqual([]);
  expect(state.favlo_workspace.selection).toBeNull();
});

test('selectPath updates selection only', () => {
  const next = selectPath(documentFromShell(galleryShell), 'Albums/zebra.png');
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'Albums/zebra.png' });
  expect(filmstripPaths(next)).toEqual([]);
});

test('preview actions sit in the pane only on a wide web window', () => {
  expect(previewActionsInPane('web', 1280)).toBe(true);
  expect(previewActionsInPane('web', 1024)).toBe(true);
  expect(previewActionsInPane('web', 1023)).toBe(false);
  expect(previewActionsInPane('web', 390)).toBe(false);
  expect(previewActionsInPane('ios', 1280)).toBe(false);
});

test('the preview shows the file name under the image', async () => {
  const view = await render(
    <ThemeProvider>
      <Preview
        props={{ fit: 'contain' }}
        selection={{ kind: 'file', path: 'Albums/zebra.png' }}
        fileUrl={(item) => `http://127.0.0.1:8130/files/${item}`}
        onSelect={() => undefined}
        onAddToChat={() => undefined}
        onRequestDelete={() => undefined}
      />
    </ThemeProvider>,
  );
  expect(view.getByText('zebra.png')).toBeTruthy();
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
  await waitFor(() => expect(view.getByText('No images in this folder')).toBeTruthy());
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
          provisioning_state: 'provisioned',
          shell: {
            id: 'gallery',
            slots: ['stage', 'rail'],
            elements: [
              { name: 'preview', slot: 'stage', initial_props: { fit: 'contain' } },
              { name: 'filmstrip', slot: 'rail', initial_props: { title: '', items: [] } },
            ],
          },
        },
      });
    }
    if (path === '/folders/pick' && method === 'POST') {
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

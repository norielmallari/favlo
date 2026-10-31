import { render, userEvent, waitFor } from '@testing-library/react-native';

import { GalleryWorkspace, groupedKinds } from '@/components/GalleryWorkspace';
import { ThemeProvider } from '@/theme/Theme';

const KINDS = [
  { kind: 'local', enabled: true },
  { kind: 'cloud', provider: 'onedrive', enabled: false },
  { kind: 'cloud', provider: 'dropbox', enabled: false },
];

test('the root form shows disabled cloud without subtypes', async () => {
  mockGallery({ root: null, images: [] });
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  expect(await view.findByLabelText('Gallery root')).toBeTruthy();
  expect(view.getByText('Folder type')).toBeTruthy();
  expect(view.getByLabelText('Folder type')).toBeTruthy();
  expect(view.queryByRole('radio', { name: 'Local' })).toBeNull();
  expect(view.queryByRole('radio', { name: 'Cloud' })).toBeNull();
  expect(view.queryByText('Cloud provider')).toBeNull();
  expect(view.queryByRole('radio', { name: 'OneDrive' })).toBeNull();
  expect(view.queryByRole('radio', { name: 'Dropbox' })).toBeNull();
  expect(view.getByRole('button', { name: 'Choose folder' })).toBeTruthy();
  expect(view.getByRole('button', { name: 'Save root' }).props.accessibilityState).toEqual(
    expect.objectContaining({ disabled: true }),
  );
});

test('folder kinds keep cloud subtypes for a later switch', () => {
  const groups = groupedKinds(KINDS);
  const cloud = groups.find((item) => item.kind === 'cloud');
  expect(cloud?.enabled).toBe(false);
  expect(cloud?.providers.map((item) => item.provider)).toEqual(['onedrive', 'dropbox']);
});

test('choose folder fills the location from this machine', async () => {
  const fetchMock = mockGallery({ root: null, images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'Choose folder' });
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/workspaces/workspace_1/root/pick' && init?.method === 'POST') {
      return json({ cancelled: false, location: '/tmp/gallery' });
    }
    return galleryResponse(path, { root: null, images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Choose folder' }));
  await waitFor(() => {
    expect(view.getByLabelText('Location').props.value).toBe('/tmp/gallery');
  });
});

test('saving a local root opens the grid', async () => {
  const fetchMock = mockGallery({ root: null, images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByLabelText('Location');
  await user.type(view.getByLabelText('Location'), '/tmp/gallery');
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/workspaces/workspace_1/root' && init?.method === 'PUT') {
      return json({
        id: 'folder_1',
        name: 'Stills',
        kind: 'local',
        provider: null,
        location: '/tmp/gallery',
        revision: 1,
      });
    }
    return galleryResponse(path, { root: localRoot(), images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Save root' }));
  expect(await view.findByLabelText('Gallery grid')).toBeTruthy();
  expect(view.getByText('No images in this folder')).toBeTruthy();
});

test('a workspace can add a skill by name and description', async () => {
  const fetchMock = mockGallery({ root: null, images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'Skills' });
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/workspaces/workspace_1/skills' && init?.method === 'POST') {
      const body = JSON.parse(String(init.body)) as { name: string; description: string };
      return json({
        id: 'skill_1',
        workspace_id: 'workspace_1',
        name: body.name,
        description: body.description,
        revision: 1,
      });
    }
    return galleryResponse(path, { root: null, images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Skills' }));
  expect(await view.findByText('No skills')).toBeTruthy();
  await user.type(view.getByLabelText('Skill name'), 'Photo captions');
  await user.type(view.getByLabelText('Skill description'), 'Write a one-line caption.');
  await user.press(view.getByRole('button', { name: 'Add skill' }));
  await waitFor(() => {
    expect(view.getByText('Photo captions')).toBeTruthy();
  });
  expect(view.getByText('Write a one-line caption.')).toBeTruthy();
  expect(view.queryByText('No skills')).toBeNull();
});

test('add to chat inserts the relative path', async () => {
  mockGallery({
    root: localRoot(),
    images: [{ path: 'horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null }],
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  expect(await view.findByRole('button', { name: 'horse.png' })).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Add to chat' }));
  await waitFor(() => {
    expect(view.getByLabelText('Message').props.value).toBe('horse.png');
  });
});

function localRoot() {
  return {
    id: 'folder_1',
    name: 'Stills',
    kind: 'local',
    provider: null,
    location: '/tmp/gallery',
    revision: 1,
  };
}

function mockGallery(options: { root: ReturnType<typeof localRoot> | null; images: object[] }) {
  const fetchMock = jest.fn(async (input: RequestInfo | URL) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    return galleryResponse(path, options);
  });
  globalThis.fetch = fetchMock as typeof fetch;
  return fetchMock;
}

function galleryResponse(
  path: string,
  options: { root: ReturnType<typeof localRoot> | null; images: object[] },
) {
  if (path === '/workspaces/workspace_1') {
    return json({
      profile: 'local',
      folder_kinds: KINDS,
      workspace: {
        id: 'workspace_1',
        name: 'Stills',
        type: 'gallery',
        session_id: 'session_ws',
        channel_id: 'channel_favlo',
        root: options.root,
        skills: [],
      },
    });
  }
  if (path.startsWith('/workspaces/workspace_1/images')) {
    return json({ path: '', folders: [], images: options.images });
  }
  if (path === '/workspaces/workspace_1/session') {
    return json({ messages: [], runs: [], cursor: 0, workspace: { id: 'workspace_1', root: options.root } });
  }
  return json({});
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

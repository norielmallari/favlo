import { render, userEvent, waitFor } from '@testing-library/react-native';

import { crossedKind, GalleryWorkspace, groupedKinds } from '@/components/GalleryWorkspace';
import { ThemeProvider } from '@/theme/Theme';

const KINDS = [
  { kind: 'local', enabled: true },
  { kind: 'cloud', provider: 'onedrive', enabled: false },
  { kind: 'cloud', provider: 'dropbox', enabled: false },
];

test('the grid is the explorer and a new folder asks for a directory', async () => {
  mockGallery({ folders: [], images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  expect(await view.findByLabelText('Gallery')).toBeTruthy();
  expect(view.getByText('No folders')).toBeTruthy();
  expect(view.queryByLabelText('Folders')).toBeNull();
  await user.press(view.getByRole('button', { name: 'New folder' }));
  expect(view.getByLabelText('New folder')).toBeTruthy();
  expect(view.getByLabelText('Folder type')).toBeTruthy();
  expect(view.queryByText('Cloud provider')).toBeNull();
  expect(view.getByRole('button', { name: 'Choose folder' })).toBeTruthy();
  expect(view.getByRole('button', { name: 'Save folder' }).props.accessibilityState).toEqual(
    expect.objectContaining({ disabled: true }),
  );
});

test('a local item dropped on a cloud folder is refused', () => {
  expect(crossedKind('local', 'cloud')).toBe('That would upload.');
  expect(crossedKind('image', 'cloud')).toBe('That would upload.');
  expect(crossedKind('cloud', 'local')).toBe('That would download.');
  expect(crossedKind('cloud', 'root')).toBeNull();
});

test('folder kinds keep cloud subtypes for a later switch', () => {
  const groups = groupedKinds(KINDS);
  const cloud = groups.find((item) => item.kind === 'cloud');
  expect(cloud?.enabled).toBe(false);
  expect(cloud?.providers.map((item) => item.provider)).toEqual(['onedrive', 'dropbox']);
});

test('choose folder fills the location from this machine', async () => {
  const fetchMock = mockGallery({ folders: [], images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'New folder' });
  await user.press(view.getByRole('button', { name: 'New folder' }));
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/workspaces/workspace_1/folders/pick' && init?.method === 'POST') {
      return json({ cancelled: false, location: '/tmp/gallery' });
    }
    return galleryResponse(path, { folders: [], images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Choose folder' }));
  await waitFor(() => {
    expect(view.getByLabelText('Location').props.value).toBe('/tmp/gallery');
  });
});

test('saving a local folder shows its tile', async () => {
  const fetchMock = mockGallery({ folders: [], images: [] });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'New folder' });
  await user.press(view.getByRole('button', { name: 'New folder' }));
  await user.type(view.getByLabelText('Folder name'), 'Camera');
  await user.type(view.getByLabelText('Location'), '/tmp/gallery');
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path === '/workspaces/workspace_1/folders' && init?.method === 'POST') {
      return json(localFolder());
    }
    return galleryResponse(path, { folders: [localFolder()], images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Save folder' }));
  expect(await view.findByText('No images in this folder')).toBeTruthy();
  expect(view.queryByRole('button', { name: 'Camera' })).toBeNull();
});

test('a workspace can add a skill by name and description', async () => {
  const fetchMock = mockGallery({ folders: [], images: [] });
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
    if (path === '/workspaces/workspace_1/skills/skill_1' && init?.method === 'PATCH') {
      const body = JSON.parse(String(init.body)) as { name: string; description: string };
      return json({
        id: 'skill_1',
        workspace_id: 'workspace_1',
        name: body.name,
        description: body.description,
        revision: 2,
      });
    }
    return galleryResponse(path, { folders: [], images: [] });
  });
  await user.press(view.getByRole('button', { name: 'Skills' }));
  expect(await view.findByText('No skills')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'New skill' }));
  await user.type(view.getByLabelText('Skill name'), 'Photo captions');
  await user.type(view.getByLabelText('Skill description'), 'Write a one-line caption.');
  await user.press(view.getByRole('button', { name: 'Create skill' }));
  await waitFor(() => {
    expect(view.getByLabelText('Skill description').props.value).toBe('Write a one-line caption.');
  });
  await user.press(view.getByRole('button', { name: 'Back to skills' }));
  expect(view.getByRole('button', { name: 'Photo captions' })).toBeTruthy();
  expect(view.queryByText('No skills')).toBeNull();
  await user.press(view.getByRole('button', { name: 'Photo captions' }));
  await user.clear(view.getByLabelText('Skill description'));
  await user.type(view.getByLabelText('Skill description'), 'Prefer the folder palette.');
  await user.press(view.getByRole('button', { name: 'Save skill' }));
  await waitFor(() => {
    expect(view.getByLabelText('Skill description').props.value).toBe('Prefer the folder palette.');
  });
});

test('move opens a list of folders', async () => {
  mockGallery({
    folders: [localFolder()],
    images: [{ path: 'horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null }],
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'Move horse.png' });
  await user.press(view.getByRole('button', { name: 'Move horse.png' }));
  expect(view.getByLabelText('Move')).toBeTruthy();
  expect(view.getByText('No destination')).toBeTruthy();
});

test('the filmstrip stays until the move finishes', async () => {
  const image = { path: 'horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null };
  const fetchMock = mockGallery({ folders: [localFolder()], images: [image] });
  let release: (value: Response) => void = () => undefined;
  const gate = new Promise<Response>((resolve) => {
    release = resolve;
  });
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path.endsWith('/images/move') && init?.method === 'POST') {
      return gate;
    }
    if (path.startsWith('/workspaces/workspace_1/folders/folder_1/images')) {
      return json({ path: '', folders: [{ path: 'Albums', name: 'Albums', kind: 'folder' }], images: [image] });
    }
    return galleryResponse(path, { folders: [localFolder()], images: [image] });
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'Move horse.png' });
  await user.press(view.getByRole('button', { name: 'Move horse.png' }));
  await user.press(view.getByRole('button', { name: 'Albums' }));
  expect(view.getByLabelText('Move')).toBeTruthy();
  expect(view.getByRole('button', { name: 'horse.png', hidden: true })).toBeTruthy();
  release(json({ path: 'Albums/horse.png' }));
  await waitFor(() => {
    expect(view.queryByLabelText('Move')).toBeNull();
  });
});

test('delete asks, then the filmstrip waits for the file to go', async () => {
  const image = { path: 'horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null };
  const fetchMock = mockGallery({ folders: [localFolder()], images: [image] });
  let removed = false;
  let release: (value: Response) => void = () => undefined;
  const gate = new Promise<Response>((resolve) => {
    release = resolve;
  });
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path.endsWith('/images') && init?.method === 'DELETE') {
      removed = true;
      return gate;
    }
    if (path.startsWith('/workspaces/workspace_1/folders/folder_1/images')) {
      return json({ path: '', folders: [], images: removed ? [] : [image] });
    }
    return galleryResponse(path, { folders: [localFolder()], images: removed ? [] : [image] });
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await view.findByRole('button', { name: 'Delete horse.png' });
  await user.press(view.getByRole('button', { name: 'Delete horse.png' }));
  expect(view.getByLabelText('Delete image')).toBeTruthy();
  expect(view.getByText('Delete horse.png?')).toBeTruthy();
  expect(view.getByRole('button', { name: 'horse.png', hidden: true })).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Delete horse.png' }));
  expect(view.getByLabelText('Delete image')).toBeTruthy();
  release(json({ path: 'horse.png', name: 'horse.png', kind: 'image' }));
  await waitFor(() => {
    expect(view.queryByLabelText('Delete image')).toBeNull();
  });
  expect(view.queryByRole('button', { name: 'horse.png' })).toBeNull();
});

test('an empty folder can be deleted after confirmation', async () => {
  const fetchMock = mockGallery({ folders: [localFolder()], images: [] });
  let removed = false;
  let release: (value: Response) => void = () => undefined;
  const gate = new Promise<Response>((resolve) => {
    release = resolve;
  });
  fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path.endsWith('/directories') && init?.method === 'DELETE') {
      removed = true;
      return gate;
    }
    if (path.includes('/images') && path.includes('path=Albums')) {
      return json({ path: 'Albums', folders: [], images: [], empty: !removed });
    }
    if (path.startsWith('/workspaces/workspace_1/folders/folder_1/images')) {
      const albums = removed ? [] : [{ path: 'Albums', name: 'Albums', kind: 'folder' }];
      return json({ path: '', folders: albums, images: [], empty: removed });
    }
    return galleryResponse(path, { folders: [localFolder()], images: [] });
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await user.press(await view.findByRole('button', { name: 'Albums' }));
  const remove = await view.findByRole('button', { name: 'Delete Albums' });
  expect(remove.props.accessibilityState).toEqual(expect.objectContaining({ disabled: false }));
  await user.press(remove);
  expect(view.getByLabelText('Delete folder')).toBeTruthy();
  expect(view.getByText('Delete Albums?')).toBeTruthy();
  await user.press(view.getByRole('button', { name: 'Delete Albums' }));
  expect(view.getByLabelText('Delete folder')).toBeTruthy();
  release(json({ path: 'Albums' }));
  await waitFor(() => {
    expect(view.queryByLabelText('Delete folder')).toBeNull();
  });
  expect(view.queryByRole('button', { name: 'Albums' })).toBeNull();
});

test('a folder that still has an image cannot be deleted', async () => {
  const image = { path: 'Albums/horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null };
  const fetchMock = mockGallery({ folders: [localFolder()], images: [] });
  fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    if (path.includes('/images') && path.includes('path=Albums')) {
      return json({ path: 'Albums', folders: [], images: [image], empty: false });
    }
    if (path.startsWith('/workspaces/workspace_1/folders/folder_1/images')) {
      return json({ path: '', folders: [{ path: 'Albums', name: 'Albums', kind: 'folder' }], images: [], empty: false });
    }
    return galleryResponse(path, { folders: [localFolder()], images: [] });
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined} />
    </ThemeProvider>,
  );
  await user.press(await view.findByRole('button', { name: 'Albums' }));
  const remove = await view.findByRole('button', { name: 'Delete Albums' });
  expect(remove.props.accessibilityState).toEqual(expect.objectContaining({ disabled: true }));
  await user.press(remove);
  expect(view.queryByLabelText('Delete folder')).toBeNull();
});

test('add to chat inserts the relative path', async () => {
  mockGallery({
    folders: [localFolder()],
    images: [{ path: 'horse.png', name: 'horse.png', media_type: 'image/png', size: 12, revision: 'abc', note: null }],
  });
  const user = userEvent.setup();
  const view = await render(
    <ThemeProvider>
      <GalleryWorkspace workspaceId="workspace_1" agentName="Favlo" onBack={() => undefined } />
    </ThemeProvider>,
  );
  expect(await view.findByRole('button', { name: 'horse.png' })).toBeTruthy();
  expect(view.queryByRole('button', { name: 'Camera' })).toBeNull();
  await user.press(view.getByRole('button', { name: 'horse.png' }));
  await user.press(view.getByRole('button', { name: 'Add to chat' }));
  await user.press(view.getByRole('button', { name: 'Chat' }));
  await waitFor(() => {
    expect(view.getByLabelText('Message').props.value).toBe('horse.png');
  });
});

function localFolder() {
  return {
    id: 'folder_1',
    name: 'Camera',
    kind: 'local',
    provider: null,
    location: '/tmp/gallery',
    parent_id: null,
    remote_id: null,
    revision: 1,
  };
}

function mockGallery(options: { folders: ReturnType<typeof localFolder>[]; images: object[] }) {
  const fetchMock = jest.fn(async (input: RequestInfo | URL) => {
    const path = String(input).replace('http://127.0.0.1:8130', '');
    return galleryResponse(path, options);
  });
  globalThis.fetch = fetchMock as typeof fetch;
  return fetchMock;
}

function galleryResponse(
  path: string,
  options: { folders: ReturnType<typeof localFolder>[]; images: object[] },
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
        skills: [],
      },
    });
  }
  if (path === '/workspaces/workspace_1/folders') {
    return json({ folders: options.folders });
  }
  if (path.startsWith('/workspaces/workspace_1/folders/folder_1/images')) {
    return json({ path: '', folders: [], images: options.images });
  }
  if (path === '/workspaces/workspace_1/focus') {
    return json({ folder_id: 'folder_1' });
  }
  if (path === '/workspaces/workspace_1/session') {
    return json({ messages: [], runs: [], cursor: 0, workspace: { id: 'workspace_1' } });
  }
  return json({});
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
}

import { render, userEvent, within } from '@testing-library/react-native';

import { Workspaces } from '@/components/Workspaces';
import { ThemeProvider } from '@/theme/Theme';

describe('workspaces column', () => {
  test('shows an empty column and a create action', async () => {
    const onCreate = jest.fn();
    const onHome = jest.fn();
    const view = await render(
      <ThemeProvider>
        <Workspaces
          workspaces={[]}
          agentName="Favlo"
          selectedId={null}
          homeSelected
          onHome={onHome}
          onSelect={() => undefined}
          onCreate={onCreate}
          onDelete={async () => undefined}
        />
      </ThemeProvider>,
    );
    expect(view.getByRole('button', { name: 'Home' })).toBeTruthy();
    expect(view.getByText('No workspaces')).toBeTruthy();
    expect(view.getByRole('button', { name: 'New workspace for Favlo' })).toBeTruthy();
    const user = userEvent.setup();
    await user.press(view.getByRole('button', { name: 'New workspace for Favlo' }));
    expect(onCreate).toHaveBeenCalled();
    await user.press(view.getByRole('button', { name: 'Home' }));
    expect(onHome).toHaveBeenCalled();
  });

  test('selects a named gallery', async () => {
    const onSelect = jest.fn();
    const view = await render(
      <ThemeProvider>
        <Workspaces
          workspaces={[{ id: 'workspace_1', name: 'Stills', type: 'gallery', session_id: 'session_1' }]}
          agentName="Favlo"
          selectedId="workspace_1"
          homeSelected={false}
          onHome={() => undefined}
          onSelect={onSelect}
          onCreate={() => undefined}
          onDelete={async () => undefined}
        />
      </ThemeProvider>,
    );
    expect(view.getByRole('button', { name: 'Stills' }).props.accessibilityState).toEqual(
      expect.objectContaining({ selected: true }),
    );
    expect(view.getByRole('button', { name: 'Delete Stills' })).toBeTruthy();
  });

  test('asks before deleting a workspace', async () => {
    const onDelete = jest.fn(async () => undefined);
    const view = await render(
      <ThemeProvider>
        <Workspaces
          workspaces={[{ id: 'workspace_1', name: 'Stills', type: 'gallery', session_id: 'session_1' }]}
          agentName="Favlo"
          selectedId={null}
          homeSelected
          onHome={() => undefined}
          onSelect={() => undefined}
          onCreate={() => undefined}
          onDelete={onDelete}
        />
      </ThemeProvider>,
    );
    const user = userEvent.setup();
    await user.press(view.getByRole('button', { name: 'Delete Stills' }));
    expect(view.getByText('Delete Stills?')).toBeTruthy();
    expect(view.getByText('Folders and files on disk stay.')).toBeTruthy();
    const dialog = within(view.getByLabelText('Delete workspace'));
    await user.press(dialog.getByRole('button', { name: 'Delete Stills' }));
    expect(onDelete).toHaveBeenCalledWith('workspace_1');
  });
});

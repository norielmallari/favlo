import { render, userEvent } from '@testing-library/react-native';

import { Workspaces } from '@/components/Workspaces';
import { ThemeProvider } from '@/theme/Theme';

describe('workspaces column', () => {
  test('shows an empty column and a create action', async () => {
    const onCreate = jest.fn();
    const view = await render(
      <ThemeProvider>
        <Workspaces workspaces={[]} selectedId={null} onSelect={() => undefined} onCreate={onCreate} />
      </ThemeProvider>,
    );
    expect(view.getByText('No workspaces')).toBeTruthy();
    expect(view.getByRole('button', { name: 'New workspace' })).toBeTruthy();
    const user = userEvent.setup();
    await user.press(view.getByRole('button', { name: 'New workspace' }));
    expect(onCreate).toHaveBeenCalled();
  });

  test('selects a named gallery', async () => {
    const onSelect = jest.fn();
    const view = await render(
      <ThemeProvider>
        <Workspaces
          workspaces={[{ id: 'workspace_1', name: 'Stills', type: 'gallery', session_id: 'session_1', root: null }]}
          selectedId="workspace_1"
          onSelect={onSelect}
          onCreate={() => undefined}
        />
      </ThemeProvider>,
    );
    expect(view.getByRole('button', { name: 'Stills' })).toBeTruthy();
    expect(view.getByTestId('selected-mark')).toBeTruthy();
  });
});

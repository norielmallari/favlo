import { render, userEvent } from '@testing-library/react-native';

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
          selectedId={null}
          homeSelected
          onHome={onHome}
          onSelect={() => undefined}
          onCreate={onCreate}
        />
      </ThemeProvider>,
    );
    expect(view.getByRole('button', { name: 'Home' })).toBeTruthy();
    expect(view.getByText('No workspaces')).toBeTruthy();
    expect(view.getByRole('button', { name: 'New workspace' })).toBeTruthy();
    const user = userEvent.setup();
    await user.press(view.getByRole('button', { name: 'New workspace' }));
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
          selectedId="workspace_1"
          homeSelected={false}
          onHome={() => undefined}
          onSelect={onSelect}
          onCreate={() => undefined}
        />
      </ThemeProvider>,
    );
    expect(view.getByRole('button', { name: 'Stills' }).props.accessibilityState).toEqual(
      expect.objectContaining({ selected: true }),
    );
  });
});

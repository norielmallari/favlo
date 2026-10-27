import { render } from '@testing-library/react-native';

import { Workspaces } from '@/components/Workspaces';
import { ThemeProvider } from '@/theme/Theme';

describe('workspaces column', () => {
  test('shows an empty column and no workspace action', async () => {
    const view = await render(
      <ThemeProvider>
        <Workspaces />
      </ThemeProvider>,
    );
    expect(view.getByText('No workspaces')).toBeTruthy();
    expect(view.queryByRole('button', { name: 'New workspace' })).toBeNull();
  });
});

import { StyleSheet } from 'react-native';
import { render } from '@testing-library/react-native';

import { ThemeProvider } from '@/theme/Theme';
import { WorkspaceAppearanceMenu } from '@/workspace/WorkspaceAppearanceMenu';

test('the dark mode row is left aligned with System and Light', async () => {
  const view = await render(
    <ThemeProvider>
      <WorkspaceAppearanceMenu
        open
        scheme="rose-pine"
        reducedMotion
        onClose={() => undefined}
        onSelectScheme={() => undefined}
      />
    </ThemeProvider>,
  );
  for (const name of ['System', 'Light', 'Dark']) {
    const row = view.getByRole('menuitemradio', { name });
    const raw = row.props.style as ((state: { pressed: boolean }) => unknown) | object;
    const flat = StyleSheet.flatten(typeof raw === 'function' ? raw({ pressed: false }) : raw);
    expect(flat.justifyContent).toBe('flex-start');
    expect(flat.alignSelf).toBe('stretch');
  }
});

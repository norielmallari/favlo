import { render, userEvent } from '@testing-library/react-native';

import { Confirmation } from '@/components/Confirmation';
import { ThemeProvider } from '@/theme/Theme';

describe('save confirmation', () => {
  test('a successful save stays until it is dismissed', async () => {
    const user = userEvent.setup();
    let message: string | null = 'Favlo was saved.';
    const view = await render(
      <ThemeProvider>
        <Confirmation message="Favlo was saved." onDismiss={() => { message = null; }} />
      </ThemeProvider>,
    );
    expect(view.getByTestId('save-confirmation').props.accessibilityLiveRegion).toBe('polite');
    expect(view.getByText('Favlo was saved.')).toBeTruthy();
    await user.press(view.getByRole('button', { name: 'Dismiss confirmation' }));
    expect(message).toBeNull();
  });
});

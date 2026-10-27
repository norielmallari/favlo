import { render } from '@testing-library/react-native';
import { Text } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { ThemeProvider } from '@/theme/Theme';

describe('button press feedback', () => {
  test('a pressed button shows a tint', async () => {
    const view = await render(
      <ThemeProvider>
        <Pressable accessibilityRole="button" accessibilityLabel="Save" testOnly_pressed>
          <Text>Save</Text>
        </Pressable>
      </ThemeProvider>,
    );
    const tint = view.getByTestId('press-feedback');
    const style = Array.isArray(tint.props.style) ? Object.assign({}, ...tint.props.style) : tint.props.style;
    expect(style.backgroundColor).toBe('rgba(24, 34, 48, 0.14)');
  });

  test('a resting button has no tint', async () => {
    const view = await render(
      <ThemeProvider>
        <Pressable accessibilityRole="button" accessibilityLabel="Save">
          <Text>Save</Text>
        </Pressable>
      </ThemeProvider>,
    );
    expect(view.queryByTestId('press-feedback')).toBeNull();
  });
});

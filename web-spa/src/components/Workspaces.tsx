import { Text, View } from 'react-native';

import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function Workspaces() {
  const { palette } = useTheme();
  return (
    <View accessibilityLabel="Workspaces" style={{ flex: 1, padding: space[4], gap: space[2] }}>
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        Workspaces
      </Text>
      <Text style={textStyle('small', palette.textSecondary)}>No workspaces</Text>
    </View>
  );
}

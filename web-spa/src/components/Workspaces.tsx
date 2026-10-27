import { Text, View } from 'react-native';

import { useTheme } from '@/theme/Theme';

export function Workspaces() {
  const { palette } = useTheme();
  return (
    <View accessibilityLabel="Workspaces" style={{ flex: 1, padding: 16, gap: 8 }}>
      <Text accessibilityRole="header" aria-level={2} style={{ color: palette.textPrimary, fontSize: 18, lineHeight: 24, fontWeight: '600' }}>
        Workspaces
      </Text>
      <Text style={{ color: palette.textSecondary, fontSize: 14, lineHeight: 20 }}>No workspaces</Text>
    </View>
  );
}

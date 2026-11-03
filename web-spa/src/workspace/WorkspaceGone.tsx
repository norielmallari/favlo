import { Text, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function WorkspaceGone() {
  const { palette } = useTheme();
  return (
    <View
      accessibilityLabel="Workspace gone"
      style={{
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 0,
        alignItems: 'center',
        justifyContent: 'center',
        padding: space[5],
      }}
    >
      <View
        testID="workspace-gone"
        style={{
          width: '100%',
          maxWidth: 420,
          alignItems: 'center',
          gap: space[4],
          paddingVertical: space[7],
          paddingHorizontal: space[6],
          backgroundColor: palette.surface,
          borderWidth: 1,
          borderColor: palette.border,
          borderRadius: radius.dialog,
        }}
      >
        <View
          style={{
            width: 72,
            height: 72,
            borderRadius: radius.pill,
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: palette.accentSubtle,
          }}
        >
          <FeatherIcon name="x-circle" size={32} color={palette.accent} />
        </View>
        <Text
          accessibilityRole="header"
          style={[textStyle('display', palette.textPrimary), { textAlign: 'center' }]}
        >
          This workspace is gone.
        </Text>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>
          This tab can be closed. Files on disk stay where they were.
        </Text>
      </View>
    </View>
  );
}

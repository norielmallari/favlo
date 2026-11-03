import { createElement } from 'react';
import { Platform, Text, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function WorkspaceCreating({ name }: { name: string }) {
  const { palette } = useTheme();
  return (
    <View
      accessibilityLabel="Creating workspace"
      style={{
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 0,
        alignItems: 'center',
        justifyContent: 'center',
        padding: space[5],
      }}
    >
      {Platform.OS === 'web'
        ? createElement('style', { 'data-testid': 'creating-activity' }, creatingActivity())
        : null}
      <View
        testID="creating-workspace"
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
          testID="creating-mark"
          style={{
            width: 72,
            height: 72,
            borderRadius: radius.pill,
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: palette.accentSubtle,
          }}
        >
          <FeatherIcon name="layers" size={32} color={palette.accent} />
        </View>
        <Text
          accessibilityRole="header"
          style={[textStyle('display', palette.textPrimary), { textAlign: 'center' }]}
        >
          {name}
        </Text>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>
          Creating this workspace.
        </Text>
      </View>
    </View>
  );
}

function creatingActivity(): string {
  return `
    @keyframes favlo-creating-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.45; }
    }
    [data-testid="creating-mark"] {
      animation: favlo-creating-pulse 2.4s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
      [data-testid="creating-mark"] {
        animation: none;
        opacity: 1;
      }
    }
  `;
}

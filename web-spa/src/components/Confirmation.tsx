import { Platform, StyleSheet, Text, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function Confirmation({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  const { palette } = useTheme();
  return (
    <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
      <View
        pointerEvents="box-none"
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 72,
          zIndex: z.toast,
          alignItems: 'center',
          paddingHorizontal: space[4],
        }}
      >
        <View
          testID="save-confirmation"
          accessibilityLiveRegion="polite"
          {...(Platform.OS === 'web' ? { role: 'status' as const } : { accessibilityRole: 'summary' as const })}
          style={{
            maxWidth: 480,
            width: '100%',
            flexDirection: 'row',
            alignItems: 'center',
            gap: space[3],
            paddingVertical: space[3],
            paddingHorizontal: space[4],
            borderRadius: radius.card,
            borderWidth: 1,
            borderColor: palette.success,
            backgroundColor: palette.successSubtle,
          }}
        >
          <Text style={[textStyle('body', palette.success), { flex: 1 }]}>{message}</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Dismiss confirmation"
            onPress={onDismiss}
            style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: space[2] }}
          >
            <Text style={textStyle('small', palette.success, '600')}>Dismiss</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

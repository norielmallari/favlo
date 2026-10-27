import { Platform, StyleSheet, Text, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';

export function Confirmation({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  const { palette } = useTheme();
  return (
    <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
      <View
        testID="save-confirmation"
        accessibilityLiveRegion="polite"
        {...(Platform.OS === 'web' ? { role: 'status' as const } : { accessibilityRole: 'summary' as const })}
        style={{
          position: 'absolute',
          left: 16,
          right: 16,
          bottom: 16,
          maxWidth: 480,
          alignSelf: 'center',
          flexDirection: 'row',
          alignItems: 'center',
          gap: 12,
          paddingVertical: 12,
          paddingHorizontal: 16,
          borderRadius: 8,
          borderWidth: 1,
          borderColor: palette.success,
          backgroundColor: palette.successSubtle,
        }}
      >
        <Text style={{ color: palette.success, fontSize: 16, lineHeight: 24, flex: 1 }}>{message}</Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Dismiss confirmation"
          onPress={onDismiss}
          style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8 }}
        >
          <Text style={{ color: palette.success, fontSize: 14, fontWeight: '600' }}>Dismiss</Text>
        </Pressable>
      </View>
    </View>
  );
}

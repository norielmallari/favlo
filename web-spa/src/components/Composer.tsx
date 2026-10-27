import { useState } from 'react';
import { StyleSheet, Text, TextInput, View, type ViewStyle } from 'react-native';

import { composerCommand } from '@/components/composer-keys';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';

type Props = {
  value: string;
  working: boolean;
  onChange: (next: string) => void;
  onSend: () => void;
};

export function Composer({ value, working, onChange, onSend }: Props) {
  const { palette } = useTheme();
  const [focused, setFocused] = useState(false);
  const ring: ViewStyle = {
    outlineStyle: 'solid',
    outlineWidth: 2,
    outlineOffset: 2,
    outlineColor: focused ? palette.focusRing : 'transparent',
  };

  return (
    <View style={styles.row}>
      <TextInput
        accessibilityLabel="Message"
        multiline
        value={value}
        onChangeText={onChange}
        editable={!working}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onKeyPress={(event) => {
          const native = event.nativeEvent as { key: string; shiftKey?: boolean; isComposing?: boolean };
          if (composerCommand(native.key, Boolean(native.shiftKey), Boolean(native.isComposing)) === 'send') {
            event.preventDefault?.();
            onSend();
          }
        }}
        style={[
          styles.input,
          ring,
          {
            color: palette.textPrimary,
            backgroundColor: palette.surface,
            borderColor: palette.borderStrong,
          },
        ]}
      />
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Send"
        accessibilityState={{ disabled: working || value.trim() === '' }}
        disabled={working || value.trim() === ''}
        onPress={onSend}
        style={[
          styles.send,
          {
            backgroundColor: working || value.trim() === '' ? palette.disabledSurface : palette.accent,
          },
        ]}
      >
        <Text style={{ color: working || value.trim() === '' ? palette.disabledText : palette.onAccent, fontSize: 14, fontWeight: '600' }}>
          Send
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', gap: 8, alignItems: 'flex-end' },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 160,
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 16,
    lineHeight: 24,
  },
  send: { minHeight: 44, minWidth: 44, borderRadius: 6, paddingHorizontal: 16, alignItems: 'center', justifyContent: 'center' },
});

import { useState } from 'react';
import { StyleSheet, Text, TextInput, View, type ViewStyle } from 'react-native';

import { composerCommand } from '@/components/composer-keys';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

type Props = {
  value: string;
  working: boolean;
  onChange: (next: string) => void;
  onSend: () => void;
  onStop?: () => void;
  agentName?: string;
};

export function Composer({ value, working, onChange, onSend, onStop, agentName }: Props) {
  const { palette } = useTheme();
  const [focused, setFocused] = useState(false);
  const ring: ViewStyle = {
    outlineStyle: 'solid',
    outlineWidth: 2,
    outlineOffset: 2,
    outlineColor: focused ? palette.focusRing : 'transparent',
  };
  const idle = !working && value.trim() === '';
  const stop = working && onStop != null;

  return (
    <View style={styles.stack}>
      <View style={styles.row}>
        <TextInput
          accessibilityLabel="Message"
          placeholder={agentName ? `Message ${agentName}` : 'Message'}
          placeholderTextColor={palette.textSecondary}
          multiline
          value={value}
          onChangeText={onChange}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyPress={(event) => {
            const native = event.nativeEvent as { key: string; shiftKey?: boolean; isComposing?: boolean };
            if (composerCommand(native.key, Boolean(native.shiftKey), Boolean(native.isComposing)) === 'send') {
              event.preventDefault?.();
              if (!working) {
                onSend();
              }
            }
          }}
          style={[
            styles.input,
            ring,
            textStyle('body', palette.textPrimary),
            {
              backgroundColor: palette.surface,
              borderColor: palette.borderStrong,
            },
          ]}
        />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={stop ? 'Stop' : 'Send'}
          accessibilityState={{ disabled: !stop && idle }}
          disabled={!stop && idle}
          onPress={stop ? onStop : onSend}
          style={[
            styles.send,
            {
              backgroundColor: !stop && idle ? palette.disabledSurface : stop ? palette.danger : palette.accent,
            },
          ]}
        >
          <Text
            style={textStyle(
              'small',
              !stop && idle ? palette.disabledText : stop ? palette.onDanger : palette.onAccent,
              '600',
            )}
          >
            {stop ? 'Stop' : 'Send'}
          </Text>
        </Pressable>
      </View>
      <Text style={textStyle('caption', palette.textSecondary)}>Return to send · Shift+Return for a line</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  stack: { gap: space[1], maxWidth: '100%', minWidth: 0 },
  row: { flexDirection: 'row', gap: space[2], alignItems: 'flex-end', maxWidth: '100%', minWidth: 0 },
  input: {
    flex: 1,
    minWidth: 0,
    minHeight: 44,
    maxHeight: 160,
    borderWidth: 1,
    borderRadius: radius.card,
    paddingHorizontal: space[3],
    paddingVertical: space[2],
  },
  send: {
    minHeight: 44,
    minWidth: 44,
    borderRadius: radius.control,
    paddingHorizontal: space[4],
    alignItems: 'center',
    justifyContent: 'center',
  },
});

import { useState } from 'react';
import { Text, View } from 'react-native';

import { MarkdownText } from '@/components/MarkdownText';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function Thinking({ text, settled }: { text: string; settled: boolean }) {
  const { palette } = useTheme();
  const [open, setOpen] = useState(!settled);
  const [wasSettled, setWasSettled] = useState(settled);
  if (settled !== wasSettled) {
    setWasSettled(settled);
    setOpen(!settled);
  }

  return (
    <View style={{ gap: space[1], maxWidth: '100%', minWidth: 0 }}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Thinking"
        accessibilityState={{ expanded: open }}
        aria-expanded={open}
        onPress={() => setOpen((value) => !value)}
        style={{
          minHeight: 44,
          flexDirection: 'row',
          alignItems: 'center',
          alignSelf: 'flex-start',
          gap: space[2],
        }}
      >
        <Text style={textStyle('small', palette.textSecondary, '600')}>Thinking</Text>
        <Text style={textStyle('small', palette.textSecondary)}>{open ? '▾' : '▸'}</Text>
      </Pressable>
      {open ? (
        <View
          testID="thinking-text"
          style={{
            paddingLeft: space[3],
            borderLeftWidth: 2,
            borderLeftColor: palette.borderStrong,
            maxWidth: '100%',
            minWidth: 0,
          }}
        >
          <MarkdownText text={text} color={palette.textSecondary} fontSize={14} lineHeight={22} />
        </View>
      ) : null}
    </View>
  );
}

import { useState } from 'react';
import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { MarkdownText } from '@/components/MarkdownText';
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
      <IconButton
        icon={open ? 'chevron-down' : 'chevron-right'}
        label="Thinking"
        onPress={() => setOpen((value) => !value)}
        background="transparent"
        border="transparent"
        color={palette.textSecondary}
        accessibilityState={{ expanded: open }}
      />
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

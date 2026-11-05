import { useState } from 'react';
import { View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { MarkdownText } from '@/components/MarkdownText';
import { type ExpandTrace, traceStartsOpen } from '@/components/expandTrace';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';

export function Thinking({
  text,
  settled,
  expandTrace = 'collapsed',
  forceOpen = false,
}: {
  text: string;
  settled: boolean;
  expandTrace?: ExpandTrace;
  forceOpen?: boolean;
}) {
  const { palette } = useTheme();
  const wantOpen = traceStartsOpen(settled, expandTrace, forceOpen);
  const [open, setOpen] = useState(wantOpen);
  const [wasWant, setWasWant] = useState(wantOpen);
  if (wantOpen !== wasWant) {
    setWasWant(wantOpen);
    setOpen(wantOpen);
  }

  return (
    <View
      style={{
        gap: space[1],
        width: '100%',
        maxWidth: '100%',
        minWidth: 0,
        alignSelf: 'stretch',
      }}
    >
      <IconButton
        icon={open ? 'chevron-down' : 'chevron-right'}
        label="Thinking"
        onPress={() => setOpen((value) => !value)}
        background="transparent"
        border="transparent"
        color={palette.textSecondary}
        accessibilityState={{ expanded: open }}
        dense
        block
      />
      {open ? (
        <View
          testID="thinking-text"
          style={{
            paddingLeft: space[2],
            borderLeftWidth: 2,
            borderLeftColor: palette.borderStrong,
            maxWidth: '100%',
            minWidth: 0,
          }}
        >
          <MarkdownText text={text} color={palette.textSecondary} role="small" />
        </View>
      ) : null}
    </View>
  );
}

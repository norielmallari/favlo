import { useState } from 'react';
import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { type ExpandTrace, traceStartsOpen } from '@/components/expandTrace';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type ToolCallRecord = { id: string; name: string; arguments: string; result?: string };

export function ToolCall({
  call,
  settled,
  expandTrace = 'collapsed',
  forceOpen = false,
}: {
  call: ToolCallRecord;
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
  const label = call.name ? `Tool call ${call.name}` : 'Tool call';

  return (
    <View
      testID="tool-call"
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
        label={label}
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
          testID="tool-call-text"
          style={{
            paddingLeft: space[2],
            borderLeftWidth: 2,
            borderLeftColor: palette.borderStrong,
            maxWidth: '100%',
            minWidth: 0,
            gap: space[1],
          }}
        >
          {call.arguments ? (
            <Text style={textStyle('small', palette.textSecondary)}>{call.arguments}</Text>
          ) : null}
          {call.result ? <Text style={textStyle('small', palette.textSecondary)}>{call.result}</Text> : null}
        </View>
      ) : null}
    </View>
  );
}

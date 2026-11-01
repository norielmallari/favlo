import { useState } from 'react';
import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type ToolCallRecord = { id: string; name: string; arguments: string; result?: string };

export function ToolCall({ call, settled }: { call: ToolCallRecord; settled: boolean }) {
  const { palette } = useTheme();
  const [open, setOpen] = useState(!settled);
  const [wasSettled, setWasSettled] = useState(settled);
  if (settled !== wasSettled) {
    setWasSettled(settled);
    setOpen(!settled);
  }
  const label = call.name ? `Tool call ${call.name}` : 'Tool call';

  return (
    <View style={{ gap: space[1], maxWidth: '100%', minWidth: 0 }}>
      <IconButton
        icon={open ? 'chevron-down' : 'chevron-right'}
        label={label}
        onPress={() => setOpen((value) => !value)}
        background="transparent"
        border="transparent"
        color={palette.textSecondary}
        accessibilityState={{ expanded: open }}
      />
      {open ? (
        <View
          testID="tool-call-text"
          style={{
            paddingLeft: space[3],
            borderLeftWidth: 2,
            borderLeftColor: palette.borderStrong,
            maxWidth: '100%',
            minWidth: 0,
            gap: space[2],
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

import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function ReplyAlert({
  message,
  disabled,
  onResend,
}: {
  message: string;
  disabled: boolean;
  onResend: () => void;
}) {
  const { palette } = useTheme();
  return (
    <View
      accessibilityRole="alert"
      accessibilityLiveRegion="polite"
      style={{
        gap: space[2],
        maxWidth: '100%',
        padding: space[3],
        borderRadius: radius.control,
        borderWidth: 1,
        borderColor: palette.danger,
        backgroundColor: palette.dangerSubtle,
      }}
    >
      <Text style={textStyle('small', palette.danger)}>{message}</Text>
      <IconButton
        icon="refresh-cw"
        label="Send again"
        disabled={disabled}
        onPress={onResend}
        background={palette.surface}
        border={palette.danger}
        color={palette.danger}
      />
    </View>
  );
}

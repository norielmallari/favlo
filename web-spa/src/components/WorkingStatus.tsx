import { Pressable, Text, View } from 'react-native';

import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

/** Activity chip. Press toggles whether thinking/tool blocks are forced open. */
export function WorkingStatus({
  label,
  expanded,
  onPress,
}: {
  label: string;
  expanded: boolean;
  onPress: () => void;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ alignSelf: 'flex-start', maxWidth: '100%' }}>
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ expanded }}
        accessibilityLabel={label}
        onPress={onPress}
        style={{
          flexDirection: 'row',
          alignItems: 'center',
          gap: space[2],
          paddingHorizontal: space[3],
          paddingVertical: space[2],
          borderRadius: radius.control,
          backgroundColor: palette.accentSubtle,
          borderWidth: 1,
          borderColor: palette.accent,
        }}
      >
        <Text
          testID="working"
          accessibilityLiveRegion="polite"
          style={textStyle('label', palette.accent, '600')}
        >
          {label}
        </Text>
      </Pressable>
    </View>
  );
}

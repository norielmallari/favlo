import { createElement } from 'react';
import { Platform, Pressable, Text, View } from 'react-native';

import { workingActivity } from '@/components/working';
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
      {Platform.OS === 'web'
        ? createElement('style', { 'data-testid': 'working-activity' }, workingActivity(palette.accent))
        : null}
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
        <View
          testID="working-dot"
          accessibilityElementsHidden
          importantForAccessibility="no"
          style={{
            width: 8,
            height: 8,
            borderRadius: 4,
            backgroundColor: palette.accent,
          }}
        />
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

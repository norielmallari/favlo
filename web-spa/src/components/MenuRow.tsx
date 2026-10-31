import { Text, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function MenuRow({
  label,
  selected,
  onPress,
  role = 'button',
  icon,
  expanded,
  selectedTestId = 'selected-mark',
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
  role?: 'button' | 'tab';
  icon?: string;
  expanded?: boolean;
  selectedTestId?: string;
}) {
  const { palette } = useTheme();
  return (
    <Pressable
      accessibilityRole={role}
      accessibilityLabel={label}
      accessibilityState={{ selected, ...(expanded == null ? {} : { expanded }) }}
      onPress={onPress}
      style={{
        minHeight: 44,
        flexDirection: 'row',
        alignItems: 'center',
        gap: space[2],
        paddingHorizontal: space[2],
        borderRadius: radius.control,
        backgroundColor: selected ? palette.accentSubtle : palette.surface,
      }}
    >
      <View
        testID={selected && selectedTestId !== '' ? selectedTestId : undefined}
        style={{
          width: 3,
          height: 20,
          borderRadius: 2,
          backgroundColor: selected ? palette.accent : 'transparent',
        }}
      />
      {icon ? <FeatherIcon name={icon} size={18} color={palette.textPrimary} /> : null}
      <Text style={textStyle('label', palette.textPrimary, selected ? '600' : '400')}>{label}</Text>
    </Pressable>
  );
}

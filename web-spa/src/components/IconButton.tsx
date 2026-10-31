import { Text, type AccessibilityState } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export const buttonRow = {
  flexDirection: 'row' as const,
  flexWrap: 'wrap' as const,
  alignItems: 'center' as const,
  gap: space[2],
  maxWidth: '100%' as const,
};

export function IconButton({
  icon,
  label,
  onPress,
  disabled = false,
  background,
  border,
  color,
  accessibilityLabel,
  accessibilityState,
  testID,
  role = 'button',
  iconOnly = false,
}: {
  icon: string;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  background: string;
  border?: string;
  color: string;
  accessibilityLabel?: string;
  accessibilityState?: AccessibilityState;
  testID?: string;
  role?: 'button' | 'link';
  iconOnly?: boolean;
}) {
  return (
    <Pressable
      testID={testID}
      accessibilityRole={role}
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={accessibilityState ?? { disabled }}
      disabled={disabled}
      onPress={onPress}
      style={{
        minHeight: 44,
        minWidth: 44,
        maxWidth: '100%',
        alignSelf: 'flex-start',
        flexDirection: 'row',
        flexGrow: 0,
        alignItems: 'center',
        gap: space[2],
        borderRadius: radius.control,
        borderWidth: 1,
        borderColor: border ?? background,
        backgroundColor: background,
        paddingHorizontal: iconOnly ? 0 : space[3],
        justifyContent: 'center',
      }}
    >
      <FeatherIcon name={icon} size={18} color={color} />
      {iconOnly ? null : <Text style={textStyle('small', color, '600')}>{label}</Text>}
    </Pressable>
  );
}

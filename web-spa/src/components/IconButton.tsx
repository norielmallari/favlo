import { Platform, Text, type AccessibilityState } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { buttonPaint, type ButtonTone } from '@/theme/buttons';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export const buttonRow = {
  flexDirection: 'row' as const,
  flexWrap: 'wrap' as const,
  alignItems: 'center' as const,
  gap: space[2],
  width: '100%' as const,
  maxWidth: '100%' as const,
  minWidth: 0,
};

type Common = {
  icon: string;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  accessibilityLabel?: string;
  accessibilityState?: AccessibilityState;
  testID?: string;
  role?: 'button' | 'link';
  iconOnly?: boolean;
  /** Full-width row (tool calls / thinking in the transcript). */
  block?: boolean;
};

type ToneProps = Common & {
  tone: ButtonTone;
  background?: undefined;
  border?: undefined;
  color?: undefined;
};

type PaintProps = Common & {
  tone?: undefined;
  background: string;
  border?: string;
  color: string;
};

export function IconButton(props: ToneProps | PaintProps) {
  const { palette } = useTheme();
  const paint =
    props.tone != null
      ? buttonPaint(palette, props.tone)
      : {
          background: props.background,
          border: props.border ?? props.background,
          color: props.color,
        };
  const {
    icon,
    label,
    onPress,
    disabled = false,
    accessibilityLabel,
    accessibilityState,
    testID,
    role = 'button',
    iconOnly = false,
    block = false,
  } = props;
  const expanded = accessibilityState?.expanded;
  const webExpanded =
    Platform.OS === 'web' && expanded !== undefined ? ({ 'aria-expanded': expanded } as const) : null;
  return (
    <Pressable
      testID={testID}
      accessibilityRole={role}
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={accessibilityState ?? { disabled }}
      disabled={disabled}
      onPress={onPress}
      {...(webExpanded ?? {})}
      style={{
        minHeight: 44,
        minWidth: iconOnly ? 44 : 0,
        maxWidth: '100%',
        width: block ? '100%' : undefined,
        alignSelf: block ? 'stretch' : 'flex-start',
        flexDirection: 'row',
        flexShrink: 1,
        flexGrow: block ? 1 : 0,
        alignItems: 'center',
        gap: space[2],
        borderRadius: radius.control,
        borderWidth: 1,
        borderColor: paint.border,
        backgroundColor: paint.background,
        paddingHorizontal: iconOnly ? 0 : space[3],
        justifyContent: block ? 'flex-start' : 'center',
      }}
    >
      <FeatherIcon name={icon} size={18} color={paint.color} />
      {iconOnly ? null : (
        <Text style={[textStyle('small', paint.color, '600'), { flexShrink: 1, minWidth: 0 }]}>{label}</Text>
      )}
    </Pressable>
  );
}

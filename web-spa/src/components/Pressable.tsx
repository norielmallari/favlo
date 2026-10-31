import {
  Pressable as NativePressable,
  StyleSheet,
  View,
  type PressableProps,
  type PressableStateCallbackType,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { useTheme } from '@/theme/Theme';
import { dark, radius } from '@/theme/tokens';

type PressState = PressableStateCallbackType & { hovered?: boolean; focused?: boolean };

const FOCUS_RING_WIDTH = 2;
const FOCUS_RING_OFFSET = 2;

/** Room outside a row so the focus ring is not clipped by the list. */
export const focusRingInset = FOCUS_RING_WIDTH + FOCUS_RING_OFFSET + 2;

export function Pressable({ children, style, ...props }: PressableProps) {
  const { palette } = useTheme();
  const night = palette.background === dark.background;
  const pressedTint = night ? 'rgba(255, 255, 255, 0.18)' : 'rgba(24, 34, 48, 0.14)';
  const hoverTint = night ? 'rgba(255, 255, 255, 0.08)' : 'rgba(24, 34, 48, 0.06)';

  return (
    <NativePressable
      {...props}
      style={(state) => [typeof style === 'function' ? style(state) : style, focusRing(palette.focusRing, state)]}
    >
      {(state) => {
        const current = state as PressState;
        const hot = current.pressed || current.hovered;
        return (
          <>
            {typeof children === 'function' ? children(state) : children}
            {hot ? (
              <View
                pointerEvents="none"
                testID="press-feedback"
                style={[
                  StyleSheet.absoluteFill,
                  { borderRadius: radius.control, backgroundColor: current.pressed ? pressedTint : hoverTint },
                ]}
              />
            ) : null}
          </>
        );
      }}
    </NativePressable>
  );
}

function focusRing(color: string, state: PressableStateCallbackType): StyleProp<ViewStyle> {
  const focused = (state as PressState).focused === true;
  return {
    outlineStyle: 'solid',
    outlineWidth: FOCUS_RING_WIDTH,
    outlineOffset: FOCUS_RING_OFFSET,
    outlineColor: focused ? color : 'transparent',
  };
}

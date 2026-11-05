import { useEffect, useRef, useState } from 'react';
import {
  Platform,
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

type PressState = PressableStateCallbackType & { hovered?: boolean };

const FOCUS_RING_WIDTH = 2;
const FOCUS_RING_OUTSET = 2;

/**
 * Room outside a focused control so the ring is not clipped by a ScrollView,
 * an overflow:hidden ancestor, or an edge-flush layout.
 */
export const focusRingInset = FOCUS_RING_WIDTH + FOCUS_RING_OUTSET + 2;

/** Padding on a clipping parent that hosts Pressable / IconButton children. */
export const focusSafePad = {
  padding: focusRingInset,
} as const;

/** Horizontal padding when only the left/right edges clip. */
export const focusSafePadX = {
  paddingHorizontal: focusRingInset,
} as const;

/** Vertical padding when only the top/bottom edges clip. */
export const focusSafePadY = {
  paddingVertical: focusRingInset,
} as const;

/** Gap between sibling focusable rows so neighboring rings do not collide. */
export const focusSafeGap = focusRingInset * 2;

export function Pressable({ children, style, ...props }: PressableProps) {
  const { palette } = useTheme();
  const night = palette.background === dark.background;
  const pressedTint = night ? 'rgba(255, 255, 255, 0.18)' : 'rgba(24, 34, 48, 0.14)';
  const hoverTint = night ? 'rgba(255, 255, 255, 0.08)' : 'rgba(24, 34, 48, 0.06)';
  const hostRef = useRef<HTMLElement | null>(null);
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }
    const node = hostRef.current;
    if (node == null || typeof node.addEventListener !== 'function') {
      return;
    }
    const onFocus = () => setFocused(true);
    const onBlur = () => setFocused(false);
    node.addEventListener('focus', onFocus);
    node.addEventListener('blur', onBlur);
    setFocused(document.activeElement === node);
    return () => {
      node.removeEventListener('focus', onFocus);
      node.removeEventListener('blur', onBlur);
    };
  }, []);

  return (
    <NativePressable
      {...props}
      ref={(node) => {
        hostRef.current = node as unknown as HTMLElement | null;
      }}
      style={(state) => [
        typeof style === 'function' ? style(state) : style,
        webControl,
        focusRingStyle(palette.focusRing, focused),
      ]}
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

/** Native buttons keep a platform appearance that clips an offset outline to the corners. */
const webControl =
  Platform.OS === 'web' ? ({ appearance: 'none' } as ViewStyle) : null;

/** A spread shadow follows the radius. An offset outline is clipped to corner ticks. */
export function focusRingStyle(color: string, on: boolean): ViewStyle {
  return {
    outlineStyle: 'none',
    boxShadow: on ? `0 0 0 ${FOCUS_RING_WIDTH}px ${color}` : 'none',
  };
}

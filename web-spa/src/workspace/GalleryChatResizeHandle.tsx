import { useRef, useState } from 'react';
import { Platform, View } from 'react-native';

import { useTheme } from '@/theme/Theme';
import { duration, radius, space } from '@/theme/tokens';
import {
  clampGalleryChatWidth,
  GALLERY_CHAT_WIDTH_DEFAULT,
  GALLERY_CHAT_WIDTH_MIN,
  GALLERY_CHAT_WIDTH_STEP,
  galleryChatWidthMax,
} from '@/workspace/gallery-chat-width';

const HIT = space[3];
const LINE = 1;
const GRIP_WIDTH = 4;
const GRIP_HEIGHT = 28;

/** Wide-web gallery splitter: hairline seam, centered grip, wide hit target. */
export function GalleryChatResizeHandle({
  value,
  rowWidth,
  onChange,
}: {
  value: number;
  rowWidth: number;
  onChange: (next: number) => void;
}) {
  const { palette } = useTheme();
  const [dragging, setDragging] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const valueRef = useRef(value);
  const rowRef = useRef(rowWidth);
  valueRef.current = value;
  rowRef.current = rowWidth;
  const max = galleryChatWidthMax(rowWidth);
  const min = Math.min(GALLERY_CHAT_WIDTH_MIN, max);
  const hot = dragging || hovered || focused;
  const ink = hot ? palette.accent : palette.borderStrong;

  function beginDrag(clientX: number) {
    if (Platform.OS !== 'web' || typeof window === 'undefined') {
      return;
    }
    const startX = clientX;
    const startW = valueRef.current;
    setDragging(true);
    const move = (event: PointerEvent) => {
      onChange(clampGalleryChatWidth(startW - (event.clientX - startX), rowRef.current));
    };
    const up = () => {
      setDragging(false);
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }

  function onKey(event: { nativeEvent?: { key?: string }; key?: string; preventDefault?: () => void }) {
    const key = event.nativeEvent?.key ?? event.key;
    if (key === 'ArrowLeft') {
      event.preventDefault?.();
      onChange(valueRef.current + GALLERY_CHAT_WIDTH_STEP);
    }
    if (key === 'ArrowRight') {
      event.preventDefault?.();
      onChange(valueRef.current - GALLERY_CHAT_WIDTH_STEP);
    }
    if (key === 'Home') {
      event.preventDefault?.();
      onChange(max);
    }
    if (key === 'End') {
      event.preventDefault?.();
      onChange(min);
    }
  }

  return (
    <View
      accessible
      accessibilityRole="adjustable"
      accessibilityLabel="Resize chat"
      accessibilityHint="Drag or use arrow keys to change the conversation width"
      accessibilityValue={{ min, max, now: value }}
      accessibilityActions={[{ name: 'increment' }, { name: 'decrement' }]}
      onAccessibilityAction={(event) => {
        if (event.nativeEvent.actionName === 'increment') {
          onChange(valueRef.current + GALLERY_CHAT_WIDTH_STEP);
        }
        if (event.nativeEvent.actionName === 'decrement') {
          onChange(valueRef.current - GALLERY_CHAT_WIDTH_STEP);
        }
      }}
      // Web focus + pointer affordances (RN-web passes these through).
      {...(Platform.OS === 'web'
        ? {
            tabIndex: 0,
            // RN-web maps adjustable → slider but omits aria-valuenow from accessibilityValue.
            'aria-valuemin': min,
            'aria-valuemax': max,
            'aria-valuenow': value,
            onFocus: () => setFocused(true),
            onBlur: () => setFocused(false),
            onMouseEnter: () => setHovered(true),
            onMouseLeave: () => setHovered(false),
            onKeyDown: onKey,
            onPointerDown: (event: { nativeEvent?: { pageX?: number }; pageX?: number }) => {
              const pageX = event.nativeEvent?.pageX ?? event.pageX;
              if (typeof pageX === 'number') {
                beginDrag(pageX);
              }
            },
            onDoubleClick: () => onChange(GALLERY_CHAT_WIDTH_DEFAULT),
          }
        : null)}
      style={{
        width: HIT,
        alignSelf: 'stretch',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1,
        cursor: 'col-resize',
        userSelect: 'none',
        backgroundColor: hot ? palette.accentSubtle : 'transparent',
        outlineStyle: 'solid',
        outlineWidth: 2,
        outlineOffset: 2,
        outlineColor: focused ? palette.focusRing : 'transparent',
        transitionProperty: 'background-color',
        transitionDuration: `${duration.fast}ms`,
      }}
    >
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          width: hot ? 2 : LINE,
          backgroundColor: ink,
          transitionProperty: 'background-color, width',
          transitionDuration: `${duration.fast}ms`,
        }}
      />
      <View
        pointerEvents="none"
        accessibilityElementsHidden
        importantForAccessibility="no-hide-descendants"
        style={{
          width: GRIP_WIDTH,
          height: GRIP_HEIGHT,
          borderRadius: radius.pill,
          backgroundColor: hot ? palette.accent : palette.surfaceRaised,
          borderWidth: 1,
          borderColor: ink,
          shadowColor: palette.textPrimary,
          shadowOpacity: hot ? 0.14 : 0.08,
          shadowRadius: hot ? 3 : 2,
          shadowOffset: { width: 0, height: 0 },
          transitionProperty: 'background-color, border-color, shadow-opacity, shadow-radius',
          transitionDuration: `${duration.fast}ms`,
        }}
      />
    </View>
  );
}

import { useRef, useState, type ReactNode } from 'react';
import { Platform, View } from 'react-native';

import { focusRingStyle } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { duration, radius, space } from '@/theme/tokens';
import {
  cycleGalleryDetent,
  galleryDockHeight,
  galleryDockHeights,
  settleGalleryDetent,
  type GalleryChatDetent,
  writeGalleryChatDetent,
} from '@/workspace/gallery-chat-dock';

const HIT = space[3];
const LINE = 1;
const GRIP_WIDTH = 28;
const GRIP_HEIGHT = 4;

/** Narrow-web gallery chat dock: grabber + children sized to a detent height. */
export function GalleryChatDock({
  detent,
  columnHeight,
  onDetentChange,
  children,
}: {
  detent: GalleryChatDetent;
  columnHeight: number;
  onDetentChange: (next: GalleryChatDetent) => void;
  children: ReactNode;
}) {
  const { palette } = useTheme();
  const [dragging, setDragging] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [dragHeight, setDragHeight] = useState<number | null>(null);
  const detentRef = useRef(detent);
  const columnRef = useRef(columnHeight);
  const heightRef = useRef(galleryDockHeight(detent, columnHeight));
  const lastY = useRef(0);
  const lastT = useRef(0);
  const velocity = useRef(0);
  detentRef.current = detent;
  columnRef.current = columnHeight;

  const heights = galleryDockHeights(columnHeight);
  const settled = heights[detent];
  const height = dragHeight ?? settled;
  heightRef.current = height;
  const hot = dragging || hovered || focused;
  const ink = hot ? palette.accent : palette.borderStrong;
  const reduceMotion =
    Platform.OS === 'web' &&
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  function commit(next: GalleryChatDetent) {
    writeGalleryChatDetent(next);
    onDetentChange(next);
  }

  function beginDrag(clientY: number) {
    if (Platform.OS !== 'web' || typeof window === 'undefined') {
      return;
    }
    const startY = clientY;
    const startH = heightRef.current;
    lastY.current = clientY;
    lastT.current = performance.now();
    velocity.current = 0;
    setDragging(true);
    setDragHeight(startH);
    const move = (event: PointerEvent) => {
      const now = performance.now();
      const dt = Math.max(1, now - lastT.current);
      // Finger up grows the dock (chat expands over the preview).
      const next = Math.max(
        heights.fullscreen,
        Math.min(columnRef.current, startH + (startY - event.clientY)),
      );
      velocity.current = (lastY.current - event.clientY) / dt;
      lastY.current = event.clientY;
      lastT.current = now;
      setDragHeight(next);
    };
    const up = () => {
      const current = heightRef.current;
      const next = settleGalleryDetent(current, columnRef.current, velocity.current);
      setDragging(false);
      setDragHeight(null);
      commit(next);
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
    if (key === 'ArrowUp') {
      event.preventDefault?.();
      const index = Math.min(
        2,
        ['fullscreen', 'peek', 'expanded'].indexOf(detentRef.current) + 1,
      );
      commit(['fullscreen', 'peek', 'expanded'][index] as GalleryChatDetent);
    }
    if (key === 'ArrowDown') {
      event.preventDefault?.();
      const index = Math.max(
        0,
        ['fullscreen', 'peek', 'expanded'].indexOf(detentRef.current) - 1,
      );
      commit(['fullscreen', 'peek', 'expanded'][index] as GalleryChatDetent);
    }
    if (key === 'Home') {
      event.preventDefault?.();
      commit('expanded');
    }
    if (key === 'End') {
      event.preventDefault?.();
      commit('fullscreen');
    }
    if (key === 'Enter' || key === ' ') {
      event.preventDefault?.();
      commit(cycleGalleryDetent(detentRef.current));
    }
  }

  const label =
    detent === 'expanded' ? 'Collapse chat' : detent === 'fullscreen' ? 'Expand chat' : 'Expand chat';

  return (
    <View
      style={
        {
          height,
          flexGrow: 0,
          flexShrink: 0,
          minHeight: heights.fullscreen,
          maxHeight: columnHeight || undefined,
          backgroundColor: palette.background,
          borderTopWidth: 1,
          borderColor: palette.border,
          overflow: 'hidden',
          ...(Platform.OS === 'web' && !dragging && !reduceMotion
            ? {
                transitionProperty: 'height',
                transitionDuration: `${duration.med}ms`,
                transitionTimingFunction: 'ease-in-out',
              }
            : null),
        } as object
      }
    >
      <View
        accessible
        accessibilityRole="adjustable"
        accessibilityLabel={label}
        accessibilityHint="Drag or use arrow keys to change the conversation height"
        accessibilityValue={{
          min: heights.fullscreen,
          max: heights.expanded,
          now: Math.round(height),
          text: detent,
        }}
        accessibilityActions={[{ name: 'increment' }, { name: 'decrement' }]}
        onAccessibilityAction={(event) => {
          if (event.nativeEvent.actionName === 'increment') {
            const index = Math.min(
              2,
              ['fullscreen', 'peek', 'expanded'].indexOf(detentRef.current) + 1,
            );
            commit(['fullscreen', 'peek', 'expanded'][index] as GalleryChatDetent);
          }
          if (event.nativeEvent.actionName === 'decrement') {
            const index = Math.max(
              0,
              ['fullscreen', 'peek', 'expanded'].indexOf(detentRef.current) - 1,
            );
            commit(['fullscreen', 'peek', 'expanded'][index] as GalleryChatDetent);
          }
        }}
        {...(Platform.OS === 'web'
          ? {
              tabIndex: 0,
              'aria-valuemin': heights.fullscreen,
              'aria-valuemax': heights.expanded,
              'aria-valuenow': Math.round(height),
              'aria-valuetext': detent,
              onFocus: () => setFocused(true),
              onBlur: () => setFocused(false),
              onMouseEnter: () => setHovered(true),
              onMouseLeave: () => setHovered(false),
              onKeyDown: onKey,
              onPointerDown: (event: { nativeEvent?: { pageY?: number }; pageY?: number }) => {
                const pageY = event.nativeEvent?.pageY ?? event.pageY;
                if (typeof pageY === 'number') {
                  beginDrag(pageY);
                }
              },
              onDoubleClick: () => commit(cycleGalleryDetent(detentRef.current)),
            }
          : null)}
        style={
          {
            height: HIT,
            alignSelf: 'stretch',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1,
            cursor: 'row-resize',
            userSelect: 'none',
            backgroundColor: hot ? palette.accentSubtle : 'transparent',
            ...focusRingStyle(palette.focusRing, focused),
            transitionProperty: 'background-color',
            transitionDuration: `${duration.fast}ms`,
          } as object
        }
      >
        <View
          pointerEvents="none"
          style={
            {
              position: 'absolute',
              left: 0,
              right: 0,
              height: hot ? 2 : LINE,
              backgroundColor: ink,
              transitionProperty: 'background-color, height',
              transitionDuration: `${duration.fast}ms`,
            } as object
          }
        />
        <View
          pointerEvents="none"
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          style={
            {
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
            } as object
          }
        />
      </View>
      <View style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, overflow: 'hidden' }}>
        {children}
      </View>
    </View>
  );
}

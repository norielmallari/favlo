import { useEffect, useMemo, useRef, useState } from 'react';
import { Platform, Text, useWindowDimensions, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { duration, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import type { ElementProps } from '@/workspace/catalog';
import { useRegisterBannerEnd } from '@/workspace/workspace-chrome';

const IMAGE = /\.(png|jpe?g|webp)$/i;
const SWIPE_THRESHOLD = 48;
const WIDE_WEB = 1024;

export function previewActionsInPane(os: string, width: number): boolean {
  return os === 'web' && width >= WIDE_WEB;
}

export function Preview({
  props,
  selection,
  fileUrl,
  onAddToChat,
  onRequestDelete,
  onSwipe,
  fullScreen = false,
  chatDetent,
  onToggleFullScreen,
  onCycleLayout,
}: ElementProps) {
  const { palette } = useTheme();
  const { width } = useWindowDimensions();
  const actionsInPane = previewActionsInPane(Platform.OS, width);
  const path = selection?.path ?? null;
  const fit = props.fit === 'cover' ? 'cover' : 'contain';
  const [opacity, setOpacity] = useState(1);
  const shownPath = useRef(path);
  const swipeOrigin = useRef<{ x: number; y: number } | null>(null);
  const reduceMotion =
    Platform.OS === 'web' &&
    typeof window !== 'undefined' &&
    Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);

  const name = path == null ? '' : (path.split('/').pop() ?? path);
  const url = path == null ? '' : fileUrl(path);
  const live = useRef({
    path,
    name,
    url,
    fullScreen,
    onAddToChat,
    onRequestDelete,
    onToggleFullScreen,
    onCycleLayout,
  });
  live.current = {
    path,
    name,
    url,
    fullScreen,
    onAddToChat,
    onRequestDelete,
    onToggleFullScreen,
    onCycleLayout,
  };

  const bannerActions = useMemo(() => {
    if (path == null) {
      return null;
    }
    const canFullScreen = onToggleFullScreen != null;
    return (
      <>
        {canFullScreen ? (
          <IconButton
            label={fullScreen ? 'Exit full screen' : 'Full screen preview'}
            icon={fullScreen ? 'minimize-2' : 'maximize-2'}
            iconOnly
            tone="ghost"
            onPress={() => live.current.onToggleFullScreen?.()}
          />
        ) : null}
        <IconButton
          label={`Download ${name}`}
          icon="download"
          iconOnly
          tone="surface"
          onPress={() => {
            void (async () => {
              if (typeof document === 'undefined') {
                return;
              }
              const current = live.current;
              const response = await fetch(current.url);
              if (!response.ok) {
                throw new Error(`Download failed (${response.status})`);
              }
              const objectUrl = URL.createObjectURL(await response.blob());
              const link = document.createElement('a');
              link.href = objectUrl;
              link.download = current.name;
              link.click();
              URL.revokeObjectURL(objectUrl);
            })();
          }}
        />
        <IconButton
          label="Add to chat"
          accessibilityLabel="Add to chat"
          icon="plus"
          iconOnly
          tone="accent"
          onPress={() => {
            const current = live.current;
            if (current.path != null) {
              current.onAddToChat(current.path);
            }
          }}
        />
        {IMAGE.test(path) ? (
          <IconButton
            label={`Delete ${name}`}
            icon="trash-2"
            iconOnly
            tone="danger"
            onPress={() => {
              const current = live.current;
              if (current.path != null) {
                current.onRequestDelete(current.path);
              }
            }}
          />
        ) : null}
      </>
    );
  }, [fullScreen, name, onToggleFullScreen, path]);

  useRegisterBannerEnd(actionsInPane ? null : bannerActions);

  useEffect(() => {
    if (path === shownPath.current) {
      return;
    }
    if (reduceMotion || path == null || shownPath.current == null) {
      shownPath.current = path;
      setOpacity(1);
      return;
    }
    setOpacity(0);
    const timer = setTimeout(() => {
      shownPath.current = path;
      setOpacity(1);
    }, duration.fast);
    return () => clearTimeout(timer);
  }, [path, reduceMotion]);

  if (path == null) {
    return (
      <View style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, alignItems: 'center', justifyContent: 'center', padding: space[4] }}>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>No media selected</Text>
      </View>
    );
  }
  const displayPath = shownPath.current ?? path;
  const displayUrl = fileUrl(displayPath);
  const displayLower = displayPath.toLowerCase();
  const displayName = displayPath.split('/').pop() ?? displayPath;
  const canCycle = onCycleLayout != null && IMAGE.test(displayPath);
  const cycleLabel = !canCycle
    ? undefined
    : chatDetent === 'fullscreen'
      ? `Exit full screen ${displayName}`
      : chatDetent === 'peek'
        ? `Expand chat ${displayName}`
        : `Expand ${displayName}`;

  function onPointerDown(event: { nativeEvent?: { pageX?: number; pageY?: number }; pageX?: number; pageY?: number }) {
    const x = event.nativeEvent?.pageX ?? event.pageX;
    const y = event.nativeEvent?.pageY ?? event.pageY;
    if (typeof x === 'number' && typeof y === 'number') {
      swipeOrigin.current = { x, y };
    }
  }

  function onPointerUp(event: { nativeEvent?: { pageX?: number; pageY?: number }; pageX?: number; pageY?: number }) {
    const origin = swipeOrigin.current;
    swipeOrigin.current = null;
    if (origin == null) {
      return;
    }
    const x = event.nativeEvent?.pageX ?? event.pageX;
    const y = event.nativeEvent?.pageY ?? event.pageY;
    if (typeof x !== 'number' || typeof y !== 'number') {
      return;
    }
    const dx = x - origin.x;
    const dy = y - origin.y;
    if (Math.abs(dx) >= SWIPE_THRESHOLD && Math.abs(dx) >= Math.abs(dy)) {
      onSwipe?.(dx < 0 ? 'next' : 'prev');
      return;
    }
    if (canCycle) {
      live.current.onCycleLayout?.();
    }
  }

  return (
    <View
      style={{
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 0,
        minHeight: 0,
        paddingTop: space[2],
        paddingHorizontal: space[2],
        position: 'relative',
      }}
      {...(Platform.OS === 'web'
        ? {
            onPointerDown,
            onPointerUp,
            onPointerCancel: () => {
              swipeOrigin.current = null;
            },
          }
        : null)}
    >
      <View
        style={
          {
            flexGrow: 1,
            flexShrink: 1,
            flexBasis: 0,
            minHeight: 0,
            width: '100%',
            opacity,
            ...(Platform.OS === 'web' && !reduceMotion
              ? {
                  transitionProperty: 'opacity',
                  transitionDuration: `${duration.fast}ms`,
                  transitionTimingFunction: 'ease-in-out',
                }
              : null),
          } as object
        }
      >
        <View
          accessibilityRole={canCycle ? 'button' : undefined}
          accessibilityLabel={cycleLabel}
          style={
            {
              flexGrow: 1,
              flexShrink: 1,
              flexBasis: 0,
              minHeight: 0,
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              position: 'relative',
              ...(Platform.OS === 'web' && canCycle ? { cursor: 'pointer' } : null),
            } as object
          }
        >
          {displayLower.endsWith('.mp3') ? (
            <audio controls src={displayUrl} style={{ width: '100%', maxWidth: 480, margin: '0 auto', display: 'block' }} />
          ) : displayLower.endsWith('.mp4') ? (
            <video
              controls
              src={displayUrl}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: fit, margin: '0 auto', display: 'block' }}
            />
          ) : (
            <img
              alt={displayName}
              src={displayUrl}
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: fit,
                display: 'block',
              }}
            />
          )}
        </View>
        <Text numberOfLines={1} style={[textStyle('caption', palette.textSecondary), { textAlign: 'center', marginTop: space[2] }]}>
          {displayName}
        </Text>
      </View>
      {actionsInPane && bannerActions != null ? (
        <View
          style={{
            position: 'absolute',
            top: space[2],
            right: space[2],
            zIndex: 2,
            flexDirection: 'row',
            alignItems: 'center',
            gap: space[2],
          }}
          {...(Platform.OS === 'web'
            ? {
                onPointerDown: (event: { stopPropagation?: () => void }) => event.stopPropagation?.(),
                onPointerUp: (event: { stopPropagation?: () => void }) => event.stopPropagation?.(),
              }
            : null)}
        >
          {bannerActions}
        </View>
      ) : null}
    </View>
  );
}

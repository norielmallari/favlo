import { useEffect, useRef, useState } from 'react';
import { Platform, ScrollView, Text, View } from 'react-native';

import { focusRingInset, focusSafeGap, Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { duration, radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import type { ElementProps } from '@/workspace/catalog';

type Item = { path: string };

export function Filmstrip({ props, selection, fileUrl, onSelect, pendingMedia = [] }: ElementProps) {
  const { palette } = useTheme();
  const title = typeof props.title === 'string' ? props.title : '';
  const items = (Array.isArray(props.items) ? props.items : []) as Item[];
  const priorPaths = useRef<Set<string>>(new Set());
  const [arrived, setArrived] = useState<Set<string>>(new Set());
  const reduceMotion =
    Platform.OS === 'web' &&
    typeof window !== 'undefined' &&
    Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);

  useEffect(() => {
    const next = new Set(items.map((item) => item.path));
    const hadItems = priorPaths.current.size > 0;
    const fresh = hadItems ? [...next].filter((path) => !priorPaths.current.has(path)) : [];
    priorPaths.current = next;
    if (fresh.length === 0 || reduceMotion) {
      return;
    }
    setArrived(new Set(fresh));
    const timer = setTimeout(() => setArrived(new Set()), duration.med);
    return () => clearTimeout(timer);
  }, [items, reduceMotion]);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof document === 'undefined' || selection?.path == null) {
      return;
    }
    const node = document.querySelector(
      `[data-filmstrip-path="${CSS.escape(selection.path)}"], [data-filmstrippath="${CSS.escape(selection.path)}"]`,
    ) as HTMLElement | null;
    if (node != null && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ inline: 'center', block: 'nearest', behavior: reduceMotion ? 'auto' : 'smooth' });
    }
  }, [selection?.path, items.length, reduceMotion]);

  if (items.length === 0 && pendingMedia.length === 0) {
    return (
      <View style={{ padding: space[3], alignItems: 'center', justifyContent: 'center' }}>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>
          Ask the agent to fill the filmstrip
        </Text>
      </View>
    );
  }

  const selectedIndex = Math.max(
    0,
    items.findIndex((item) => item.path === selection?.path),
  );
  const positionLabel =
    items.length === 0
      ? `${pendingMedia.length} generating`
      : `${selectedIndex + 1} of ${items.length}`;
  const snapStyle =
    Platform.OS === 'web'
      ? ({ scrollSnapType: 'x mandatory', WebkitOverflowScrolling: 'touch' } as object)
      : undefined;
  const edgePad = Math.max(space[3], focusRingInset);

  return (
    <View style={{ gap: space[1], paddingVertical: Math.max(space[2], focusRingInset), alignItems: 'stretch' }}>
      {title ? (
        <Text style={[textStyle('label', palette.textPrimary), { textAlign: 'center', paddingHorizontal: space[3] }]}>
          {title}
        </Text>
      ) : null}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={snapStyle}
        contentContainerStyle={{
          gap: Math.max(space[2], focusSafeGap),
          paddingHorizontal: edgePad,
          paddingVertical: focusRingInset,
          alignItems: 'center',
        }}
      >
        {items.map((item) => {
          const selected = selection?.path === item.path;
          const name = item.path.split('/').pop() ?? item.path;
          const lower = item.path.toLowerCase();
          const justArrived = arrived.has(item.path);
          const mediaLabel = mediaPlaceholder(lower);
          const frameStyle =
            Platform.OS === 'web'
              ? ({ scrollSnapAlign: 'center' } as object)
              : undefined;
          return (
            <View
              key={item.path}
              {...(Platform.OS === 'web'
                ? ({ dataSet: { filmstripPath: item.path } } as object)
                : null)}
              style={frameStyle}
            >
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={name}
                accessibilityState={{ selected }}
                onPress={() => onSelect(item.path)}
                style={{
                  width: 96,
                  borderWidth: selected ? 2 : 1,
                  borderColor: selected ? palette.accent : palette.border,
                  borderRadius: radius.control,
                  backgroundColor: palette.surface,
                  opacity: selected ? 1 : justArrived && !reduceMotion ? 0.55 : 0.72,
                  ...(Platform.OS === 'web' && justArrived && !reduceMotion
                    ? ({
                        transitionProperty: 'opacity',
                        transitionDuration: `${duration.med}ms`,
                        transitionTimingFunction: 'ease-in-out',
                      } as object)
                    : null),
                }}
              >
                <View style={{ overflow: 'hidden', borderRadius: radius.control - 1 }}>
                  {mediaLabel ? (
                    <View style={{ height: 72, alignItems: 'center', justifyContent: 'center', padding: space[1] }}>
                      <Text style={[textStyle('caption', palette.textPrimary), { textAlign: 'center' }]}>
                        {mediaLabel}
                      </Text>
                      <Text numberOfLines={2} style={[textStyle('caption', palette.textSecondary), { textAlign: 'center' }]}>
                        {name}
                      </Text>
                    </View>
                  ) : (
                    <View style={{ width: 96, height: 72, alignItems: 'center', justifyContent: 'center' }}>
                      <img
                        alt={name}
                        src={fileUrl(item.path)}
                        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                      />
                    </View>
                  )}
                </View>
              </Pressable>
            </View>
          );
        })}
        {pendingMedia.map((item) => (
          <View
            key={item.task_id}
            {...(Platform.OS === 'web' ? { 'data-filmstrip-pending': item.task_id } : null)}
            accessibilityLabel={`Generating ${item.kind}`}
            style={{
              width: 96,
              height: 72,
              borderWidth: 1,
              borderColor: palette.border,
              borderRadius: radius.control,
              borderStyle: 'dashed',
              backgroundColor: palette.surface,
              alignItems: 'center',
              justifyContent: 'center',
              padding: space[1],
              ...(Platform.OS === 'web'
                ? ({ scrollSnapAlign: 'center' } as object)
                : null),
            }}
          >
            <Text style={[textStyle('caption', palette.textPrimary), { textAlign: 'center' }]}>
              {item.kind === 'video' ? 'Video' : item.kind === 'audio' ? 'Audio' : 'Image'}
            </Text>
            <Text style={[textStyle('caption', palette.textSecondary), { textAlign: 'center' }]}>
              Generating…
            </Text>
          </View>
        ))}
      </ScrollView>
      <Text
        accessibilityLiveRegion="polite"
        style={[textStyle('caption', palette.textSecondary), { textAlign: 'center' }]}
      >
        {positionLabel}
      </Text>
    </View>
  );
}

function mediaPlaceholder(path: string): string | null {
  if (path.endsWith('.mp4')) {
    return 'Video';
  }
  if (path.endsWith('.mp3') || path.endsWith('.wav')) {
    return 'Audio';
  }
  return null;
}

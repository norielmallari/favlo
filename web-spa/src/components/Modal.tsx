import { useEffect, useRef, type ReactNode } from 'react';
import { Platform, ScrollView, Text, useWindowDimensions, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function Modal({
  label,
  closeLabel,
  children,
  onClose,
}: {
  label: string;
  closeLabel: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const { palette } = useTheme();
  const { height } = useWindowDimensions();
  const panel = useRef<View>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }
    const node = panel.current as unknown as HTMLElement | null;
    if (node == null) {
      return;
    }
    const hidden: Element[] = [];
    for (const child of document.body.children) {
      if (child.contains(node)) {
        continue;
      }
      child.setAttribute('inert', '');
      hidden.push(child);
    }
    const focusable = node.querySelector<HTMLElement>('input, textarea, button, [tabindex]:not([tabindex="-1"])');
    (focusable ?? node).focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close.current();
        return;
      }
      if (event.key !== 'Tab') {
        return;
      }
      const items = [...node.querySelectorAll<HTMLElement>('input, textarea, button, [tabindex]:not([tabindex="-1"])')];
      const first = items[0];
      const last = items.at(-1);
      if (first == null || last == null) {
        return;
      }
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      for (const child of hidden) {
        child.removeAttribute('inert');
      }
    };
  }, []);

  return (
    <View
      accessibilityViewIsModal
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: z.modal,
        backgroundColor: palette.overlay,
        justifyContent: 'center',
        padding: space[4],
      }}
    >
      <View
        ref={panel}
        accessibilityLabel={label}
        {...(Platform.OS === 'web' ? { role: 'dialog' as const, tabIndex: -1 } : {})}
        style={{
          backgroundColor: palette.surface,
          borderRadius: radius.dialog,
          maxWidth: 480,
          alignSelf: 'center',
          width: '100%',
          maxHeight: height - 32,
          overflow: 'hidden',
        }}
      >
        <ScrollView contentContainerStyle={{ padding: space[4], gap: space[3] }}>
          {children}
          <Pressable accessibilityRole="button" accessibilityLabel={closeLabel} onPress={onClose} style={{ minHeight: 44, justifyContent: 'center' }}>
            <Text style={textStyle('small', palette.accent, '600')}>Close</Text>
          </Pressable>
        </ScrollView>
      </View>
    </View>
  );
}

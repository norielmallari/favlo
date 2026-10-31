import { createElement, useEffect, useState, type ReactNode } from 'react';
import { Platform, Text, useWindowDimensions, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme, type Preference } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

const ORDER: Preference[] = ['system', 'light', 'dark'];
const APPEARANCE: Record<Preference, string> = { system: 'System', light: 'Light', dark: 'Dark' };

export function Frame({
  children,
  onSkip,
  link,
  title = 'Favlo',
}: {
  children: ReactNode;
  onSkip?: () => void;
  link: { label: string; onPress: () => void };
  title?: string;
}) {
  const { palette, preference, setPreference } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < 1024;
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof document === 'undefined') {
      return;
    }
    document.title = title;
  }, [title]);

  useEffect(() => {
    if (Platform.OS !== 'web' || !menu) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenu(false);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [menu]);

  return (
    <View style={{ flex: 1, width: '100%', maxWidth: '100%', backgroundColor: palette.background }}>
      {Platform.OS === 'web' && onSkip
        ? createElement(
            'style',
            null,
            '[data-testid="skip-to-message"]:not(:focus-within){position:absolute;left:-10000px;width:1px;height:1px;overflow:hidden}',
          )
        : null}
      <View
        testID="app-banner"
        {...(Platform.OS === 'web' ? { role: 'banner' as const } : {})}
        style={{
          minHeight: 48,
          width: '100%',
          maxWidth: '100%',
          paddingHorizontal: space[3],
          paddingVertical: 4,
          flexDirection: 'row',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: narrow ? space[2] : space[3],
          borderBottomWidth: 1,
          borderBottomColor: palette.border,
          backgroundColor: palette.surface,
          zIndex: z.banner,
        }}
      >
        {onSkip ? (
          <Pressable
            testID="skip-to-message"
            accessibilityRole="link"
            accessibilityLabel="Skip to message"
            onPress={onSkip}
            style={target}
          >
            <Text style={textStyle('small', palette.accent)}>Skip to message</Text>
          </Pressable>
        ) : null}
        <Text accessibilityRole="header" style={textStyle('title', palette.textPrimary)}>
          Favlo
        </Text>
        <View style={{ marginLeft: narrow ? 0 : 'auto', position: 'relative' }}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Appearance, ${APPEARANCE[preference]}`}
            accessibilityState={{ expanded: menu }}
            onPress={() => setMenu((open) => !open)}
            style={[
              target,
              {
                minWidth: 44,
                alignItems: 'center',
                borderRadius: radius.control,
                borderWidth: 1,
                borderColor: palette.borderStrong,
                backgroundColor: palette.surface,
                paddingHorizontal: space[2],
              },
            ]}
          >
            <AppearanceMark preference={preference} color={palette.textPrimary} />
          </Pressable>
          {menu ? (
            <View
              accessibilityRole="menu"
              accessibilityLabel="Appearance"
              style={{
                position: 'absolute',
                top: 48,
                ...(narrow ? { left: 0 } : { right: 0 }),
                zIndex: z.modal,
                minWidth: 140,
                maxWidth: 240,
                padding: space[1],
                borderRadius: radius.control,
                borderWidth: 1,
                borderColor: palette.borderStrong,
                backgroundColor: palette.surfaceRaised,
              }}
            >
              {ORDER.map((item) => {
                const selected = item === preference;
                return (
                  <Pressable
                    key={item}
                    accessibilityRole="menuitem"
                    accessibilityLabel={APPEARANCE[item]}
                    accessibilityState={{ selected }}
                    onPress={() => {
                      setPreference(item);
                      setMenu(false);
                    }}
                    style={{
                      minHeight: 44,
                      justifyContent: 'center',
                      paddingHorizontal: space[3],
                      borderRadius: radius.control,
                      backgroundColor: selected ? palette.accentSubtle : palette.surfaceRaised,
                    }}
                  >
                    <Text style={textStyle('label', palette.textPrimary, selected ? '600' : '400')}>{APPEARANCE[item]}</Text>
                  </Pressable>
                );
              })}
            </View>
          ) : null}
        </View>
        <Pressable accessibilityRole="button" accessibilityLabel={link.label} onPress={link.onPress} style={target}>
          <Text style={textStyle('small', palette.accent, '600')}>{link.label}</Text>
        </Pressable>
      </View>
      {children}
    </View>
  );
}

const target = { minHeight: 44, justifyContent: 'center' as const };

function AppearanceMark({ preference, color }: { preference: Preference; color: string }) {
  if (Platform.OS !== 'web') {
    const glyph = preference === 'light' ? '☀' : preference === 'dark' ? '☾' : '◐';
    return <Text style={{ color, fontSize: 18, lineHeight: 22 }}>{glyph}</Text>;
  }
  const shape = preference === 'light' ? sun(color) : preference === 'dark' ? moon(color) : contrast(color);
  return createElement('svg', {
    width: 22,
    height: 22,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: color,
    strokeWidth: 1.75,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    'aria-hidden': true,
    focusable: 'false',
  }, shape);
}

function sun(color: string): ReactNode {
  return [
    createElement('circle', { key: 'disk', cx: 12, cy: 12, r: 4, stroke: color }),
    createElement('path', {
      key: 'rays',
      d: 'M12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8',
      stroke: color,
    }),
  ];
}

function moon(color: string): ReactNode {
  return createElement('path', {
    d: 'M21 14.5A8.5 8.5 0 1 1 9.5 3a7 7 0 0 0 11.5 11.5z',
    fill: color,
    stroke: 'none',
  });
}

function contrast(color: string): ReactNode {
  return [
    createElement('circle', { key: 'ring', cx: 12, cy: 12, r: 8, stroke: color }),
    createElement('path', { key: 'half', d: 'M12 4a8 8 0 1 0 0 16z', fill: color, stroke: 'none' }),
  ];
}

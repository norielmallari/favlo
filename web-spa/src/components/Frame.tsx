import { createElement, type ReactNode } from 'react';
import { Platform, Text, useWindowDimensions, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme, type Preference } from '@/theme/Theme';

const ORDER: Preference[] = ['system', 'light', 'dark'];
const APPEARANCE: Record<Preference, string> = { system: 'System', light: 'Light', dark: 'Dark' };

export function nextAppearance(current: Preference): Preference {
  const index = ORDER.indexOf(current);
  return ORDER[(index + 1) % ORDER.length];
}

export function Frame({
  children,
  onSkip,
  link,
}: {
  children: ReactNode;
  onSkip?: () => void;
  link: { label: string; onPress: () => void };
}) {
  const { palette, preference, setPreference } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < 1024;
  return (
    <View style={{ flex: 1, width: '100%', maxWidth: '100%', backgroundColor: palette.background }}>
      <View
        testID="app-banner"
        {...(Platform.OS === 'web' ? { role: 'banner' as const } : {})}
        style={{
          minHeight: 48,
          width: '100%',
          maxWidth: '100%',
          paddingHorizontal: 12,
          paddingVertical: 4,
          flexDirection: 'row',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: narrow ? 8 : 12,
          borderBottomWidth: 1,
          borderBottomColor: palette.border,
          backgroundColor: palette.surface,
        }}
      >
        {onSkip ? (
          <Pressable accessibilityRole="link" accessibilityLabel="Skip to message" onPress={onSkip} style={target}>
            <Text style={{ color: palette.accent, fontSize: 14 }}>Skip to message</Text>
          </Pressable>
        ) : null}
        <Text accessibilityRole="header" style={{ color: palette.textPrimary, fontSize: 18, lineHeight: 24, fontWeight: '600' }}>
          Favlo
        </Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Appearance, ${APPEARANCE[preference]}`}
          onPress={() => setPreference(nextAppearance(preference))}
          style={[
            target,
            {
              marginLeft: narrow ? 0 : 'auto',
              minWidth: 44,
              alignItems: 'center',
              borderRadius: 6,
              borderWidth: 1,
              borderColor: palette.borderStrong,
              backgroundColor: palette.surface,
              paddingHorizontal: 8,
            },
          ]}
        >
          <AppearanceMark preference={preference} color={palette.textPrimary} />
        </Pressable>
        <Pressable accessibilityRole="button" accessibilityLabel={link.label} onPress={link.onPress} style={target}>
          <Text style={{ color: palette.accent, fontSize: 14, fontWeight: '600' }}>{link.label}</Text>
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
    createElement('path', { key: 'half', d: 'M12 4a8 8 0 0 0 0 16z', fill: color, stroke: 'none' }),
  ];
}

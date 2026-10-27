import type { ReactNode } from 'react';
import { Platform, Text, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme, type Preference } from '@/theme/Theme';

const CHOICES: Preference[] = ['system', 'light', 'dark'];

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
  return (
    <View style={{ flex: 1, backgroundColor: palette.background }}>
      <View
        {...(Platform.OS === 'web' ? { role: 'banner' as const } : {})}
        style={{
          minHeight: 48,
          paddingHorizontal: 12,
          flexDirection: 'row',
          alignItems: 'center',
          gap: 12,
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
        <View accessibilityRole="radiogroup" accessibilityLabel="Appearance" style={{ flexDirection: 'row', gap: 4, marginLeft: 'auto' }}>
          {CHOICES.map((choice) => {
            const selected = preference === choice;
            const label = choice[0].toUpperCase() + choice.slice(1);
            return (
              <Pressable
                key={choice}
                accessibilityRole="radio"
                accessibilityLabel={label}
                accessibilityState={{ selected }}
                aria-checked={selected}
                onPress={() => setPreference(choice)}
                style={[
                  target,
                  {
                    borderRadius: 6,
                    borderWidth: 1,
                    borderColor: selected ? palette.accent : palette.borderStrong,
                    backgroundColor: selected ? palette.accentSubtle : palette.surface,
                    paddingHorizontal: 8,
                  },
                ]}
              >
                <Text style={{ color: palette.textPrimary, fontSize: 14 }}>{label}</Text>
              </Pressable>
            );
          })}
        </View>
        <Pressable accessibilityRole="button" accessibilityLabel={link.label} onPress={link.onPress} style={target}>
          <Text style={{ color: palette.accent, fontSize: 14, fontWeight: '600' }}>{link.label}</Text>
        </Pressable>
      </View>
      {children}
    </View>
  );
}

const target = { minHeight: 44, justifyContent: 'center' as const };

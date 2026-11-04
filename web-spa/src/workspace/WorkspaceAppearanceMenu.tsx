import { useEffect } from 'react';
import { Appearance, Platform, Text, useColorScheme, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import { useTheme, type Preference } from '@/theme/Theme';
import {
  WORKSPACE_SCHEMES,
  workspaceSchemeSwatches,
  type WorkspaceScheme,
} from '@/theme/workspace-schemes';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

const ORDER: Preference[] = ['system', 'light', 'dark'];
const APPEARANCE: Record<Preference, string> = { system: 'System', light: 'Light', dark: 'Dark' };
const target = { minHeight: 44, justifyContent: 'center' as const };

export function WorkspaceAppearanceMenu({
  open,
  scheme,
  reducedMotion,
  onClose,
  onSelectScheme,
}: {
  open: boolean;
  scheme: WorkspaceScheme;
  reducedMotion: boolean;
  onClose: () => void;
  onSelectScheme: (scheme: WorkspaceScheme) => void;
}) {
  const { palette, preference, setPreference } = useTheme();
  const system = useColorScheme();

  useEffect(() => {
    if (Platform.OS !== 'web' || !open) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  const mode = preference === 'system' ? (system ?? Appearance.getColorScheme() ?? 'light') : preference;
  const resolved = mode === 'dark' ? 'dark' : 'light';

  return (
    <View
      accessibilityRole="menu"
      accessibilityLabel="Appearance"
      style={{
        position: 'absolute',
        top: 48,
        left: 0,
        zIndex: z.modal,
        minWidth: 220,
        maxWidth: 280,
        padding: space[1],
        borderRadius: radius.control,
        borderWidth: 1,
        borderColor: palette.borderStrong,
        backgroundColor: palette.surfaceRaised,
        opacity: reducedMotion ? 1 : 1,
      }}
    >
      {WORKSPACE_SCHEMES.map((item) => {
        const selected = item.id === scheme;
        const swatches = workspaceSchemeSwatches(item.id, resolved);
        return (
          <Pressable
            key={item.id}
            accessibilityRole="menuitemradio"
            accessibilityLabel={item.label}
            accessibilityState={{ checked: selected, selected }}
            {...(Platform.OS === 'web' ? ({ 'aria-checked': selected } as const) : null)}
            onPress={() => {
              onSelectScheme(item.id);
              onClose();
            }}
            style={{
              minHeight: 44,
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: space[3],
              paddingHorizontal: space[3],
              borderRadius: radius.control,
              backgroundColor: palette.surfaceRaised,
            }}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: space[2], flexShrink: 1 }}>
              <FeatherIcon
                name={selected ? 'check' : 'circle'}
                size={16}
                color={selected ? palette.accent : palette.borderStrong}
              />
              <Text style={textStyle('label', palette.textPrimary, selected ? '600' : '400')}>{item.label}</Text>
            </View>
            <View style={{ flexDirection: 'row', gap: 4 }}>
              {[swatches.background, swatches.surface, swatches.accent].map((color, index) => (
                <View
                  key={`${item.id}-${index}`}
                  accessibilityElementsHidden
                  importantForAccessibility="no-hide-descendants"
                  style={{
                    width: 14,
                    height: 14,
                    borderRadius: 4,
                    backgroundColor: color,
                    borderWidth: 1,
                    borderColor: palette.border,
                  }}
                />
              ))}
            </View>
          </Pressable>
        );
      })}
      <View
        accessibilityElementsHidden
        style={{
          height: 1,
          marginVertical: space[1],
          marginHorizontal: space[2],
          backgroundColor: palette.border,
        }}
      />
      {ORDER.map((item) => {
        const selected = item === preference;
        return (
          <Pressable
            key={item}
            accessibilityRole="menuitemradio"
            accessibilityLabel={APPEARANCE[item]}
            accessibilityState={{ checked: selected, selected }}
            {...(Platform.OS === 'web' ? ({ 'aria-checked': selected } as const) : null)}
            onPress={() => {
              setPreference(item);
              onClose();
            }}
            style={[
              target,
              {
                flexDirection: 'row',
                alignItems: 'center',
                gap: space[2],
                paddingHorizontal: space[3],
                borderRadius: radius.control,
                backgroundColor: palette.surfaceRaised,
              },
            ]}
          >
            <FeatherIcon
              name={item === 'light' ? 'sun' : item === 'dark' ? 'moon' : 'monitor'}
              size={18}
              color={palette.textPrimary}
            />
            <Text style={textStyle('label', palette.textPrimary, selected ? '600' : '400')}>
              {APPEARANCE[item]}
            </Text>
            {selected ? <FeatherIcon name="check" size={16} color={palette.accent} /> : null}
          </Pressable>
        );
      })}
    </View>
  );
}

import { useEffect, useState } from 'react';
import { Platform, Text, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import {
  GALLERY_SCHEMES,
  gallerySchemeLabel,
  type GalleryScheme,
} from '@/theme/gallery-schemes';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

const target = { minHeight: 44, justifyContent: 'center' as const };

export function GalleryThemeMenu({
  scheme,
  narrow,
  onSelect,
}: {
  scheme: GalleryScheme;
  narrow: boolean;
  onSelect: (scheme: GalleryScheme) => void;
}) {
  const { palette } = useTheme();
  const [menu, setMenu] = useState(false);
  const label = gallerySchemeLabel(scheme);

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
    <View style={{ position: 'relative' }}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Gallery theme, ${label}`}
        accessibilityState={{ expanded: menu }}
        onPress={() => setMenu((open) => !open)}
        style={[
          target,
          {
            flexDirection: 'row',
            alignItems: 'center',
            gap: space[2],
            borderRadius: radius.control,
            borderWidth: 1,
            borderColor: palette.borderStrong,
            backgroundColor: palette.surface,
            paddingHorizontal: space[2],
          },
        ]}
      >
        <FeatherIcon name="droplet" size={18} color={palette.textPrimary} />
        {narrow ? null : (
          <Text style={textStyle('label', palette.textPrimary)}>{label}</Text>
        )}
      </Pressable>
      {menu ? (
        <View
          accessibilityRole="menu"
          accessibilityLabel="Gallery theme"
          style={{
            position: 'absolute',
            top: 48,
            ...(narrow ? { left: 0 } : { right: 0 }),
            zIndex: z.modal,
            minWidth: 160,
            padding: space[1],
            borderRadius: radius.control,
            borderWidth: 1,
            borderColor: palette.borderStrong,
            backgroundColor: palette.surfaceRaised,
          }}
        >
          {GALLERY_SCHEMES.map((item) => {
            const selected = item.id === scheme;
            return (
              <Pressable
                key={item.id}
                accessibilityRole="menuitem"
                accessibilityLabel={item.label}
                accessibilityState={{ selected }}
                onPress={() => {
                  setMenu(false);
                  onSelect(item.id);
                }}
                style={{
                  minHeight: 44,
                  justifyContent: 'center',
                  paddingHorizontal: space[3],
                  borderRadius: radius.control,
                  backgroundColor: selected ? palette.accentSubtle : palette.surfaceRaised,
                }}
              >
                <Text style={textStyle('label', palette.textPrimary, selected ? '600' : '400')}>
                  {item.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      ) : null}
    </View>
  );
}

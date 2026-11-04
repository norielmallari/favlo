import { useEffect, useState } from 'react';
import { Platform, Text, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import type { WorkspaceApp } from '@/workspace/workspace-apps';
import { useWorkspaceChromeActions } from '@/workspace/workspace-chrome';
import { WorkspaceAppearanceMenu } from '@/workspace/WorkspaceAppearanceMenu';
import type { WorkspaceScheme } from '@/theme/workspace-schemes';

const target = { minHeight: 44, justifyContent: 'center' as const };

export function WorkspaceTitleMenu({
  app,
  name,
  scheme,
  reducedMotion,
  onSelectScheme,
}: {
  app: WorkspaceApp;
  name: string;
  scheme: WorkspaceScheme;
  reducedMotion: boolean;
  onSelectScheme: (scheme: WorkspaceScheme) => void;
}) {
  const { palette } = useTheme();
  const actions = useWorkspaceChromeActions();
  const [menu, setMenu] = useState(false);
  const [appearance, setAppearance] = useState(false);

  useEffect(() => {
    if (Platform.OS !== 'web' || (!menu && !appearance)) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setAppearance(false);
        setMenu(false);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [menu, appearance]);

  const open = menu || appearance;

  return (
    <View style={{ position: 'relative', flexGrow: 0, flexShrink: 1, minWidth: 0, maxWidth: 280 }}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`${name}, ${app.name} workspace menu`}
        accessibilityState={{ expanded: open }}
        onPress={() => {
          if (appearance) {
            setAppearance(false);
            setMenu(false);
            return;
          }
          setMenu((value) => !value);
        }}
        style={[
          target,
          {
            flexDirection: 'row',
            alignItems: 'center',
            alignSelf: 'flex-start',
            gap: space[1],
            maxWidth: '100%',
            paddingHorizontal: space[2],
            borderRadius: radius.control,
          },
        ]}
      >
        <Text
          accessibilityRole="header"
          aria-level={2}
          numberOfLines={1}
          style={[textStyle('title', palette.textPrimary), { flexShrink: 1, minWidth: 0 }]}
        >
          {name}
        </Text>
        <FeatherIcon name="chevron-down" size={18} color={palette.textSecondary} />
      </Pressable>
      {menu && !appearance ? (
        <View
          accessibilityRole="menu"
          accessibilityLabel={`${app.name} workspace menu`}
          style={{
            position: 'absolute',
            top: 48,
            left: 0,
            zIndex: z.modal,
            minWidth: 200,
            maxWidth: 280,
            padding: space[1],
            borderRadius: radius.control,
            borderWidth: 1,
            borderColor: palette.borderStrong,
            backgroundColor: palette.surfaceRaised,
          }}
        >
          <Pressable
            accessibilityRole="menuitem"
            accessibilityLabel="Skills"
            onPress={() => {
              setMenu(false);
              actions.openSkills();
            }}
            style={[target, { paddingHorizontal: space[3], borderRadius: radius.control }]}
          >
            <Text style={textStyle('label', palette.textPrimary)}>Skills…</Text>
          </Pressable>
          <Pressable
            accessibilityRole="menuitem"
            accessibilityLabel="Appearance"
            onPress={() => {
              setMenu(false);
              setAppearance(true);
            }}
            style={[target, { paddingHorizontal: space[3], borderRadius: radius.control }]}
          >
            <Text style={textStyle('label', palette.textPrimary)}>Appearance…</Text>
          </Pressable>
          <Pressable
            accessibilityRole="menuitem"
            accessibilityLabel="Open Favlo chat"
            onPress={() => {
              setMenu(false);
              if (Platform.OS === 'web') {
                window.open('/', '_blank', 'noopener');
              }
            }}
            style={[target, { paddingHorizontal: space[3], borderRadius: radius.control }]}
          >
            <Text style={textStyle('label', palette.textPrimary)}>Open Favlo chat</Text>
          </Pressable>
          <View
            accessibilityElementsHidden
            style={{
              height: 1,
              marginVertical: space[1],
              marginHorizontal: space[2],
              backgroundColor: palette.border,
            }}
          />
          <Pressable
            accessibilityRole="menuitem"
            accessibilityLabel="Delete workspace"
            onPress={() => {
              setMenu(false);
              actions.requestDelete();
            }}
            style={[target, { paddingHorizontal: space[3], borderRadius: radius.control }]}
          >
            <Text style={textStyle('label', palette.danger)}>Delete workspace</Text>
          </Pressable>
        </View>
      ) : null}
      <WorkspaceAppearanceMenu
        open={appearance}
        scheme={scheme}
        reducedMotion={reducedMotion}
        onClose={() => setAppearance(false)}
        onSelectScheme={onSelectScheme}
      />
    </View>
  );
}

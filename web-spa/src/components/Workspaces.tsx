import { Text, View } from 'react-native';

import { MenuRow } from '@/components/MenuRow';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type Workspace = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  root: {
    id: string;
    kind: string;
    provider: string | null;
    location: string | null;
    name: string;
  } | null;
};

export function Workspaces({
  workspaces,
  selectedId,
  onSelect,
  onCreate,
}: {
  workspaces: Workspace[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
}) {
  const { palette } = useTheme();
  return (
    <View accessibilityLabel="Workspaces" style={{ flex: 1, padding: space[4], gap: space[2] }}>
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        Workspaces
      </Text>
      {workspaces.length === 0 ? (
        <Text style={textStyle('small', palette.textSecondary)}>No workspaces</Text>
      ) : (
        <View style={{ gap: 8 }}>
          {workspaces.map((item) => (
            <MenuRow
              key={item.id}
              label={item.name}
              selected={item.id === selectedId}
              onPress={() => onSelect(item.id)}
            />
          ))}
        </View>
      )}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="New workspace"
        onPress={onCreate}
        style={{
          minHeight: 44,
          minWidth: 44,
          borderRadius: radius.control,
          borderWidth: 1,
          borderColor: palette.accent,
          backgroundColor: palette.accent,
          paddingHorizontal: space[3],
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Text style={textStyle('small', palette.onAccent, '600')}>New workspace</Text>
      </Pressable>
    </View>
  );
}

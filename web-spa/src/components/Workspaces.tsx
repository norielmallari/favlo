import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { MenuRow } from '@/components/MenuRow';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type Workspace = {
  id: string;
  name: string;
  type: string;
  session_id: string;
};

export function Workspaces({
  workspaces,
  selectedId,
  homeSelected,
  onHome,
  onSelect,
  onCreate,
}: {
  workspaces: Workspace[];
  selectedId: string | null;
  homeSelected: boolean;
  onHome: () => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
}) {
  const { palette } = useTheme();
  return (
    <View accessibilityLabel="Workspaces" style={{ gap: space[2], paddingLeft: space[4] }}>
      <MenuRow label="Home" icon="home" selected={homeSelected} onPress={onHome} />
      {workspaces.length === 0 ? (
        <Text style={textStyle('small', palette.textSecondary)}>No workspaces</Text>
      ) : (
        <View style={{ gap: 8 }}>
          {workspaces.map((item) => (
            <MenuRow
              key={item.id}
              label={item.name}
              icon={item.type === 'gallery' ? 'image' : 'box'}
              selected={item.id === selectedId}
              selectedTestId=""
              onPress={() => onSelect(item.id)}
            />
          ))}
        </View>
      )}
      <IconButton icon="plus" label="New workspace" onPress={onCreate} background={palette.accent} color={palette.onAccent} />
    </View>
  );
}

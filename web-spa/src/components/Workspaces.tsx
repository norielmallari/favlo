import { useState } from 'react';
import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { MenuRow } from '@/components/MenuRow';
import { Modal } from '@/components/Modal';
import { focusRingInset, focusSafeGap } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type Workspace = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  icon?: string;
  provisioning_state?: string;
};

export function Workspaces({
  workspaces,
  agentName,
  selectedId,
  homeSelected,
  onHome,
  onSelect,
  onCreate,
  onDelete,
}: {
  workspaces: Workspace[];
  agentName: string;
  selectedId: string | null;
  homeSelected: boolean;
  onHome: () => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => Promise<void>;
}) {
  const { palette } = useTheme();
  const [pending, setPending] = useState<Workspace | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  return (
    <View
      accessibilityLabel="Workspaces"
      style={{
        gap: space[2],
        paddingLeft: space[4],
        paddingRight: focusRingInset,
        paddingVertical: focusRingInset,
      }}
    >
      <MenuRow label="Home" icon="home" selected={homeSelected} onPress={onHome} />
      {workspaces.length === 0 ? (
        <Text style={textStyle('small', palette.textSecondary)}>No workspaces</Text>
      ) : (
        <View style={{ gap: focusSafeGap }}>
          {workspaces.map((item) => (
            <View key={item.id} style={{ gap: 2 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: space[1], minWidth: 0 }}>
                <View style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minWidth: 0 }}>
                  <MenuRow
                    label={item.name}
                    icon={item.icon || 'layers'}
                    selected={item.id === selectedId}
                    selectedTestId=""
                    onPress={() => onSelect(item.id)}
                  />
                </View>
                <IconButton
                  icon="trash-2"
                  label={`Delete ${item.name}`}
                  iconOnly
                  tone="dangerGhost"
                  onPress={() => {
                    setNotice(null);
                    setPending(item);
                  }}
                />
              </View>
              {item.provisioning_state === 'new' || item.provisioning_state === 'provisioning' ? (
                <Text testID="workspace-creating" style={textStyle('small', palette.textSecondary)}>
                  Creating
                </Text>
              ) : null}
            </View>
          ))}
        </View>
      )}
      <IconButton
        icon="plus"
        label={`New workspace for ${agentName}`}
        onPress={onCreate}
        block
        background={palette.accent}
        color={palette.onAccent}
      />
      {pending ? (
        <Modal label="Delete workspace" closeLabel="Close delete workspace" onClose={() => setPending(null)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {pending.name}?
          </Text>
          <Text style={textStyle('body', palette.textSecondary)}>Folders and files on disk stay.</Text>
          {notice ? <Text style={textStyle('body', palette.danger)}>{notice}</Text> : null}
          <IconButton
            icon="trash-2"
            label={`Delete ${pending.name}`}
            tone="danger"
            onPress={() => {
              const id = pending.id;
              void onDelete(id)
                .then(() => setPending(null))
                .catch((caught: Error) => setNotice(caught.message));
            }}
          />
        </Modal>
      ) : null}
    </View>
  );
}

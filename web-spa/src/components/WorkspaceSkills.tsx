import { useState } from 'react';
import { Text, View } from 'react-native';

import { api, idempotencyKey } from '@/api/client';
import { Modal } from '@/components/Modal';
import { FeatherIcon } from '@/components/FeatherIcon';
import { buttonRow, IconButton } from '@/components/IconButton';
import { Pressable } from '@/components/Pressable';
import { SkillForm } from '@/components/SkillForm';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type WorkspaceSkill = {
  id: string;
  workspace_id: string;
  name: string;
  description: string;
  definition?: string;
  revision: number;
};

export function WorkspaceSkills({
  workspaceId,
  skills,
  onChange,
  onClose,
}: {
  workspaceId: string;
  skills: WorkspaceSkill[];
  onChange: (skills: WorkspaceSkill[]) => void;
  onClose: () => void;
}) {
  const { palette } = useTheme();
  const [pane, setPane] = useState<'list' | 'form'>('list');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [token, setToken] = useState(0);
  const selected = skills.find((item) => item.id === selectedId) ?? null;

  function openNew() {
    setCreating(true);
    setSelectedId(null);
    setToken((current) => current + 1);
    setPane('form');
  }

  function openSkill(skill: WorkspaceSkill) {
    setCreating(false);
    setSelectedId(skill.id);
    setPane('form');
  }

  async function save(fields: { name: string; description: string }) {
    if (creating || selected == null) {
      const created = await api<WorkspaceSkill>(`/workspaces/${workspaceId}/skills`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify(fields),
      });
      onChange([...skills.filter((item) => item.id !== created.id), created]);
      setCreating(false);
      setSelectedId(created.id);
      return;
    }
    const saved = await api<WorkspaceSkill>(`/workspaces/${workspaceId}/skills/${selected.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ ...fields, expected_revision: selected.revision }),
    });
    onChange(skills.map((item) => (item.id === saved.id ? saved : item)));
  }

  async function remove() {
    if (selected == null) {
      return;
    }
    await api(`/workspaces/${workspaceId}/skills/${selected.id}`, { method: 'DELETE' });
    onChange(skills.filter((item) => item.id !== selected.id));
    setSelectedId(null);
    setCreating(false);
    setPane('list');
  }

  return (
    <Modal label="Skills" closeLabel="Close skills" onClose={onClose}>
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        Skills
      </Text>
      <Text style={textStyle('small', palette.textSecondary)}>
        Any agent in this workspace uses these skills.
      </Text>
      {pane === 'list' ? (
        <View accessibilityLabel="Workspace skills" style={{ gap: space[3] }}>
          {skills.length === 0 ? (
            <Text style={textStyle('small', palette.textSecondary)}>No skills</Text>
          ) : (
            skills.map((item) => (
              <Pressable
                key={item.id}
                accessibilityRole="button"
                accessibilityLabel={item.name}
                onPress={() => openSkill(item)}
                style={[barButton(palette.surface, palette.borderStrong), { flexDirection: 'row', gap: space[2], alignSelf: 'flex-start' }]}
              >
                <FeatherIcon name="book-open" size={18} color={palette.textPrimary} />
                <Text style={textStyle('body', palette.textPrimary, '600')}>{item.name}</Text>
              </Pressable>
            ))
          )}
          <IconButton icon="plus" label="New skill" onPress={openNew} background={palette.accent} color={palette.onAccent} />
        </View>
      ) : (
        <View style={{ gap: space[3] }}>
          <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to skills" onPress={() => setPane('list')} background="transparent" border="transparent" color={palette.accent} />
          <SkillForm
            key={creating ? `new-${token}` : `${selected?.id}:${selected?.revision}`}
            skill={creating ? null : selected}
            onSave={save}
            onDelete={selected == null ? undefined : remove}
          />
        </View>
      )}
    </Modal>
  );
}

function barButton(background: string, border: string) {
  return {
    minHeight: 44,
    minWidth: 44,
    borderRadius: radius.control,
    borderWidth: 1,
    borderColor: border,
    backgroundColor: background,
    paddingHorizontal: space[3],
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  };
}

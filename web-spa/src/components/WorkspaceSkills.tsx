import { useState } from 'react';
import { Text, TextInput, View } from 'react-native';

import { api, idempotencyKey } from '@/api/client';
import { Modal } from '@/components/Modal';
import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type WorkspaceSkill = {
  id: string;
  workspace_id: string;
  name: string;
  description: string;
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
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const canAdd = name.trim() !== '';

  async function addSkill() {
    if (!canAdd) {
      return;
    }
    try {
      const created = await api<WorkspaceSkill>(`/workspaces/${workspaceId}/skills`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ name: name.trim(), description: description.trim() }),
      });
      onChange([...skills.filter((item) => item.id !== created.id), created]);
      setName('');
      setDescription('');
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The skill was not added.');
    }
  }

  async function removeSkill(skill: WorkspaceSkill) {
    try {
      await api(`/workspaces/${workspaceId}/skills/${skill.id}`, { method: 'DELETE' });
      onChange(skills.filter((item) => item.id !== skill.id));
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The skill was not deleted.');
    }
  }

  return (
    <Modal label="Skills" closeLabel="Close skills" onClose={onClose}>
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        Skills
      </Text>
      <Text style={textStyle('small', palette.textSecondary)}>
        Any agent in this workspace uses these skills.
      </Text>
      {skills.length === 0 ? (
        <Text style={textStyle('small', palette.textSecondary)}>No skills</Text>
      ) : (
        <View accessibilityLabel="Skill list" style={{ gap: space[3] }}>
          {skills.map((item) => (
            <View key={item.id} style={{ gap: space[1] }}>
              <Text style={textStyle('body', palette.textPrimary, '600')}>{item.name}</Text>
              {item.description ? (
                <Text style={textStyle('small', palette.textSecondary)}>{item.description}</Text>
              ) : null}
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`Delete ${item.name}`}
                onPress={() => void removeSkill(item)}
                style={barButton(palette.surface, palette.borderStrong)}
              >
                <Text style={textStyle('small', palette.textPrimary)}>Delete</Text>
              </Pressable>
            </View>
          ))}
        </View>
      )}
      <TextInput
        accessibilityLabel="Skill name"
        value={name}
        onChangeText={setName}
        style={fieldStyle(palette)}
      />
      <TextInput
        accessibilityLabel="Skill description"
        value={description}
        onChangeText={setDescription}
        multiline
        style={[fieldStyle(palette), { minHeight: 88, paddingVertical: 12 }]}
      />
      {error ? (
        <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={textStyle('small', palette.danger)}>
          {error}
        </Text>
      ) : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Add skill"
        accessibilityState={{ disabled: !canAdd }}
        disabled={!canAdd}
        onPress={() => void addSkill()}
        style={barButton(canAdd ? palette.accent : palette.disabledSurface, palette.accent)}
      >
        <Text style={textStyle('small', canAdd ? palette.onAccent : palette.disabledText, '600')}>Add skill</Text>
      </Pressable>
    </Modal>
  );
}

function fieldStyle(palette: { textPrimary: string; surface: string; borderStrong: string }) {
  return {
    ...textStyle('body', palette.textPrimary),
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: radius.control,
    paddingHorizontal: 12,
    backgroundColor: palette.surface,
  };
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

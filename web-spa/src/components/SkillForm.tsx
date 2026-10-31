import { useState } from 'react';
import { Switch, Text, TextInput, View } from 'react-native';

import { buttonRow, IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { radius } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type SkillFields = {
  name: string;
  description: string;
  definition?: string;
  add_by_default?: boolean;
};

export function SkillForm({
  skill,
  onSave,
  onDelete,
  showDefault = false,
}: {
  skill: SkillFields | null;
  onSave: (fields: SkillFields) => Promise<void>;
  onDelete?: () => Promise<void>;
  showDefault?: boolean;
}) {
  const { palette } = useTheme();
  const [name, setName] = useState(skill?.name ?? '');
  const [description, setDescription] = useState(skill?.description ?? '');
  const [definition, setDefinition] = useState(skill?.definition ?? '');
  const [addByDefault, setAddByDefault] = useState(skill?.add_by_default ?? false);
  const [error, setError] = useState<string | null>(null);
  const ready = name.trim().length > 0;

  async function save() {
    if (!ready) {
      return;
    }
    try {
      await onSave({
        name: name.trim(),
        description: description.trim(),
        definition: definition.trim(),
        ...(showDefault ? { add_by_default: addByDefault } : {}),
      });
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The skill was not saved.');
    }
  }

  async function remove() {
    if (onDelete == null) {
      return;
    }
    try {
      await onDelete();
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The skill was not deleted.');
    }
  }

  return (
    <View style={{ gap: 8 }}>
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        {skill?.name ?? 'New skill'}
      </Text>
      <Text style={textStyle('label', palette.textPrimary)}>Skill name</Text>
      <TextInput accessibilityLabel="Skill name" value={name} onChangeText={setName} style={fieldStyle(palette)} />
      <Text style={textStyle('label', palette.textPrimary)}>Skill description</Text>
      <TextInput
        accessibilityLabel="Skill description"
        multiline
        value={description}
        onChangeText={setDescription}
        style={[fieldStyle(palette), { minHeight: 88, paddingVertical: 12 }]}
      />
      <Text style={textStyle('label', palette.textPrimary)}>Skill definition</Text>
      <TextInput
        accessibilityLabel="Skill definition"
        multiline
        value={definition}
        onChangeText={setDefinition}
        style={[fieldStyle(palette), { minHeight: 88, paddingVertical: 12 }]}
      />
      {showDefault ? (
        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12, minHeight: 44 }}>
          <Text onPress={() => setAddByDefault((value) => !value)} style={textStyle('body', palette.textPrimary)}>
            Add to new agents
          </Text>
          <Switch
            accessibilityLabel="Add to new agents"
            accessibilityState={{ checked: addByDefault }}
            value={addByDefault}
            onValueChange={setAddByDefault}
            trackColor={{ false: palette.borderStrong, true: palette.accent }}
            thumbColor={palette.onAccent}
          />
        </View>
      ) : null}
      <View style={buttonRow}>
        <IconButton
          icon={skill == null ? 'plus' : 'check'}
          label={skill == null ? 'Create skill' : 'Save skill'}
          disabled={!ready}
          onPress={() => void save()}
          background={ready ? palette.accent : palette.disabledSurface}
          color={ready ? palette.onAccent : palette.disabledText}
        />
        {skill != null && onDelete != null ? (
          <IconButton
            icon="trash-2"
            label="Delete skill"
            onPress={() => void remove()}
            background={palette.surface}
            border={palette.borderStrong}
            color={palette.textPrimary}
          />
        ) : null}
      </View>
      {error ? (
        <Text accessibilityRole="alert" style={textStyle('small', palette.danger)}>
          {error}
        </Text>
      ) : null}
    </View>
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

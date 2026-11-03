import { useState } from 'react';
import { Text, TextInput, View } from 'react-native';

import { api, idempotencyKey } from '@/api/client';
import { IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export function FolderSetup({
  workspaceId,
  onBound,
}: {
  workspaceId: string;
  onBound: () => void;
}) {
  const { palette } = useTheme();
  const [location, setLocation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function choose() {
    setError(null);
    try {
      const picked = await api<{ cancelled?: boolean; location?: string }>(
        `/folders/pick`,
        { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey() }, body: '{}' },
      );
      if (!picked.cancelled && picked.location) {
        setLocation(picked.location);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The folder was not chosen.');
    }
  }

  async function save() {
    if (!location.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api(`/workspaces/${workspaceId}/folders`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ kind: 'local', location: location.trim(), name: 'Library' }),
      });
      onBound();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The folder was not saved.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={{ flex: 1, padding: space[4], gap: space[3], justifyContent: 'center' }}>
      <Text style={textStyle('title', palette.textPrimary)}>Choose a local folder</Text>
      <Text style={textStyle('body', palette.textSecondary)}>
        Favlo indexes images, mp3, and mp4 in that folder for the gallery filmstrip.
      </Text>
      <IconButton
        label="Choose folder"
        icon="folder"
        onPress={() => void choose()}
        background={palette.accent}
        color={palette.onAccent}
      />
      <TextInput
        accessibilityLabel="Folder location"
        value={location}
        onChangeText={setLocation}
        placeholder="Folder location"
        placeholderTextColor={palette.textSecondary}
        style={{
          borderWidth: 1,
          borderColor: palette.borderStrong,
          borderRadius: 8,
          padding: space[3],
          color: palette.textPrimary,
          backgroundColor: palette.surface,
        }}
      />
      <IconButton
        label="Save folder"
        icon="check"
        disabled={busy || !location.trim()}
        background={busy || !location.trim() ? palette.disabledSurface : palette.accent}
        border={busy || !location.trim() ? palette.borderStrong : palette.accent}
        color={busy || !location.trim() ? palette.disabledText : palette.onAccent}
        onPress={() => void save()}
      />
      {error ? <Text style={textStyle('body', palette.danger)}>{error}</Text> : null}
    </View>
  );
}

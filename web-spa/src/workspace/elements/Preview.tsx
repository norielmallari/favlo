import { Text, View } from 'react-native';

import { IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import type { ElementProps } from '@/workspace/catalog';

export function Preview({ props, selection, fileUrl, onAddToChat }: ElementProps) {
  const { palette } = useTheme();
  const path = selection?.path ?? null;
  const fit = props.fit === 'cover' ? 'cover' : 'contain';
  if (path == null) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', padding: space[4] }}>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>No media selected</Text>
      </View>
    );
  }
  const lower = path.toLowerCase();
  const url = fileUrl(path);
  const name = path.split('/').pop() ?? path;
  return (
    <View style={{ flex: 1, padding: space[3], gap: space[2] }}>
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        {lower.endsWith('.mp3') ? (
          // @ts-expect-error web audio element
          <audio controls src={url} style={{ width: '100%', maxWidth: 480, margin: '0 auto', display: 'block' }} />
        ) : lower.endsWith('.mp4') ? (
          // @ts-expect-error web video element
          <video
            controls
            src={url}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: fit, margin: '0 auto', display: 'block' }}
          />
        ) : (
          // @ts-expect-error web img element
          <img
            alt={name}
            src={url}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: fit, margin: '0 auto', display: 'block' }}
          />
        )}
      </View>
      <View style={{ flexDirection: 'row', gap: space[2], justifyContent: 'center' }}>
        <IconButton
          label={`Download ${name}`}
          icon="download"
          background={palette.surface}
          border={palette.borderStrong}
          color={palette.textPrimary}
          onPress={() => {
            void (async () => {
              if (typeof document === 'undefined') {
                return;
              }
              const response = await fetch(url);
              if (!response.ok) {
                throw new Error(`Download failed (${response.status})`);
              }
              const objectUrl = URL.createObjectURL(await response.blob());
              const link = document.createElement('a');
              link.href = objectUrl;
              link.download = name;
              link.click();
              URL.revokeObjectURL(objectUrl);
            })();
          }}
        />
        <IconButton
          label="Add to chat"
          accessibilityLabel="Add to chat"
          icon="plus"
          background={palette.accent}
          color={palette.onAccent}
          onPress={() => onAddToChat(path)}
        />
      </View>
    </View>
  );
}

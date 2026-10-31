import { Text, View } from 'react-native';

import { buttonRow, IconButton } from '@/components/IconButton';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export const STARTERS = [
  { label: 'Generate an image', fill: 'Generate an image of a chestnut horse in a garden.', icon: 'image' },
  { label: 'List registered models', fill: 'Which models are registered?', icon: 'list' },
  { label: 'Set up a model', fill: 'Help me set up a model.', icon: 'cpu' },
];

export function EmptyState({
  name,
  onFill,
  compact = false,
}: {
  name: string;
  onFill: (text: string) => void;
  compact?: boolean;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ gap: space[3], maxWidth: '100%' }}>
      <Text
        accessibilityRole="header"
        aria-level={2}
        style={[textStyle(compact ? 'title' : 'display', palette.textPrimary), { maxWidth: '100%' }]}
      >
        {name}
      </Text>
      <Text style={textStyle('body', palette.textSecondary)}>Chat, images, music, and voice on this machine.</Text>
      <View style={buttonRow}>
        {STARTERS.map((item) => (
          <IconButton
            key={item.label}
            icon={item.icon}
            label={item.label}
            onPress={() => onFill(item.fill)}
            background={palette.surface}
            border={palette.borderStrong}
            color={palette.textPrimary}
          />
        ))}
      </View>
    </View>
  );
}

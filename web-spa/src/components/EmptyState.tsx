import { Text, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export const STARTERS = [
  { label: 'Generate an image', fill: 'Generate an image of a chestnut horse in a garden.' },
  { label: 'List registered models', fill: 'Which models are registered?' },
  { label: 'Set up a model', fill: 'Help me set up a model.' },
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
      <View
        style={{
          flexDirection: compact ? 'column' : 'row',
          flexWrap: compact ? 'nowrap' : 'wrap',
          gap: space[2],
          maxWidth: '100%',
        }}
      >
        {STARTERS.map((item) => (
          <Pressable
            key={item.label}
            accessibilityRole="button"
            accessibilityLabel={item.label}
            onPress={() => onFill(item.fill)}
            style={{
              minHeight: 44,
              borderRadius: radius.pill,
              borderWidth: 1,
              borderColor: palette.borderStrong,
              backgroundColor: palette.surface,
              paddingHorizontal: space[3],
              justifyContent: 'center',
              alignItems: 'center',
              ...(compact ? { alignSelf: 'stretch' } : null),
            }}
          >
            <Text style={textStyle('small', palette.textPrimary, '600')}>{item.label}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

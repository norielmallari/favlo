import { ScrollView, Text, View } from 'react-native';

import { Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import type { ElementProps } from '@/workspace/catalog';

type Item = { path: string };

export function Filmstrip({ props, selection, fileUrl, onSelect }: ElementProps) {
  const { palette } = useTheme();
  const title = typeof props.title === 'string' ? props.title : '';
  const items = (Array.isArray(props.items) ? props.items : []) as Item[];
  if (items.length === 0) {
    return (
      <View style={{ padding: space[3], alignItems: 'center', justifyContent: 'center' }}>
        <Text style={[textStyle('body', palette.textSecondary), { textAlign: 'center' }]}>
          Ask the agent to fill the filmstrip
        </Text>
      </View>
    );
  }
  return (
    <View style={{ gap: space[2], paddingVertical: space[2], alignItems: 'stretch' }}>
      {title ? (
        <Text style={[textStyle('label', palette.textPrimary), { textAlign: 'center', paddingHorizontal: space[3] }]}>
          {title}
        </Text>
      ) : null}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={{
          gap: space[2],
          paddingHorizontal: space[3],
          flexGrow: 1,
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        {items.map((item) => {
          const selected = selection?.path === item.path;
          const name = item.path.split('/').pop() ?? item.path;
          const lower = item.path.toLowerCase();
          return (
            <Pressable
              key={item.path}
              accessibilityRole="button"
              accessibilityLabel={name}
              onPress={() => onSelect(item.path)}
              style={{
                width: 96,
                borderWidth: selected ? 2 : 1,
                borderColor: selected ? palette.accent : palette.border,
                borderRadius: radius.control,
                overflow: 'hidden',
                backgroundColor: palette.surface,
              }}
            >
              {lower.endsWith('.mp3') || lower.endsWith('.mp4') ? (
                <View style={{ height: 72, alignItems: 'center', justifyContent: 'center', padding: space[1] }}>
                  <Text style={[textStyle('caption', palette.textPrimary), { textAlign: 'center' }]}>
                    {lower.endsWith('.mp3') ? 'Audio' : 'Video'}
                  </Text>
                  <Text numberOfLines={2} style={[textStyle('caption', palette.textSecondary), { textAlign: 'center' }]}>
                    {name}
                  </Text>
                </View>
              ) : (
                <View style={{ width: 96, height: 72, alignItems: 'center', justifyContent: 'center' }}>
                  {/* @ts-expect-error web img */}
                  <img
                    alt={name}
                    src={fileUrl(item.path)}
                    style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                  />
                </View>
              )}
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

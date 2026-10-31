import { useEffect, useState, type ReactNode } from 'react';
import { View } from 'react-native';

type IconView = (props: { name: string; size: number; color: string }) => ReactNode;

export function FeatherIcon({
  name,
  size = 22,
  color,
}: {
  name: string;
  size?: number;
  color: string;
}) {
  const [Icon, setIcon] = useState<IconView | null>(null);

  useEffect(() => {
    let live = true;
    void import('@react-native-vector-icons/feather')
      .then((mod) => {
        if (live) {
          setIcon(() => mod.Feather as unknown as IconView);
        }
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  if (Icon == null) {
    return <View accessibilityElementsHidden style={{ width: size, height: size }} />;
  }
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
      <Icon name={name} size={size} color={color} />
    </View>
  );
}

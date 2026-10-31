import { Platform, type TextStyle } from 'react-native';

import { type } from '@/theme/tokens';

export type TypeRole = 'caption' | 'small' | 'body' | 'label' | 'title' | 'display';

const weights: Record<TypeRole, NonNullable<TextStyle['fontWeight']>> = {
  caption: type.weight.medium,
  small: type.weight.regular,
  body: type.weight.regular,
  label: type.weight.medium,
  title: type.weight.semibold,
  display: type.weight.semibold,
};

export function textStyle(
  role: TypeRole,
  color?: string,
  weight?: TextStyle['fontWeight'],
): TextStyle {
  return {
    ...(Platform.OS === 'web' ? { fontFamily: type.family.ui } : null),
    fontSize: type.size[role],
    lineHeight: type.line[role],
    fontWeight: weight ?? weights[role],
    ...(color ? { color } : null),
  };
}

export function monoStyle(color?: string): TextStyle {
  return {
    fontFamily: Platform.OS === 'web' ? type.family.mono : 'monospace',
    fontSize: type.size.mono,
    lineHeight: type.line.mono,
    ...(color ? { color } : null),
  };
}

export const measureStyle = {
  width: '100%' as const,
  maxWidth: Platform.OS === 'web' ? `${type.measure}ch` : type.measure * 8,
  alignSelf: 'center' as const,
};

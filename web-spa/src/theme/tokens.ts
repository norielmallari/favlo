export type Palette = {
  background: string;
  surface: string;
  surfaceRaised: string;
  surfaceHover: string;
  border: string;
  borderStrong: string;
  textPrimary: string;
  textSecondary: string;
  accent: string;
  accentHover: string;
  accentSubtle: string;
  onAccent: string;
  focusRing: string;
  success: string;
  successSubtle: string;
  warning: string;
  warningSubtle: string;
  danger: string;
  dangerSubtle: string;
  onDanger: string;
  disabledText: string;
  disabledSurface: string;
  overlay: string;
};

export const light: Palette = {
  background: '#F6F8FC',
  surface: '#FFFFFF',
  surfaceRaised: '#FFFFFF',
  surfaceHover: '#EDF1F7',
  border: '#D8DFEA',
  borderStrong: '#718096',
  textPrimary: '#182230',
  textSecondary: '#526176',
  accent: '#2563EB',
  accentHover: '#1D4ED8',
  accentSubtle: '#EAF1FF',
  onAccent: '#FFFFFF',
  focusRing: '#1D4ED8',
  success: '#166534',
  successSubtle: '#E7F6EC',
  warning: '#92400E',
  warningSubtle: '#FDF1DC',
  danger: '#B91C1C',
  dangerSubtle: '#FDECEC',
  onDanger: '#FFFFFF',
  disabledText: '#7A879B',
  disabledSurface: '#E6EBF2',
  overlay: 'rgba(15, 23, 42, 0.48)',
};

export const dark: Palette = {
  background: '#0F1117',
  surface: '#171B24',
  surfaceRaised: '#202633',
  surfaceHover: '#293244',
  border: '#364155',
  borderStrong: '#7B8AA3',
  textPrimary: '#E8EDF5',
  textSecondary: '#A8B5C8',
  accent: '#60A5FA',
  accentHover: '#93C5FD',
  accentSubtle: '#172B49',
  onAccent: '#0F172A',
  focusRing: '#93C5FD',
  success: '#4ADE80',
  successSubtle: '#12301F',
  warning: '#FBBF24',
  warningSubtle: '#35260C',
  danger: '#F87171',
  dangerSubtle: '#3A1818',
  onDanger: '#0F172A',
  disabledText: '#7A869A',
  disabledSurface: '#232A38',
  overlay: 'rgba(0, 0, 0, 0.64)',
};

export function contrast(foreground: string, background: string): number {
  const ratio = (left: number, right: number) => {
    const lighter = Math.max(left, right);
    const darker = Math.min(left, right);
    return (lighter + 0.05) / (darker + 0.05);
  };
  return ratio(luminance(foreground), luminance(background));
}

function luminance(hex: string): number {
  const value = hex.replace('#', '');
  const channels = [0, 2, 4].map((index) => {
    const channel = Number.parseInt(value.slice(index, index + 2), 16) / 255;
    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

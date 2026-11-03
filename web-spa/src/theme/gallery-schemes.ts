import type { Palette } from '@/theme/tokens';

/**
 * Gallery-only color schemes. The Favlo app stays on the blue palette in tokens.ts.
 * Light and dark pairs follow the published Rosé Pine Dawn/Moon and Catppuccin Latte/Mocha roles.
 * A few text and button colors are darkened so small text stays at WCAG AA on the surface it sits on.
 * Rosé Pine: https://rosepinetheme.com/palette/
 * Catppuccin: https://catppuccin.com/palette/
 */
export type GalleryScheme = 'favlo' | 'rose-pine' | 'catppuccin';

export const GALLERY_SCHEMES: { id: GalleryScheme; label: string }[] = [
  { id: 'favlo', label: 'Favlo' },
  { id: 'rose-pine', label: 'Rosé Pine' },
  { id: 'catppuccin', label: 'Catppuccin' },
];

export function isGalleryScheme(value: string | undefined): value is GalleryScheme {
  return value === 'favlo' || value === 'rose-pine' || value === 'catppuccin';
}

export function gallerySchemeLabel(scheme: GalleryScheme): string {
  return GALLERY_SCHEMES.find((item) => item.id === scheme)?.label ?? 'Favlo';
}

const rosePineDawn: Palette = {
  background: '#faf4ed',
  surface: '#fffaf3',
  surfaceRaised: '#f2e9e1',
  surfaceHover: '#f4ede8',
  border: '#dfdad9',
  borderStrong: '#797593',
  textPrimary: '#464261',
  textSecondary: '#575279',
  accent: '#286983',
  accentHover: '#1f5468',
  accentSubtle: '#dfdad9',
  onAccent: '#ffffff',
  focusRing: '#286983',
  success: '#286983',
  successSubtle: '#f4ede8',
  warning: '#8a5a10',
  warningSubtle: '#f4ede8',
  danger: '#9a445c',
  dangerSubtle: '#f4ede8',
  onDanger: '#ffffff',
  disabledText: '#564f68',
  disabledSurface: '#f2e9e1',
  overlay: 'rgba(35, 33, 54, 0.48)',
};

const rosePineMoon: Palette = {
  background: '#232136',
  surface: '#2a273f',
  surfaceRaised: '#393552',
  surfaceHover: '#44415a',
  border: '#44415a',
  borderStrong: '#56526e',
  textPrimary: '#e0def4',
  textSecondary: '#a8a3bc',
  accent: '#c4a7e7',
  accentHover: '#d0b8f0',
  accentSubtle: '#44415a',
  onAccent: '#232136',
  focusRing: '#c4a7e7',
  success: '#9ccfd8',
  successSubtle: '#2a273f',
  warning: '#f6c177',
  warningSubtle: '#393552',
  danger: '#eb6f92',
  dangerSubtle: '#2a273f',
  onDanger: '#232136',
  disabledText: '#a8a3bc',
  disabledSurface: '#393552',
  overlay: 'rgba(0, 0, 0, 0.64)',
};

const catppuccinLatte: Palette = {
  background: '#eff1f5',
  surface: '#e6e9ef',
  surfaceRaised: '#dce0e8',
  surfaceHover: '#ccd0da',
  border: '#bcc0cc',
  borderStrong: '#9ca0b0',
  textPrimary: '#4c4f69',
  textSecondary: '#5c5f77',
  accent: '#7329d4',
  accentHover: '#6424b8',
  accentSubtle: '#ccd0da',
  onAccent: '#ffffff',
  focusRing: '#7329d4',
  success: '#26731a',
  successSubtle: '#e6e9ef',
  warning: '#8a5a10',
  warningSubtle: '#e6e9ef',
  danger: '#c20e36',
  dangerSubtle: '#e6e9ef',
  onDanger: '#ffffff',
  disabledText: '#4c4f69',
  disabledSurface: '#dce0e8',
  overlay: 'rgba(17, 17, 27, 0.48)',
};

const catppuccinMocha: Palette = {
  background: '#1e1e2e',
  surface: '#181825',
  surfaceRaised: '#313244',
  surfaceHover: '#45475a',
  border: '#45475a',
  borderStrong: '#585b70',
  textPrimary: '#cdd6f4',
  textSecondary: '#a6adc8',
  accent: '#cba6f7',
  accentHover: '#ddc8fb',
  accentSubtle: '#45475a',
  onAccent: '#11111b',
  focusRing: '#cba6f7',
  success: '#a6e3a1',
  successSubtle: '#313244',
  warning: '#f9e2af',
  warningSubtle: '#313244',
  danger: '#f38ba8',
  dangerSubtle: '#313244',
  onDanger: '#11111b',
  disabledText: '#a6adc8',
  disabledSurface: '#313244',
  overlay: 'rgba(0, 0, 0, 0.64)',
};

export const gallerySchemes: Record<Exclude<GalleryScheme, 'favlo'>, { light: Palette; dark: Palette }> = {
  'rose-pine': { light: rosePineDawn, dark: rosePineMoon },
  catppuccin: { light: catppuccinLatte, dark: catppuccinMocha },
};

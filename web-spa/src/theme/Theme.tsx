import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Appearance, Platform, useColorScheme } from 'react-native';

import { dark, light, type Palette } from '@/theme/tokens';

export type Preference = 'system' | 'light' | 'dark';

type ThemeValue = {
  preference: Preference;
  palette: Palette;
  setPreference: (next: Preference) => void;
};

const ThemeContext = createContext<ThemeValue | null>(null);
const STORAGE_KEY = 'favlo.theme';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const system = useColorScheme();
  const [preference, setPreference] = useState<Preference>('system');

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof document === 'undefined') {
      return;
    }
    document.documentElement.lang = 'en';
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
      return;
    }
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      setPreference(stored);
    }
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
      return;
    }
    localStorage.setItem(STORAGE_KEY, preference);
  }, [preference]);

  const palette = useMemo(() => {
    const scheme = preference === 'system' ? (system ?? Appearance.getColorScheme()) : preference;
    return scheme === 'dark' ? dark : light;
  }, [preference, system]);

  const value = useMemo(
    () => ({ preference, palette, setPreference }),
    [preference, palette],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeValue {
  const value = useContext(ThemeContext);
  if (value == null) {
    throw new Error('ThemeProvider is missing');
  }
  return value;
}

import { createContext, createElement, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Appearance, Platform, useColorScheme } from 'react-native';

import { workspaceSchemes, type WorkspaceScheme } from '@/theme/workspace-schemes';
import { focusRingCss, scrollbarStyle } from '@/theme/scrollbar';
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

  return (
    <ThemeContext.Provider value={value}>
      {Platform.OS === 'web'
        ? createElement('style', { 'data-testid': 'scrollbar' }, `${scrollbarStyle(palette)}\n${focusRingCss(palette.focusRing)}`)
        : null}
      {children}
    </ThemeContext.Provider>
  );
}

export function WorkspaceSchemeProvider({
  scheme,
  children,
}: {
  scheme: WorkspaceScheme;
  children: ReactNode;
}) {
  const parent = useTheme();
  const system = useColorScheme();
  const palette = useMemo(() => {
    if (scheme === 'favlo') {
      return parent.palette;
    }
    const mode = parent.preference === 'system' ? (system ?? Appearance.getColorScheme()) : parent.preference;
    const pair = workspaceSchemes[scheme];
    return mode === 'dark' ? pair.dark : pair.light;
  }, [parent.palette, parent.preference, scheme, system]);
  const value = useMemo(
    () => ({ preference: parent.preference, palette, setPreference: parent.setPreference }),
    [palette, parent.preference, parent.setPreference],
  );

  useEffect(() => {
    if (Platform.OS !== 'web' || scheme === 'favlo' || typeof document === 'undefined') {
      return;
    }
    const root = document.documentElement;
    const previous = root.style.getPropertyValue('--favlo-focus-ring');
    root.style.setProperty('--favlo-focus-ring', palette.focusRing);
    const node = document.createElement('style');
    node.dataset.testid = 'workspace-scrollbar';
    node.textContent = scrollbarStyle(palette);
    document.head.appendChild(node);
    return () => {
      if (previous) {
        root.style.setProperty('--favlo-focus-ring', previous);
      } else {
        root.style.removeProperty('--favlo-focus-ring');
      }
      node.remove();
    };
  }, [palette, scheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeValue {
  const value = useContext(ThemeContext);
  if (value == null) {
    throw new Error('ThemeProvider is missing');
  }
  return value;
}

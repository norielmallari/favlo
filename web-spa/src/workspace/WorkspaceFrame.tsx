import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Platform, useWindowDimensions, View } from 'react-native';

import { FeatherIcon } from '@/components/FeatherIcon';
import { IconButton } from '@/components/IconButton';
import { focusRingInset } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';
import { space, z } from '@/theme/tokens';
import type { WorkspaceApp } from '@/workspace/workspace-apps';
import {
  useWorkspaceBannerEnd,
  WorkspaceBannerProvider,
  WorkspaceChromeProvider,
} from '@/workspace/workspace-chrome';
import { WorkspaceTitleMenu } from '@/workspace/WorkspaceTitleMenu';
import type { WorkspaceScheme } from '@/theme/workspace-schemes';

const FAVLO_TITLE = 'Favlo';
const FAVLO_THEME = '#FFFFFF';
const FAVLO_ICON =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2563EB"/><text x="16" y="22" text-anchor="middle" font-size="16" font-family="system-ui,sans-serif" font-weight="700" fill="#FFFFFF">F</text></svg>`,
  );

export function WorkspaceFrame({
  app,
  name,
  closed,
  scheme,
  onBack,
  onScheme,
  children,
}: {
  app: WorkspaceApp;
  name: string;
  closed?: boolean;
  scheme: WorkspaceScheme;
  onBack?: () => void;
  onScheme: (scheme: WorkspaceScheme) => void;
  children: ReactNode;
}) {
  return (
    <WorkspaceChromeProvider>
      <WorkspaceFrameShell
        app={app}
        name={name}
        closed={closed}
        scheme={scheme}
        onBack={onBack}
        onScheme={onScheme}
      >
        {children}
      </WorkspaceFrameShell>
    </WorkspaceChromeProvider>
  );
}

function WorkspaceFrameShell({
  app,
  name,
  closed,
  scheme,
  onBack,
  onScheme,
  children,
}: {
  app: WorkspaceApp;
  name: string;
  closed?: boolean;
  scheme: WorkspaceScheme;
  onBack?: () => void;
  onScheme: (scheme: WorkspaceScheme) => void;
  children: ReactNode;
}) {
  const { palette } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < 1024;
  const bannerEnd = useWorkspaceBannerEnd();
  const [bannerHosted, setBannerHostedState] = useState(false);
  const setBannerHosted = useCallback((next: boolean) => {
    setBannerHostedState((current) => (current === next ? current : next));
  }, []);
  const [reducedMotion, setReducedMotion] = useState(false);
  const titleName = closed ? 'This workspace is gone' : name;
  const documentTitle = `${titleName} — ${app.name}`;

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof window === 'undefined') {
      return;
    }
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => setReducedMotion(media.matches);
    sync();
    media.addEventListener('change', sync);
    return () => media.removeEventListener('change', sync);
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof document === 'undefined') {
      return;
    }
    const previousTitle = document.title;
    const previousIcon = document.querySelector<HTMLLinkElement>("link[rel='icon']");
    const previousIconHref = previousIcon?.getAttribute('href') ?? null;
    const previousTheme = document.querySelector<HTMLMetaElement>("meta[name='theme-color']");
    const previousThemeContent = previousTheme?.getAttribute('content') ?? null;

    document.title = documentTitle;

    let icon = previousIcon;
    if (icon == null) {
      icon = document.createElement('link');
      icon.rel = 'icon';
      document.head.appendChild(icon);
    }
    icon.setAttribute('href', appMarkIcon(app.mark, palette.accent, palette.onAccent));

    let theme = previousTheme;
    if (theme == null) {
      theme = document.createElement('meta');
      theme.name = 'theme-color';
      document.head.appendChild(theme);
    }
    theme.setAttribute('content', palette.surface);

    return () => {
      document.title = previousTitle || FAVLO_TITLE;
      if (icon != null) {
        if (previousIconHref != null) {
          icon.setAttribute('href', previousIconHref);
        } else {
          icon.setAttribute('href', FAVLO_ICON);
        }
      }
      if (theme != null) {
        theme.setAttribute('content', previousThemeContent ?? FAVLO_THEME);
      }
    };
  }, [app.mark, documentTitle, palette.accent, palette.onAccent, palette.surface]);

  const canvas = app.bar === 'canvas';

  const banner = (
      <View
        testID="app-banner"
        {...(Platform.OS === 'web' ? { role: 'banner' as const } : {})}
        style={{
          minHeight: 44,
          width: '100%',
          maxWidth: '100%',
          paddingHorizontal: Math.max(space[3], focusRingInset),
          paddingVertical: Math.max(canvas ? 4 : 0, focusRingInset),
          flexDirection: 'row',
          alignItems: 'center',
          gap: space[2],
          borderBottomWidth: canvas ? 0 : 1,
          borderBottomColor: palette.borderStrong,
          backgroundColor: canvas ? palette.background : palette.surface,
          zIndex: z.banner,
        }}
      >
        {onBack ? (
          <IconButton
            label="Back to channel"
            icon="arrow-left"
            onPress={onBack}
            iconOnly={narrow}
            tone="ghost"
          />
        ) : null}
        <AppMark glyph={app.mark} />
        <WorkspaceTitleMenu
          app={app}
          name={titleName}
          scheme={scheme}
          reducedMotion={reducedMotion}
          onSelectScheme={onScheme}
        />
        <View style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minWidth: 0 }} />
        {bannerEnd != null ? (
          <View
            style={{
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'flex-end',
              gap: space[1],
              flexShrink: 0,
            }}
          >
            {bannerEnd}
          </View>
        ) : null}
      </View>
  );

  return (
    <WorkspaceBannerProvider banner={banner} hosted={bannerHosted} setHosted={setBannerHosted}>
      <View
        testID="app-frame"
        style={{
          flex: 1,
          width: '100%',
          maxWidth: '100%',
          backgroundColor: palette.background,
        }}
      >
        {bannerHosted ? null : banner}
        {children}
      </View>
    </WorkspaceBannerProvider>
  );
}

function AppMark({ glyph }: { glyph: WorkspaceApp['mark'] }) {
  const { palette } = useTheme();
  return (
    <View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={{
        width: 32,
        height: 32,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: palette.accent,
      }}
    >
      <FeatherIcon name={glyph} size={18} color={palette.onAccent} />
    </View>
  );
}

/** Same Feather glyph as the header mark, drawn at 18px inside the 32px tile. */
export function appMarkIcon(glyph: WorkspaceApp['mark'], accent: string, onAccent: string): string {
  const mark =
    glyph === 'terminal'
      ? `<path d="M4 8l6 4-6 4M12 18h8" fill="none" stroke="${onAccent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`
      : `<g transform="translate(7 7) scale(0.75)" fill="none" stroke="${onAccent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M14.31 8L20.05 17.94M9.69 8h11.48M7.38 12L13.12 2.06M9.69 16L3.95 6.06M14.31 16H2.83M16.62 12l-5.74 9.94"/></g>`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="${accent}"/>${mark}</svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

export function favloDefaultIcon(): string {
  return FAVLO_ICON;
}

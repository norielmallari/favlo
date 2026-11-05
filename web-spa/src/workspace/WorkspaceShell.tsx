import { useEffect } from 'react';
import { View } from 'react-native';

import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { COMPONENTS } from '@/workspace/catalog';
import type { PlacementState } from '@/workspace/placement';
import { filmstripPaths, selectPath } from '@/workspace/placement';

export function WorkspaceShell({
  state,
  slots,
  fileUrl,
  onChange,
  onAddToChat,
  onRequestDelete,
  fullScreen = false,
  chatDetent,
  onToggleFullScreen,
  onCycleLayout,
  hideFilmstrip = false,
  pendingMedia = [],
  folderFiles = null,
}: {
  state: PlacementState;
  slots: string[];
  fileUrl: (path: string) => string;
  onChange: (next: PlacementState) => void;
  onAddToChat: (path: string) => void;
  onRequestDelete: (path: string) => void;
  fullScreen?: boolean;
  chatDetent?: 'fullscreen' | 'peek' | 'expanded';
  onToggleFullScreen?: () => void;
  onCycleLayout?: () => void;
  hideFilmstrip?: boolean;
  pendingMedia?: { task_id: string; kind: string; status: string }[];
  folderFiles?: string[] | null;
}) {
  const { palette } = useTheme();
  const workspace = state?.favlo_workspace;
  const elements = Array.isArray(workspace?.elements) ? workspace.elements : [];
  const selection = workspace?.selection ?? null;
  const order = slots.filter((slot, index) => slots.indexOf(slot) === index);

  function stepSelection(direction: 'prev' | 'next') {
    const paths = folderFiles ?? filmstripPaths(state);
    if (paths.length === 0) {
      return;
    }
    const current = selection?.path ?? paths[0];
    const index = Math.max(0, paths.indexOf(current));
    const next =
      direction === 'next'
        ? paths[Math.min(paths.length - 1, index + 1)]
        : paths[Math.max(0, index - 1)];
    onChange(selectPath(state, next));
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (event.key === 'ArrowRight') {
        stepSelection('next');
      }
      if (event.key === 'ArrowLeft') {
        stepSelection('prev');
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('keydown', onKey);
      return () => document.removeEventListener('keydown', onKey);
    }
    return undefined;
    // state and selection are read inside stepSelection via closure of render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folderFiles, onChange, selection?.path, state]);

  function renderSlot(items: typeof elements) {
    return items.map((item) => {
      const Component = COMPONENTS[item.element];
      if (Component == null) {
        return null;
      }
      if (hideFilmstrip && item.element === 'filmstrip') {
        return null;
      }
      return (
        <Component
          key={item.key}
          props={
            item.element === 'filmstrip' && folderFiles != null
              ? { ...item.props, items: folderFiles.map((path) => ({ path })) }
              : item.props
          }
          selection={selection}
          fileUrl={fileUrl}
          onSelect={(path) => onChange(selectPath(state, path))}
          onAddToChat={onAddToChat}
          onRequestDelete={onRequestDelete}
          onSwipe={item.element === 'preview' ? stepSelection : undefined}
          fullScreen={fullScreen}
          chatDetent={item.element === 'preview' ? chatDetent : undefined}
          onToggleFullScreen={item.element === 'preview' ? onToggleFullScreen : undefined}
          onCycleLayout={item.element === 'preview' ? onCycleLayout : undefined}
          pendingMedia={item.element === 'filmstrip' ? pendingMedia : undefined}
        />
      );
    });
  }

  return (
    <View
      style={{
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 0,
        minHeight: 0,
        backgroundColor: palette.background,
        flexDirection: 'column',
        gap: space[1],
      }}
    >
      {order.map((slot) => {
        const items = elements.filter((item) => item.slot === slot);
        if (items.length === 0) {
          return null;
        }
        if (hideFilmstrip && items.every((item) => item.element === 'filmstrip')) {
          return null;
        }
        const rail = slot === 'rail';
        return (
          <View
            key={slot}
            accessibilityLabel={rail ? 'Filmstrip' : undefined}
            style={rail ? undefined : { flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0 }}
          >
            {renderSlot(items)}
          </View>
        );
      })}
    </View>
  );
}

import { useEffect } from 'react';
import { View } from 'react-native';

import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { CATALOG } from '@/workspace/catalog';
import type { PlacementState } from '@/workspace/placement';
import { filmstripPaths, selectPath } from '@/workspace/placement';

export function WorkspaceShell({
  state,
  fileUrl,
  onChange,
  onAddToChat,
}: {
  state: PlacementState;
  fileUrl: (path: string) => string;
  onChange: (next: PlacementState) => void;
  onAddToChat: (path: string) => void;
}) {
  const { palette } = useTheme();
  const workspace = state?.favlo_workspace;
  const elements = Array.isArray(workspace?.elements) && workspace.elements.length > 0
    ? workspace.elements
    : [
        { key: 'preview', element: 'preview', slot: 'stage', props: { fit: 'contain' } },
        { key: 'filmstrip', element: 'filmstrip', slot: 'rail', props: { title: '', items: [] } },
      ];
  const selection = workspace?.selection ?? null;
  const stage = elements.filter((item) => item.slot === 'stage');
  const rail = elements.filter((item) => item.slot === 'rail');

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      const paths = filmstripPaths(state);
      if (paths.length === 0) {
        return;
      }
      const current = selection?.path ?? paths[0];
      const index = Math.max(0, paths.indexOf(current));
      if (event.key === 'ArrowRight') {
        const next = paths[Math.min(paths.length - 1, index + 1)];
        onChange(selectPath(state, next));
      }
      if (event.key === 'ArrowLeft') {
        const next = paths[Math.max(0, index - 1)];
        onChange(selectPath(state, next));
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('keydown', onKey);
      return () => document.removeEventListener('keydown', onKey);
    }
    return undefined;
  }, [onChange, selection?.path, state]);

  function renderSlot(items: typeof stage) {
    return items.map((item) => {
      const entry = CATALOG[item.element];
      if (entry == null) {
        return null;
      }
      const Component = entry.component;
      return (
        <Component
          key={item.key}
          props={item.props}
          selection={selection}
          fileUrl={fileUrl}
          onSelect={(path) => onChange(selectPath(state, path))}
          onAddToChat={onAddToChat}
        />
      );
    });
  }

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: palette.background,
        flexDirection: 'column',
        gap: space[2],
      }}
    >
      <View style={{ flex: 1, minHeight: 200 }}>{renderSlot(stage)}</View>
      {rail.length > 0 ? <View accessibilityLabel="Filmstrip">{renderSlot(rail)}</View> : null}
    </View>
  );
}

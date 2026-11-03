import {
  documentFromShell,
  ensureFilmstripSelection,
  type PlacementState,
} from '@/workspace/placement';

function strip(paths: string[], selection: string | null): PlacementState {
  return {
    favlo_workspace: {
      version: 1,
      guide: 'gallery',
      elements: [
        { key: 'preview', element: 'preview', slot: 'stage', props: { fit: 'contain' } },
        {
          key: 'filmstrip',
          element: 'filmstrip',
          slot: 'rail',
          props: { title: '', items: paths.map((path) => ({ path })) },
        },
      ],
      selection: selection ? { kind: 'file', path: selection } : null,
    },
  };
}

test('ensureFilmstripSelection picks the first path when selection is missing', () => {
  const next = ensureFilmstripSelection(strip(['a.png', 'b.png'], null));
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'a.png' });
});

test('ensureFilmstripSelection keeps a path that is still on the strip', () => {
  const next = ensureFilmstripSelection(strip(['a.png', 'b.png'], 'b.png'));
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'b.png' });
});

test('ensureFilmstripSelection replaces a path that left the strip', () => {
  const next = ensureFilmstripSelection(strip(['a.png'], 'gone.png'));
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'a.png' });
});

test('ensureFilmstripSelection clears selection when the strip is empty', () => {
  const next = ensureFilmstripSelection(strip([], 'a.png'));
  expect(next.favlo_workspace.selection).toBeNull();
  expect(
    ensureFilmstripSelection(
      documentFromShell({
        id: 'gallery',
        elements: [
          { name: 'preview', slot: 'stage', initial_props: { fit: 'contain' } },
          { name: 'filmstrip', slot: 'rail', initial_props: { title: '', items: [] } },
        ],
      }),
    ).favlo_workspace.selection,
  ).toBeNull();
});

import {
  documentFromShell,
  ensureFilmstripSelection,
  ensureShellElements,
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

test('ensureFilmstripSelection keeps a folder file the placement list omits', () => {
  const next = ensureFilmstripSelection(strip(['a.png'], 'late.png'), ['a.png', 'late.png']);
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'late.png' });
});

test('ensureShellElements puts a dropped preview back on stage', () => {
  const shell = {
    id: 'gallery',
    elements: [
      { name: 'preview', slot: 'stage', initial_props: { fit: 'contain' } },
      { name: 'filmstrip', slot: 'rail', initial_props: { title: '', items: [] } },
    ],
  };
  const dropped: PlacementState = {
    favlo_workspace: {
      version: 1,
      guide: 'gallery',
      elements: [
        {
          key: 'filmstrip',
          element: 'filmstrip',
          slot: 'rail',
          props: { title: '', items: [{ path: 'horse.png' }] },
        },
      ],
      selection: { kind: 'file', path: 'horse.png' },
    },
  };
  const next = ensureShellElements(dropped, shell);
  expect(next.favlo_workspace.elements.map((item) => item.element)).toEqual(['preview', 'filmstrip']);
  expect(next.favlo_workspace.elements[1]?.props).toEqual({
    title: '',
    items: [{ path: 'horse.png' }],
  });
  expect(next.favlo_workspace.selection).toEqual({ kind: 'file', path: 'horse.png' });
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

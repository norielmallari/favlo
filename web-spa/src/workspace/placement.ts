export type PlacementState = {
  favlo_workspace: {
    version: number;
    guide: string;
    elements: {
      key: string;
      element: string;
      slot: string;
      props: Record<string, unknown>;
    }[];
    selection: { kind: string; path: string } | null;
  };
};

export type ShellDocument = {
  id: string;
  slots?: string[];
  elements?: { name: string; slot: string; initial_props?: Record<string, unknown> }[];
};

/** First placement from the shell document. An empty document paints nothing. */
export function documentFromShell(shell: ShellDocument | null | undefined): PlacementState {
  return {
    favlo_workspace: {
      version: 1,
      guide: shell?.id || '',
      elements: (shell?.elements ?? []).map((item) => ({
        key: item.name,
        element: item.name,
        slot: item.slot,
        props: item.initial_props ?? {},
      })),
      selection: null,
    },
  };
}

export function mergeSnapshot(
  local: PlacementState,
  incoming: PlacementState,
  folderPaths: readonly string[] = [],
): PlacementState {
  const next = incoming.favlo_workspace;
  const prior = local.favlo_workspace;
  const selection =
    next.selection != null
      ? next.selection
      : prior.selection;
  return ensureFilmstripSelection(
    {
      favlo_workspace: {
        ...next,
        selection,
      },
    },
    folderPaths,
  );
}

export function selectPath(state: PlacementState, path: string): PlacementState {
  return {
    favlo_workspace: {
      ...state.favlo_workspace,
      selection: { kind: 'file', path },
    },
  };
}

export function filmstripPaths(state: PlacementState): string[] {
  const strip = state.favlo_workspace.elements.find((item) => item.element === 'filmstrip');
  const items = (strip?.props?.items as { path?: string }[] | undefined) ?? [];
  return items.map((item) => String(item.path || '')).filter(Boolean);
}

/**
 * Paint every shell element. A saved placement that dropped preview still
 * shows the stage image; placement props win when that element is present.
 */
export function ensureShellElements(
  state: PlacementState,
  shell: ShellDocument | null | undefined,
  folderPaths: readonly string[] = [],
): PlacementState {
  const shellItems = shell?.elements ?? [];
  if (shellItems.length === 0) {
    return ensureFilmstripSelection(state, folderPaths);
  }
  const placed = new Map(state.favlo_workspace.elements.map((item) => [item.element, item]));
  const shellNames = new Set(shellItems.map((item) => item.name));
  const elements = shellItems.map((item) => {
    const existing = placed.get(item.name);
    if (existing != null) {
      return existing;
    }
    return {
      key: item.name,
      element: item.name,
      slot: item.slot,
      props: { ...(item.initial_props ?? {}) },
    };
  });
  for (const extra of state.favlo_workspace.elements) {
    if (!shellNames.has(extra.element)) {
      elements.push(extra);
    }
  }
  return ensureFilmstripSelection(
    {
      favlo_workspace: {
        ...state.favlo_workspace,
        guide: state.favlo_workspace.guide || shell?.id || '',
        elements,
      },
    },
    folderPaths,
  );
}

/**
 * Keep selection on a placement path, or on a folder file the strip is showing.
 * A folder file is valid even when the saved placement list does not name it.
 */
export function ensureFilmstripSelection(
  state: PlacementState,
  folderPaths: readonly string[] = [],
): PlacementState {
  const paths = filmstripPaths(state);
  const current = state.favlo_workspace.selection?.path;
  if (current && (paths.includes(current) || folderPaths.includes(current))) {
    return state;
  }
  if (paths.length === 0) {
    if (state.favlo_workspace.selection == null) {
      return state;
    }
    return {
      favlo_workspace: {
        ...state.favlo_workspace,
        selection: null,
      },
    };
  }
  return {
    favlo_workspace: {
      ...state.favlo_workspace,
      selection: { kind: 'file', path: paths[0] },
    },
  };
}

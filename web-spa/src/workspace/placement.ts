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
): PlacementState {
  const next = incoming.favlo_workspace;
  const prior = local.favlo_workspace;
  const selection =
    next.selection != null
      ? next.selection
      : prior.selection;
  return ensureFilmstripSelection({
    favlo_workspace: {
      ...next,
      selection,
    },
  });
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

/** When the strip has items, keep selection on one of them so preview has a path. */
export function ensureFilmstripSelection(state: PlacementState): PlacementState {
  const paths = filmstripPaths(state);
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
  const current = state.favlo_workspace.selection?.path;
  if (current && paths.includes(current)) {
    return state;
  }
  return {
    favlo_workspace: {
      ...state.favlo_workspace,
      selection: { kind: 'file', path: paths[0] },
    },
  };
}

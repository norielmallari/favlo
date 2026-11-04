export type WorkspaceBarStyle = 'canvas' | 'toolbar';

export type WorkspaceAppId = 'gallery' | 'coding';

export type WorkspaceApp = {
  id: WorkspaceAppId;
  name: string;
  mark: 'aperture' | 'terminal';
  defaultScheme: 'favlo' | 'rose-pine' | 'catppuccin';
  bar: WorkspaceBarStyle;
};

const APPS: Record<WorkspaceAppId, WorkspaceApp> = {
  gallery: {
    id: 'gallery',
    name: 'Gallery',
    mark: 'aperture',
    defaultScheme: 'rose-pine',
    bar: 'canvas',
  },
  coding: {
    id: 'coding',
    name: 'Coding',
    mark: 'terminal',
    defaultScheme: 'catppuccin',
    bar: 'toolbar',
  },
};

export function workspaceAppForRenderer(renderer: string | undefined): WorkspaceApp {
  if (renderer === 'chat') {
    return APPS.coding;
  }
  return APPS.gallery;
}

export function workspaceApp(id: WorkspaceAppId): WorkspaceApp {
  return APPS[id];
}

import type { ComponentType } from 'react';

import { Filmstrip } from '@/workspace/elements/Filmstrip';
import { Preview } from '@/workspace/elements/Preview';

export type ElementProps = {
  props: Record<string, unknown>;
  selection: { kind: string; path: string } | null;
  fileUrl: (path: string) => string;
  onSelect: (path: string) => void;
  onAddToChat: (path: string) => void;
};

export const CATALOG: Record<
  string,
  { name: string; version: number; slots: string[]; component: ComponentType<ElementProps> }
> = {
  preview: { name: 'preview', version: 1, slots: ['stage'], component: Preview },
  filmstrip: { name: 'filmstrip', version: 1, slots: ['rail'], component: Filmstrip },
};

export const RESERVED = ['transcript', 'file_list', 'file_view', 'diff', 'terminal_log'] as const;

export function catalogNames(): string[] {
  return Object.keys(CATALOG).sort();
}

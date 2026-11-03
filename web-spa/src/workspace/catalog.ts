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

/** Local painters. The shell document names them; an unknown name is skipped. */
export const COMPONENTS: Record<string, ComponentType<ElementProps>> = {
  preview: Preview,
  filmstrip: Filmstrip,
};

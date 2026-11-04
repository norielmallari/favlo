import type { ComponentType } from 'react';

import { Filmstrip } from '@/workspace/elements/Filmstrip';
import { Preview } from '@/workspace/elements/Preview';

export type PendingMedia = {
  task_id: string;
  kind: string;
  status: string;
};

export type ElementProps = {
  props: Record<string, unknown>;
  selection: { kind: string; path: string } | null;
  fileUrl: (path: string) => string;
  onSelect: (path: string) => void;
  onAddToChat: (path: string) => void;
  onRequestDelete: (path: string) => void;
  /** Local only: swipe the preview to the previous or next filmstrip item. */
  onSwipe?: (direction: 'prev' | 'next') => void;
  /** Local only: preview is filling the column above the composer. */
  fullScreen?: boolean;
  /** Local only: toggle the full-screen preview detent. */
  onToggleFullScreen?: () => void;
  /** Host media tasks still working; shown as filmstrip placeholders. */
  pendingMedia?: PendingMedia[];
};

/** Local painters. The shell document names them; an unknown name is skipped. */
export const COMPONENTS: Record<string, ComponentType<ElementProps>> = {
  preview: Preview,
  filmstrip: Filmstrip,
};

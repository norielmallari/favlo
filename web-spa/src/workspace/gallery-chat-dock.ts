import { Platform } from 'react-native';

/** Resting heights for the narrow gallery chat dock. */
export type GalleryChatDetent = 'peek' | 'expanded' | 'fullscreen';

export const GALLERY_CHAT_DETENT_KEY = 'favlo.galleryChatDetent';
export const GALLERY_CHAT_DETENTS: GalleryChatDetent[] = ['fullscreen', 'peek', 'expanded'];

/** Preview keeps at least this much height when the dock is not fullscreen. */
export const GALLERY_PREVIEW_FLOOR = 120;
/** Filmstrip row plus the "n of N" caption. */
export const GALLERY_STRIP_HEIGHT = 110;
/** Grabber hit target (matches the borders rule). */
export const GALLERY_DOCK_GRABBER = 12;
/** Composer row without the Return caption. */
export const GALLERY_COMPOSER_LINE = 56;
/** One short turn above the composer in Peek. */
export const GALLERY_PEEK_TURN = 88;
/** Peek may not take more than this share of the column. */
export const GALLERY_PEEK_MAX_FRACTION = 1 / 3;
/** Drag velocity (px/ms) that jumps to the next detent. */
export const GALLERY_DOCK_FLING = 0.45;

export function isGalleryChatDetent(value: unknown): value is GalleryChatDetent {
  return value === 'peek' || value === 'expanded' || value === 'fullscreen';
}

/** Height of the dock at each detent for a column of `columnHeight`. */
export function galleryDockHeights(columnHeight: number): Record<GalleryChatDetent, number> {
  const height = Math.max(0, Math.round(columnHeight));
  const fullscreen = Math.min(height, GALLERY_DOCK_GRABBER + GALLERY_COMPOSER_LINE);
  const peekNatural = GALLERY_DOCK_GRABBER + GALLERY_PEEK_TURN + GALLERY_COMPOSER_LINE;
  const peekCap = Math.floor(height * GALLERY_PEEK_MAX_FRACTION);
  const peek = Math.min(height, Math.max(fullscreen, Math.min(peekNatural, peekCap || peekNatural)));
  const galleryReserve = GALLERY_PREVIEW_FLOOR + GALLERY_STRIP_HEIGHT;
  const expanded = Math.min(height, Math.max(peek, height - galleryReserve));
  return {
    fullscreen,
    peek: Math.min(height, Math.max(fullscreen, peek)),
    expanded: Math.min(height, Math.max(peek, expanded)),
  };
}

export function galleryDockHeight(detent: GalleryChatDetent, columnHeight: number): number {
  return galleryDockHeights(columnHeight)[detent];
}

/** Nearest detent by absolute height. */
export function nearestGalleryDetent(height: number, columnHeight: number): GalleryChatDetent {
  const heights = galleryDockHeights(columnHeight);
  let best: GalleryChatDetent = 'peek';
  let bestGap = Number.POSITIVE_INFINITY;
  for (const detent of GALLERY_CHAT_DETENTS) {
    const gap = Math.abs(heights[detent] - height);
    if (gap < bestGap) {
      best = detent;
      bestGap = gap;
    }
  }
  return best;
}

/**
 * Settle after a drag. Positive velocity means the dock is growing (finger moved up).
 * A fling jumps one detent in that direction; otherwise the nearest height wins.
 */
export function settleGalleryDetent(
  height: number,
  columnHeight: number,
  velocityPxPerMs: number,
): GalleryChatDetent {
  const nearest = nearestGalleryDetent(height, columnHeight);
  if (Math.abs(velocityPxPerMs) < GALLERY_DOCK_FLING) {
    return nearest;
  }
  const index = GALLERY_CHAT_DETENTS.indexOf(nearest);
  if (velocityPxPerMs > 0) {
    return GALLERY_CHAT_DETENTS[Math.min(GALLERY_CHAT_DETENTS.length - 1, index + 1)];
  }
  return GALLERY_CHAT_DETENTS[Math.max(0, index - 1)];
}

export function cycleGalleryDetent(current: GalleryChatDetent): GalleryChatDetent {
  const index = GALLERY_CHAT_DETENTS.indexOf(current);
  return GALLERY_CHAT_DETENTS[(index + 1) % GALLERY_CHAT_DETENTS.length];
}

export function readGalleryChatDetent(): GalleryChatDetent {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return 'peek';
  }
  const raw = localStorage.getItem(GALLERY_CHAT_DETENT_KEY);
  return isGalleryChatDetent(raw) ? raw : 'peek';
}

export function writeGalleryChatDetent(detent: GalleryChatDetent): void {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return;
  }
  localStorage.setItem(GALLERY_CHAT_DETENT_KEY, detent);
}

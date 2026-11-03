import { Platform } from 'react-native';

export const GALLERY_CHAT_WIDTH_KEY = 'favlo.galleryChatWidth';
export const GALLERY_CHAT_WIDTH_MIN = 320;
export const GALLERY_CHAT_WIDTH_MAX_CAP = 560;
export const GALLERY_CHAT_WIDTH_DEFAULT = 420;
export const GALLERY_CHAT_WIDTH_FRACTION = 0.45;
export const GALLERY_CHAT_WIDTH_STEP = 24;

/** Upper bound for the gallery Conversation column on a wide row. */
export function galleryChatWidthMax(rowWidth: number): number {
  if (!(rowWidth > 0)) {
    return GALLERY_CHAT_WIDTH_MAX_CAP;
  }
  return Math.min(GALLERY_CHAT_WIDTH_MAX_CAP, Math.floor(rowWidth * GALLERY_CHAT_WIDTH_FRACTION));
}

/** Clamp a chat width to the wide-layout min/max for ``rowWidth``. */
export function clampGalleryChatWidth(width: number, rowWidth: number): number {
  const max = galleryChatWidthMax(rowWidth);
  const min = Math.min(GALLERY_CHAT_WIDTH_MIN, max);
  if (!Number.isFinite(width)) {
    return Math.min(Math.max(GALLERY_CHAT_WIDTH_DEFAULT, min), max);
  }
  return Math.max(min, Math.min(max, Math.round(width)));
}

export function readGalleryChatWidth(rowWidth = 0): number {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return clampGalleryChatWidth(GALLERY_CHAT_WIDTH_DEFAULT, rowWidth);
  }
  const raw = localStorage.getItem(GALLERY_CHAT_WIDTH_KEY);
  const parsed = raw == null ? GALLERY_CHAT_WIDTH_DEFAULT : Number(raw);
  return clampGalleryChatWidth(parsed, rowWidth);
}

export function writeGalleryChatWidth(width: number, rowWidth = 0): void {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return;
  }
  localStorage.setItem(GALLERY_CHAT_WIDTH_KEY, String(clampGalleryChatWidth(width, rowWidth)));
}

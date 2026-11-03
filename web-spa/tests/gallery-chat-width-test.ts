import { Platform } from 'react-native';

import {
  clampGalleryChatWidth,
  GALLERY_CHAT_WIDTH_DEFAULT,
  GALLERY_CHAT_WIDTH_KEY,
  GALLERY_CHAT_WIDTH_MAX_CAP,
  GALLERY_CHAT_WIDTH_MIN,
  galleryChatWidthMax,
  readGalleryChatWidth,
  writeGalleryChatWidth,
} from '@/workspace/gallery-chat-width';

test('gallery chat width max respects the fraction and the hard cap', () => {
  expect(galleryChatWidthMax(2000)).toBe(GALLERY_CHAT_WIDTH_MAX_CAP);
  expect(galleryChatWidthMax(800)).toBe(360);
});

test('gallery chat width clamps to the wide-layout window', () => {
  expect(clampGalleryChatWidth(100, 1280)).toBe(GALLERY_CHAT_WIDTH_MIN);
  expect(clampGalleryChatWidth(900, 1280)).toBe(GALLERY_CHAT_WIDTH_MAX_CAP);
  expect(clampGalleryChatWidth(400, 1280)).toBe(400);
  expect(clampGalleryChatWidth(400, 600)).toBe(270);
});

test('gallery chat width persists through localStorage', () => {
  const store = new Map<string, string>();
  const previous = Platform.OS;
  Object.defineProperty(Platform, 'OS', { configurable: true, get: () => 'web' });
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => {
        store.set(key, value);
      },
      clear: () => store.clear(),
    },
  });
  try {
    expect(readGalleryChatWidth(1280)).toBe(GALLERY_CHAT_WIDTH_DEFAULT);
    writeGalleryChatWidth(480, 1280);
    expect(store.get(GALLERY_CHAT_WIDTH_KEY)).toBe('480');
    expect(readGalleryChatWidth(1280)).toBe(480);
    writeGalleryChatWidth(900, 1280);
    expect(readGalleryChatWidth(1280)).toBe(GALLERY_CHAT_WIDTH_MAX_CAP);
  } finally {
    Object.defineProperty(Platform, 'OS', { configurable: true, get: () => previous });
  }
});

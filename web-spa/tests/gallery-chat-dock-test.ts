import { Platform } from 'react-native';

import {
  cycleGalleryDetent,
  GALLERY_CHAT_DETENT_KEY,
  GALLERY_COMPOSER_LINE,
  GALLERY_DOCK_GRABBER,
  GALLERY_PEEK_TURN,
  galleryDockHeight,
  galleryDockHeights,
  nearestGalleryDetent,
  readGalleryChatDetent,
  settleGalleryDetent,
  writeGalleryChatDetent,
} from '@/workspace/gallery-chat-dock';

test('gallery dock heights keep peek between fullscreen and expanded', () => {
  const heights = galleryDockHeights(800);
  expect(heights.fullscreen).toBe(GALLERY_DOCK_GRABBER + GALLERY_COMPOSER_LINE);
  expect(heights.peek).toBeGreaterThan(heights.fullscreen);
  expect(heights.peek).toBeLessThanOrEqual(Math.floor(800 / 3));
  expect(heights.expanded).toBeGreaterThanOrEqual(heights.peek);
  expect(heights.expanded).toBeLessThanOrEqual(800);
});

test('gallery dock peek uses the natural turn height on a tall column', () => {
  const natural = GALLERY_DOCK_GRABBER + GALLERY_PEEK_TURN + GALLERY_COMPOSER_LINE;
  expect(galleryDockHeight('peek', 900)).toBe(natural);
  const short = galleryDockHeights(natural);
  expect(short.peek).toBe(short.fullscreen);
  expect(short.peek).toBeLessThanOrEqual(natural);
});

test('nearest gallery detent picks the closest resting height', () => {
  const heights = galleryDockHeights(900);
  expect(nearestGalleryDetent(heights.peek, 900)).toBe('peek');
  expect(nearestGalleryDetent(heights.expanded - 4, 900)).toBe('expanded');
  expect(nearestGalleryDetent(heights.fullscreen + 2, 900)).toBe('fullscreen');
});

test('settle gallery detent flings to the next resting height', () => {
  const heights = galleryDockHeights(900);
  expect(settleGalleryDetent(heights.peek, 900, 0)).toBe('peek');
  expect(settleGalleryDetent(heights.peek, 900, 0.8)).toBe('expanded');
  expect(settleGalleryDetent(heights.peek, 900, -0.8)).toBe('fullscreen');
});

test('cycle gallery detent walks fullscreen → peek → expanded', () => {
  expect(cycleGalleryDetent('fullscreen')).toBe('peek');
  expect(cycleGalleryDetent('peek')).toBe('expanded');
  expect(cycleGalleryDetent('expanded')).toBe('fullscreen');
});

test('gallery chat detent persists through localStorage', () => {
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
    expect(readGalleryChatDetent()).toBe('peek');
    writeGalleryChatDetent('expanded');
    expect(store.get(GALLERY_CHAT_DETENT_KEY)).toBe('expanded');
    expect(readGalleryChatDetent()).toBe('expanded');
    writeGalleryChatDetent('fullscreen');
    expect(readGalleryChatDetent()).toBe('fullscreen');
  } finally {
    Object.defineProperty(Platform, 'OS', { configurable: true, get: () => previous });
  }
});

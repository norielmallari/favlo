import { keepLatest } from '@/components/ChatScreen';

test('a fresh transcript stays with the latest message', () => {
  expect(keepLatest(true, false, 900)).toBe(true);
  expect(keepLatest(true, true, 0)).toBe(true);
  expect(keepLatest(true, true, 64)).toBe(true);
});

test('scrolling up leaves the latest message', () => {
  expect(keepLatest(true, true, 200)).toBe(false);
  expect(keepLatest(false, true, 200)).toBe(false);
  expect(keepLatest(false, true, 0)).toBe(true);
});

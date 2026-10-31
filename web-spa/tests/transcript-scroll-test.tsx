import { keepLatest, transcriptFrame } from '@/components/chat-transcript';

test('a fresh transcript stays with the latest message', () => {
  expect(keepLatest(true, false, 900)).toBe(true);
  expect(keepLatest(true, true, 0)).toBe(true);
  expect(keepLatest(true, true, 64)).toBe(true);
});

test('a short transcript rests on the composer', () => {
  expect(transcriptFrame(16)).toEqual({
    flexGrow: 1,
    justifyContent: 'flex-end',
    gap: 16,
    maxWidth: '100%',
  });
});

test('scrolling up leaves the latest message', () => {
  expect(keepLatest(true, true, 200)).toBe(false);
  expect(keepLatest(false, true, 200)).toBe(false);
  expect(keepLatest(false, true, 0)).toBe(true);
});

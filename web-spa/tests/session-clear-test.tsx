import { clearedTranscript } from '@/components/ChatScreen';

test('a cleared session drops that transcript', () => {
  const event = { type: 'session.cleared', stream_id: 'session:session_favlo' };
  expect(clearedTranscript(event, 'session_favlo')).toBe(true);
  expect(clearedTranscript(event, 'session_other')).toBe(false);
  expect(clearedTranscript({ type: 'message.created', stream_id: 'session:session_favlo' }, 'session_favlo')).toBe(
    false,
  );
});

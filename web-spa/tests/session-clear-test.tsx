import { clearedTranscript } from '@/components/ChatScreen';

test('a cleared session drops the replaced transcript only', () => {
  const event = {
    type: 'session.cleared',
    stream_id: 'session:session_new',
    payload: { session_id: '22222222-2222-4222-8222-222222222222', previous_session_id: '11111111-1111-4111-8111-111111111111' },
  };
  expect(clearedTranscript(event, '11111111-1111-4111-8111-111111111111')).toBe(true);
  expect(clearedTranscript(event, '22222222-2222-4222-8222-222222222222')).toBe(false);
  expect(clearedTranscript(event, 'session_other')).toBe(false);
  expect(clearedTranscript({ type: 'message.created', stream_id: 'session:11111111-1111-4111-8111-111111111111' }, '11111111-1111-4111-8111-111111111111')).toBe(
    false,
  );
});

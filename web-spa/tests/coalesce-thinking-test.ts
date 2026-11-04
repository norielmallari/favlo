import {
  coalesceThinkingTurns,
  isThinkingOnlyMessage,
  type ChatMessage,
} from '@/components/chat-transcript';

test('thinking-only turns merge into one Thinking block', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'First thought.' },
    { id: '2', role: 'assistant', body: '', reasoning: 'Second thought.' },
  ];
  expect(isThinkingOnlyMessage(messages[0]!)).toBe(true);
  const shown = coalesceThinkingTurns(messages);
  expect(shown).toHaveLength(1);
  expect(shown[0]?.id).toBe('2');
  expect(shown[0]?.reasoning).toBe('First thought.\n\nSecond thought.');
  expect(shown[0]?.body).toBe('');
});

test('a reply or tool call breaks the merge', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'Alone.' },
    { id: '2', role: 'assistant', body: 'Hello.', reasoning: 'With reply.' },
    { id: '3', role: 'assistant', body: '', reasoning: 'After.' },
    {
      id: '4',
      role: 'assistant',
      body: '',
      reasoning: 'Tool turn.',
      toolCalls: [{ id: 'c1', name: 'list', arguments: '{}' }],
    },
    { id: '5', role: 'assistant', body: '', reasoning: 'Later.' },
  ];
  const shown = coalesceThinkingTurns(messages);
  expect(shown.map((item) => item.id)).toEqual(['1', '2', '3', '4', '5']);
});

test('a user message breaks the merge', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'A.' },
    { id: '2', role: 'user', body: 'Hi' },
    { id: '3', role: 'assistant', body: '', reasoning: 'B.' },
  ];
  expect(coalesceThinkingTurns(messages)).toHaveLength(3);
});

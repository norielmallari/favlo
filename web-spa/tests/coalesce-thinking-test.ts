import {
  coalesceThinkingTurns,
  isThinkingOnlyMessage,
  visibleReplySteps,
  storedSteps,
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

test('a thinking-only turn folds into the next reply', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'Alone.' },
    { id: '2', role: 'assistant', body: 'Hello.', reasoning: 'With reply.' },
  ];
  const shown = coalesceThinkingTurns(messages);
  expect(shown).toHaveLength(1);
  expect(shown[0]?.id).toBe('2');
  expect(shown[0]?.body).toBe('Hello.');
  expect(shown[0]?.reasoning).toBe('Alone.\n\nWith reply.');
});

test('a thinking-only turn folds into a tool turn and hidden tools coalesce Thinking', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'Before tool.' },
    {
      id: '2',
      role: 'assistant',
      body: 'Done.',
      reasoning: 'After tool.',
      toolCalls: [{ id: 'c1', name: 'gallery_create_folder', arguments: '{}' }],
    },
  ];
  const shown = coalesceThinkingTurns(messages);
  expect(shown).toHaveLength(1);
  const display = visibleReplySteps(storedSteps(shown[0]!), 'collapsed', 'hidden');
  expect(display).toEqual([
    { kind: 'thinking', text: 'Before tool.\n\nAfter tool.', settled: true },
    { kind: 'reply', text: 'Done.' },
  ]);
});

test('a user message breaks the merge', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'A.' },
    { id: '2', role: 'user', body: 'Hi' },
    { id: '3', role: 'assistant', body: '', reasoning: 'B.' },
  ];
  expect(coalesceThinkingTurns(messages)).toHaveLength(3);
});

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
      toolCalls: [{ id: 'c1', name: 'gallery_delete_image', arguments: '{}' }],
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

test('a whitespace thinking row with a trace folds into the tool reply', () => {
  const messages: ChatMessage[] = [
    {
      id: 'think',
      role: 'assistant',
      body: '\n\n',
      reasoning: 'Before the tool.',
      trace: [{ kind: 'thinking', text: 'Before the tool.' }],
    },
    {
      id: 'reply',
      role: 'assistant',
      body: '\n\nThank you.',
      reasoning: 'After the tool.',
      toolCalls: [{ id: 'c1', name: 'favlo_media_remember', arguments: '{}' }],
      trace: [
        { kind: 'thinking', text: 'After the tool.' },
        { kind: 'tool', call: { id: 'c1', name: 'favlo_media_remember', arguments: '{}' } },
        { kind: 'reply', text: '\n\nThank you.' },
      ],
    },
  ];
  expect(isThinkingOnlyMessage(messages[0]!)).toBe(true);
  const shown = coalesceThinkingTurns(messages);
  expect(shown).toHaveLength(1);
  const display = visibleReplySteps(storedSteps(shown[0]!), 'collapsed', 'hidden');
  expect(display.map((step) => step.kind)).toEqual(['thinking', 'reply']);
  expect(display[0]).toMatchObject({
    kind: 'thinking',
    text: 'Before the tool.\n\nAfter the tool.',
  });
});

test('a user message breaks the merge', () => {
  const messages: ChatMessage[] = [
    { id: '1', role: 'assistant', body: '', reasoning: 'A.' },
    { id: '2', role: 'user', body: 'Hi' },
    { id: '3', role: 'assistant', body: '', reasoning: 'B.' },
  ];
  expect(coalesceThinkingTurns(messages)).toHaveLength(3);
});

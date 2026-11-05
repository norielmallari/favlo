import { useState } from 'react';
import { Pressable, Text } from 'react-native';

import { render, userEvent, waitFor } from '@testing-library/react-native';

import { useAguiChat } from '@/components/use-agui-chat';
import type { ReplyStep } from '@/components/chat-transcript';

const DELTAS = Array.from({ length: 60 }, (_, index) => `w${index} `);

function thinking(steps: ReplyStep[]): string {
  return steps
    .filter((step) => step.kind === 'thinking')
    .map((step) => step.text)
    .join('');
}

function Probe() {
  const [steps, setSteps] = useState<ReplyStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const chat = useAguiChat({
    threadId: 'thread',
    onMessages: () => undefined,
    onSteps: setSteps,
    onWorking: () => undefined,
    onRunId: () => undefined,
    onError: setError,
    onStatus: () => undefined,
  });
  return (
    <>
      <Pressable accessibilityLabel="Send" onPress={() => void chat.send('hello')} />
      <Text>{error ?? thinking(steps)}</Text>
    </>
  );
}

test('a burst of reasoning tokens in one read stays one update', async () => {
  const sse = DELTAS.map(
    (delta) => `data: ${JSON.stringify({ type: 'REASONING_MESSAGE_CONTENT', delta })}\n\n`,
  ).join('');
  const fetchMock = jest.fn(async (url: string) => {
    if (url.includes('/cancel')) {
      return { ok: true, json: async () => ({}) };
    }
    return {
      ok: true,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        },
      }),
    };
  });
  const prior = globalThis.fetch;
  globalThis.fetch = fetchMock as typeof fetch;
  try {
    const view = await render(<Probe />);
    await userEvent.press(view.getByLabelText('Send'));
    await waitFor(() => {
      expect(view.getByText(DELTAS.join(''))).toBeTruthy();
    });
  } finally {
    globalThis.fetch = prior;
  }
});

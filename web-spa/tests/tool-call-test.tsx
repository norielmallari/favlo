import { render, userEvent, waitFor } from '@testing-library/react-native';

import { showsMessage } from '@/components/ChatScreen';
import {
  LiveReply,
  mergeToolCall,
  noteReasoning,
  noteReply,
  noteTool,
  readToolCalls,
  type ReplyStep,
} from '@/components/chat-transcript';
import { ToolCall } from '@/components/ToolCall';
import { ThemeProvider } from '@/theme/Theme';

const call = { id: 'call_1', name: 'conversation_list_conversations', arguments: '{"kind": "all"}' };

function Host({ settled }: { settled: boolean }) {
  return (
    <ThemeProvider>
      <ToolCall call={call} settled={settled} />
    </ThemeProvider>
  );
}

test('a tool call stays on screen without reply text', () => {
  expect(showsMessage(undefined, undefined, [call])).toBe(true);
  expect(showsMessage('', '', [])).toBe(false);
  expect(readToolCalls(JSON.stringify([call]))).toEqual([call]);
  expect(mergeToolCall([], call)).toEqual([call]);
  expect(mergeToolCall([call], { ...call, arguments: '{"kind": "all", "limit": 1}' })).toEqual([
    { ...call, arguments: '{"kind": "all", "limit": 1}' },
  ]);
});

test('thinking closes on a tool call and the tool call closes when thinking starts', async () => {
  let steps: ReplyStep[] = [];
  steps = noteReasoning(steps, 'First look.');
  steps = noteTool(steps, call);
  steps = noteTool(steps, { ...call, arguments: '{"kind": "all", "limit": 1}' });
  const view = await render(
    <ThemeProvider>
      <LiveReply steps={steps} />
    </ThemeProvider>,
  );
  expect(view.getByRole('button', { name: 'Thinking' }).props.accessibilityState).toMatchObject({ expanded: false });
  expect(view.queryByText('First look.')).toBeNull();
  const tool = view.getByRole('button', { name: 'Tool call conversation_list_conversations' });
  expect(tool.props.accessibilityState).toMatchObject({ expanded: true });
  expect(view.getByText('{"kind": "all", "limit": 1}')).toBeTruthy();
  steps = noteReasoning(steps, 'After the tool.');
  view.rerender(
    <ThemeProvider>
      <LiveReply steps={steps} />
    </ThemeProvider>,
  );
  await waitFor(() => {
    expect(view.getByRole('button', { name: 'Tool call conversation_list_conversations' }).props.accessibilityState).toMatchObject({
      expanded: false,
    });
  });
  const thinking = view.getAllByRole('button', { name: 'Thinking' });
  expect(thinking).toHaveLength(2);
  expect(thinking[0].props.accessibilityState).toMatchObject({ expanded: false });
  expect(thinking[1].props.accessibilityState).toMatchObject({ expanded: true });
  expect(view.getByText('After the tool.')).toBeTruthy();
  expect(view.queryByText('First look.')).toBeNull();
});

test('thinking, a tool call, and the reply stay in arrival order', async () => {
  let steps: ReplyStep[] = [];
  steps = noteReasoning(steps, 'First look.');
  steps = noteReply(steps, 'Hello');
  steps = noteTool(steps, call);
  steps = noteReasoning(steps, 'After the tool.');
  steps = noteReply(steps, 'Done');
  const view = await render(
    <ThemeProvider>
      <LiveReply steps={steps} speaker="Favlo" />
    </ThemeProvider>,
  );
  const thinking = view.getAllByRole('button', { name: 'Thinking' });
  expect(thinking).toHaveLength(2);
  expect(thinking[0].props.accessibilityState).toMatchObject({ expanded: false });
  expect(thinking[1].props.accessibilityState).toMatchObject({ expanded: false });
  const tool = view.getByRole('button', { name: 'Tool call conversation_list_conversations' });
  expect(tool.props.accessibilityState).toMatchObject({ expanded: false });
  const tree = JSON.stringify(view.toJSON());
  const hello = tree.indexOf('Favlo said: Hello');
  const first = tree.indexOf('Thinking');
  const toolAt = tree.indexOf('Tool call conversation_list_conversations');
  const done = tree.indexOf('Done');
  expect(first).toBeGreaterThanOrEqual(0);
  expect(first).toBeLessThan(hello);
  expect(hello).toBeLessThan(toolAt);
  expect(toolAt).toBeLessThan(done);
  expect(view.getByText('Hello')).toBeTruthy();
  expect(view.getByText('Done')).toBeTruthy();
  expect(view.queryByText('First look.')).toBeNull();
});

test('a tool call starts open and closes when the reply is stored', async () => {
  const user = userEvent.setup();
  const view = await render(<Host settled={false} />);
  const button = view.getByRole('button', { name: 'Tool call conversation_list_conversations' });
  expect(button.props.accessibilityState).toMatchObject({ expanded: true });
  expect(view.getByText('{"kind": "all"}')).toBeTruthy();
  await user.press(button);
  expect(view.queryByText('{"kind": "all"}')).toBeNull();
  view.rerender(<Host settled />);
  await waitFor(() => {
    expect(view.queryByText('{"kind": "all"}')).toBeNull();
  });
  expect(view.getByRole('button', { name: 'Tool call conversation_list_conversations' }).props.accessibilityState).toMatchObject({
    expanded: false,
  });
  await user.press(view.getByRole('button', { name: 'Tool call conversation_list_conversations' }));
  expect(view.getByText('{"kind": "all"}')).toBeTruthy();
});

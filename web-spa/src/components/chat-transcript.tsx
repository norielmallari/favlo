import { useCallback, useState } from 'react';
import { Platform, Text, View } from 'react-native';

import { humanText, MessageBody } from '@/components/MessageBody';
import { IconButton } from '@/components/IconButton';
import { type ExpandTrace } from '@/components/expandTrace';
import { focusSafePadX, focusSafePadY } from '@/components/Pressable';
import { Thinking } from '@/components/Thinking';
import { ToolCall, type ToolCallRecord } from '@/components/ToolCall';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

export type { ToolCallRecord };
export type ChatMessage = {
  id: string;
  role: string;
  body: string;
  reasoning?: string;
  toolCalls?: ToolCallRecord[];
  trace?: StoredPart[];
};
export const LATEST_SLACK = 64;

export function readToolCalls(value: unknown): ToolCallRecord[] {
  if (value == null || value === '') {
    return [];
  }
  let raw = value;
  if (typeof value === 'string') {
    try {
      raw = JSON.parse(value) as unknown;
    } catch {
      return [];
    }
  }
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.flatMap((item) => {
    if (item == null || typeof item !== 'object') {
      return [];
    }
    const row = item as { id?: unknown; name?: unknown; arguments?: unknown; result?: unknown };
    if (typeof row.id !== 'string' || row.id === '') {
      return [];
    }
    const call: ToolCallRecord = {
      id: row.id,
      name: typeof row.name === 'string' ? row.name : '',
      arguments: typeof row.arguments === 'string' ? row.arguments : '',
    };
    if (typeof row.result === 'string' && row.result !== '') {
      call.result = row.result;
    }
    return [call];
  });
}

export function mergeToolCall(current: ToolCallRecord[], incoming: ToolCallRecord): ToolCallRecord[] {
  const index = current.findIndex((item) => item.id === incoming.id);
  if (index < 0) {
    return [...current, incoming];
  }
  const prior = current[index];
  const next = current.slice();
  next[index] = {
    id: incoming.id,
    name: incoming.name || prior.name,
    arguments: incoming.arguments,
    result: incoming.result ?? prior.result,
  };
  return next;
}

export function toolCallDelta(agui?: {
  id?: string;
  name?: string;
  arguments?: string;
  result?: string;
}): ToolCallRecord | null {
  if (!agui?.id) {
    return null;
  }
  return {
    id: agui.id,
    name: agui.name ?? '',
    arguments: agui.arguments ?? '',
    result: agui.result || undefined,
  };
}

export type StoredPart =
  | { kind: 'thinking'; text: string }
  | { kind: 'tool'; call: ToolCallRecord }
  | { kind: 'reply'; text: string };

export type ReplyStep =
  | { kind: 'thinking'; text: string; settled: boolean }
  | { kind: 'tool'; call: ToolCallRecord; settled: boolean }
  | { kind: 'reply'; text: string };

function settleOpen(steps: ReplyStep[]): ReplyStep[] {
  return steps.map((step) => (step.kind !== 'reply' && !step.settled ? { ...step, settled: true } : step));
}

export function noteReasoning(steps: ReplyStep[], delta: string): ReplyStep[] {
  if (!delta) {
    return steps;
  }
  const last = steps[steps.length - 1];
  if (last?.kind === 'thinking' && !last.settled) {
    const next = steps.slice();
    next[next.length - 1] = { kind: 'thinking', text: last.text + delta, settled: false };
    return next;
  }
  if (!delta.trim()) {
    return steps;
  }
  return [...settleOpen(steps), { kind: 'thinking', text: delta, settled: false }];
}

export function noteReply(steps: ReplyStep[], delta: string): ReplyStep[] {
  if (!delta) {
    return steps;
  }
  const last = steps[steps.length - 1];
  if (last?.kind === 'reply') {
    const next = steps.slice();
    next[next.length - 1] = { kind: 'reply', text: last.text + delta };
    return next;
  }
  if (!delta.trim()) {
    return steps;
  }
  return [...settleOpen(steps), { kind: 'reply', text: delta }];
}

export function noteTool(steps: ReplyStep[], call: ToolCallRecord): ReplyStep[] {
  const index = steps.findIndex((step) => step.kind === 'tool' && step.call.id === call.id);
  if (index >= 0) {
    const prior = steps[index];
    if (prior.kind !== 'tool') {
      return steps;
    }
    const next = steps.slice();
    next[index] = {
      kind: 'tool',
      settled: prior.settled,
      call: {
        id: call.id,
        name: call.name || prior.call.name,
        arguments: call.arguments,
        result: call.result ?? prior.call.result,
      },
    };
    return next;
  }
  return [
    ...steps.map((step) => (step.kind === 'thinking' && !step.settled ? { ...step, settled: true } : step)),
    { kind: 'tool', call, settled: false },
  ];
}

export function replyVisible(steps: ReplyStep[]): boolean {
  return steps.some((step) => step.kind === 'tool' || step.text.trim() !== '');
}

/**
 * Display-only: drop hidden tools/thinking, then merge consecutive Thinking rows.
 * With Expand tool calls = Not shown, thinking around a tool becomes one block.
 */
export function visibleReplySteps(
  steps: ReplyStep[],
  expandTrace: ExpandTrace = 'collapsed',
  expandTools: ExpandTrace = 'collapsed',
): ReplyStep[] {
  const shown: ReplyStep[] = [];
  for (const step of steps) {
    if (step.kind === 'tool') {
      if (expandTools === 'hidden') {
        continue;
      }
      shown.push(step);
      continue;
    }
    if (step.kind === 'thinking') {
      if (expandTrace === 'hidden' || !step.text.trim()) {
        continue;
      }
      const last = shown[shown.length - 1];
      if (last?.kind === 'thinking') {
        shown[shown.length - 1] = {
          kind: 'thinking',
          text: `${last.text.trim()}\n\n${step.text.trim()}`,
          settled: last.settled && step.settled,
        };
        continue;
      }
      shown.push({ kind: 'thinking', text: step.text.trim(), settled: step.settled });
      continue;
    }
    if (!step.text.trim()) {
      continue;
    }
    shown.push(step);
  }
  return shown;
}

export function LiveReply({
  steps,
  speaker,
  measured = false,
  expandTrace = 'collapsed',
  expandTools,
  forceOpen = false,
  imageNames = false,
}: {
  steps: ReplyStep[];
  speaker?: string;
  measured?: boolean;
  expandTrace?: ExpandTrace;
  expandTools?: ExpandTrace;
  forceOpen?: boolean;
  imageNames?: boolean;
}) {
  const toolExpand = expandTools ?? expandTrace;
  const display = visibleReplySteps(steps, expandTrace, toolExpand);
  return (
    <View style={{ width: '100%', maxWidth: '100%', minWidth: 0, alignSelf: 'stretch', gap: space[2] }}>
      {display.map((step, index) => {
        if (step.kind === 'thinking') {
          return (
            <Thinking
              key={`thinking-${index}`}
              text={step.text}
              settled={step.settled}
              expandTrace={expandTrace}
              forceOpen={forceOpen}
            />
          );
        }
        if (step.kind === 'tool') {
          return (
            <ToolCall
              key={`tool-${step.call.id}-${index}`}
              call={step.call}
              settled={step.settled}
              expandTrace={toolExpand}
              forceOpen={forceOpen}
            />
          );
        }
        return (
          <View
            key={`reply-${index}`}
            accessibilityLabel={speaker ? `${speaker} said: ${humanText(step.text, imageNames)}` : undefined}
            style={[{ width: '100%', maxWidth: '100%', alignSelf: 'stretch' }, measured ? measureStyle : null]}
          >
            <MessageBody body={step.text} imageNames={imageNames} />
          </View>
        );
      })}
    </View>
  );
}

export function showsMessage(text?: string, reasoning?: string | null, tools?: unknown): boolean {
  return Boolean(text) || Boolean(reasoning && reasoning.trim()) || readToolCalls(tools).length > 0;
}

export const SESSION_CLEAR = '/session clear';
export const SESSION_ARCHIVE = '/session archive';

export function sessionCommand(text: string): 'clear' | 'archive' | null {
  const trimmed = text.trim();
  if (trimmed === SESSION_CLEAR) {
    return 'clear';
  }
  if (trimmed === SESSION_ARCHIVE) {
    return 'archive';
  }
  return null;
}

export type SentMessage = {
  message_id?: string;
  text?: string;
  run_id?: string;
  cleared?: boolean;
  session_id?: string;
  previous_session_id?: string;
};

export function clearedTranscript(
  event: {
    type: string;
    stream_id?: string;
    payload?: { session_id?: string; previous_session_id?: string };
  },
  sessionId: string | undefined,
): boolean {
  if (sessionId == null || event.type !== 'session.cleared') {
    return false;
  }
  // Clear only when this view is still on the replaced session. The birth
  // event on the new stream must not wipe messages loaded after that clear.
  return event.payload?.previous_session_id === sessionId;
}

export function keepLatest(following: boolean, placed: boolean, fromBottom: number): boolean {
  if (!placed) {
    return following;
  }
  return fromBottom <= LATEST_SLACK;
}

export function transcriptFrame(gap: number) {
  return {
    flexGrow: 1,
    justifyContent: 'flex-end' as const,
    gap,
    maxWidth: '100%' as const,
    // Room for Pressable focus rings on full-width Thinking / tool rows.
    ...focusSafePadX,
    ...focusSafePadY,
  };
}

export function readTrace(value: unknown): StoredPart[] {
  if (value == null || value === '') {
    return [];
  }
  let raw = value;
  if (typeof value === 'string') {
    try {
      raw = JSON.parse(value) as unknown;
    } catch {
      return [];
    }
  }
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.flatMap((item) => {
    if (item == null || typeof item !== 'object') {
      return [];
    }
    const row = item as { kind?: unknown; text?: unknown; id?: unknown; name?: unknown; arguments?: unknown; result?: unknown };
    if (row.kind === 'thinking' && typeof row.text === 'string' && row.text.trim()) {
      return [{ kind: 'thinking' as const, text: row.text }];
    }
    if (row.kind === 'reply' && typeof row.text === 'string' && row.text.trim()) {
      return [{ kind: 'reply' as const, text: row.text }];
    }
    if (row.kind === 'tool' && typeof row.id === 'string' && row.id !== '') {
      const call: ToolCallRecord = {
        id: row.id,
        name: typeof row.name === 'string' ? row.name : '',
        arguments: typeof row.arguments === 'string' ? row.arguments : '',
      };
      if (typeof row.result === 'string' && row.result !== '') {
        call.result = row.result;
      }
      return [{ kind: 'tool' as const, call }];
    }
    return [];
  });
}

/** Keep one bubble when the optimistic send and the stored message arrive as two ids. */
export function includeCreatedMessage(current: ChatMessage[], message: ChatMessage): ChatMessage[] {
  if (current.some((item) => item.id === message.id)) {
    return current;
  }
  const last = current[current.length - 1];
  if (last != null && last.role === message.role && last.body === message.body) {
    return [...current.slice(0, -1), message];
  }
  return [...current, message];
}

export function chatMessage(item: {
  id: string;
  role: string;
  body: string;
  reasoning?: string | null;
  tool_calls?: unknown;
  trace?: unknown;
}): ChatMessage {
  const reasoning = item.reasoning?.trim() ?? '';
  const toolCalls = readToolCalls(item.tool_calls);
  const trace = readTrace(item.trace);
  return {
    id: item.id,
    role: item.role,
    body: item.body,
    ...(reasoning ? { reasoning } : {}),
    ...(toolCalls.length ? { toolCalls } : {}),
    ...(trace.length ? { trace } : {}),
  };
}

export function storedSteps(message: {
  body: string;
  reasoning?: string;
  toolCalls?: ToolCallRecord[];
  trace?: StoredPart[];
}): ReplyStep[] {
  if (message.trace?.length) {
    return message.trace.map((part) =>
      part.kind === 'tool'
        ? { kind: 'tool', call: part.call, settled: true }
        : part.kind === 'thinking'
          ? { kind: 'thinking', text: part.text, settled: true }
          : { kind: 'reply', text: part.text },
    );
  }
  const steps: ReplyStep[] = [];
  if (message.reasoning?.trim()) {
    steps.push({ kind: 'thinking', text: message.reasoning, settled: true });
  }
  for (const call of message.toolCalls ?? []) {
    steps.push({ kind: 'tool', call, settled: true });
  }
  if (message.body.trim()) {
    steps.push({ kind: 'reply', text: message.body });
  }
  return steps;
}

/** True when the turn is only Thinking (no tools, no visible reply or media). */
export function isThinkingOnlyMessage(message: ChatMessage): boolean {
  if (message.role !== 'assistant') {
    return false;
  }
  const steps = storedSteps(message);
  if (steps.length === 0) {
    return false;
  }
  return steps.every((step) => step.kind === 'thinking' && step.text.trim() !== '');
}

function thinkingTexts(message: ChatMessage): string[] {
  return storedSteps(message).flatMap((step) =>
    step.kind === 'thinking' && step.text.trim() ? [step.text.trim()] : [],
  );
}

/**
 * Display-only: fold a thinking-only assistant turn into the next assistant turn.
 * Stored messages stay separate. A user message breaks the run.
 */
export function coalesceThinkingTurns(messages: ChatMessage[]): ChatMessage[] {
  const shown: ChatMessage[] = [];
  for (const item of messages) {
    const prior = shown[shown.length - 1];
    if (prior != null && isThinkingOnlyMessage(prior) && item.role === 'assistant') {
      const priorText = thinkingTexts(prior).join('\n\n');
      if (isThinkingOnlyMessage(item)) {
        shown[shown.length - 1] = {
          id: item.id,
          role: 'assistant',
          body: '',
          reasoning: [priorText, ...thinkingTexts(item)].filter(Boolean).join('\n\n'),
        };
        continue;
      }
      shown[shown.length - 1] = foldThinkingInto(item, priorText);
      continue;
    }
    shown.push(item);
  }
  return shown;
}

function foldThinkingInto(message: ChatMessage, priorText: string): ChatMessage {
  if (!priorText) {
    return message;
  }
  if (message.trace?.length) {
    const trace = message.trace.map((part) => ({ ...part }));
    const index = trace.findIndex((part) => part.kind === 'thinking');
    if (index >= 0) {
      const part = trace[index];
      if (part.kind === 'thinking') {
        trace[index] = {
          kind: 'thinking',
          text: `${priorText}\n\n${part.text.trim()}`.trim(),
        };
      }
    } else {
      trace.unshift({ kind: 'thinking', text: priorText });
    }
    return { ...message, trace, reasoning: undefined };
  }
  const next = thinkingTexts(message);
  return {
    ...message,
    reasoning: [priorText, ...next].filter(Boolean).join('\n\n'),
  };
}

export function readDraft(sessionId: string): string {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return '';
  }
  return localStorage.getItem(`favlo.draft.${sessionId}`) ?? '';
}

export function writeDraft(sessionId: string, value: string) {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return;
  }
  localStorage.setItem(`favlo.draft.${sessionId}`, value);
}

export function TranscriptTurn({
  name,
  role,
  body,
  reasoning,
  toolCalls,
  trace,
  expandTrace = 'collapsed',
  expandTools,
  imageNames = false,
}: {
  name: string;
  role: string;
  body: string;
  reasoning?: string;
  toolCalls?: ToolCallRecord[];
  trace?: StoredPart[];
  expandTrace?: ExpandTrace;
  expandTools?: ExpandTrace;
  imageNames?: boolean;
}) {
  const toolExpand = expandTools ?? expandTrace;
  const { palette } = useTheme();
  const [empty, setEmpty] = useState(false);
  const [copied, setCopied] = useState(false);
  const reportEmpty = useCallback(() => setEmpty(true), []);
  const mine = role === 'user';
  const steps = storedSteps({ body, reasoning, toolCalls, trace });
  const display = visibleReplySteps(steps, expandTrace, toolExpand);
  const replyOnly = display.length === 1 && display[0]?.kind === 'reply';
  if (empty) {
    return null;
  }
  return (
    <View
      accessibilityLabel={`${name} said: ${humanText(body, imageNames)}`}
      style={{
        alignItems: mine ? 'flex-end' : 'stretch',
        width: '100%',
        maxWidth: '100%',
      }}
    >
      <View
        style={{
          width: mine ? undefined : '100%',
          maxWidth: mine ? '85%' : '100%',
          alignSelf: mine ? undefined : 'stretch',
          gap: space[1],
          paddingVertical: mine ? space[1] : 0,
          paddingHorizontal: mine ? space[2] : 0,
          borderRadius: radius.card,
          backgroundColor: mine ? palette.accentSubtle : 'transparent',
        }}
      >
        <Text style={textStyle('caption', palette.textPrimary, '600')}>{name}</Text>
        {display.map((step, index) => {
          if (step.kind === 'thinking') {
            return (
              <Thinking
                key={`thinking-${index}`}
                text={step.text}
                settled
                expandTrace={expandTrace}
              />
            );
          }
          if (step.kind === 'tool') {
            return (
              <ToolCall
                key={`tool-${step.call.id}-${index}`}
                call={step.call}
                settled
                expandTrace={toolExpand}
              />
            );
          }
          return (
            <MessageBody
              key={`reply-${index}`}
              body={step.text}
              onEmpty={replyOnly ? reportEmpty : undefined}
              imageNames={imageNames}
            />
          );
        })}
        {!mine && humanText(body, imageNames) ? (
          <IconButton
            icon={copied ? 'check' : 'copy'}
            label={copied ? 'Copied' : 'Copy'}
            accessibilityLabel="Copy"
            dense
            onPress={() => {
              if (Platform.OS === 'web') {
                void navigator.clipboard.writeText(humanText(body, imageNames)).then(() => setCopied(true));
              }
            }}
            background="transparent"
            border="transparent"
            color={palette.textSecondary}
          />
        ) : null}
      </View>
    </View>
  );
}

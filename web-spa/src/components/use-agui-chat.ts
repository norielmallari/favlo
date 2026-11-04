import { useCallback, useRef } from 'react';

import { api, idempotencyKey } from '@/api/client';
import { runAgUi, type AguiEvent } from '@/api/agui';
import {
  noteReasoning,
  noteReply,
  noteTool,
  type ChatMessage,
  type ReplyStep,
  type ToolCallRecord,
} from '@/components/chat-transcript';

type Hydrate = {
  messages: ChatMessage[];
  state?: Record<string, unknown>;
};

type Options = {
  threadId: string | null;
  state?: Record<string, unknown>;
  onMessages: (update: (current: ChatMessage[]) => ChatMessage[]) => void;
  onSteps: (update: (current: ReplyStep[]) => ReplyStep[]) => void;
  onWorking: (value: boolean) => void;
  onRunId: (value: string | null) => void;
  onError: (value: string | null) => void;
  onStatus: (value: string | null) => void;
  onState?: (state: Record<string, unknown>) => void;
  onSessionId?: (sessionId: string) => void;
  hydrate?: () => Promise<Hydrate>;
};

export function useAguiChat(options: Options) {
  const abortRef = useRef<AbortController | null>(null);
  const stateRef = useRef(options.state);
  stateRef.current = options.state;
  const toolsRef = useRef<Map<string, ToolCallRecord>>(new Map());
  const threadRef = useRef(options.threadId);
  threadRef.current = options.threadId;

  const stop = useCallback(async () => {
    // Signal the server first so MAF closes its stream/client, then abort SSE.
    // Aborting first used to drop the cancel handle and leave the turn running.
    const threadId = threadRef.current;
    if (threadId) {
      try {
        await api(`/sessions/${threadId}/cancel`, { method: 'POST' });
      } catch {
        /* still abort the browser stream */
      }
    }
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const applyEvent = useCallback(
    (event: AguiEvent) => {
      const type = event.type;
      if (type === 'RUN_STARTED') {
        options.onWorking(true);
        options.onError(null);
        options.onStatus(null);
        options.onSteps(() => []);
        toolsRef.current = new Map();
        if (event.runId) {
          options.onRunId(event.runId);
        }
        return;
      }
      if (type === 'TEXT_MESSAGE_CONTENT' && event.delta) {
        options.onSteps((current) => noteReply(current, event.delta ?? ''));
        return;
      }
      if (
        (type === 'REASONING_MESSAGE_CONTENT' || type === 'THINKING_TEXT_MESSAGE_CONTENT') &&
        event.delta
      ) {
        options.onSteps((current) => noteReasoning(current, event.delta ?? ''));
        return;
      }
      if (type === 'TOOL_CALL_START') {
        const id = event.toolCallId || `tool-${toolsRef.current.size}`;
        const call: ToolCallRecord = {
          id,
          name: event.toolCallName || '',
          arguments: '',
        };
        toolsRef.current.set(id, call);
        options.onSteps((current) => noteTool(current, call));
        return;
      }
      if (type === 'TOOL_CALL_ARGS' && event.toolCallId) {
        const prior = toolsRef.current.get(event.toolCallId) ?? {
          id: event.toolCallId,
          name: '',
          arguments: '',
        };
        const call = {
          ...prior,
          arguments: prior.arguments + (event.delta ?? ''),
        };
        toolsRef.current.set(event.toolCallId, call);
        options.onSteps((current) => noteTool(current, call));
        return;
      }
      if (type === 'TOOL_CALL_RESULT' && event.toolCallId) {
        const prior = toolsRef.current.get(event.toolCallId) ?? {
          id: event.toolCallId,
          name: '',
          arguments: '',
        };
        const call = {
          ...prior,
          result: typeof event.content === 'string' ? event.content : prior.result,
        };
        toolsRef.current.set(event.toolCallId, call);
        options.onSteps((current) => noteTool(current, call));
        return;
      }
      if (type === 'STATE_SNAPSHOT' && event.snapshot && typeof event.snapshot === 'object') {
        options.onState?.(event.snapshot as Record<string, unknown>);
        return;
      }
      if (type === 'RUN_FINISHED') {
        options.onWorking(false);
        options.onRunId(null);
        options.onSteps(() => []);
        const result = event.result as
          | { status?: string; session_id?: string }
          | undefined;
        if (result?.status === 'archived' && result.session_id) {
          options.onMessages(() => []);
          options.onStatus(null);
          options.onSessionId?.(result.session_id);
          return;
        }
        if (result?.status === 'cancelled') {
          options.onStatus('The reply stopped.');
          return;
        }
        options.onStatus('Reply ready');
        return;
      }
      if (type === 'RUN_ERROR') {
        options.onWorking(false);
        options.onRunId(null);
        options.onSteps(() => []);
        options.onError(event.message || 'The reply failed.');
      }
    },
    [options],
  );

  const send = useCallback(
    async (text: string) => {
      if (options.threadId == null || text.trim() === '') {
        return;
      }
      await stop();
      const controller = new AbortController();
      abortRef.current = controller;
      const runId = globalThis.crypto.randomUUID();
      const userId = globalThis.crypto.randomUUID();
      options.onMessages((current) => [
        ...current,
        { id: userId, role: 'user', body: text },
      ]);
      options.onWorking(true);
      options.onError(null);
      options.onStatus(null);
      options.onSteps(() => []);
      options.onRunId(runId);
      try {
        let failed = false;
        for await (const event of runAgUi({
          threadId: options.threadId,
          runId,
          messages: [{ id: userId, role: 'user', content: text }],
          state: stateRef.current,
          signal: controller.signal,
        })) {
          if (event.type === 'RUN_ERROR') {
            failed = true;
          }
          applyEvent(event);
        }
        if (!failed && options.hydrate) {
          const snap = await options.hydrate();
          options.onMessages(() => snap.messages);
          if (snap.state) {
            options.onState?.(snap.state);
          }
        }
      } catch (caught) {
        if ((caught as Error)?.name === 'AbortError') {
          options.onWorking(false);
          options.onRunId(null);
          options.onSteps(() => []);
          options.onStatus('The reply stopped.');
          return;
        }
        options.onWorking(false);
        options.onRunId(null);
        options.onSteps(() => []);
        options.onError(caught instanceof Error ? caught.message : 'The reply failed.');
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
      }
    },
    [applyEvent, options, stop],
  );

  const clear = useCallback(async () => {
    if (options.threadId == null) {
      return;
    }
    await stop();
    const result = await api<{ status: string; session_id: string }>(
      `/sessions/${options.threadId}/clear`,
      { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey() } },
    );
    options.onMessages(() => []);
    options.onSteps(() => []);
    options.onWorking(false);
    options.onRunId(null);
    options.onError(null);
    options.onStatus(null);
    options.onSessionId?.(result.session_id);
    return result.session_id;
  }, [options, stop]);

  const archive = useCallback(async () => {
    if (options.threadId == null) {
      return;
    }
    await stop();
    const runId = globalThis.crypto.randomUUID();
    await api(`/sessions/${options.threadId}/archive`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey() },
      body: JSON.stringify({ run_id: runId }),
    });
    const controller = new AbortController();
    abortRef.current = controller;
    options.onWorking(true);
    options.onError(null);
    options.onStatus(null);
    options.onSteps(() => []);
    options.onRunId(runId);
    try {
      let archived = false;
      for await (const event of runAgUi({
        threadId: options.threadId,
        runId,
        messages: [],
        favloArchive: true,
        state: stateRef.current,
        signal: controller.signal,
      })) {
        if (event.type === 'RUN_FINISHED') {
          const result = event.result as { status?: string } | undefined;
          archived = result?.status === 'archived';
        }
        applyEvent(event);
      }
      if (!archived && options.hydrate) {
        const snap = await options.hydrate();
        options.onMessages(() => snap.messages);
        if (snap.state) {
          options.onState?.(snap.state);
        }
      }
    } catch (caught) {
      if ((caught as Error)?.name === 'AbortError') {
        options.onWorking(false);
        options.onRunId(null);
        options.onSteps(() => []);
        options.onStatus('The reply stopped.');
        return;
      }
      options.onWorking(false);
      options.onRunId(null);
      options.onSteps(() => []);
      options.onError(caught instanceof Error ? caught.message : 'The reply failed.');
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }, [applyEvent, options, stop]);

  return { send, clear, archive, stop, applyEvent };
}

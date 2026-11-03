import { API_URL } from '@/api/client';

export function agUiUrl(): string {
  return `${API_URL}/ag-ui`;
}

export type AguiEvent = {
  type: string;
  runId?: string;
  threadId?: string;
  messageId?: string;
  delta?: string;
  toolCallId?: string;
  toolCallName?: string;
  content?: string;
  role?: string;
  result?: unknown;
  snapshot?: unknown;
  message?: string;
  code?: string;
};

export type AguiRunInput = {
  threadId: string;
  runId: string;
  messages: { role: string; content: string }[];
  state?: Record<string, unknown>;
  favlo_archive?: boolean;
  signal?: AbortSignal;
};

/** Post one AG-UI run and yield decoded SSE events. */
export async function* runAgUi(input: AguiRunInput): AsyncGenerator<AguiEvent> {
  const response = await fetch(agUiUrl(), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      threadId: input.threadId,
      runId: input.runId,
      messages: input.messages,
      state: input.state,
      favlo_archive: input.favlo_archive,
    }),
    signal: input.signal,
  });
  if (!response.ok) {
    let message = 'The request failed.';
    try {
      const body = (await response.json()) as { message?: string };
      if (body.message) {
        message = body.message;
      }
    } catch {
      /* keep default */
    }
    throw new Error(message);
  }
  if (response.body == null) {
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() ?? '';
    for (const chunk of chunks) {
      const event = decodeSseChunk(chunk);
      if (event != null) {
        yield event;
      }
    }
  }
  if (buffer.trim()) {
    const event = decodeSseChunk(buffer);
    if (event != null) {
      yield event;
    }
  }
}

function decodeSseChunk(chunk: string): AguiEvent | null {
  const lines = chunk.split('\n');
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return JSON.parse(dataLines.join('\n')) as AguiEvent;
  } catch {
    return null;
  }
}

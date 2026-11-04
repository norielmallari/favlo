import { useCallback, useEffect, useRef, useState, type ComponentRef } from 'react';
import {
  Platform,
  ScrollView,
  Text,
  useWindowDimensions,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';

import { api, idempotencyKey, socketUrl } from '@/api/client';
import { spokenError } from '@/reply-error';
import {
  chatMessage,
  includeCreatedMessage,
  clearedTranscript,
  keepLatest,
  sessionCommand,
  LiveReply,
  readDraft,
  showsMessage,
  transcriptFrame,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
  type ReplyStep,
} from '@/components/chat-transcript';
import { useAguiChat } from '@/components/use-agui-chat';
import { BannerPortal } from '@/components/Frame';
import { Composer } from '@/components/Composer';
import { ReplyAlert } from '@/components/ReplyAlert';
import { buttonRow, IconButton } from '@/components/IconButton';
import { Modal } from '@/components/Modal';
import { WorkspaceSkills, type WorkspaceSkill } from '@/components/WorkspaceSkills';
import { readExpandTrace, workingLabel } from '@/components/expandTrace';
import { WorkingStatus } from '@/components/WorkingStatus';
import { useTheme } from '@/theme/Theme';
import { space, type as typeScale } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  channel_id: string;
  skills: WorkspaceSkill[];
  expand_thinking?: string | null;
  expand_tools?: string | null;
};

type Snapshot = {
  messages: { id: string; role: string; body: string; reasoning?: string | null; tool_calls?: unknown; trace?: unknown }[];
  runs: { id: string; status: string; error: string | null }[];
  cursor: number;
};

type AgentEvent = {
  event_id: string;
  stream_id?: string;
  run_id?: string;
  sequence?: number;
  type: string;
  payload?: {
    message_id?: string;
    role?: string;
    text?: string;
    reasoning?: string;
    tool_calls?: unknown;
    message?: string;
    status?: string;
    task_id?: string;
    task_ids?: string[];
    agui?: { type?: string; delta?: string; id?: string; name?: string; arguments?: string; result?: string };
    trace?: unknown;
  };
};

const LATEST_SLACK = 64;

export function CodingWorkspace({
  workspaceId,
  agentName,
  onBack,
  onDeleted,
}: {
  workspaceId: string;
  agentName: string;
  onBack?: () => void;
  onDeleted?: () => void;
}) {
  const { palette } = useTheme();
  const { width, height } = useWindowDimensions();
  const narrow = width < 1024;
  const [detail, setDetail] = useState<WorkspaceDetail | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [steps, setSteps] = useState<ReplyStep[]>([]);
  const [working, setWorking] = useState(false);
  const [traceForcedOpen, setTraceForcedOpen] = useState(false);
  useEffect(() => {
    if (!working) {
      setTraceForcedOpen(false);
    }
  }, [working]);
  const [, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const cursorRef = useRef(0);
  const seen = useRef(new Set<string>());
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const openRun = useRef<string | null>(null);
  const [listen, setListen] = useState(false);
  const sessionId = detail?.session_id;
  const chat = useAguiChat({
    threadId: sessionId ?? null,
    onMessages: setMessages,
    onSteps: setSteps,
    onWorking: setWorking,
    onRunId: setRunId,
    onError: setError,
    onStatus: setStatus,
    onSessionId: (nextId) => {
      setDetail((value) => (value == null ? value : { ...value, session_id: nextId }));
    },
    hydrate: async () => {
      if (sessionId == null) {
        return { messages: [] };
      }
      const snap = await api<Snapshot>(`/sessions/${sessionId}`);
      return { messages: snap.messages.map((item) => chatMessage(item)) };
    },
  });

  const loadWorkspace = useCallback(async () => {
    const body = await api<{ workspace: WorkspaceDetail }>(`/workspaces/${workspaceId}`);
    const workspace = { ...body.workspace, skills: body.workspace.skills ?? [] };
    setDetail(workspace);
  }, [workspaceId]);

  useEffect(() => {
    void loadWorkspace().catch((caught: Error) => setNotice(caught.message));
  }, [loadWorkspace]);

  useEffect(() => {
    if (sessionId == null) {
      return;
    }
    let cancelled = false;
    setListen(false);
    setDraft(readDraft(sessionId));
    void api<Snapshot>(`/workspaces/${workspaceId}/session`)
      .then((snapshot) => {
        if (cancelled) {
          return;
        }
        cursorRef.current = snapshot.cursor;
        setListen(true);
        setMessages(snapshot.messages.map(chatMessage));
        const latest = snapshot.runs.at(-1);
        const running = latest?.status === 'running' ? latest.id : null;
        openRun.current = running;
        setWorking(running != null);
        setRunId(running);
        setError(latest?.status === 'failed' ? latest.error : null);
        setSteps([]);
      })
      .catch((caught: Error) => {
        if (!cancelled) {
          setNotice(caught.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, workspaceId]);

  const showLatest = useCallback(() => {
    if (!followLatest.current) {
      return;
    }
    const scroll = () => transcriptRef.current?.scrollToEnd({ animated: false });
    scroll();
    if (Platform.OS === 'web') {
      requestAnimationFrame(() => {
        scroll();
        requestAnimationFrame(scroll);
      });
    }
  }, []);

  useEffect(() => {
    showLatest();
  }, [messages, steps, width, height, showLatest]);

  const noteScroll = useCallback((event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    const fromBottom = contentSize.height - layoutMeasurement.height - contentOffset.y;
    if (fromBottom <= LATEST_SLACK) {
      placedLatest.current = true;
    }
    followLatest.current = keepLatest(followLatest.current, placedLatest.current, fromBottom);
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web' || sessionId == null || !listen) {
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      socket = new WebSocket(socketUrl());
      socket.onopen = () => {
        socket?.send(
          JSON.stringify({
            action: 'subscribe',
            stream_id: `session:${sessionId}`,
            cursor: cursorRef.current,
          }),
        );
      };
      socket.onmessage = (event) => applyEvent(JSON.parse(String(event.data)) as AgentEvent);
      socket.onclose = () => {
        if (!stopped) {
          timer = setTimeout(connect, 1000);
        }
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, [sessionId, listen]);

  function applyEvent(event: AgentEvent) {
    if (seen.current.has(event.event_id)) {
      return;
    }
    seen.current.add(event.event_id);
    if (event.sequence != null && event.sequence > cursorRef.current) {
      cursorRef.current = event.sequence;
    }
    if (event.stream_id !== `session:${sessionId}`) {
      return;
    }
    if (
      event.type === 'message.created' &&
      event.payload?.message_id &&
      showsMessage(event.payload.text, event.payload.reasoning, event.payload.tool_calls)
    ) {
      const id = event.payload.message_id;
      const role = event.payload.role ?? 'assistant';
      const body = event.payload.text ?? '';
      setMessages((currentMessages) =>
        includeCreatedMessage(
          currentMessages,
          chatMessage({
            id,
            role,
            body,
            reasoning: event.payload?.reasoning,
            tool_calls: event.payload?.tool_calls,
            trace: event.payload?.trace,
          }),
        ),
      );
      if (role === 'assistant') {
        const delivery =
          Array.isArray(event.payload?.task_ids) ||
          (typeof event.payload?.task_id === 'string' && event.payload.task_id.length > 0);
        if (!delivery && (event.run_id == null || event.run_id === openRun.current)) {
          openRun.current = null;
          setSteps([]);
          setWorking(false);
          setRunId(null);
          setStatus('Reply ready');
        }
      }
    }
    if (clearedTranscript(event, sessionId)) {
      openRun.current = null;
      setMessages([]);
      setSteps([]);
      setWorking(false);
      setRunId(null);
      setError(null);
      setStatus(null);
    }
    if (event.type === 'run.started') {
      openRun.current = event.run_id ?? null;
      setWorking(true);
      setError(null);
      setSteps([]);
      setStatus(null);
      setRunId(event.run_id ?? null);
    }
    if (event.type === 'run.completed' && sessionId != null) {
      if (event.payload?.status === 'archived') {
        openRun.current = null;
        setMessages([]);
        setSteps([]);
        setWorking(false);
        setRunId(null);
        setError(null);
        setStatus(null);
        return;
      }
      void api<Snapshot>(`/workspaces/${workspaceId}/session`)
        .then((snapshot) => {
          setMessages(snapshot.messages.map(chatMessage));
          setSteps([]);
          setWorking(false);
          setRunId(null);
          setStatus('Reply ready');
        })
        .catch((caught: Error) => setNotice(caught.message));
      return;
    }
    if (event.type === 'run.delta') {
      return;
    }
    if (event.type === 'run.failed' || event.type === 'run.cancelled') {
      if (event.run_id != null && openRun.current != null && event.run_id !== openRun.current) {
        return;
      }
      openRun.current = null;
      setWorking(false);
      setSteps([]);
      setRunId(null);
      setError(event.type === 'run.failed' ? (event.payload?.message ?? 'The reply failed.') : null);
      setStatus(event.type === 'run.cancelled' ? 'The reply stopped.' : null);
    }
  }

  function updateDraft(next: string) {
    setDraft(next);
    if (sessionId != null) {
      writeDraft(sessionId, next);
    }
  }

  async function send() {
    const command = sessionCommand(draft);
    if (draft.trim() === '' || (working && command !== 'clear')) {
      return;
    }
    const text = draft;
    updateDraft('');
    try {
      if (command === 'clear') {
        await chat.clear();
        return;
      }
      if (command === 'archive') {
        await chat.archive();
        return;
      }
      openRun.current = null;
      await chat.send(text);
    } catch (caught) {
      updateDraft(text);
      setNotice(caught instanceof Error ? caught.message : 'The message was not sent.');
    }
  }

  async function resend() {
    const last = [...messages].reverse().find((item) => item.role === 'user');
    if (working || last == null || last.body.trim() === '') {
      return;
    }
    try {
      openRun.current = null;
      await chat.send(last.body);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The message was not sent.');
    }
  }

  async function stop() {
    try {
      await chat.stop();
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The reply did not stop.');
    }
  }

  async function runDeleteWorkspace() {
    try {
      await api(`/workspaces/${workspaceId}`, { method: 'DELETE' });
      setConfirmDelete(false);
      onDeleted?.();
      onBack?.();
    } catch (caught) {
      setConfirmDelete(false);
      setNotice(caught instanceof Error ? caught.message : 'The workspace was not deleted.');
    }
  }

  const skillsButton = (
    <View style={{ marginRight: space[2] }}>
      <IconButton
        icon="book-open"
        label="Skills"
        onPress={() => setSkillsOpen(true)}
        background={palette.surface}
        border={palette.borderStrong}
        color={palette.textPrimary}
      />
    </View>
  );

  const workspaceBar = (
    <>
      {onBack ? (
        <IconButton
          icon="arrow-left"
          label="Back to channel"
          onPress={onBack}
          background={palette.surface}
          border={palette.borderStrong}
          color={palette.textPrimary}
        />
      ) : null}
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        {detail?.name ?? 'Coding'}
      </Text>
      <Text style={textStyle('small', palette.textSecondary)}>Coding</Text>
      {Platform.OS === 'web' ? null : skillsButton}
      <IconButton
        icon="trash-2"
        label="Delete workspace"
        onPress={() => setConfirmDelete(true)}
        background={palette.surface}
        border={palette.borderStrong}
        color={palette.textPrimary}
      />
    </>
  );

  return (
    <View
      accessibilityLabel="Coding workspace"
      style={{
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        overflow: 'hidden',
        ...(narrow && Platform.OS === 'web' ? { height: Math.max(320, height - 56), maxHeight: height - 56 } : null),
      }}
    >
      {Platform.OS === 'web' ? <BannerPortal hostId="banner-actions">{skillsButton}</BannerPortal> : null}
      {narrow ? (
        <View style={[buttonRow, { padding: space[3], borderBottomWidth: 1, borderBottomColor: palette.border }]}>{workspaceBar}</View>
      ) : (
        <BannerPortal>{workspaceBar}</BannerPortal>
      )}
      {notice ? (
        <Text
          accessibilityRole="alert"
          accessibilityLiveRegion="polite"
          style={[textStyle('small', palette.danger), { paddingHorizontal: space[3] }]}
        >
          {notice}
        </Text>
      ) : null}
      {skillsOpen ? (
        <WorkspaceSkills
          workspaceId={workspaceId}
          skills={detail?.skills ?? []}
          onChange={(skills) => setDetail((value) => (value == null ? value : { ...value, skills }))}
          onClose={() => setSkillsOpen(false)}
        />
      ) : null}
      <View
        accessibilityLabel="Conversation"
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          maxWidth: readingWidth,
          alignSelf: 'center',
          width: '100%',
          padding: space[3],
          gap: space[2],
        }}
      >
        <ScrollView
          ref={transcriptRef}
          testID="transcript"
          style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, minWidth: 0 }}
          contentContainerStyle={transcriptFrame(space[4])}
          onContentSizeChange={showLatest}
          onScroll={noteScroll}
          scrollEventThrottle={16}
        >
          {messages.map((item) => (
            <TranscriptTurn
              key={item.id}
              name={item.role === 'user' ? 'You' : agentName}
              role={item.role}
              body={item.body}
              reasoning={item.reasoning}
              toolCalls={item.toolCalls}
              trace={item.trace}
              expandTrace={readExpandTrace(detail?.expand_thinking)}
              expandTools={readExpandTrace(detail?.expand_tools, 'hidden')}
            />
          ))}
          <LiveReply
            steps={steps}
            speaker={agentName}
            measured
            expandTrace={readExpandTrace(detail?.expand_thinking)}
            expandTools={readExpandTrace(detail?.expand_tools, 'hidden')}
            forceOpen={traceForcedOpen}
          />
        </ScrollView>
        {working ? (
          <WorkingStatus
            label={workingLabel(
              steps,
              readExpandTrace(detail?.expand_thinking),
              readExpandTrace(detail?.expand_tools, 'hidden'),
            )}
            expanded={traceForcedOpen}
            onPress={() => setTraceForcedOpen((value) => !value)}
          />
        ) : null}
        {status ? (
          <Text
            accessibilityLiveRegion="polite"
            style={
              Platform.OS === 'web'
                ? { position: 'absolute', width: 1, height: 1, overflow: 'hidden' }
                : textStyle('small', palette.textSecondary)
            }
          >
            {status}
          </Text>
        ) : null}
        {error ? <ReplyAlert message={spokenError(error)} disabled={working} onResend={() => void resend()} /> : null}
        <Composer
          value={draft}
          working={working}
          agentName={agentName}
          onChange={updateDraft}
          onSend={() => void send()}
          onStop={() => void stop()}
        />
      </View>
      {confirmDelete ? (
        <Modal label="Delete workspace" closeLabel="Close delete workspace" onClose={() => setConfirmDelete(false)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {detail?.name ?? 'workspace'}?
          </Text>
          <Text style={textStyle('small', palette.textSecondary)}>Folders and files on disk stay.</Text>
          <IconButton
            icon="trash-2"
            label={`Delete ${detail?.name ?? 'workspace'}`}
            onPress={() => void runDeleteWorkspace()}
            tone="danger"
          />
        </Modal>
      ) : null}
    </View>
  );
}

const readingWidth =
  Platform.OS === 'web'
    ? (`calc(${typeScale.measure}ch + ${space[3] * 2}px)` as const)
    : typeScale.measure * 8 + space[3] * 2;

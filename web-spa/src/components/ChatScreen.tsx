import { useFocusEffect, useRouter } from 'expo-router';
import { useCallback, useEffect, useRef, useState, type ComponentRef } from 'react';
import {
  Platform,
  ScrollView,
  Text,
  TextInput,
  useWindowDimensions,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';

import { api, idempotencyKey, socketUrl } from '@/api/client';
import { FAVLO_AGENT_ID } from '@/api/seed-ids';
import { spokenError } from '@/reply-error';
import {
  chatMessage,
  clearedTranscript,
  keepLatest,
  sessionCommand,
  LATEST_SLACK,
  LiveReply,
  readDraft,
  replyVisible,
  showsMessage,
  transcriptFrame,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
  type ReplyStep,
} from '@/components/chat-transcript';
import { useAguiChat } from '@/components/use-agui-chat';
import { Composer } from '@/components/Composer';
import { ReplyAlert } from '@/components/ReplyAlert';
import { Confirmation } from '@/components/Confirmation';
import { EmptyState } from '@/components/EmptyState';
import { Frame } from '@/components/Frame';
import { buttonRow, IconButton } from '@/components/IconButton';
import { CodingWorkspace } from '@/components/CodingWorkspace';
import { WorkspaceScreen } from '@/workspace/WorkspaceScreen';
import { MenuRow } from '@/components/MenuRow';
import { Modal } from '@/components/Modal';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { AgentForm, type Provider, type ToolGroup } from '@/components/SettingsScreen';
import { Workspaces, type Workspace } from '@/components/Workspaces';
import { readExpandTrace, workingLabel } from '@/components/expandTrace';
import { WorkingStatus } from '@/components/WorkingStatus';
import { useTheme } from '@/theme/Theme';
import { Picker } from '@react-native-picker/picker';
import { radius, space, z } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

export { clearedTranscript, keepLatest, showsMessage };

type Channel = {
  id: string;
  name: string;
  agent_id: string;
  session_id: string;
  expand_trace?: string | null;
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
  type: string;
  payload?: {
    message_id?: string;
    role?: string;
    text?: string;
    reasoning?: string;
    tool_calls?: unknown;
    message?: string;
    status?: string;
    tool_calls?: unknown;
    trace?: unknown;
    session_id?: string;
    previous_session_id?: string;
    agui?: { type?: string; delta?: string; id?: string; name?: string; arguments?: string; result?: string };
  };
};

export function ChatScreen() {
  const router = useRouter();
  const { palette } = useTheme();
  const { width, height } = useWindowDimensions();
  const wide = width >= 1024;
  const drawerWidth = width < 600 ? width : 360;
  const cursorRef = useRef(0);
  const seen = useRef(new Set<string>());
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const openRun = useRef<string | null>(null);
  const expandedSeed = useRef(false);
  const [channels, setChannels] = useState<Channel[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
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
  const [saved, setSaved] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<'channels' | null>(null);
  const [workspaceOpen, setWorkspaceOpen] = useState(() => readOpenWorkspace() != null);
  const [creating, setCreating] = useState(false);
  const [welcomeAgentId, setWelcomeAgentId] = useState<string | null>(null);
  const [createToken, setCreateToken] = useState(0);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<string | null>(() => readOpenWorkspace());
  const [expanded, setExpanded] = useState<string | null>(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [workspaceName, setWorkspaceName] = useState('');
  const [workspaceType, setWorkspaceType] = useState<'gallery' | 'coding'>('gallery');
  const [workspaceTypeFocused, setWorkspaceTypeFocused] = useState(false);
  const [workspaceLocation, setWorkspaceLocation] = useState('');
  const [workspacePickBusy, setWorkspacePickBusy] = useState(false);

  function replaceModels(id: string, models: Provider['models']) {
    setProviders((rows) => rows.map((item) => (item.id === id ? { ...item, models } : item)));
  }

  const channel = channels.find((item) => item.id === selected) ?? null;
  const workspace = workspaces.find((item) => item.id === selectedWorkspace) ?? null;

  const chat = useAguiChat({
    threadId: channel?.session_id ?? null,
    onMessages: setMessages,
    onSteps: setSteps,
    onWorking: setWorking,
    onRunId: setRunId,
    onError: setError,
    onStatus: setStatus,
    onSessionId: (sessionId) => {
      if (channel == null) {
        return;
      }
      setChannels((rows) =>
        rows.map((item) => (item.id === channel.id ? { ...item, session_id: sessionId } : item)),
      );
    },
    hydrate: async () => {
      if (channel == null) {
        return { messages: [] };
      }
      const snap = await api<Snapshot & { state?: Record<string, unknown> }>(
        `/sessions/${channel.session_id}`,
      );
      return {
        messages: snap.messages.map((item) => chatMessage(item)),
        state: snap.state,
      };
    },
  });

  const loadChannels = useCallback(async () => {
    const body = await api<{ channels: Channel[] }>('/channels');
    setChannels(body.channels);
    setSelected((current) => {
      if (current != null && body.channels.some((item) => item.id === current)) {
        return current;
      }
      const favlo = body.channels.find((item) => item.agent_id === FAVLO_AGENT_ID);
      return favlo?.id ?? body.channels[0]?.id ?? null;
    });
  }, []);

  const loadWorkspaces = useCallback(async (channelId: string) => {
    const body = await api<{ workspaces: Workspace[] }>(
      `/workspaces?channel_id=${encodeURIComponent(channelId)}`,
    );
    setWorkspaces(body.workspaces ?? []);
    setSelectedWorkspace((current) => {
      const next =
        current != null && (body.workspaces ?? []).some((item) => item.id === current) ? current : null;
      if (next == null) {
        writeOpenWorkspace(null);
        setWorkspaceOpen(false);
      } else {
        writeOpenWorkspace(next);
        setWorkspaceOpen(true);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    void loadChannels().catch((caught: Error) => setNotice(caught.message));
  }, [loadChannels]);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void loadChannels().catch((caught: Error) => setNotice(caught.message));
      void Promise.all([
        api<{ providers: Provider[] }>('/providers'),
        api<{ groups: ToolGroup[] }>('/tool-groups'),
      ])
        .then(([listedProviders, listedGroups]) => {
          if (!active) {
            return;
          }
          setProviders(listedProviders.providers);
          setGroups(listedGroups.groups);
        })
        .catch(() => {
          if (active) {
            setProviders([]);
            setGroups([]);
          }
        });
      return () => {
        active = false;
      };
    }, [loadChannels]),
  );

  useEffect(() => {
    if (expandedSeed.current || selected == null) {
      return;
    }
    expandedSeed.current = true;
    setExpanded(selected);
  }, [selected]);

  useEffect(() => {
    if (expanded == null) {
      setWorkspaces([]);
      return;
    }
    void loadWorkspaces(expanded).catch((caught: Error) => setNotice(caught.message));
  }, [expanded, loadWorkspaces]);

  useEffect(() => {
    if (channel == null) {
      return;
    }
    const stored = readDraft(channel.session_id);
    setDraft(stored);
    void api<Snapshot>(`/sessions/${channel.session_id}`).then((snapshot) => {
      cursorRef.current = snapshot.cursor;
      setMessages(snapshot.messages.map(chatMessage));
      const latest = snapshot.runs.at(-1);
      const running = latest?.status === 'running' ? latest.id : null;
      openRun.current = running;
      setWorking(running != null);
      setRunId(running);
      setError(latest?.status === 'failed' ? latest.error : null);
      setSteps([]);
    }).catch((caught: Error) => setNotice(caught.message));
  }, [channel]);

  useEffect(() => {
    followLatest.current = true;
    placedLatest.current = false;
  }, [channel?.session_id]);

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
    if (Platform.OS !== 'web') {
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      socket = new WebSocket(socketUrl());
      socket.onopen = () => {
        socket?.send(JSON.stringify({ action: 'subscribe', stream_id: 'app', cursor: 0 }));
        if (channel != null) {
          socket?.send(
            JSON.stringify({
              action: 'subscribe',
              stream_id: `session:${channel.session_id}`,
              cursor: cursorRef.current,
            }),
          );
        }
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
  }, [channel?.session_id]);

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDrawer(null);
        setCreating(false);
        setCreatingWorkspace(false);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  function applyEvent(event: AgentEvent) {
    if (seen.current.has(event.event_id)) {
      return;
    }
    seen.current.add(event.event_id);
    if (event.type === 'agent.created' || event.type === 'agent.deleted') {
      void loadChannels();
      return;
    }
    if (event.type === 'workspace.created' || event.type === 'workspace.deleted' || event.type === 'workspace.root_set') {
      if (channel != null) {
        void loadWorkspaces(channel.id);
      }
      if (event.type === 'workspace.deleted' && event.payload && 'id' in (event.payload as object)) {
        const deleted = (event.payload as { id?: string }).id;
        if (deleted != null && deleted === selectedWorkspace) {
          writeOpenWorkspace(null);
          setSelectedWorkspace(null);
          setWorkspaceOpen(false);
        }
      }
      return;
    }
    if (event.stream_id !== `session:${channel?.session_id}`) {
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
      setMessages((current) =>
        current.some((item) => item.id === id)
          ? current
          : [
              ...current,
              chatMessage({
                id,
                role,
                body,
                reasoning: event.payload?.reasoning,
                tool_calls: event.payload?.tool_calls,
                trace: event.payload?.trace,
              }),
            ],
      );
      if (role === 'assistant') {
        openRun.current = null;
        setSteps([]);
        setWorking(false);
        setRunId(null);
        setStatus('Reply ready');
      }
    }
    if (clearedTranscript(event, channel?.session_id)) {
      openRun.current = null;
      setMessages([]);
      setSteps([]);
      setWorking(false);
      setRunId(null);
      setError(null);
      setStatus(null);
      setCreating(false);
      setWelcomeAgentId(null);
    }
    if (event.type === 'run.started') {
      openRun.current = event.run_id ?? null;
      setWorking(true);
      setError(null);
      setSteps([]);
      setStatus(null);
      setRunId(event.run_id ?? null);
    }
    if (event.type === 'run.delta') {
      return;
    }
    if (event.type === 'run.failed' || event.type === 'run.cancelled') {
      if (event.run_id != null && event.run_id === openRun.current) {
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
    if (channel != null) {
      writeDraft(channel.session_id, next);
    }
  }

  async function send() {
    const command = sessionCommand(draft);
    if (channel == null || draft.trim() === '' || (working && command !== 'clear')) {
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
    if (channel == null || working || last == null || last.body.trim() === '') {
      return;
    }
    try {
      openRun.current = null;
      await chat.send(last.body);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The message was not sent.');
    }
  }

  function resetWorkspaceForm() {
    setWorkspaceName('');
    setWorkspaceType('gallery');
    setWorkspaceLocation('');
    setWorkspacePickBusy(false);
  }

  async function pickWorkspaceFolder() {
    setWorkspacePickBusy(true);
    setNotice(null);
    try {
      const picked = await api<{ cancelled?: boolean; location?: string }>('/folders/pick', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: '{}',
      });
      if (!picked.cancelled && picked.location) {
        setWorkspaceLocation(picked.location);
      }
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The folder was not chosen.');
    } finally {
      setWorkspacePickBusy(false);
    }
  }

  async function createWorkspace() {
    if (channel == null || workspaceName.trim() === '') {
      return;
    }
    if (workspaceType === 'gallery' && workspaceLocation.trim() === '') {
      return;
    }
    try {
      const created = await api<Workspace>('/workspaces', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({
          channel_id: channel.id,
          name: workspaceName.trim(),
          type: workspaceType,
          ...(workspaceType === 'gallery' ? { location: workspaceLocation.trim() } : {}),
        }),
      });
      setCreatingWorkspace(false);
      resetWorkspaceForm();
      setWorkspaces((current) => [...current, created]);
      writeOpenWorkspace(created.id);
      setSelectedWorkspace(created.id);
      setWorkspaceOpen(true);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The workspace was not created.');
    }
  }

  function closeWorkspace() {
    writeOpenWorkspace(null);
    setWorkspaceOpen(false);
    setSelectedWorkspace(null);
  }

  async function stop() {
    try {
      await chat.stop();
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The reply did not stop.');
    }
  }

  const hasReply = messages.some((item) => item.role === 'assistant') || replyVisible(steps);
  const conversation = hasReply || working;
  const greeting = !conversation && channel != null && channel.agent_id === welcomeAgentId;
  const expandTrace = readExpandTrace(channel?.expand_trace);
  const transcript = (
    <>
      {messages.map((item) => (
        <TranscriptTurn
          key={item.id}
          name={item.role === 'user' ? 'You' : channel?.name ?? 'Agent'}
          role={item.role}
          body={item.body}
          reasoning={item.reasoning}
          toolCalls={item.toolCalls}
          trace={item.trace}
          expandTrace={expandTrace}
        />
      ))}
      <LiveReply
        steps={steps}
        speaker={channel?.name ?? 'Agent'}
        measured
        expandTrace={expandTrace}
        forceOpen={traceForcedOpen}
      />
    </>
  );

  return (
    <Frame
      brand={!workspaceOpen}
      title={workspaceOpen && workspace != null ? workspace.name : 'Favlo'}
      onSkip={() => {
        if (Platform.OS === 'web') {
          document.querySelector<HTMLElement>('[aria-label="Message"]')?.focus();
        }
      }}
      link={{ label: 'Configuration', onPress: () => router.push('/settings') }}
    >
      <View style={{ flex: 1, minWidth: 0, minHeight: 0, maxWidth: '100%' }}>
        {!wide && !workspaceOpen ? (
          <View style={[buttonRow, { padding: space[2] }]}>
            <IconButton icon="menu" label="Channels" onPress={() => setDrawer('channels')} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
          </View>
        ) : null}
        <View style={{ flex: 1, flexDirection: 'row', minWidth: 0, minHeight: 0, maxWidth: '100%' }}>
          {!wide && drawer != null ? (
            <Pressable
              accessible={false}
              onPress={() => setDrawer(null)}
              style={{ position: 'absolute', inset: 0, zIndex: z.drawer - 1, backgroundColor: palette.overlay }}
            />
          ) : null}
          {!workspaceOpen && (wide || drawer === 'channels') && (
            <View
              accessibilityLabel="Channels"
              style={{
                width: wide ? 280 : drawerWidth,
                borderRightWidth: 1,
                borderRightColor: palette.border,
                backgroundColor: palette.surface,
                padding: space[3],
                gap: space[2],
                ...(wide ? {} : { position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: z.drawer }),
              }}
            >
              <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>Channels</Text>
              <ScrollView style={{ flex: 1 }} contentContainerStyle={menuRows}>
              {channels.map((item) => (
                <View key={item.id} style={{ gap: space[2] }}>
                  <MenuRow
                    label={item.name}
                    icon={item.id === expanded ? 'chevron-down' : 'chevron-right'}
                    expanded={item.id === expanded}
                    selected={false}
                    selectedTestId=""
                    onPress={() => setExpanded((current) => (current === item.id ? null : item.id))}
                  />
                  {item.id === expanded ? (
                    <Workspaces
                      workspaces={workspaces}
                      selectedId={workspaceOpen ? selectedWorkspace : null}
                      homeSelected={item.id === selected && !workspaceOpen}
                      onHome={() => {
                        setSelected(item.id);
                        writeOpenWorkspace(null);
                        setSelectedWorkspace(null);
                        setWorkspaceOpen(false);
                        setDrawer(null);
                      }}
                      onSelect={(id) => {
                        setSelected(item.id);
                        writeOpenWorkspace(id);
                        setSelectedWorkspace(id);
                        setWorkspaceOpen(true);
                        setDrawer(null);
                      }}
                      onCreate={() => {
                        setWorkspaceName('');
                        setCreatingWorkspace(true);
                        setDrawer(null);
                      }}
                    />
                  ) : null}
                </View>
              ))}
              </ScrollView>
              <IconButton icon="user-plus" label="New agent" onPress={() => { setCreateToken((value) => value + 1); setCreating(true); }} background={palette.accent} color={palette.onAccent} />
              {!wide ? (
                <IconButton icon="x" label="Close" accessibilityLabel="Close channels" onPress={() => setDrawer(null)} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
            </View>
          )}
          {workspaceOpen && workspace != null ? (
            <View style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
              {workspace.type === 'coding' ? (
                <CodingWorkspace
                  workspaceId={workspace.id}
                  agentName={channel?.name ?? 'Agent'}
                  onBack={closeWorkspace}
                  onDeleted={closeWorkspace}
                />
              ) : (
                <WorkspaceScreen
                  workspaceId={workspace.id}
                  agentName={channel?.name ?? 'Agent'}
                  onBack={closeWorkspace}
                  onDeleted={closeWorkspace}
                />
              )}
            </View>
          ) : (
          <View accessibilityLabel="Conversation" style={{ flex: 1, minWidth: 0, minHeight: 0, maxWidth: '100%', padding: space[4], overflow: 'hidden' }}>
            <View testID="reading-column" style={[{ flex: 1, minWidth: 0, minHeight: 0, width: '100%', gap: space[3] }, measureStyle]}>
            {channel == null ? (
              <Text style={textStyle('body', palette.textPrimary)}>Create an agent to start a conversation.</Text>
            ) : greeting ? null : (
              <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>{channel.name}</Text>
            )}
            {!greeting && channel != null ? (
              <ScrollView
                ref={transcriptRef}
                testID="transcript"
                style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, minWidth: 0, maxWidth: '100%' }}
                contentContainerStyle={transcriptFrame(space[4])}
                onContentSizeChange={showLatest}
                onScroll={noteScroll}
                scrollEventThrottle={16}
              >
                {transcript}
              </ScrollView>
            ) : (
              <View style={{ flex: 1, justifyContent: 'flex-end', gap: space[4], maxWidth: '100%' }}>
                {channel != null ? (
                  <EmptyState
                    name={channel.name}
                    compact={width < 600}
                    onFill={(text) => {
                      updateDraft(text);
                      if (Platform.OS === 'web') {
                        document.querySelector<HTMLElement>('[aria-label="Message"]')?.focus();
                      }
                    }}
                  />
                ) : null}
                {transcript}
              </View>
            )}
            {working ? (
              <WorkingStatus
                label={workingLabel(steps)}
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
            {error ? (
              <ReplyAlert message={spokenError(error)} disabled={working} onResend={() => void resend()} />
            ) : null}
            {notice ? (
              <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={textStyle('small', palette.danger)}>{notice}</Text>
            ) : null}
            <Composer
              value={draft}
              working={working}
              agentName={channel?.name}
              onChange={updateDraft}
              onSend={() => void send()}
              onStop={() => void stop()}
            />
            {greeting ? <View style={{ flex: 1 }} /> : null}
            </View>
          </View>
          )}
        </View>
        {creatingWorkspace ? (
          <Modal
            label="New workspace"
            closeLabel="Close new workspace"
            onClose={() => {
              setCreatingWorkspace(false);
              resetWorkspaceForm();
            }}
          >
            <View style={{ gap: space[3] }}>
              <Text style={textStyle('label', palette.textPrimary)}>Name</Text>
              <TextInput
                accessibilityLabel="Workspace name"
                value={workspaceName}
                onChangeText={setWorkspaceName}
                style={{
                  ...textStyle('body', palette.textPrimary),
                  minHeight: 44,
                  borderWidth: 1,
                  borderColor: palette.borderStrong,
                  borderRadius: radius.control,
                  paddingHorizontal: 12,
                  backgroundColor: palette.surface,
                }}
              />
              <Text style={textStyle('label', palette.textPrimary)}>Type</Text>
              <Picker
                accessibilityLabel="Workspace type"
                selectedValue={workspaceType}
                onValueChange={(value) => {
                  const next = String(value) as 'gallery' | 'coding';
                  setWorkspaceType(next);
                  if (next === 'coding') {
                    setWorkspaceLocation('');
                  }
                }}
                onFocus={() => setWorkspaceTypeFocused(true)}
                onBlur={() => setWorkspaceTypeFocused(false)}
                style={{
                  ...textStyle('body', palette.textPrimary),
                  width: '100%',
                  maxWidth: '100%',
                  minWidth: 0,
                  minHeight: 44,
                  borderWidth: 1,
                  borderColor: workspaceTypeFocused ? palette.focusRing : palette.borderStrong,
                  borderRadius: radius.control,
                  paddingHorizontal: 12,
                  backgroundColor: palette.surface,
                }}
              >
                <Picker.Item label="Gallery" value="gallery" color={palette.textPrimary} />
                <Picker.Item label="Coding" value="coding" color={palette.textPrimary} />
              </Picker>
              {workspaceType === 'gallery' ? (
                <View style={{ gap: space[2] }}>
                  <Text style={textStyle('label', palette.textPrimary)}>Local folder</Text>
                  <Text style={textStyle('small', palette.textSecondary)}>
                    Favlo indexes images, mp3, and mp4 in that folder for the gallery filmstrip.
                  </Text>
                  <IconButton
                    icon="folder"
                    label="Choose folder"
                    disabled={workspacePickBusy}
                    onPress={() => void pickWorkspaceFolder()}
                    tone="accent"
                  />
                  <TextInput
                    accessibilityLabel="Folder location"
                    value={workspaceLocation}
                    onChangeText={setWorkspaceLocation}
                    placeholder="Folder location"
                    placeholderTextColor={palette.textSecondary}
                    style={{
                      ...textStyle('body', palette.textPrimary),
                      minHeight: 44,
                      borderWidth: 1,
                      borderColor: palette.borderStrong,
                      borderRadius: radius.control,
                      paddingHorizontal: 12,
                      backgroundColor: palette.surface,
                    }}
                  />
                </View>
              ) : null}
              <IconButton
                icon="plus"
                label="Create workspace"
                disabled={
                  workspaceName.trim() === '' ||
                  (workspaceType === 'gallery' && workspaceLocation.trim() === '')
                }
                onPress={() => void createWorkspace()}
                background={
                  workspaceName.trim() === '' ||
                  (workspaceType === 'gallery' && workspaceLocation.trim() === '')
                    ? palette.disabledSurface
                    : palette.accent
                }
                color={
                  workspaceName.trim() === '' ||
                  (workspaceType === 'gallery' && workspaceLocation.trim() === '')
                    ? palette.disabledText
                    : palette.onAccent
                }
              />
            </View>
          </Modal>
        ) : null}
        {creating ? (
          <Modal label="New agent" closeLabel="Close new agent" onClose={() => setCreating(false)}>
            <AgentForm
              key={createToken}
              agent={null}
              groups={groups}
              providers={providers}
              onModels={replaceModels}
              onSuccess={setSaved}
              onSaved={(agent) => {
                setCreating(false);
                setWelcomeAgentId(agent.id);
                void loadChannels();
              }}
            />
          </Modal>
        ) : null}
        {saved ? <Confirmation message={saved} onDismiss={() => setSaved(null)} /> : null}
      </View>
    </Frame>
  );
}

const menuRows = { gap: focusRingInset * 2, padding: focusRingInset };
const OPEN_WORKSPACE_KEY = 'favlo.openWorkspace';

function readOpenWorkspace(): string | null {
  if (Platform.OS !== 'web' || typeof sessionStorage === 'undefined') {
    return null;
  }
  return sessionStorage.getItem(OPEN_WORKSPACE_KEY);
}

function writeOpenWorkspace(id: string | null) {
  if (Platform.OS !== 'web' || typeof sessionStorage === 'undefined') {
    return;
  }
  if (id == null) {
    sessionStorage.removeItem(OPEN_WORKSPACE_KEY);
    return;
  }
  sessionStorage.setItem(OPEN_WORKSPACE_KEY, id);
}

function barButton(background: string, border: string) {
  return {
    minHeight: 44,
    minWidth: 44,
    borderRadius: radius.control,
    borderWidth: 1,
    borderColor: border,
    backgroundColor: background,
    paddingHorizontal: space[3],
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  };
}

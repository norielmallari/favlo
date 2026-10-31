import { useFocusEffect, useRouter } from 'expo-router';
import { createElement, useCallback, useEffect, useRef, useState, type ComponentRef } from 'react';
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
import {
  chatMessage,
  clearedTranscript,
  keepLatest,
  LATEST_SLACK,
  readDraft,
  showsMessage,
  transcriptFrame,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
} from '@/components/chat-transcript';
import { Composer } from '@/components/Composer';
import { Confirmation } from '@/components/Confirmation';
import { EmptyState } from '@/components/EmptyState';
import { Frame } from '@/components/Frame';
import { buttonRow, IconButton } from '@/components/IconButton';
import { GalleryWorkspace } from '@/components/GalleryWorkspace';
import { MarkdownText } from '@/components/MarkdownText';
import { humanText } from '@/components/MessageBody';
import { MenuRow } from '@/components/MenuRow';
import { Modal } from '@/components/Modal';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { AgentForm, type Provider, type Settings, type ToolGroup } from '@/components/SettingsScreen';
import { Thinking } from '@/components/Thinking';
import { transcriptScrollbar } from '@/components/scrollbar';
import { Workspaces, type Workspace } from '@/components/Workspaces';
import { workingShimmer } from '@/components/working';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

export { clearedTranscript, keepLatest, showsMessage };

type Channel = { id: string; name: string; agent_id: string; session_id: string };
type Snapshot = {
  messages: { id: string; role: string; body: string; reasoning?: string | null }[];
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
    message?: string;
    status?: string;
    agui?: { type?: string; delta?: string };
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
  const expandedSeed = useRef(false);
  const [channels, setChannels] = useState<Channel[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [live, setLive] = useState('');
  const [thinking, setThinking] = useState('');
  const [working, setWorking] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<'channels' | null>(null);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [welcomeAgentId, setWelcomeAgentId] = useState<string | null>(null);
  const [createToken, setCreateToken] = useState(0);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [workspaceName, setWorkspaceName] = useState('');

  const channel = channels.find((item) => item.id === selected) ?? null;
  const workspace = workspaces.find((item) => item.id === selectedWorkspace) ?? null;

  const loadChannels = useCallback(async () => {
    const body = await api<{ channels: Channel[] }>('/channels');
    setChannels(body.channels);
    setSelected((current) => {
      if (current != null && body.channels.some((item) => item.id === current)) {
        return current;
      }
      const favlo = body.channels.find((item) => item.agent_id === 'agent_favlo');
      return favlo?.id ?? body.channels[0]?.id ?? null;
    });
  }, []);

  const loadWorkspaces = useCallback(async (channelId: string) => {
    const body = await api<{ workspaces: Workspace[] }>(
      `/workspaces?channel_id=${encodeURIComponent(channelId)}`,
    );
    setWorkspaces(body.workspaces ?? []);
    setSelectedWorkspace((current) =>
      current != null && (body.workspaces ?? []).some((item) => item.id === current) ? current : null,
    );
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
        api<Settings>('/settings'),
      ])
        .then(([listedProviders, listedGroups, nextSettings]) => {
          if (!active) {
            return;
          }
          setProviders(listedProviders.providers);
          setGroups(listedGroups.groups);
          setSettings(nextSettings);
        })
        .catch(() => {
          if (active) {
            setProviders([]);
            setGroups([]);
            setSettings(null);
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
      setWorking(latest?.status === 'running');
      setRunId(latest?.status === 'running' ? latest.id : null);
      setError(latest?.status === 'failed' ? latest.error : null);
      setLive('');
      setThinking('');
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
    transcriptRef.current?.scrollToEnd({ animated: false });
  }, []);

  useEffect(() => {
    showLatest();
  }, [messages, live, thinking, width, height, showLatest]);

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
          setSelectedWorkspace(null);
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
      showsMessage(event.payload.text, event.payload.reasoning)
    ) {
      const id = event.payload.message_id;
      const role = event.payload.role ?? 'assistant';
      const body = event.payload.text ?? '';
      setMessages((current) =>
        current.some((item) => item.id === id)
          ? current
          : [...current, chatMessage({ id, role, body, reasoning: event.payload?.reasoning })],
      );
      if (role === 'assistant') {
        setLive('');
        setThinking('');
        setWorking(false);
        setRunId(null);
        setStatus('Reply ready');
      }
    }
    if (clearedTranscript(event, channel?.session_id)) {
      setMessages([]);
      setLive('');
      setThinking('');
      setError(null);
      setStatus(null);
      setCreating(false);
      setWelcomeAgentId(null);
    }
    if (event.type === 'run.started') {
      setWorking(true);
      setError(null);
      setThinking('');
      setStatus(null);
      setRunId(event.run_id ?? null);
    }
    if (event.type === 'run.delta' && event.payload?.agui?.type === 'TEXT_MESSAGE_CONTENT') {
      setLive((current) => current + (event.payload?.agui?.delta ?? ''));
    }
    if (event.type === 'run.delta' && event.payload?.agui?.type === 'REASONING_MESSAGE_CONTENT') {
      setThinking((current) => current + (event.payload?.agui?.delta ?? ''));
    }
    if (event.type === 'run.failed' || event.type === 'run.cancelled') {
      setWorking(false);
      setLive('');
      setThinking('');
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
    if (channel == null || draft.trim() === '' || working) {
      return;
    }
    const text = draft;
    updateDraft('');
    try {
      const result = await api<{ message_id: string; text: string; run_id?: string }>(`/sessions/${channel.session_id}/messages`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ text }),
      });
      setMessages((current) =>
        current.some((item) => item.id === result.message_id)
          ? current
          : [...current, { id: result.message_id, role: 'user', body: result.text }],
      );
      setWorking(true);
      setRunId(result.run_id ?? null);
      setError(null);
      setStatus(null);
    } catch (caught) {
      updateDraft(text);
      setNotice(caught instanceof Error ? caught.message : 'The message was not sent.');
    }
  }

  async function createWorkspace() {
    if (channel == null || workspaceName.trim() === '') {
      return;
    }
    try {
      const created = await api<Workspace>('/workspaces', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ channel_id: channel.id, name: workspaceName.trim(), type: 'gallery' }),
      });
      setCreatingWorkspace(false);
      setWorkspaceName('');
      setWorkspaces((current) => [...current, created]);
      setSelectedWorkspace(created.id);
      setWorkspaceOpen(true);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The workspace was not created.');
    }
  }

  async function stop() {
    if (runId == null) {
      return;
    }
    try {
      await api(`/runs/${runId}/cancel`, { method: 'POST' });
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The reply did not stop.');
    }
  }

  const shown = humanText(live);
  const liveThinking = thinking.trim();
  const hasReply = messages.some((item) => item.role === 'assistant') || shown !== '' || liveThinking !== '';
  const conversation = hasReply || working;
  const greeting = !conversation && channel != null && channel.agent_id === welcomeAgentId;
  const transcript = (
    <>
      {messages.map((item) => (
        <TranscriptTurn
          key={item.id}
          name={item.role === 'user' ? 'You' : channel?.name ?? 'Agent'}
          role={item.role}
          body={item.body}
          reasoning={item.reasoning}
        />
      ))}
      {liveThinking !== '' ? <Thinking text={liveThinking} settled={false} /> : null}
      {shown !== '' ? (
        <View accessibilityLabel={`${channel?.name ?? 'Agent'} said: ${shown}`} style={measureStyle}>
          <MarkdownText text={shown} color={palette.textPrimary} />
        </View>
      ) : null}
    </>
  );

  return (
    <Frame
      onSkip={() => {
        if (Platform.OS === 'web') {
          document.querySelector<HTMLElement>('[aria-label="Message"]')?.focus();
        }
      }}
      link={{ label: 'Configuration', onPress: () => router.push('/settings') }}
    >
      <View style={{ flex: 1, minWidth: 0, maxWidth: '100%' }}>
        {!wide && !workspaceOpen ? (
          <View style={[buttonRow, { padding: space[2] }]}>
            <IconButton icon="menu" label="Channels" onPress={() => setDrawer('channels')} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
          </View>
        ) : null}
        <View style={{ flex: 1, flexDirection: 'row', minWidth: 0, maxWidth: '100%' }}>
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
                        setSelectedWorkspace(null);
                        setWorkspaceOpen(false);
                        setDrawer(null);
                      }}
                      onSelect={(id) => {
                        setSelected(item.id);
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
            <View style={{ flex: 1, minWidth: 0 }}>
              <GalleryWorkspace
                workspaceId={workspace.id}
                agentName={channel?.name ?? 'Agent'}
                onBack={() => {
                  setWorkspaceOpen(false);
                  setSelectedWorkspace(null);
                }}
              />
            </View>
          ) : (
          <View accessibilityLabel="Conversation" style={{ flex: 1, minWidth: 0, maxWidth: '100%', padding: space[4], overflow: 'hidden' }}>
            <View testID="reading-column" style={[{ flex: 1, minWidth: 0, width: '100%', gap: space[3] }, measureStyle]}>
            {channel == null ? (
              <Text style={textStyle('body', palette.textPrimary)}>Create an agent to start a conversation.</Text>
            ) : greeting ? null : (
              <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>{channel.name}</Text>
            )}
            {!greeting && channel != null && Platform.OS === 'web'
              ? createElement('style', { 'data-testid': 'transcript-scrollbar' }, transcriptScrollbar(palette.borderStrong, palette.background, palette.textSecondary))
              : null}
            {!greeting && channel != null ? (
              <ScrollView
                ref={transcriptRef}
                testID="transcript"
                style={{ flex: 1, minWidth: 0, maxWidth: '100%' }}
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
              <>
                {Platform.OS === 'web'
                  ? createElement('style', { 'data-testid': 'working-shimmer' }, workingShimmer(palette.textSecondary))
                  : null}
                <Text
                  testID="working"
                  accessibilityLiveRegion="polite"
                  style={textStyle('small', palette.textSecondary)}
                >
                  Working
                </Text>
              </>
            ) : null}
            {status ? (
              <Text
                accessibilityLiveRegion="polite"
                {...(Platform.OS === 'web' ? { role: 'status' as const } : {})}
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
              <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={textStyle('small', palette.danger)}>{error}</Text>
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
          <Modal label="New workspace" closeLabel="Close new workspace" onClose={() => setCreatingWorkspace(false)}>
            <View style={{ gap: space[3] }}>
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
              <IconButton
                icon="plus"
                label="Create workspace"
                disabled={workspaceName.trim() === ''}
                onPress={() => void createWorkspace()}
                background={workspaceName.trim() === '' ? palette.disabledSurface : palette.accent}
                color={workspaceName.trim() === '' ? palette.disabledText : palette.onAccent}
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
              settings={settings}
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

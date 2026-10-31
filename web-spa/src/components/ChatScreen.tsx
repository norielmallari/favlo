import { useFocusEffect, useRouter } from 'expo-router';
import { createElement, useCallback, useEffect, useRef, useState, type ComponentRef } from 'react';
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
import { Composer } from '@/components/Composer';
import { Confirmation } from '@/components/Confirmation';
import { EmptyState } from '@/components/EmptyState';
import { Frame } from '@/components/Frame';
import { MarkdownText } from '@/components/MarkdownText';
import { humanText, MessageBody } from '@/components/MessageBody';
import { MenuRow } from '@/components/MenuRow';
import { Modal } from '@/components/Modal';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { AgentForm, type Provider, type Settings, type ToolGroup } from '@/components/SettingsScreen';
import { Thinking } from '@/components/Thinking';
import { transcriptScrollbar } from '@/components/scrollbar';
import { workingShimmer } from '@/components/working';
import { useTheme } from '@/theme/Theme';
import { radius, space, z } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

type Channel = { id: string; name: string; agent_id: string; session_id: string };
type ChatMessage = { id: string; role: string; body: string; reasoning?: string };
type Snapshot = {
  messages: { id: string; role: string; body: string; reasoning?: string | null }[];
  runs: { id: string; status: string; error: string | null }[];
  cursor: number;
};
const LATEST_SLACK = 64;

export function showsMessage(text?: string, reasoning?: string | null): boolean {
  return Boolean(text) || Boolean(reasoning && reasoning.trim());
}

export function clearedTranscript(
  event: { type: string; stream_id?: string },
  sessionId: string | undefined,
): boolean {
  return sessionId != null && event.type === 'session.cleared' && event.stream_id === `session:${sessionId}`;
}

export function keepLatest(following: boolean, placed: boolean, fromBottom: number): boolean {
  if (!placed) {
    return following;
  }
  return fromBottom <= LATEST_SLACK;
}

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
  const [creating, setCreating] = useState(false);
  const [createToken, setCreateToken] = useState(0);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);

  const channel = channels.find((item) => item.id === selected) ?? null;

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
        {!wide ? (
          <View style={{ flexDirection: 'row', gap: space[2], padding: space[2] }}>
            <Pressable accessibilityRole="button" accessibilityLabel="Channels" onPress={() => setDrawer('channels')} style={barButton(palette.surface, palette.borderStrong)}>
              <Text style={textStyle('small', palette.textPrimary)}>Channels</Text>
            </Pressable>
          </View>
        ) : null}
        <View style={{ flex: 1, flexDirection: 'row', minWidth: 0, maxWidth: '100%' }}>
          {!wide && drawer === 'channels' ? (
            <Pressable
              accessible={false}
              onPress={() => setDrawer(null)}
              style={{ position: 'absolute', inset: 0, zIndex: z.drawer - 1, backgroundColor: palette.overlay }}
            />
          ) : null}
          {(wide || drawer === 'channels') && (
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
              <View style={menuRows}>
              {channels.map((item) => (
                <MenuRow
                  key={item.id}
                  label={item.name}
                  selected={item.id === selected}
                  onPress={() => {
                    setSelected(item.id);
                    setDrawer(null);
                  }}
                />
              ))}
              </View>
              <Pressable accessibilityRole="button" accessibilityLabel="New agent" onPress={() => { setCreateToken((value) => value + 1); setCreating(true); }} style={barButton(palette.accent, palette.accent)}>
                <Text style={textStyle('small', palette.onAccent, '600')}>New agent</Text>
              </Pressable>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Close channels" onPress={() => setDrawer(null)} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={textStyle('small', palette.accent)}>Close</Text>
                </Pressable>
              ) : null}
            </View>
          )}
          <View accessibilityLabel="Conversation" style={{ flex: 1, minWidth: 0, maxWidth: '100%', padding: space[4], overflow: 'hidden' }}>
            <View testID="reading-column" style={[{ flex: 1, minWidth: 0, width: '100%', gap: space[3] }, measureStyle]}>
            {channel == null ? (
              <Text style={textStyle('body', palette.textPrimary)}>Create an agent to start a conversation.</Text>
            ) : hasReply ? (
              <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>{channel.name}</Text>
            ) : null}
            {hasReply && Platform.OS === 'web'
              ? createElement('style', { 'data-testid': 'transcript-scrollbar' }, transcriptScrollbar(palette.borderStrong, palette.background, palette.textSecondary))
              : null}
            {hasReply ? (
              <ScrollView
                ref={transcriptRef}
                testID="transcript"
                style={{ flex: 1, minWidth: 0, maxWidth: '100%' }}
                contentContainerStyle={{ gap: space[4], maxWidth: '100%' }}
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
            {hasReply ? null : <View style={{ flex: 1 }} />}
            </View>
          </View>
        </View>
        {creating ? (
          <Modal label="New agent" closeLabel="Close new agent" onClose={() => setCreating(false)}>
            <AgentForm
              key={createToken}
              agent={null}
              groups={groups}
              providers={providers}
              settings={settings}
              onSuccess={setSaved}
              onSaved={() => {
                setCreating(false);
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

function TranscriptTurn({
  name,
  role,
  body,
  reasoning,
}: {
  name: string;
  role: string;
  body: string;
  reasoning?: string;
}) {
  const { palette } = useTheme();
  const [empty, setEmpty] = useState(false);
  const [copied, setCopied] = useState(false);
  const reportEmpty = useCallback(() => setEmpty(true), []);
  const mine = role === 'user';
  if (empty) {
    return null;
  }
  return (
    <View
      accessibilityLabel={`${name} said: ${humanText(body)}`}
      style={{ alignItems: mine ? 'flex-end' : 'flex-start', maxWidth: '100%' }}
    >
      <View
        style={{
          maxWidth: mine ? '85%' : '100%',
          gap: space[1],
          padding: mine ? space[3] : 0,
          borderRadius: radius.card,
          backgroundColor: mine ? palette.accentSubtle : 'transparent',
        }}
      >
        <Text style={textStyle('label', palette.textPrimary, '600')}>{name}</Text>
        {reasoning ? <Thinking text={reasoning} settled /> : null}
        <MessageBody body={body} onEmpty={reportEmpty} />
        {!mine && humanText(body) ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Copy"
            onPress={() => {
              if (Platform.OS === 'web') {
                void navigator.clipboard.writeText(humanText(body)).then(() => setCopied(true));
              }
            }}
            style={{ minHeight: 44, justifyContent: 'center', alignSelf: 'flex-start' }}
          >
            <Text style={textStyle('caption', palette.textSecondary, '600')}>{copied ? 'Copied' : 'Copy'}</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

function chatMessage(item: { id: string; role: string; body: string; reasoning?: string | null }): ChatMessage {
  const reasoning = item.reasoning?.trim() ?? '';
  return reasoning
    ? { id: item.id, role: item.role, body: item.body, reasoning }
    : { id: item.id, role: item.role, body: item.body };
}

function readDraft(sessionId: string): string {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return '';
  }
  return localStorage.getItem(`favlo.draft.${sessionId}`) ?? '';
}

function writeDraft(sessionId: string, value: string) {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return;
  }
  localStorage.setItem(`favlo.draft.${sessionId}`, value);
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

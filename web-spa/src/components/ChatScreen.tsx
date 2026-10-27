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
import { Composer } from '@/components/Composer';
import { Confirmation } from '@/components/Confirmation';
import { Frame } from '@/components/Frame';
import { MessageBody } from '@/components/MessageBody';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { Workspaces } from '@/components/Workspaces';
import { useTheme } from '@/theme/Theme';

type Channel = { id: string; name: string; agent_id: string; session_id: string };
type ChatMessage = { id: string; role: string; body: string };
type Snapshot = {
  messages: { id: string; role: string; body: string }[];
  runs: { status: string; error: string | null }[];
  cursor: number;
};
const LATEST_SLACK = 64;

export function keepLatest(following: boolean, placed: boolean, fromBottom: number): boolean {
  if (!placed) {
    return following;
  }
  return fromBottom <= LATEST_SLACK;
}

type AgentEvent = {
  event_id: string;
  stream_id?: string;
  type: string;
  payload?: {
    message_id?: string;
    role?: string;
    text?: string;
    message?: string;
    status?: string;
    agui?: { type?: string; delta?: string };
  };
};

export function ChatScreen() {
  const router = useRouter();
  const { palette } = useTheme();
  const { width } = useWindowDimensions();
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
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<'channels' | 'workspaces' | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [persona, setPersona] = useState('');
  const [hasDefault, setHasDefault] = useState(false);
  const [channelWidth, setChannelWidth] = useState(280);

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
      void api<{ default_model_id: string | null }>('/settings')
        .then((body) => {
          if (active) {
            setHasDefault(body.default_model_id != null);
          }
        })
        .catch(() => {
          if (active) {
            setHasDefault(false);
          }
        });
      return () => {
        active = false;
      };
    }, []),
  );

  useEffect(() => {
    if (channel == null) {
      return;
    }
    const stored = readDraft(channel.session_id);
    setDraft(stored);
    void api<Snapshot>(`/sessions/${channel.session_id}`).then((snapshot) => {
      cursorRef.current = snapshot.cursor;
      setMessages(snapshot.messages.map((item) => ({ id: item.id, role: item.role, body: item.body })));
      const latest = snapshot.runs.at(-1);
      setWorking(latest?.status === 'running');
      setError(latest?.status === 'failed' ? latest.error : null);
      setLive('');
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
  }, [messages, live, showLatest]);

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
    if (event.type === 'agent.created') {
      void loadChannels();
      return;
    }
    if (event.stream_id !== `session:${channel?.session_id}`) {
      return;
    }
    if (event.type === 'message.created' && event.payload?.message_id && event.payload.text) {
      const id = event.payload.message_id;
      const role = event.payload.role ?? 'assistant';
      const body = event.payload.text;
      setMessages((current) => (current.some((item) => item.id === id) ? current : [...current, { id, role, body }]));
      if (role === 'assistant') {
        setLive('');
        setWorking(false);
      }
    }
    if (event.type === 'run.started') {
      setWorking(true);
      setError(null);
    }
    if (event.type === 'run.delta' && event.payload?.agui?.type === 'TEXT_MESSAGE_CONTENT') {
      setLive((current) => current + (event.payload?.agui?.delta ?? ''));
    }
    if (event.type === 'run.failed' || event.type === 'run.cancelled') {
      setWorking(false);
      setLive('');
      setError(event.type === 'run.failed' ? (event.payload?.message ?? 'The reply failed.') : null);
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
      const result = await api<{ message_id: string; text: string }>(`/sessions/${channel.session_id}/messages`, {
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
      setError(null);
    } catch (caught) {
      updateDraft(text);
      setNotice(caught instanceof Error ? caught.message : 'The message was not sent.');
    }
  }

  async function createAgent() {
    try {
      await api('/agents', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ name, persona, default_model_id: null }),
      });
      setCreating(false);
      setName('');
      setPersona('');
      setSaved(`${name.trim()} was created.`);
      await loadChannels();
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The agent was not created.');
    }
  }

  const hasReply = messages.some((item) => item.role === 'assistant') || live !== '';
  const transcript = (
    <>
      {messages.map((item) => (
        <View
          key={item.id}
          accessibilityLabel={`${item.role === 'user' ? 'You' : channel?.name ?? 'Agent'} said: ${item.body}`}
        >
          <Text style={{ color: palette.textPrimary, fontSize: 16, lineHeight: 24, fontWeight: '600' }}>
            {item.role === 'user' ? 'You' : channel?.name}
          </Text>
          <MessageBody body={item.body} />
        </View>
      ))}
      {live !== '' ? (
        <Text accessibilityLabel={`${channel?.name ?? 'Agent'} said: ${live}`} style={{ color: palette.textPrimary, fontSize: 16, lineHeight: 24 }}>
          {live}
        </Text>
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
      <View style={{ flex: 1 }}>
        {!wide ? (
          <View style={{ flexDirection: 'row', gap: 8, padding: 8 }}>
            <Pressable accessibilityRole="button" accessibilityLabel="Channels" onPress={() => setDrawer('channels')} style={barButton(palette.surface, palette.borderStrong)}>
              <Text style={{ color: palette.textPrimary, fontSize: 14 }}>Channels</Text>
            </Pressable>
            <Pressable accessibilityRole="button" accessibilityLabel="Workspaces" onPress={() => setDrawer('workspaces')} style={barButton(palette.surface, palette.borderStrong)}>
              <Text style={{ color: palette.textPrimary, fontSize: 14 }}>Workspaces</Text>
            </Pressable>
          </View>
        ) : null}
        <View style={{ flex: 1, flexDirection: 'row' }}>
          {(wide || drawer === 'channels') && (
            <View
              accessibilityLabel="Channels"
              style={{
                width: wide ? channelWidth : drawerWidth,
                borderRightWidth: 1,
                borderRightColor: palette.border,
                backgroundColor: palette.surface,
                padding: 12,
                gap: 8,
                ...(wide ? {} : { position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: 2 }),
              }}
            >
              <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>Channels</Text>
              <View style={menuRows}>
              {channels.map((item) => (
                <Pressable
                  key={item.id}
                  accessibilityRole="button"
                  accessibilityLabel={item.name}
                  accessibilityState={{ selected: item.id === selected }}
                  onPress={() => {
                    setSelected(item.id);
                    setDrawer(null);
                  }}
                  style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: item.id === selected ? palette.accentSubtle : palette.surface }}
                >
                  <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{item.name}</Text>
                </Pressable>
              ))}
              </View>
              <Pressable accessibilityRole="button" accessibilityLabel="New agent" onPress={() => setCreating(true)} style={barButton(palette.accent, palette.accent)}>
                <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New agent</Text>
              </Pressable>
              {wide ? (
                <Pressable
                  accessibilityRole="adjustable"
                  accessibilityLabel="Channels width"
                  accessibilityValue={{ min: 220, max: 480, now: channelWidth }}
                  aria-valuemin={220}
                  aria-valuemax={480}
                  aria-valuenow={channelWidth}
                  accessibilityActions={[{ name: 'increment' }, { name: 'decrement' }]}
                  onAccessibilityAction={(event) => {
                    const next = event.nativeEvent.actionName === 'increment' ? channelWidth + 16 : channelWidth - 16;
                    setChannelWidth(Math.min(480, Math.max(220, next)));
                  }}
                  style={{ minHeight: 44, width: 44, justifyContent: 'center' }}
                >
                  <Text style={{ color: palette.textSecondary, fontSize: 12 }}>Width</Text>
                </Pressable>
              ) : (
                <Pressable accessibilityRole="button" accessibilityLabel="Close channels" onPress={() => setDrawer(null)} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Close</Text>
                </Pressable>
              )}
            </View>
          )}
          <View accessibilityLabel="Conversation" style={{ flex: 1, padding: 16, gap: 12 }}>
            {channel == null ? (
              <Text style={{ color: palette.textPrimary, fontSize: 16, lineHeight: 24 }}>Create an agent to start a conversation.</Text>
            ) : (
              <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>{channel.name}</Text>
            )}
            {hasReply ? (
              <ScrollView
                ref={transcriptRef}
                testID="transcript"
                style={{ flex: 1 }}
                contentContainerStyle={{ gap: 16 }}
                onContentSizeChange={showLatest}
                onScroll={noteScroll}
                scrollEventThrottle={16}
              >
                {transcript}
              </ScrollView>
            ) : (
              <View style={{ flex: 1, justifyContent: 'flex-end', gap: 16 }}>{transcript}</View>
            )}
            {working ? (
              <Text accessibilityLiveRegion="polite" style={{ color: palette.textSecondary, fontSize: 14 }}>Working</Text>
            ) : null}
            {error ? (
              <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text>
            ) : null}
            {notice ? (
              <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={{ color: palette.danger, fontSize: 14 }}>{notice}</Text>
            ) : null}
            <Composer value={draft} working={working} onChange={updateDraft} onSend={() => void send()} />
            {hasReply ? null : <View style={{ flex: 1 }} />}
          </View>
          {(wide || drawer === 'workspaces') && (
            <View
              style={{
                width: wide ? 280 : drawerWidth,
                borderLeftWidth: 1,
                borderLeftColor: palette.border,
                backgroundColor: palette.surface,
                ...(wide ? {} : { position: 'absolute', top: 0, bottom: 0, right: 0, zIndex: 2 }),
              }}
            >
              <Workspaces />
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Close workspaces" onPress={() => setDrawer(null)} style={{ minHeight: 44, paddingHorizontal: 16, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Close</Text>
                </Pressable>
              ) : null}
            </View>
          )}
        </View>
        {creating ? (
          <View accessibilityViewIsModal style={{ position: 'absolute', inset: 0, backgroundColor: palette.overlay, justifyContent: 'center', padding: 16 }}>
            <View style={{ backgroundColor: palette.surface, borderRadius: 8, padding: 16, gap: 12, maxWidth: 480, alignSelf: 'center', width: '100%' }}>
              <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>New agent</Text>
              <TextInput accessibilityLabel="Agent name" value={name} onChangeText={setName} style={field(palette)} />
              <TextInput accessibilityLabel="Persona" multiline value={persona} onChangeText={setPersona} style={[field(palette), { minHeight: 88 }]} />
              <Text style={{ color: palette.textSecondary, fontSize: 12 }}>
                {hasDefault
                  ? 'This agent uses the application default.'
                  : 'Choose a model in configuration. This agent uses the application default.'}
              </Text>
              <Pressable accessibilityRole="button" accessibilityLabel="Create agent" onPress={() => void createAgent()} style={barButton(palette.accent, palette.accent)}>
                <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>Create agent</Text>
              </Pressable>
              <Pressable accessibilityRole="button" accessibilityLabel="Close new agent" onPress={() => setCreating(false)} style={{ minHeight: 44, justifyContent: 'center' }}>
                <Text style={{ color: palette.accent, fontSize: 14 }}>Close</Text>
              </Pressable>
            </View>
          </View>
        ) : null}
        {saved ? <Confirmation message={saved} onDismiss={() => setSaved(null)} /> : null}
      </View>
    </Frame>
  );
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

function heading(color: string) {
  return { color, fontSize: 18, lineHeight: 24, fontWeight: '600' as const };
}

function field(palette: { textPrimary: string; surface: string; borderStrong: string }) {
  return {
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: 6,
    paddingHorizontal: 12,
    color: palette.textPrimary,
    backgroundColor: palette.surface,
    fontSize: 16,
  };
}

function barButton(background: string, border: string) {
  return {
    minHeight: 44,
    minWidth: 44,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: border,
    backgroundColor: background,
    paddingHorizontal: 12,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
  };
}

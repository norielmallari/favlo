import { Picker } from '@react-native-picker/picker';
import { createElement, useCallback, useEffect, useRef, useState, type ComponentRef } from 'react';
import {
  Image,
  Platform,
  ScrollView,
  Text,
  TextInput,
  useWindowDimensions,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';

import { API_URL, api, idempotencyKey, socketUrl } from '@/api/client';
import {
  chatMessage,
  clearedTranscript,
  keepLatest,
  readDraft,
  showsMessage,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
} from '@/components/chat-transcript';
import { Composer } from '@/components/Composer';
import { MarkdownText } from '@/components/MarkdownText';
import { humanText } from '@/components/MessageBody';
import { Pressable } from '@/components/Pressable';
import { transcriptScrollbar } from '@/components/scrollbar';
import { Thinking } from '@/components/Thinking';
import { WorkspaceSkills, type WorkspaceSkill } from '@/components/WorkspaceSkills';
import { workingShimmer } from '@/components/working';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

export type FolderKind = {
  kind: string;
  provider?: string;
  enabled: boolean;
};

type Root = {
  id: string;
  name: string;
  kind: string;
  provider: string | null;
  location: string | null;
  revision: number;
};

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  channel_id: string;
  root: Root | null;
  skills: WorkspaceSkill[];
};

type Listing = {
  path: string;
  folders: { path: string; name: string; kind: string }[];
  images: {
    path: string;
    name: string;
    media_type: string;
    size: number;
    revision: string;
    note: Record<string, unknown> | null;
  }[];
};

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

const LATEST_SLACK = 64;

export function GalleryWorkspace({
  workspaceId,
  agentName,
  onBack,
}: {
  workspaceId: string;
  agentName: string;
  onBack: () => void;
}) {
  const { palette } = useTheme();
  const { width, height } = useWindowDimensions();
  const treeOpen = width >= 1440;
  const [detail, setDetail] = useState<WorkspaceDetail | null>(null);
  const [kinds, setKinds] = useState<FolderKind[]>([]);
  const [chosenKind, setChosenKind] = useState<FolderKind | null>(null);
  const [location, setLocation] = useState('');
  const [folderName, setFolderName] = useState('');
  const [path, setPath] = useState('');
  const [listing, setListing] = useState<Listing | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [treeDrawer, setTreeDrawer] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [live, setLive] = useState('');
  const [thinking, setThinking] = useState('');
  const [working, setWorking] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const cursorRef = useRef(0);
  const pathRef = useRef('');
  const seen = useRef(new Set<string>());
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const sessionId = detail?.session_id;
  const root = detail?.root ?? null;

  const loadWorkspace = useCallback(async () => {
    const body = await api<{
      workspace: WorkspaceDetail;
      folder_kinds: FolderKind[];
    }>(`/workspaces/${workspaceId}`);
    setDetail({ ...body.workspace, skills: body.workspace.skills ?? [] });
    setKinds(body.folder_kinds);
    setFolderName(body.workspace.root?.name || body.workspace.name);
    if (body.workspace.root?.location) {
      setLocation(body.workspace.root.location);
    }
    const local = body.folder_kinds.find((item) => item.kind === 'local' && item.enabled);
    setChosenKind((current) => current ?? local ?? null);
  }, [workspaceId]);

  const loadListing = useCallback(
    async (nextPath: string) => {
      const body = await api<Listing>(
        `/workspaces/${workspaceId}/images?path=${encodeURIComponent(nextPath)}`,
      );
      setListing(body);
      pathRef.current = nextPath;
      setPath(nextPath);
      setSelected((current) =>
        body.images.some((item) => item.path === current) ? current : body.images[0]?.path ?? null,
      );
    },
    [workspaceId],
  );

  useEffect(() => {
    void loadWorkspace().catch((caught: Error) => setNotice(caught.message));
  }, [loadWorkspace]);

  useEffect(() => {
    if (root == null) {
      setListing(null);
      return;
    }
    void loadListing('').catch((caught: Error) => setNotice(caught.message));
  }, [root?.id, loadListing, root]);

  useEffect(() => {
    if (sessionId == null) {
      return;
    }
    setDraft(readDraft(sessionId));
    void api<Snapshot>(`/workspaces/${workspaceId}/session`)
      .then((snapshot) => {
        cursorRef.current = snapshot.cursor;
        setMessages(snapshot.messages.map(chatMessage));
        const latest = snapshot.runs.at(-1);
        setWorking(latest?.status === 'running');
        setRunId(latest?.status === 'running' ? latest.id : null);
        setError(latest?.status === 'failed' ? latest.error : null);
        setLive('');
        setThinking('');
      })
      .catch((caught: Error) => setNotice(caught.message));
  }, [sessionId, workspaceId]);

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
    if (Platform.OS !== 'web' || sessionId == null) {
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
  }, [sessionId]);

  function applyEvent(event: AgentEvent) {
    if (seen.current.has(event.event_id)) {
      return;
    }
    seen.current.add(event.event_id);
    if (event.type === 'gallery.updated') {
      void loadListing(pathRef.current).catch((caught: Error) => setNotice(caught.message));
      return;
    }
    if (event.stream_id !== `session:${sessionId}`) {
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
        void loadListing(pathRef.current).catch((caught: Error) => setNotice(caught.message));
      }
    }
    if (clearedTranscript(event, sessionId)) {
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
    if (sessionId != null) {
      writeDraft(sessionId, next);
    }
  }

  async function pickRoot() {
    try {
      const body = await api<{ cancelled: boolean; location?: string | null }>(
        `/workspaces/${workspaceId}/root/pick`,
        { method: 'POST' },
      );
      if (body.cancelled || body.location == null || body.location === '') {
        return;
      }
      setLocation(body.location);
      setNotice(null);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The folder was not chosen.');
    }
  }

  async function saveRoot() {
    if (chosenKind == null || !chosenKind.enabled) {
      return;
    }
    try {
      const rootRow = await api<Root>(`/workspaces/${workspaceId}/root`, {
        method: 'PUT',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({
          name: folderName.trim() || detail?.name,
          kind: chosenKind.kind,
          provider: chosenKind.provider,
          location: location.trim(),
        }),
      });
      setDetail((current) => (current == null ? current : { ...current, root: rootRow }));
      setNotice(null);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The gallery root was not saved.');
    }
  }

  async function send() {
    if (draft.trim() === '' || working || root == null) {
      return;
    }
    const text = draft;
    updateDraft('');
    try {
      const result = await api<{ message_id: string; text: string; run_id?: string }>(
        `/workspaces/${workspaceId}/session/messages`,
        {
          method: 'POST',
          headers: { 'Idempotency-Key': idempotencyKey() },
          body: JSON.stringify({ text }),
        },
      );
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
  const selectedImage = listing?.images.find((item) => item.path === selected) ?? null;
  const fileUrl = selectedImage
    ? `${API_URL}/workspaces/${workspaceId}/file?path=${encodeURIComponent(selectedImage.path)}`
    : null;

  const tree = (
    <View accessibilityLabel="Folders" style={{ flex: 1, gap: space[2], padding: space[3] }}>
      <Text accessibilityRole="header" aria-level={3} style={textStyle('title', palette.textPrimary)}>
        Folders
      </Text>
      {groupedKinds(kinds).map((group) => {
        const active = group.kind === 'local' && root != null;
        return (
          <View key={group.kind} style={{ gap: space[1] }}>
            <Text
              style={textStyle('small', group.enabled ? palette.textPrimary : palette.disabledText)}
            >
              {kindLabel(group.kind)}
              {group.enabled ? ', enabled' : ', disabled'}
            </Text>
            {active && root?.location ? (
              <Text style={textStyle('caption', palette.textSecondary)}>{root.location}</Text>
            ) : null}
            {group.enabled
              ? group.providers.map((item) => (
                  <Text
                    key={item.provider}
                    style={[
                      textStyle('small', item.enabled ? palette.textPrimary : palette.disabledText),
                      { paddingLeft: space[3] },
                    ]}
                  >
                    {providerLabel(item.provider)}
                  </Text>
                ))
              : null}
          </View>
        );
      })}
      {listing != null ? (
        <View style={{ gap: space[1] }}>
          {path ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Parent folder"
              onPress={() => {
                const parts = path.split('/');
                void loadListing(parts.slice(0, -1).join('/'));
              }}
              style={barButton(palette.surface, palette.borderStrong)}
            >
              <Text style={textStyle('small', palette.textPrimary)}>..</Text>
            </Pressable>
          ) : null}
          {listing.folders.map((item) => (
            <Pressable
              key={item.path}
              accessibilityRole="button"
              accessibilityLabel={item.name}
              onPress={() => void loadListing(item.path)}
              style={barButton(palette.surface, palette.borderStrong)}
            >
              <Text style={textStyle('small', palette.textPrimary)}>{item.name}</Text>
            </Pressable>
          ))}
        </View>
      ) : null}
    </View>
  );

  const grid = (
    <View accessibilityLabel="Gallery grid" style={{ flex: 1, minWidth: 0, padding: space[3], gap: space[3] }}>
      <Text accessibilityRole="header" aria-level={3} style={textStyle('title', palette.textPrimary)}>
        Gallery
      </Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space[2] }}>
        {(listing?.images ?? []).map((item) => (
          <Pressable
            key={item.path}
            accessibilityRole="button"
            accessibilityLabel={item.name}
            accessibilityState={{ selected: item.path === selected }}
            onPress={() => setSelected(item.path)}
            style={{
              width: 96,
              height: 96,
              borderRadius: radius.control,
              borderWidth: 2,
              borderColor: item.path === selected ? palette.accent : palette.border,
              overflow: 'hidden',
            }}
          >
            <Image
              accessibilityIgnoresInvertColors
              source={{
                uri: `${API_URL}/workspaces/${workspaceId}/file?path=${encodeURIComponent(item.path)}`,
              }}
              style={{ width: 96, height: 96 }}
            />
          </Pressable>
        ))}
      </View>
      {selectedImage && fileUrl ? (
        <View accessibilityLabel="Preview" style={{ gap: space[2], flex: 1, minHeight: 120 }}>
          <Image
            accessibilityLabel={selectedImage.name}
            source={{ uri: fileUrl }}
            style={{ width: '100%', flex: 1, minHeight: 160, borderRadius: radius.card }}
            resizeMode="contain"
          />
          <View style={{ flexDirection: 'row', gap: space[2] }}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Download"
              onPress={() => {
                if (Platform.OS === 'web') {
                  const link = document.createElement('a');
                  link.href = fileUrl;
                  link.download = selectedImage.name;
                  link.click();
                }
              }}
              style={barButton(palette.surface, palette.borderStrong)}
            >
              <Text style={textStyle('small', palette.textPrimary)}>Download</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Add to chat"
              onPress={() => {
                const next = draft.trim() === '' ? selectedImage.path : `${draft.trim()}\n${selectedImage.path}`;
                updateDraft(next);
              }}
              style={barButton(palette.accent, palette.accent)}
            >
              <Text style={textStyle('small', palette.onAccent, '600')}>Add to chat</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <Text style={textStyle('small', palette.textSecondary)}>No images in this folder</Text>
      )}
    </View>
  );

  const transcript = (
    <View
      accessibilityLabel="Conversation"
      style={{ flex: 1, minWidth: 0, padding: space[3], gap: space[2] }}
    >
      {Platform.OS === 'web'
        ? createElement('style', { 'data-testid': 'transcript-scrollbar' }, transcriptScrollbar(palette.borderStrong, palette.background, palette.textSecondary))
        : null}
      <ScrollView
        ref={transcriptRef}
        testID="transcript"
        style={{ flex: 1, minWidth: 0 }}
        contentContainerStyle={{ gap: space[4] }}
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
          />
        ))}
        {liveThinking !== '' ? <Thinking text={liveThinking} settled={false} /> : null}
        {shown !== '' ? (
          <View accessibilityLabel={`${agentName} said: ${shown}`} style={measureStyle}>
            <MarkdownText text={shown} color={palette.textPrimary} />
          </View>
        ) : null}
      </ScrollView>
      {working ? (
        <>
          {Platform.OS === 'web'
            ? createElement('style', { 'data-testid': 'working-shimmer' }, workingShimmer(palette.textSecondary))
            : null}
          <Text testID="working" accessibilityLiveRegion="polite" style={textStyle('small', palette.textSecondary)}>
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
        <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={textStyle('small', palette.danger)}>
          {error}
        </Text>
      ) : null}
      <Composer
        value={draft}
        working={working}
        agentName={agentName}
        onChange={updateDraft}
        onSend={() => void send()}
        onStop={() => void stop()}
      />
    </View>
  );

  return (
    <View accessibilityLabel="Gallery workspace" style={{ flex: 1, minWidth: 0 }}>
      <View
        style={{
          flexDirection: 'row',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: space[2],
          padding: space[3],
          borderBottomWidth: 1,
          borderBottomColor: palette.border,
        }}
      >
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Back to channel"
          onPress={onBack}
          style={barButton(palette.surface, palette.borderStrong)}
        >
          <Text style={textStyle('small', palette.textPrimary)}>Back to channel</Text>
        </Pressable>
        <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
          {detail?.name ?? 'Gallery'}
        </Text>
        <Text style={textStyle('small', palette.textSecondary)}>Gallery</Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Skills"
          onPress={() => setSkillsOpen(true)}
          style={barButton(palette.surface, palette.borderStrong)}
        >
          <Text style={textStyle('small', palette.textPrimary)}>Skills</Text>
        </Pressable>
        {!treeOpen && root != null ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Folders"
            onPress={() => setTreeDrawer(true)}
            style={barButton(palette.surface, palette.borderStrong)}
          >
            <Text style={textStyle('small', palette.textPrimary)}>Folders</Text>
          </Pressable>
        ) : null}
      </View>
      {notice ? (
        <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={[textStyle('small', palette.danger), { paddingHorizontal: space[3] }]}>
          {notice}
        </Text>
      ) : null}
      {skillsOpen ? (
        <WorkspaceSkills
          workspaceId={workspaceId}
          skills={detail?.skills ?? []}
          onChange={(skills) =>
            setDetail((current) => (current == null ? current : { ...current, skills }))
          }
          onClose={() => setSkillsOpen(false)}
        />
      ) : null}
      {root == null ? (
        <RootForm
          kinds={kinds}
          chosen={chosenKind}
          onChoose={setChosenKind}
          name={folderName}
          onName={setFolderName}
          location={location}
          onLocation={setLocation}
          onPick={() => void pickRoot()}
          onSave={() => void saveRoot()}
        />
      ) : (
        <View style={{ flex: 1, flexDirection: width < 720 ? 'column' : 'row', minWidth: 0 }}>
          {!treeOpen && treeDrawer ? (
            <Pressable
              accessible={false}
              onPress={() => setTreeDrawer(false)}
              style={{ position: 'absolute', inset: 0, zIndex: 20, backgroundColor: palette.overlay }}
            />
          ) : null}
          {(treeOpen || treeDrawer) && (
            <View
              style={{
                width: 220,
                borderRightWidth: 1,
                borderRightColor: palette.border,
                backgroundColor: palette.surface,
                ...(treeOpen ? {} : { position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: 21 }),
              }}
            >
              {tree}
            </View>
          )}
          {grid}
          <View
            style={{
              width: width < 720 ? undefined : 280,
              flex: 1,
              minWidth: 0,
              minHeight: width < 720 ? 240 : undefined,
              borderLeftWidth: width < 720 ? 0 : 1,
              borderTopWidth: width < 720 ? 1 : 0,
              borderColor: palette.border,
            }}
          >
            {transcript}
          </View>
        </View>
      )}
    </View>
  );
}

function RootForm({
  kinds,
  chosen,
  onChoose,
  name,
  onName,
  location,
  onLocation,
  onPick,
  onSave,
}: {
  kinds: FolderKind[];
  chosen: FolderKind | null;
  onChoose: (item: FolderKind) => void;
  name: string;
  onName: (value: string) => void;
  location: string;
  onLocation: (value: string) => void;
  onPick: () => void;
  onSave: () => void;
}) {
  const { palette } = useTheme();
  const [kindFocused, setKindFocused] = useState(false);
  const [providerFocused, setProviderFocused] = useState(false);
  const groups = groupedKinds(kinds);
  const cloud = groups.find((group) => group.kind === 'cloud');
  const local = chosen?.kind === 'local' && chosen.enabled;
  const showProviders = chosen?.kind === 'cloud' && (cloud?.enabled ?? false);
  return (
    <View accessibilityLabel="Gallery root" style={{ flex: 1, padding: space[4], gap: space[3], maxWidth: 520 }}>
      <Text style={textStyle('body', palette.textPrimary)}>Set the gallery root before the grid opens.</Text>
      <TextInput
        accessibilityLabel="Folder name"
        value={name}
        onChangeText={onName}
        style={fieldStyle(palette)}
      />
      <Text style={textStyle('small', palette.textPrimary)}>Folder type</Text>
      <Picker
        accessibilityLabel="Folder type"
        selectedValue={chosen?.kind ?? 'local'}
        onValueChange={(value) => {
          const next = groups.find((group) => group.kind === String(value));
          if (next == null || !next.enabled) {
            return;
          }
          onChoose(pickKind(kinds, next.kind));
        }}
        onFocus={() => setKindFocused(true)}
        onBlur={() => setKindFocused(false)}
        style={pickerStyle(palette, kindFocused)}
      >
        {groups.map((group) => (
          <Picker.Item
            key={group.kind}
            label={kindLabel(group.kind)}
            value={group.kind}
            enabled={group.enabled}
            color={group.enabled ? palette.textPrimary : palette.disabledText}
          />
        ))}
      </Picker>
      {showProviders ? (
        <>
          <Text style={textStyle('small', palette.textPrimary)}>Cloud provider</Text>
          <Picker
            accessibilityLabel="Cloud provider"
            selectedValue={chosen?.provider ?? ''}
            onValueChange={(value) => {
              const item = cloud?.providers.find((row) => row.provider === String(value));
              if (item == null || !item.enabled) {
                return;
              }
              onChoose(item);
            }}
            onFocus={() => setProviderFocused(true)}
            onBlur={() => setProviderFocused(false)}
            style={pickerStyle(palette, providerFocused)}
          >
            {cloud?.providers.map((item) => (
              <Picker.Item
                key={item.provider}
                label={providerLabel(item.provider)}
                value={item.provider ?? ''}
                enabled={item.enabled}
                color={item.enabled ? palette.textPrimary : palette.disabledText}
              />
            ))}
          </Picker>
        </>
      ) : null}
      {local ? (
        <>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Choose folder"
            onPress={onPick}
            style={barButton(palette.surface, palette.borderStrong)}
          >
            <Text style={textStyle('small', palette.textPrimary)}>Choose folder</Text>
          </Pressable>
          <TextInput
            accessibilityLabel="Location"
            value={location}
            onChangeText={onLocation}
            autoCapitalize="none"
            style={fieldStyle(palette)}
          />
        </>
      ) : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Save root"
        accessibilityState={{ disabled: !local || location.trim() === '' }}
        disabled={!local || location.trim() === ''}
        onPress={onSave}
        style={barButton(local && location.trim() !== '' ? palette.accent : palette.disabledSurface, palette.accent)}
      >
        <Text style={textStyle('small', local && location.trim() !== '' ? palette.onAccent : palette.disabledText, '600')}>
          Save root
        </Text>
      </Pressable>
    </View>
  );
}

export function groupedKinds(kinds: FolderKind[]): { kind: string; enabled: boolean; providers: FolderKind[] }[] {
  const groups: { kind: string; enabled: boolean; providers: FolderKind[] }[] = [];
  for (const item of kinds) {
    if (item.kind === 'local') {
      if (groups.some((group) => group.kind === 'local')) {
        continue;
      }
      groups.push({ kind: 'local', enabled: item.enabled, providers: [] });
      continue;
    }
    if (item.kind !== 'cloud') {
      continue;
    }
    const cloud = groups.find((group) => group.kind === 'cloud');
    if (cloud) {
      cloud.providers.push(item);
      cloud.enabled = cloud.enabled || item.enabled;
      continue;
    }
    groups.push({ kind: 'cloud', enabled: item.enabled, providers: [item] });
  }
  return groups;
}

export function kindLabel(kind: string): string {
  if (kind === 'local') {
    return 'Local';
  }
  if (kind === 'cloud') {
    return 'Cloud';
  }
  return kind;
}

export function providerLabel(provider: string | undefined): string {
  if (provider === 'onedrive') {
    return 'OneDrive';
  }
  if (provider === 'dropbox') {
    return 'Dropbox';
  }
  return provider ?? '';
}

function pickKind(kinds: FolderKind[], kind: string): FolderKind {
  if (kind === 'local') {
    return kinds.find((item) => item.kind === 'local') ?? { kind: 'local', enabled: true };
  }
  const cloud = kinds.filter((item) => item.kind === 'cloud');
  return cloud.find((item) => item.enabled) ?? cloud[0] ?? { kind: 'cloud', enabled: false };
}

function pickerStyle(
  palette: { textPrimary: string; surface: string; borderStrong: string; focusRing: string },
  focused: boolean,
) {
  return {
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: radius.control,
    paddingHorizontal: 12,
    color: palette.textPrimary,
    backgroundColor: palette.surface,
    fontSize: 16,
    outlineStyle: 'solid' as const,
    outlineWidth: 2,
    outlineOffset: 2,
    outlineColor: focused ? palette.focusRing : 'transparent',
  };
}

function fieldStyle(palette: { textPrimary: string; surface: string; borderStrong: string }) {
  return {
    ...textStyle('body', palette.textPrimary),
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: radius.control,
    paddingHorizontal: 12,
    backgroundColor: palette.surface,
  };
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

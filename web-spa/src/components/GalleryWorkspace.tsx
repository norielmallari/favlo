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
  transcriptFrame,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
} from '@/components/chat-transcript';
import { BannerPortal } from '@/components/Frame';
import { Composer } from '@/components/Composer';
import { FeatherIcon } from '@/components/FeatherIcon';
import { buttonRow, IconButton } from '@/components/IconButton';
import { MarkdownText } from '@/components/MarkdownText';
import { humanText } from '@/components/MessageBody';
import { Modal } from '@/components/Modal';
import { Pressable } from '@/components/Pressable';
import { transcriptScrollbar } from '@/components/scrollbar';
import { Thinking } from '@/components/Thinking';
import { WorkspaceSkills, type WorkspaceSkill } from '@/components/WorkspaceSkills';
import { workingShimmer } from '@/components/working';
import { useTheme } from '@/theme/Theme';
import { radius, space, type as typeScale } from '@/theme/tokens';
import { measureStyle, textStyle } from '@/theme/type';

export type FolderKind = {
  kind: string;
  provider?: string;
  enabled: boolean;
};

type FolderRow = {
  id: string;
  name: string;
  kind: string;
  provider: string | null;
  location: string | null;
  parent_id: string | null;
  remote_id: string | null;
  revision: number;
};

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  channel_id: string;
  skills: WorkspaceSkill[];
};

type ImageFile = {
  path: string;
  name: string;
  media_type: string;
  size: number;
  revision: string;
  note: Record<string, unknown> | null;
};

type Listing = {
  path: string;
  folders: { path: string; name: string; kind: string }[];
  images: ImageFile[];
  empty?: boolean;
};

type Crumb = {
  key: string;
  label: string;
  scope: 'rows' | 'disk';
  folderId: string | null;
  path: string;
  kind: 'root' | 'cloud' | 'local' | 'directory';
};

type Source = {
  type: 'image' | 'directory' | 'row';
  id: string;
  name: string;
  path: string;
  kind: string;
  revision: number;
  folderId: string;
};

type Dest = {
  key: string;
  label: string;
  kind: 'cloud' | 'local' | 'root' | 'disk';
  folderId: string | null;
  path: string;
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

export function crossedKind(source: string, dest: string): string | null {
  const localSource = source === 'local' || source === 'image' || source === 'directory';
  if (localSource && dest === 'cloud') {
    return 'That would upload.';
  }
  if (source === 'cloud' && (dest === 'local' || dest === 'disk')) {
    return 'That would download.';
  }
  if (localSource && dest === 'root') {
    return 'That would upload.';
  }
  return null;
}

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
  const narrow = width < 1024;
  const [detail, setDetail] = useState<WorkspaceDetail | null>(null);
  const [kinds, setKinds] = useState<FolderKind[]>([]);
  const [crumbs, setCrumbs] = useState<Crumb[]>([]);
  const [rows, setRows] = useState<FolderRow[]>([]);
  const [listing, setListing] = useState<Listing | null>(null);
  const [rootFolderId, setRootFolderId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Source | null>(null);
  const [moveFor, setMoveFor] = useState<Source | null>(null);
  const [sheet, setSheet] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [folderName, setFolderName] = useState('');
  const [folderKind, setFolderKind] = useState('local');
  const [provider, setProvider] = useState('onedrive');
  const [remoteId, setRemoteId] = useState('');
  const [location, setLocation] = useState('');
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
  const [deleteFor, setDeleteFor] = useState<Source | null>(null);
  const [confirmFolder, setConfirmFolder] = useState(false);
  const cursorRef = useRef(0);
  const seen = useRef(new Set<string>());
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const placeRef = useRef<Crumb | null>(null);
  const reloadRef = useRef<() => void>(() => undefined);
  const moving = useRef(false);
  const deleting = useRef(false);
  const sessionId = detail?.session_id;
  const current = crumbs.at(-1) ?? null;
  placeRef.current = current;

  const loadWorkspace = useCallback(async () => {
    const body = await api<{
      workspace: WorkspaceDetail;
      folder_kinds: FolderKind[];
    }>(`/workspaces/${workspaceId}`);
    const workspace = { ...body.workspace, skills: body.workspace.skills ?? [] };
    setDetail(workspace);
    setKinds(body.folder_kinds);
    setCrumbs((existing) =>
      existing.length > 0
        ? existing
        : [{ key: 'root', label: workspace.name, scope: 'rows', folderId: null, path: '', kind: 'root' }],
    );
  }, [workspaceId]);

  const loadPlace = useCallback(
    async (place: Crumb) => {
      if (place.scope === 'rows') {
        const query = place.kind === 'cloud' && place.folderId ? `?parent_id=${encodeURIComponent(place.folderId)}` : '';
        const body = await api<{ folders: FolderRow[] }>(`/workspaces/${workspaceId}/folders${query}`);
        const root = place.kind === 'root' ? rootLocalFolder(body.folders, place.label) : null;
        if (root) {
          await api(`/workspaces/${workspaceId}/focus`, {
            method: 'PUT',
            body: JSON.stringify({ folder_id: root.id }),
          });
          const images = await api<Listing>(`/workspaces/${workspaceId}/folders/${root.id}/images?path=`);
          setRootFolderId(root.id);
          setRows(body.folders.filter((row) => row.id !== root.id));
          setListing(images);
          keepImage(images.images, root.id, setSelected);
          return;
        }
        setRootFolderId(null);
        setRows(body.folders);
        setListing(null);
        setSelected(null);
        return;
      }
      if (place.folderId == null) {
        return;
      }
      await api(`/workspaces/${workspaceId}/focus`, {
        method: 'PUT',
        body: JSON.stringify({ folder_id: place.folderId }),
      });
      const body = await api<Listing>(
        `/workspaces/${workspaceId}/folders/${place.folderId}/images?path=${encodeURIComponent(place.path)}`,
      );
      setListing(body);
      setRows([]);
      keepImage(body.images, place.folderId, setSelected);
    },
    [workspaceId],
  );

  reloadRef.current = () => {
    const place = placeRef.current;
    if (place) {
      void loadPlace(place).catch((caught: Error) => setNotice(caught.message));
    }
  };

  useEffect(() => {
    void loadWorkspace().catch((caught: Error) => setNotice(caught.message));
  }, [loadWorkspace]);

  useEffect(() => {
    const place = placeRef.current;
    if (place == null) {
      return;
    }
    setSelected(null);
    setMoveFor(null);
    void loadPlace(place).catch((caught: Error) => setNotice(caught.message));
  }, [current?.key, loadPlace]);

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
      if (!moving.current && !deleting.current) {
        reloadRef.current();
      }
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
      setMessages((currentMessages) =>
        currentMessages.some((item) => item.id === id)
          ? currentMessages
          : [...currentMessages, chatMessage({ id, role, body, reasoning: event.payload?.reasoning })],
      );
      if (role === 'assistant') {
        setLive('');
        setThinking('');
        setWorking(false);
        setRunId(null);
        setStatus('Reply ready');
        reloadRef.current();
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
      setLive((value) => value + (event.payload?.agui?.delta ?? ''));
    }
    if (event.type === 'run.delta' && event.payload?.agui?.type === 'REASONING_MESSAGE_CONTENT') {
      setThinking((value) => value + (event.payload?.agui?.delta ?? ''));
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

  function openCrumb(index: number) {
    setCrumbs((existing) => existing.slice(0, index + 1));
    setNotice(null);
  }

  async function openRow(row: FolderRow) {
    if (row.kind === 'local') {
      try {
        await api<Listing>(
          `/workspaces/${workspaceId}/folders/${row.id}/images?path=`,
        );
      } catch (caught) {
        setNotice(caught instanceof Error ? caught.message : 'That folder did not open.');
        return;
      }
    }
    setCrumbs((existing) => [
      ...existing,
      {
        key: row.id,
        label: row.name,
        scope: row.kind === 'local' ? 'disk' : 'rows',
        folderId: row.id,
        path: '',
        kind: row.kind === 'local' ? 'local' : 'cloud',
      },
    ]);
    setNotice(null);
  }

  function openDirectory(path: string, name: string) {
    const folderId = current?.folderId ?? rootFolderId;
    if (folderId == null) {
      return;
    }
    setCrumbs((existing) => [
      ...existing,
      {
        key: `${folderId}:${path}`,
        label: name,
        scope: 'disk',
        folderId,
        path,
        kind: 'directory',
      },
    ]);
  }

  async function pickDirectory() {
    try {
      const body = await api<{ cancelled: boolean; location?: string | null }>(
        `/workspaces/${workspaceId}/folders/pick`,
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

  async function saveFolder() {
    if (current == null) {
      return;
    }
    try {
      if (current.scope === 'disk' && current.folderId) {
        const path = current.path ? `${current.path}/${folderName.trim()}` : folderName.trim();
        await api(`/workspaces/${workspaceId}/folders/${current.folderId}/directories`, {
          method: 'POST',
          body: JSON.stringify({ path }),
        });
      } else {
        await api(`/workspaces/${workspaceId}/folders`, {
          method: 'POST',
          headers: { 'Idempotency-Key': idempotencyKey() },
          body: JSON.stringify({
            name: folderName.trim(),
            kind: current.kind === 'cloud' ? 'cloud' : folderKind,
            provider: (current.kind === 'cloud' ? 'cloud' : folderKind) === 'cloud' ? provider : null,
            location: folderKind === 'local' && current.kind !== 'cloud' ? location.trim() : null,
            remote_id: remoteId.trim() || null,
            parent_id: current.kind === 'cloud' ? current.folderId : null,
          }),
        });
      }
      setSheet(false);
      setNotice(null);
      await loadPlace(current);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The folder was not saved.');
    }
  }

  const destinations = moveFor == null || current == null ? [] : destinationsFor(moveFor, crumbs, rows, listing);

  async function runMove(source: Source, dest: Dest) {
    const refused = crossedKind(source.kind, dest.kind);
    if (refused) {
      setNotice(refused);
      setMoveFor(null);
      return;
    }
    if (source.type === 'row' && dest.folderId === source.id) {
      setNotice('A folder cannot be moved into itself.');
      setMoveFor(null);
      return;
    }
    moving.current = true;
    try {
      if (source.type === 'image' || source.type === 'directory') {
        const name = source.name;
        const to = dest.path ? `${dest.path}/${name}` : name;
        if (to === source.path) {
          setMoveFor(null);
          return;
        }
        if (source.type === 'directory' && (dest.path === source.path || dest.path.startsWith(`${source.path}/`))) {
          setNotice('A folder cannot be moved into itself.');
          setMoveFor(null);
          return;
        }
        if (source.type === 'image') {
          await api(`/workspaces/${workspaceId}/folders/${source.folderId}/images/move`, {
            method: 'POST',
            body: JSON.stringify({ path: source.path, to }),
          });
        } else {
          await api(`/workspaces/${workspaceId}/folders/${source.folderId}/directories`, {
            method: 'PATCH',
            body: JSON.stringify({ path: source.path, to }),
          });
        }
      } else {
        await api(`/workspaces/${workspaceId}/folders/${source.id}`, {
          method: 'PATCH',
          body: JSON.stringify({
            expected_revision: source.revision,
            move: true,
            parent_id: dest.kind === 'root' ? null : dest.folderId,
          }),
        });
      }
      if (current) {
        await loadPlace(current);
      }
      setNotice(null);
      setMoveFor(null);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The move was refused.');
      setMoveFor(null);
    } finally {
      moving.current = false;
    }
  }

  async function runDelete(source: Source) {
    if (deleting.current) {
      return;
    }
    deleting.current = true;
    try {
      await api(`/workspaces/${workspaceId}/folders/${source.folderId}/images`, {
        method: 'DELETE',
        body: JSON.stringify({ path: source.path }),
      });
      if (current) {
        await loadPlace(current);
      }
      setNotice(null);
      setDeleteFor(null);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The image was not deleted.');
      setDeleteFor(null);
    } finally {
      deleting.current = false;
    }
  }

  async function runDeleteFolder() {
    const parent = crumbs.at(-2);
    if (deleting.current || current?.kind !== 'directory' || current.folderId == null || parent == null || !folderEmpty) {
      setConfirmFolder(false);
      return;
    }
    deleting.current = true;
    try {
      await api(`/workspaces/${workspaceId}/folders/${current.folderId}/directories`, {
        method: 'DELETE',
        body: JSON.stringify({ path: current.path }),
      });
      placeRef.current = parent;
      setCrumbs((existing) => existing.slice(0, -1));
      await loadPlace(parent);
      setNotice(null);
      setConfirmFolder(false);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : 'The folder was not deleted.');
      setConfirmFolder(false);
    } finally {
      deleting.current = false;
    }
  }

  async function send() {
    if (draft.trim() === '' || working) {
      return;
    }
    const text = draft;
    const place = crumbs.at(-1);
    const folderId = place?.scope === 'disk' ? place.folderId : rootFolderId;
    const onDisk = folderId != null && place?.kind !== 'cloud';
    const folder = onDisk ? (place?.scope === 'disk' ? place.path : '') : undefined;
    const image = onDisk && selected?.type === 'image' ? selected.path : undefined;
    updateDraft('');
    try {
      const result = await api<{ message_id: string; text: string; run_id?: string }>(
        `/workspaces/${workspaceId}/session/messages`,
        {
          method: 'POST',
          headers: { 'Idempotency-Key': idempotencyKey() },
          body: JSON.stringify({
            text,
            ...(folder !== undefined ? { folder } : {}),
            ...(image ? { image } : {}),
          }),
        },
      );
      setMessages((currentMessages) =>
        currentMessages.some((item) => item.id === result.message_id)
          ? currentMessages
          : [...currentMessages, { id: result.message_id, role: 'user', body: result.text }],
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
  const galleryFolderId = current?.scope === 'disk' ? current.folderId : rootFolderId;
  const selectedImage = selected?.type === 'image' ? selected : null;
  const fileUrl =
    selectedImage && galleryFolderId
      ? `${API_URL}/workspaces/${workspaceId}/folders/${galleryFolderId}/file?path=${encodeURIComponent(selectedImage.path)}`
      : null;
  const cloudOnly = current?.kind === 'cloud';
  const makingDirectory = current?.scope === 'disk';
  const canSave =
    folderName.trim() !== '' &&
    (makingDirectory || cloudOnly || folderKind === 'cloud' || location.trim() !== '');

  useEffect(() => {
    if (Platform.OS !== 'web' || galleryFolderId == null) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement) {
        const tag = target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable) {
          return;
        }
      }
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
        return;
      }
      const images = listing?.images ?? [];
      const index = images.findIndex((item) => item.path === selected?.path);
      const next = event.key === 'ArrowRight' ? index + 1 : index - 1;
      const item = images[next];
      if (!item) {
        return;
      }
      event.preventDefault();
      setSelected(imageSource(item, galleryFolderId));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [galleryFolderId, listing, selected]);

  const emptyFolder =
    (listing?.images.length ?? 0) === 0 && (listing?.folders.length ?? 0) === 0 && rows.length === 0 && listing != null;
  const folderEmpty = current?.kind === 'directory' && listing?.path === current.path && listing.empty === true;
  const noFolders = current?.scope === 'rows' && rows.length === 0 && listing == null;

  const stage = (
    <View accessibilityLabel="Gallery" style={{ flex: 1, minWidth: 0, backgroundColor: palette.background }}>
      <ScrollView horizontal contentContainerStyle={{ alignItems: 'center', gap: space[1], paddingHorizontal: space[3], paddingVertical: space[2] }} style={{ flexGrow: 0 }}>
        <View accessibilityLabel="Breadcrumb" style={{ flexDirection: 'row', alignItems: 'center', gap: space[1] }}>
          {crumbs.map((crumb, index) => {
            const last = index === crumbs.length - 1;
            return (
              <View key={crumb.key} style={{ flexDirection: 'row', alignItems: 'center', gap: space[1] }}>
                {index > 0 ? <Text style={textStyle('small', palette.textSecondary)}>{'>'}</Text> : null}
                {last ? (
                  <Text accessibilityRole="text" style={textStyle('small', palette.textPrimary, '600')}>{crumb.label}</Text>
                ) : (
                  <Pressable accessibilityRole="button" accessibilityLabel={crumb.label} onPress={() => openCrumb(index)} style={{ minHeight: 44, justifyContent: 'center' }}>
                    <Text style={textStyle('small', palette.accent)}>{crumb.label}</Text>
                  </Pressable>
                )}
              </View>
            );
          })}
        </View>
      </ScrollView>
      <View style={{ flex: 1, minHeight: 160, alignItems: 'center', justifyContent: 'center', padding: space[3] }}>
        {selectedImage && fileUrl ? (
          <Image
            accessibilityLabel={selectedImage.name}
            accessibilityIgnoresInvertColors
            source={{ uri: fileUrl }}
            style={{ width: '100%', height: '100%', borderRadius: 8 }}
            resizeMode="contain"
          />
        ) : (
          <Text style={textStyle('small', palette.textSecondary)}>
            {noFolders ? 'No folders' : emptyFolder ? 'No images in this folder' : ''}
          </Text>
        )}
        {(selectedImage && fileUrl) || current?.kind === 'directory' ? (
          <View style={{ position: 'absolute', top: space[3], right: space[3], flexDirection: 'row', gap: space[2] }}>
            {selectedImage && fileUrl ? (
              <>
            <IconButton
              iconOnly
              icon="download"
              label="Download"
              onPress={() => {
                if (Platform.OS === 'web') {
                  const link = document.createElement('a');
                  link.href = fileUrl;
                  link.download = selectedImage.name;
                  link.click();
                }
              }}
              background={palette.surface}
              border={palette.borderStrong}
              color={palette.textPrimary}
            />
            <IconButton
              iconOnly
              icon="message-circle"
              label="Add to chat"
              onPress={() => {
                const next = draft.trim() === '' ? selectedImage.path : `${draft.trim()}\n${selectedImage.path}`;
                updateDraft(next);
              }}
              background={palette.surface}
              border={palette.borderStrong}
              color={palette.textPrimary}
            />
            <IconButton
              iconOnly
              icon="move"
              label="Move"
              accessibilityLabel={`Move ${selectedImage.name}`}
              onPress={() => setMoveFor(selectedImage)}
              background={palette.surface}
              border={palette.borderStrong}
              color={palette.textPrimary}
            />
            <IconButton
              iconOnly
              icon="trash-2"
              label="Delete"
              accessibilityLabel={`Delete ${selectedImage.name}`}
              onPress={() => setDeleteFor(selectedImage)}
              background={palette.surface}
              border={palette.borderStrong}
              color={palette.textPrimary}
            />
              </>
            ) : null}
            {current?.kind === 'directory' ? (
            <IconButton
              iconOnly
              icon="trash-2"
              label="Delete folder"
              accessibilityLabel={`Delete ${current.label}`}
              disabled={!folderEmpty}
              onPress={() => setConfirmFolder(true)}
              background={palette.surface}
              border={palette.borderStrong}
              color={folderEmpty ? palette.textPrimary : palette.disabledText}
            />
            ) : null}
          </View>
        ) : null}
      </View>
      <ScrollView
        horizontal
        accessibilityLabel="Filmstrip"
        style={{ flexGrow: 0, borderTopWidth: 1, borderTopColor: palette.border }}
        contentContainerStyle={{ gap: space[2], padding: space[3], alignItems: 'center' }}
      >
        {rows.map((row) => (
          <Pressable
            key={row.id}
            accessibilityRole="button"
            accessibilityLabel={row.name}
            onPress={() => void openRow(row)}
            style={thumbStyle(false, palette)}
          >
            <FeatherIcon name="folder" size={22} color={palette.textPrimary} />
            <Text numberOfLines={1} style={textStyle('caption', palette.textPrimary)}>{row.name}</Text>
          </Pressable>
        ))}
        {(listing?.folders ?? []).map((item) => (
          <Pressable
            key={item.path}
            accessibilityRole="button"
            accessibilityLabel={item.name}
            onPress={() => openDirectory(item.path, item.name)}
            style={thumbStyle(false, palette)}
          >
            <FeatherIcon name="folder" size={22} color={palette.textPrimary} />
            <Text numberOfLines={1} style={textStyle('caption', palette.textPrimary)}>{item.name}</Text>
          </Pressable>
        ))}
        {(listing?.images ?? []).map((item) => {
          const active = selected?.path === item.path;
          return (
            <Pressable
              key={item.path}
              accessibilityRole="button"
              accessibilityLabel={item.name}
              accessibilityState={{ selected: active }}
              onPress={() => galleryFolderId && setSelected(imageSource(item, galleryFolderId))}
              style={thumbStyle(active, palette)}
            >
              <Image
                accessibilityIgnoresInvertColors
                source={{ uri: `${API_URL}/workspaces/${workspaceId}/folders/${galleryFolderId}/file?path=${encodeURIComponent(item.path)}` }}
                style={{ width: 80, height: 80 }}
              />
            </Pressable>
          );
        })}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="New folder"
          onPress={() => {
            setFolderName('');
            setFolderKind(cloudOnly ? 'cloud' : 'local');
            setLocation('');
            setRemoteId('');
            setSheet(true);
          }}
          style={thumbStyle(false, palette)}
        >
          <FeatherIcon name="folder-plus" size={22} color={palette.textPrimary} />
          <Text numberOfLines={1} style={textStyle('caption', palette.textPrimary)}>New folder</Text>
        </Pressable>
      </ScrollView>
    </View>
  );

  const transcript = (
    <View accessibilityLabel="Conversation" style={{ flex: 1, minWidth: 0, padding: space[3], gap: space[2] }}>
      {Platform.OS === 'web'
        ? createElement('style', { 'data-testid': 'transcript-scrollbar' }, transcriptScrollbar(palette.borderStrong, palette.background, palette.textSecondary))
        : null}
      <ScrollView
        ref={transcriptRef}
        testID="transcript"
        style={{ flex: 1, minWidth: 0 }}
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
        <Text accessibilityLiveRegion="polite" {...(Platform.OS === 'web' ? { role: 'status' as const } : {})} style={Platform.OS === 'web' ? { position: 'absolute', width: 1, height: 1, overflow: 'hidden' } : textStyle('small', palette.textSecondary)}>
          {status}
        </Text>
      ) : null}
      {error ? (
        <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={textStyle('small', palette.danger)}>
          {error}
        </Text>
      ) : null}
      <Composer value={draft} working={working} agentName={agentName} onChange={updateDraft} onSend={() => void send()} onStop={() => void stop()} />
    </View>
  );

  const skillsButton = (
    <View style={{ marginRight: space[2] }}>
      <IconButton icon="book-open" label="Skills" onPress={() => setSkillsOpen(true)} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
    </View>
  );

  const workspaceBar = (
    <>
      <IconButton icon="arrow-left" label="Back to channel" onPress={onBack} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
      <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
        {detail?.name ?? 'Gallery'}
      </Text>
      <Text style={textStyle('small', palette.textSecondary)}>Gallery</Text>
      {Platform.OS === 'web' ? null : skillsButton}
      {narrow && !chatOpen ? (
        <IconButton icon="message-square" label="Chat" onPress={() => setChatOpen(true)} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
      ) : null}
      {narrow && chatOpen ? (
        <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to gallery" onPress={() => setChatOpen(false)} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
      ) : null}
    </>
  );

  return (
    <View accessibilityLabel="Gallery workspace" style={{ flex: 1, minWidth: 0 }}>
      {Platform.OS === 'web' ? <BannerPortal hostId="banner-actions">{skillsButton}</BannerPortal> : null}
      {narrow ? (
        <View style={[buttonRow, { padding: space[3], borderBottomWidth: 1, borderBottomColor: palette.border }]}>{workspaceBar}</View>
      ) : (
        <BannerPortal>{workspaceBar}</BannerPortal>
      )}
      {notice ? (
        <Text accessibilityRole="alert" accessibilityLiveRegion="polite" style={[textStyle('small', palette.danger), { paddingHorizontal: space[3] }]}>
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
      {narrow && chatOpen ? (
        transcript
      ) : (
        <View style={{ flex: 1, flexDirection: narrow ? 'column' : 'row', minWidth: 0 }}>
          {stage}
          {narrow ? null : (
            <View style={{ width: chatColumnWidth, minWidth: 0, borderLeftWidth: 1, borderColor: palette.border }}>{transcript}</View>
          )}
        </View>
      )}
      {moveFor ? (
        <Modal label="Move" closeLabel="Close move" onClose={() => setMoveFor(null)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>Move to</Text>
          {destinations.length === 0 ? (
            <Text style={textStyle('small', palette.textSecondary)}>No destination</Text>
          ) : (
            <View accessibilityLabel="Move to" style={{ gap: space[1] }}>
              {destinations.map((dest) => (
                <Pressable
                  key={dest.key}
                  accessibilityRole="button"
                  accessibilityLabel={dest.label}
                  onPress={() => void runMove(moveFor, dest)}
                  style={{
                    minHeight: 44,
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: space[2],
                    paddingHorizontal: space[2],
                    borderRadius: radius.control,
                  }}
                >
                  <FeatherIcon name="folder" size={18} color={palette.textPrimary} />
                  <Text style={textStyle('body', palette.textPrimary)}>{dest.label}</Text>
                </Pressable>
              ))}
            </View>
          )}
        </Modal>
      ) : null}
      {deleteFor ? (
        <Modal label="Delete image" closeLabel="Close delete" onClose={() => setDeleteFor(null)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {deleteFor.name}?
          </Text>
          <IconButton
            icon="trash-2"
            label={`Delete ${deleteFor.name}`}
            onPress={() => void runDelete(deleteFor)}
            background={palette.danger}
            color={palette.onAccent}
          />
        </Modal>
      ) : null}
      {confirmFolder && current?.kind === 'directory' ? (
        <Modal label="Delete folder" closeLabel="Close delete folder" onClose={() => setConfirmFolder(false)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {current.label}?
          </Text>
          <IconButton
            icon="trash-2"
            label={`Delete ${current.label}`}
            onPress={() => void runDeleteFolder()}
            background={palette.danger}
            color={palette.onAccent}
          />
        </Modal>
      ) : null}
      {sheet ? (
        <Modal label="New folder" closeLabel="Close new folder" onClose={() => setSheet(false)}>
          <TextInput accessibilityLabel="Folder name" value={folderName} onChangeText={setFolderName} style={fieldStyle(palette)} />
          {makingDirectory ? null : (
            <>
              <Text style={textStyle('small', palette.textPrimary)}>Folder type</Text>
              <Picker
                accessibilityLabel="Folder type"
                selectedValue={cloudOnly ? 'cloud' : folderKind}
                onValueChange={(value) => setFolderKind(String(value))}
                style={pickerStyle(palette)}
              >
                {cloudOnly ? null : <Picker.Item label="Local" value="local" />}
                <Picker.Item label="Cloud" value="cloud" />
              </Picker>
              {(cloudOnly || folderKind === 'cloud') ? (
                <>
                  <Text style={textStyle('small', palette.textPrimary)}>Cloud provider</Text>
                  <Picker accessibilityLabel="Cloud provider" selectedValue={provider} onValueChange={(value) => setProvider(String(value))} style={pickerStyle(palette)}>
                    {kinds.filter((item) => item.kind === 'cloud').map((item) => (
                      <Picker.Item key={item.provider} label={providerLabel(item.provider)} value={item.provider ?? ''} />
                    ))}
                  </Picker>
                  <TextInput accessibilityLabel="Remote folder id" value={remoteId} onChangeText={setRemoteId} autoCapitalize="none" style={fieldStyle(palette)} />
                </>
              ) : (
                <>
                  <IconButton icon="folder" label="Choose folder" onPress={() => void pickDirectory()} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
                  <TextInput accessibilityLabel="Location" value={location} onChangeText={setLocation} autoCapitalize="none" style={fieldStyle(palette)} />
                </>
              )}
            </>
          )}
          <IconButton
            icon="check"
            label="Save folder"
            disabled={!canSave}
            onPress={() => void saveFolder()}
            background={canSave ? palette.accent : palette.disabledSurface}
            color={canSave ? palette.onAccent : palette.disabledText}
          />
        </Modal>
      ) : null}
    </View>
  );
}

function imageSource(item: ImageFile, folderId: string): Source {
  return {
    type: 'image',
    id: item.path,
    name: item.name,
    path: item.path,
    kind: 'image',
    revision: 0,
    folderId,
  };
}

function keepImage(
  images: ImageFile[],
  folderId: string,
  setSelected: (value: Source | null | ((current: Source | null) => Source | null)) => void,
) {
  setSelected((current) => {
    if (current?.type === 'image' && current.folderId === folderId && images.some((item) => item.path === current.path)) {
      return current;
    }
    const first = images[0];
    return first ? imageSource(first, folderId) : null;
  });
}

function destinationsFor(source: Source, crumbs: Crumb[], rows: FolderRow[], listing: Listing | null): Dest[] {
  const current = crumbs.at(-1);
  if (current == null) {
    return [];
  }
  const found: Dest[] = [];
  if (source.type === 'row' && source.kind === 'cloud') {
    for (const row of rows) {
      if (row.kind === 'cloud' && row.id !== source.id) {
        found.push({ key: `row:${row.id}`, label: row.name, kind: 'cloud', folderId: row.id, path: '' });
      }
    }
    crumbs.slice(0, -1).forEach((crumb, index) => {
      if (crumb.kind === 'root') {
        found.push({ key: `crumb:${index}`, label: crumb.label, kind: 'root', folderId: null, path: '' });
      } else if (crumb.kind === 'cloud') {
        found.push({ key: `crumb:${index}`, label: crumb.label, kind: 'cloud', folderId: crumb.folderId, path: '' });
      }
    });
    return found;
  }
  if (source.type === 'image' || source.type === 'directory') {
    for (const folder of listing?.folders ?? []) {
      if (source.type === 'directory' && (folder.path === source.path || folder.path.startsWith(`${source.path}/`))) {
        continue;
      }
      found.push({ key: `dir:${folder.path}`, label: folder.name, kind: 'disk', folderId: source.folderId, path: folder.path });
    }
    crumbs.slice(0, -1).forEach((crumb, index) => {
      if (crumb.scope === 'disk' && crumb.folderId === source.folderId) {
        found.push({ key: `crumb:${index}`, label: crumb.label, kind: 'disk', folderId: crumb.folderId, path: crumb.path });
      }
    });
  }
  return found;
}

export function rootLocalFolder(folders: FolderRow[], workspaceName: string): FolderRow | null {
  const locals = folders.filter((row) => row.kind === 'local' && !row.parent_id);
  if (locals.length === 1) {
    return locals[0];
  }
  return locals.find((row) => row.name === workspaceName) ?? null;
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

/** Side chat matches the reading measure, plus the column padding. */
const chatColumnWidth =
  Platform.OS === 'web'
    ? (`calc(${typeScale.measure}ch + ${space[3] * 2}px)` as const)
    : typeScale.measure * 8 + space[3] * 2;

function thumbStyle(selected: boolean, palette: { accent: string; surface: string; textPrimary: string }) {
  return {
    width: 80,
    height: 80,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: selected ? palette.accent : 'transparent',
    overflow: 'hidden' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    backgroundColor: palette.surface,
  };
}

function pickerStyle(palette: { textPrimary: string; surface: string; borderStrong: string }) {
  return {
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: radius.control,
    paddingHorizontal: 12,
    color: palette.textPrimary,
    backgroundColor: palette.surface,
    fontSize: 16,
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


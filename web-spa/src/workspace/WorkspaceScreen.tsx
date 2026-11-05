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

import { API_URL, api, socketUrl } from '@/api/client';
import { spokenError } from '@/reply-error';
import {
  chatMessage,
  clearedTranscript,
  includeCreatedMessage,
  keepLatest,
  sessionCommand,
  LiveReply,
  readDraft,
  showsMessage,
  coalesceThinkingTurns,
  transcriptFrame,
  TranscriptTurn,
  writeDraft,
  type ChatMessage,
  type ReplyStep,
} from '@/components/chat-transcript';
import { Composer } from '@/components/Composer';
import { ReplyAlert } from '@/components/ReplyAlert';
import { IconButton } from '@/components/IconButton';
import { Modal } from '@/components/Modal';
import { useAguiChat } from '@/components/use-agui-chat';
import { WorkspaceSkills, type WorkspaceSkill } from '@/components/WorkspaceSkills';
import { readExpandTrace, workingLabel } from '@/components/expandTrace';
import { WorkingStatus } from '@/components/WorkingStatus';
import { useKeyboardInset } from '@/hooks/useKeyboardInset';
import { isWorkspaceScheme, type WorkspaceScheme } from '@/theme/workspace-schemes';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import { FolderSetup } from '@/workspace/FolderSetup';
import { GalleryChatDock } from '@/workspace/GalleryChatDock';
import { useHostWorkspaceBanner, useRegisterWorkspaceChrome } from '@/workspace/workspace-chrome';
import { GalleryChatResizeHandle } from '@/workspace/GalleryChatResizeHandle';
import {
  cycleGalleryDetent,
  GALLERY_DOCK_GAP,
  GALLERY_DOCK_PAD,
  type GalleryChatDetent,
  readGalleryChatDetent,
  writeGalleryChatDetent,
} from '@/workspace/gallery-chat-dock';
import {
  clampGalleryChatWidth,
  readGalleryChatWidth,
  writeGalleryChatWidth,
} from '@/workspace/gallery-chat-width';
import { WorkspaceShell } from '@/workspace/WorkspaceShell';
import { documentFromShell, ensureShellElements, type PlacementState, type ShellDocument } from '@/workspace/placement';

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  channel_id: string;
  skills: WorkspaceSkill[];
  expand_thinking?: string | null;
  expand_tools?: string | null;
  theme?: string;
  shell?: ShellDocument;
};

type PendingMedia = { task_id: string; kind: string; status: string };

type Snapshot = {
  messages: { id: string; role: string; body: string; reasoning?: string | null; tool_calls?: unknown; trace?: unknown }[];
  runs: { id: string; status: string; error: string | null }[];
  cursor: number;
  state?: PlacementState;
  pending_media?: PendingMedia[];
};

const MEDIA_DONE = new Set(['completed', 'failed', 'cancelled']);

const LATEST_SLACK = 64;

export function WorkspaceScreen({
  workspaceId,
  agentName,
  onTheme,
  onBack,
  onDeleted,
}: {
  workspaceId: string;
  agentName: string;
  onTheme?: (scheme: WorkspaceScheme) => void;
  onBack?: () => void;
  onDeleted?: () => void;
}) {
  const { palette } = useTheme();
  const { width, height } = useWindowDimensions();
  const narrow = width < 1024;
  const wideChat = !narrow && Platform.OS === 'web';
  const columnBanner = useHostWorkspaceBanner(wideChat);
  const keyboardInset = useKeyboardInset();
  const [rowWidth, setRowWidth] = useState(width);
  const [columnHeight, setColumnHeight] = useState(height);
  const [chatWidth, setChatWidth] = useState(() => readGalleryChatWidth(width));
  const [detent, setDetent] = useState<GalleryChatDetent>(() => readGalleryChatDetent());
  const detentBeforeFocus = useRef<GalleryChatDetent | null>(null);
  const rowWidthRef = useRef(rowWidth);
  rowWidthRef.current = rowWidth;
  useEffect(() => {
    // Read the saved width again once the row is known. Clamping the in-memory
    // value would forget a width that was saved against a wider window.
    setChatWidth(readGalleryChatWidth(rowWidth));
  }, [rowWidth]);
  const [detail, setDetail] = useState<WorkspaceDetail | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [pulseToken, setPulseToken] = useState(0);
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
  const [status, setStatus] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  useRegisterWorkspaceChrome({
    openSkills: () => setSkillsOpen(true),
    requestDelete: () => setConfirmDelete(true),
  });
  const [confirmImage, setConfirmImage] = useState<string | null>(null);
  const [removingImage, setRemovingImage] = useState(false);
  const [placement, setPlacement] = useState<PlacementState>(documentFromShell(null));
  const shellRef = useRef<ShellDocument | null>(null);
  shellRef.current = detail?.shell ?? null;
  const [bound, setBound] = useState(false);
  const [folderId, setFolderId] = useState<string | null>(null);
  const [folderFiles, setFolderFiles] = useState<string[] | null>(null);
  const folderIdRef = useRef<string | null>(null);
  folderIdRef.current = folderId;
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const openRun = useRef<string | null>(null);
  const cursorRef = useRef(0);
  const seen = useRef(new Set<string>());
  const [pendingMedia, setPendingMedia] = useState<PendingMedia[]>([]);
  const sessionId = detail?.session_id ?? null;

  const applyDetent = useCallback((next: GalleryChatDetent) => {
    setDetent(next);
    writeGalleryChatDetent(next);
  }, []);
  const cycleLayout = useCallback(() => {
    setDetent((current) => {
      const next = cycleGalleryDetent(current);
      writeGalleryChatDetent(next);
      return next;
    });
  }, []);
  const toggleFullScreen = useCallback(() => {
    setDetent((current) => {
      if (current === 'fullscreen') {
        const prior = detentBeforeFocus.current ?? 'peek';
        detentBeforeFocus.current = null;
        writeGalleryChatDetent(prior);
        return prior;
      }
      detentBeforeFocus.current = current;
      writeGalleryChatDetent('fullscreen');
      return 'fullscreen';
    });
  }, []);

  const folderFilesRef = useRef<string[] | null>(null);
  folderFilesRef.current = folderFiles;

  const commitPlacement = useCallback((next: PlacementState) => {
    setPlacement(ensureShellElements(next, shellRef.current, folderFilesRef.current ?? []));
  }, []);

  const chat = useAguiChat({
    threadId: sessionId,
    state: placement as unknown as Record<string, unknown>,
    onMessages: setMessages,
    onSteps: setSteps,
    onWorking: setWorking,
    onRunId: setRunId,
    onError: setError,
    onStatus: setStatus,
    onState: (next) => {
      if (next && typeof next === 'object' && 'favlo_workspace' in next) {
        commitPlacement(next as PlacementState);
      }
    },
    onSessionId: (nextId) => {
      setDetail((value) => (value == null ? value : { ...value, session_id: nextId }));
      commitPlacement(documentFromShell(shellRef.current));
    },
    hydrate: async () => {
      if (sessionId == null) {
        return { messages: [] };
      }
      const snap = await api<Snapshot>(`/sessions/${sessionId}`);
      setPendingMedia(snap.pending_media ?? []);
      if (snap.state?.favlo_workspace) {
        const placed = ensureShellElements(
          snap.state,
          shellRef.current,
          folderFilesRef.current ?? [],
        );
        setPlacement(placed);
        return {
          messages: snap.messages.map((item) => chatMessage(item)),
          state: placed as unknown as Record<string, unknown>,
        };
      }
      commitPlacement(documentFromShell(shellRef.current));
      return {
        messages: snap.messages.map((item) => chatMessage(item)),
        state: documentFromShell(shellRef.current) as unknown as Record<string, unknown>,
      };
    },
  });

  const load = useCallback(async () => {
    const body = await api<{ workspace: WorkspaceDetail }>(`/workspaces/${workspaceId}`);
    const workspace = { ...body.workspace, skills: body.workspace.skills ?? [] };
    const listed = await api<{ folders: { id: string; kind: string }[] }>(`/workspaces/${workspaceId}/folders`);
    const local = (listed.folders ?? []).find((item) => item.kind === 'local');
    setBound(local != null);
    setFolderId(local?.id ?? null);
    const snap = await api<Snapshot>(`/sessions/${workspace.session_id}`);
    cursorRef.current = snap.cursor;
    setPendingMedia(snap.pending_media ?? []);
    setMessages(snap.messages.map((item) => chatMessage(item)));
    shellRef.current = workspace.shell ?? null;
    if (snap.state?.favlo_workspace) {
      setPlacement(ensureShellElements(snap.state, workspace.shell, folderFilesRef.current ?? []));
    } else {
      setPlacement(documentFromShell(workspace.shell));
    }
    setDraft(readDraft(workspace.session_id));
    if (isWorkspaceScheme(workspace.theme)) {
      onTheme?.(workspace.theme);
    }
    // Set detail last so the socket effect subscribes with the loaded cursor.
    setDetail(workspace);
  }, [onTheme, workspaceId]);

  useEffect(() => {
    void load().catch((caught: Error) => setNotice(caught.message));
  }, [load]);

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
      socket.onmessage = (event) => {
        const payload = JSON.parse(String(event.data)) as {
          event_id: string;
          type: string;
          stream_id?: string;
          run_id?: string;
          sequence?: number;
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
            kind?: string;
            session_id?: string;
            previous_session_id?: string;
            trace?: unknown;
            theme?: string;
            state?: PlacementState;
          };
        };
        if (seen.current.has(payload.event_id)) {
          return;
        }
        seen.current.add(payload.event_id);
        if (typeof payload.sequence === 'number') {
          cursorRef.current = Math.max(cursorRef.current, payload.sequence);
        }
        if (clearedTranscript(payload, sessionId)) {
          openRun.current = null;
          setMessages([]);
          setSteps([]);
          setWorking(false);
          setRunId(null);
          setPendingMedia([]);
          commitPlacement(documentFromShell(shellRef.current));
          if (payload.payload?.session_id) {
            setDetail((value) =>
              value == null ? value : { ...value, session_id: payload.payload!.session_id! },
            );
          }
        }
        if (payload.type === 'workspace.theme_set' && isWorkspaceScheme(payload.payload?.theme)) {
          onTheme?.(payload.payload.theme);
        }
        if (payload.type === 'gallery.updated') {
          if (payload.payload?.state?.favlo_workspace) {
            commitPlacement(payload.payload.state);
          }
          const listed = folderIdRef.current;
          if (listed != null) {
            void loadFolderFiles(listed);
          }
        }
        if (payload.type === 'media.pending' && payload.payload?.task_id) {
          const taskId = payload.payload.task_id;
          const kind = payload.payload.kind ?? 'image';
          const status = payload.payload.status ?? 'working';
          if (MEDIA_DONE.has(status)) {
            setPendingMedia((current) => current.filter((item) => item.task_id !== taskId));
          } else {
            setPendingMedia((current) => {
              const next = current.filter((item) => item.task_id !== taskId);
              next.push({ task_id: taskId, kind, status });
              return next;
            });
          }
        }
        if (
          payload.type === 'message.created' &&
          payload.payload?.message_id &&
          showsMessage(payload.payload.text, payload.payload.reasoning, payload.payload.tool_calls)
        ) {
          const id = payload.payload.message_id;
          const role = payload.payload.role ?? 'assistant';
          const body = payload.payload.text ?? '';
          setMessages((current) =>
            includeCreatedMessage(
              current,
              chatMessage({
                id,
                role,
                body,
                reasoning: payload.payload?.reasoning,
                tool_calls: payload.payload?.tool_calls,
                trace: payload.payload?.trace,
              }),
            ),
          );
          if (role === 'assistant' && Array.isArray(payload.payload.task_ids)) {
            setPendingMedia((current) =>
              current.filter((item) => !(payload.payload?.task_ids ?? []).includes(item.task_id)),
            );
          }
        }
        if (payload.type === 'run.started') {
          openRun.current = payload.run_id ?? null;
          setWorking(true);
          setRunId(payload.run_id ?? null);
          setError(null);
          setStatus(null);
        }
        if (payload.type === 'run.completed' && payload.payload?.status === 'archived') {
          openRun.current = null;
          setMessages([]);
          setSteps([]);
          setWorking(false);
          setRunId(null);
          setError(null);
          setStatus(null);
          setPendingMedia([]);
          return;
        }
        if (payload.type === 'run.completed') {
          if (payload.run_id != null && openRun.current != null && payload.run_id !== openRun.current) {
            return;
          }
          openRun.current = null;
          setWorking(false);
          setRunId(null);
          setSteps([]);
          setStatus('Reply ready');
          void load();
          return;
        }
        if (payload.type === 'run.failed' || payload.type === 'run.cancelled') {
          if (payload.run_id != null && openRun.current != null && payload.run_id !== openRun.current) {
            return;
          }
          openRun.current = null;
          setWorking(false);
          setSteps([]);
          setRunId(null);
          setError(
            payload.type === 'run.failed'
              ? (payload.payload?.message ?? 'The reply failed.')
              : null,
          );
          setStatus(payload.type === 'run.cancelled' ? 'The reply stopped.' : null);
        }
      };
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
  }, [load, onTheme, sessionId]);

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
        commitPlacement(documentFromShell(shellRef.current));
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

  const loadFolderFiles = useCallback(async (id: string) => {
    const listing = await api<{ images?: { path?: string }[] }>(
      `/workspaces/${workspaceId}/folders/${id}/images`,
    );
    const names = (listing.images ?? [])
      .map((item) => String(item.path || ''))
      .filter(Boolean);
    setFolderFiles(names);
  }, [workspaceId]);

  useEffect(() => {
    if (folderId == null) {
      setFolderFiles(null);
      return;
    }
    void loadFolderFiles(folderId).catch((caught: Error) => setNotice(caught.message));
  }, [folderId, loadFolderFiles]);

  function fileUrl(path: string) {
    return `${API_URL}/workspaces/${workspaceId}/file?path=${encodeURIComponent(path)}`;
  }

  async function deleteImage(path: string) {
    if (folderId == null) {
      setNotice('Bind a local folder first.');
      return;
    }
    const body = await api<{ state?: PlacementState }>(
      `/workspaces/${workspaceId}/folders/${folderId}/images`,
      { method: 'DELETE', body: JSON.stringify({ path }) },
    );
    if (body.state?.favlo_workspace) {
      commitPlacement(body.state);
    }
  }

  const showLatest = useCallback(() => {
    if (!followLatest.current) {
      return;
    }
    transcriptRef.current?.scrollToEnd({ animated: false });
    placedLatest.current = true;
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

  function applyChatWidth(next: number) {
    const clamped = clampGalleryChatWidth(next, rowWidthRef.current);
    setChatWidth(clamped);
    writeGalleryChatWidth(clamped, rowWidthRef.current);
  }

  const chatPadding = narrow ? GALLERY_DOCK_PAD : space[4];
  const chatInner = (
    <>
      {detent === 'fullscreen' && narrow ? null : (
        <ScrollView
          ref={transcriptRef}
          testID="transcript"
          style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, minWidth: 0, maxWidth: '100%' }}
          contentContainerStyle={
            narrow
              ? { ...transcriptFrame(space[4]), paddingBottom: 0, minHeight: '100%' }
              : transcriptFrame(space[4])
          }
          onContentSizeChange={showLatest}
          onLayout={showLatest}
          onScroll={noteScroll}
          scrollEventThrottle={16}
        >
          {coalesceThinkingTurns(messages).map((item) => (
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
              imageNames={detail?.type === 'gallery'}
            />
          ))}
          <LiveReply
            steps={steps}
            speaker={agentName}
            measured
            expandTrace={readExpandTrace(detail?.expand_thinking)}
            expandTools={readExpandTrace(detail?.expand_tools, 'hidden')}
            forceOpen={traceForcedOpen}
            imageNames={detail?.type === 'gallery'}
          />
        </ScrollView>
      )}
      {working && !(detent === 'fullscreen' && narrow) ? (
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
      {error ? (
        <ReplyAlert
          message={spokenError(error)}
          disabled={working}
          onResend={() =>
            void chat.send([...messages].reverse().find((item) => item.role === 'user')?.body ?? '')
          }
        />
      ) : null}
      <Composer
        value={draft}
        agentName={agentName}
        compact={narrow}
        flushTop={narrow}
        singleLine={narrow && detent !== 'expanded'}
        pulseToken={pulseToken}
        onChange={updateDraft}
        onSend={() => void send()}
        onStop={() => {
          void chat.stop();
        }}
        working={working}
      />
    </>
  );

  const chatColumn = (
    <View
      accessibilityLabel="Conversation"
      style={{
        ...(narrow
          ? { flexGrow: 1, flexShrink: 1, flexBasis: 0, minWidth: 0, maxWidth: '100%' as const }
          : {
              width: chatWidth,
              flexGrow: 0,
              flexShrink: 0,
              flexBasis: chatWidth,
              minWidth: chatWidth,
              maxWidth: chatWidth,
            }),
        alignSelf: 'stretch',
        minHeight: 0,
        padding: chatPadding,
        gap: narrow ? GALLERY_DOCK_GAP : space[3],
        overflow: 'hidden',
      }}
    >
      {chatInner}
    </View>
  );

  const galleryPane = (
    <View
      accessibilityLabel="Gallery"
      style={{
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 0,
        minWidth: 0,
        minHeight: 0,
        borderRightWidth: !narrow && Platform.OS !== 'web' ? 1 : 0,
        borderColor: palette.border,
      }}
    >
      {!bound ? (
        <FolderSetup
          workspaceId={workspaceId}
          onBound={() => {
            setBound(true);
            void load();
          }}
        />
      ) : (
        <WorkspaceShell
          state={placement}
          slots={detail?.shell?.slots ?? []}
          fileUrl={fileUrl}
          onChange={commitPlacement}
          chatDetent={narrow ? detent : undefined}
          onCycleLayout={narrow ? cycleLayout : undefined}
          onAddToChat={(path) => {
            updateDraft(draft ? `${draft}\n${path}` : path);
            setPulseToken((value) => value + 1);
          }}
          onRequestDelete={setConfirmImage}
          fullScreen={narrow && detent === 'fullscreen'}
          hideFilmstrip={narrow && detent === 'fullscreen'}
          pendingMedia={pendingMedia}
          folderFiles={folderFiles}
          onToggleFullScreen={narrow ? toggleFullScreen : undefined}
        />
      )}
    </View>
  );

  return (
    <View accessibilityLabel="Gallery workspace" style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minWidth: 0, minHeight: 0 }}>
      <View
        style={{
          flexGrow: 1,
          flexShrink: 1,
          flexBasis: 0,
          minHeight: 0,
          minWidth: 0,
          flexDirection: narrow ? 'column' : 'row',
          paddingBottom: narrow ? keyboardInset : 0,
        }}
        onLayout={(event) => {
          setRowWidth(event.nativeEvent.layout.width);
          setColumnHeight(event.nativeEvent.layout.height);
        }}
      >
        <View
          style={{
            flexGrow: 1,
            flexShrink: 1,
            flexBasis: 0,
            minWidth: 0,
            minHeight: 0,
          }}
        >
          {columnBanner}
          {notice ? (
            <Text style={[textStyle('body', palette.danger), { padding: space[3] }]}>{notice}</Text>
          ) : null}
          {galleryPane}
        </View>
        {wideChat ? (
          <GalleryChatResizeHandle value={chatWidth} rowWidth={rowWidth} onChange={applyChatWidth} />
        ) : null}
        {narrow ? (
          <GalleryChatDock
            detent={detent}
            columnHeight={Math.max(0, columnHeight - keyboardInset)}
            onDetentChange={(next) => applyDetent(next)}
          >
            {chatColumn}
          </GalleryChatDock>
        ) : (
          chatColumn
        )}
      </View>
      {skillsOpen && detail != null ? (
        <WorkspaceSkills
          workspaceId={workspaceId}
          skills={detail.skills}
          onChange={(skills) => setDetail((value) => (value == null ? value : { ...value, skills }))}
          onClose={() => setSkillsOpen(false)}
        />
      ) : null}
      {confirmImage != null ? (
        <Modal label="Delete image" closeLabel="Close delete image" onClose={() => setConfirmImage(null)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {confirmImage.split('/').pop() ?? confirmImage}?
          </Text>
          <Text style={textStyle('body', palette.textSecondary)}>The file moves to the Trash.</Text>
          <IconButton
            label={`Move ${confirmImage.split('/').pop() ?? confirmImage} to Trash`}
            icon="trash-2"
            tone="danger"
            disabled={removingImage}
            onPress={() => {
              const path = confirmImage;
              setRemovingImage(true);
              void deleteImage(path)
                .then(() => setConfirmImage(null))
                .catch((caught: Error) => {
                  setConfirmImage(null);
                  setNotice(caught.message);
                })
                .finally(() => setRemovingImage(false));
            }}
          />
        </Modal>
      ) : null}
      {confirmDelete ? (
        <Modal label="Delete workspace" closeLabel="Close delete workspace" onClose={() => setConfirmDelete(false)}>
          <Text accessibilityRole="header" aria-level={2} style={textStyle('title', palette.textPrimary)}>
            Delete {detail?.name ?? 'workspace'}?
          </Text>
          <Text style={textStyle('body', palette.textSecondary)}>Folders and files on disk stay.</Text>
          <IconButton
            label={`Delete ${detail?.name ?? 'workspace'}`}
            icon="trash-2"
            tone="danger"
            onPress={() => {
              void api(`/workspaces/${workspaceId}`, { method: 'DELETE' })
                .then(() => {
                  onDeleted?.();
                  onBack?.();
                })
                .catch((caught: Error) => setNotice(caught.message));
            }}
          />
        </Modal>
      ) : null}
    </View>
  );
}

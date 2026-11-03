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
import { BannerPortal } from '@/components/Frame';
import { Composer } from '@/components/Composer';
import { ReplyAlert } from '@/components/ReplyAlert';
import { IconButton } from '@/components/IconButton';
import { Modal } from '@/components/Modal';
import { useAguiChat } from '@/components/use-agui-chat';
import { WorkspaceSkills, type WorkspaceSkill } from '@/components/WorkspaceSkills';
import { readExpandTrace, workingLabel } from '@/components/expandTrace';
import { WorkingStatus } from '@/components/WorkingStatus';
import { useTheme } from '@/theme/Theme';
import { space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';
import { FolderSetup } from '@/workspace/FolderSetup';
import { GalleryChatResizeHandle } from '@/workspace/GalleryChatResizeHandle';
import {
  clampGalleryChatWidth,
  readGalleryChatWidth,
  writeGalleryChatWidth,
} from '@/workspace/gallery-chat-width';
import { WorkspaceShell } from '@/workspace/WorkspaceShell';
import { ensureFilmstripSelection, initialGallery, type PlacementState } from '@/workspace/placement';

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  session_id: string;
  channel_id: string;
  skills: WorkspaceSkill[];
  expand_trace?: string | null;
};

type Snapshot = {
  messages: { id: string; role: string; body: string; reasoning?: string | null; tool_calls?: unknown; trace?: unknown }[];
  runs: { id: string; status: string; error: string | null }[];
  cursor: number;
  state?: PlacementState;
};

const LATEST_SLACK = 64;

export function WorkspaceScreen({
  workspaceId,
  agentName,
  onBack,
  onDeleted,
}: {
  workspaceId: string;
  agentName: string;
  onBack: () => void;
  onDeleted?: () => void;
}) {
  const { palette } = useTheme();
  const { width, height } = useWindowDimensions();
  const narrow = width < 1024;
  const [pane, setPane] = useState<'workspace' | 'chat'>(narrow ? 'workspace' : 'chat');
  const [rowWidth, setRowWidth] = useState(width);
  const [chatWidth, setChatWidth] = useState(() => readGalleryChatWidth(width));
  const rowWidthRef = useRef(rowWidth);
  rowWidthRef.current = rowWidth;
  useEffect(() => {
    if (narrow) {
      setPane('workspace');
    }
  }, [narrow]);
  useEffect(() => {
    setChatWidth((current) => clampGalleryChatWidth(current, rowWidth));
  }, [rowWidth]);
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
  const [status, setStatus] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [placement, setPlacement] = useState<PlacementState>(initialGallery());
  const [bound, setBound] = useState(false);
  const transcriptRef = useRef<ComponentRef<typeof ScrollView>>(null);
  const followLatest = useRef(true);
  const placedLatest = useRef(false);
  const openRun = useRef<string | null>(null);
  const cursorRef = useRef(0);
  const seen = useRef(new Set<string>());
  const sessionId = detail?.session_id ?? null;

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
        setPlacement(ensureFilmstripSelection(next as PlacementState));
      }
    },
    onSessionId: (nextId) => {
      setDetail((value) => (value == null ? value : { ...value, session_id: nextId }));
      setPlacement(initialGallery());
    },
    hydrate: async () => {
      if (sessionId == null) {
        return { messages: [] };
      }
      const snap = await api<Snapshot>(`/sessions/${sessionId}`);
      if (snap.state?.favlo_workspace) {
        const placed = ensureFilmstripSelection(snap.state);
        setPlacement(placed);
        return {
          messages: snap.messages.map((item) => chatMessage(item)),
          state: placed as unknown as Record<string, unknown>,
        };
      }
      setPlacement(initialGallery());
      return {
        messages: snap.messages.map((item) => chatMessage(item)),
        state: initialGallery() as unknown as Record<string, unknown>,
      };
    },
  });

  const load = useCallback(async () => {
    const body = await api<{ workspace: WorkspaceDetail }>(`/workspaces/${workspaceId}`);
    const workspace = { ...body.workspace, skills: body.workspace.skills ?? [] };
    const listed = await api<{ folders: { id: string; kind: string }[] }>(`/workspaces/${workspaceId}/folders`);
    setBound((listed.folders ?? []).some((item) => item.kind === 'local'));
    const snap = await api<Snapshot>(`/sessions/${workspace.session_id}`);
    cursorRef.current = snap.cursor;
    setMessages(snap.messages.map((item) => chatMessage(item)));
    if (snap.state?.favlo_workspace) {
      setPlacement(ensureFilmstripSelection(snap.state));
    } else {
      setPlacement(initialGallery());
    }
    setDraft(readDraft(workspace.session_id));
    // Set detail last so the socket effect subscribes with the loaded cursor.
    setDetail(workspace);
  }, [workspaceId]);

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
            session_id?: string;
            previous_session_id?: string;
            trace?: unknown;
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
          setPlacement(initialGallery());
          if (payload.payload?.session_id) {
            setDetail((value) =>
              value == null ? value : { ...value, session_id: payload.payload!.session_id! },
            );
          }
        }
        if (payload.type === 'run.started' && payload.run_id !== openRun.current) {
          setWorking(true);
        }
        if (
          payload.type === 'run.completed' ||
          (payload.type === 'message.created' &&
            payload.payload?.role === 'assistant' &&
            showsMessage(payload.payload.text, payload.payload.reasoning, payload.payload.tool_calls))
        ) {
          void load();
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
  }, [load, sessionId]);

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
        setPlacement(initialGallery());
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

  function fileUrl(path: string) {
    return `${API_URL}/workspaces/${workspaceId}/file?path=${encodeURIComponent(path)}`;
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
        minHeight: 0,
        padding: space[4],
        gap: space[3],
        overflow: 'hidden',
      }}
    >
      <ScrollView
        ref={transcriptRef}
        testID="transcript"
        style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, minWidth: 0, maxWidth: '100%' }}
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
            expandTrace={readExpandTrace(detail?.expand_trace)}
          />
        ))}
        <LiveReply
          steps={steps}
          speaker={agentName}
          measured
          expandTrace={readExpandTrace(detail?.expand_trace)}
          forceOpen={traceForcedOpen}
        />
      </ScrollView>
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
        onChange={updateDraft}
        onSend={() => void send()}
        onStop={() => {
          void chat.stop();
        }}
        working={working}
      />
    </View>
  );

  return (
    <View accessibilityLabel="Gallery workspace" style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
      <BannerPortal>
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            gap: space[2],
            flexShrink: 1,
            minWidth: 0,
            maxWidth: '100%',
          }}
        >
          <IconButton
            label="Back to channel"
            icon="arrow-left"
            onPress={onBack}
            iconOnly={narrow}
            background="transparent"
            border="transparent"
            color={palette.accent}
          />
          <Text
            accessibilityRole="header"
            aria-level={2}
            numberOfLines={1}
            style={[textStyle('title', palette.textPrimary), { flexShrink: 1, minWidth: 0 }]}
          >
            {detail?.name ?? 'Workspace'}
          </Text>
          {narrow ? null : (
            <Text style={textStyle('caption', palette.textSecondary)}>Gallery</Text>
          )}
          {narrow ? null : <View style={{ flex: 1 }} />}
          {narrow ? (
            <IconButton
              label={pane === 'chat' ? 'Back to workspace' : 'Chat'}
              icon={pane === 'chat' ? 'image' : 'message-circle'}
              onPress={() => setPane((current) => (current === 'chat' ? 'workspace' : 'chat'))}
              iconOnly
              background="transparent"
              border="transparent"
              color={palette.accent}
            />
          ) : null}
          <IconButton
            label="Skills"
            icon="book"
            onPress={() => setSkillsOpen(true)}
            iconOnly={narrow}
            background="transparent"
            border="transparent"
            color={palette.textPrimary}
          />
          <IconButton
            label="Delete workspace"
            icon="trash-2"
            onPress={() => setConfirmDelete(true)}
            iconOnly={narrow}
            background="transparent"
            border="transparent"
            color={palette.textPrimary}
          />
        </View>
      </BannerPortal>
      {notice ? (
        <Text style={[textStyle('body', palette.danger), { padding: space[3] }]}>{notice}</Text>
      ) : null}
      <View
        style={{ flexGrow: 1, flexShrink: 1, flexBasis: 0, minHeight: 0, minWidth: 0, flexDirection: narrow ? 'column' : 'row' }}
        onLayout={(event) => setRowWidth(event.nativeEvent.layout.width)}
      >
        {(!narrow || pane === 'workspace') && (
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
                fileUrl={fileUrl}
                onChange={setPlacement}
                onAddToChat={(path) => updateDraft(draft ? `${draft}\n${path}` : path)}
              />
            )}
          </View>
        )}
        {!narrow && Platform.OS === 'web' ? (
          <GalleryChatResizeHandle value={chatWidth} rowWidth={rowWidth} onChange={applyChatWidth} />
        ) : null}
        {(!narrow || pane === 'chat') && chatColumn}
      </View>
      {skillsOpen && detail != null ? (
        <WorkspaceSkills
          workspaceId={workspaceId}
          skills={detail.skills}
          onChange={(skills) => setDetail((value) => (value == null ? value : { ...value, skills }))}
          onClose={() => setSkillsOpen(false)}
        />
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
                  onBack();
                })
                .catch((caught: Error) => setNotice(caught.message));
            }}
          />
        </Modal>
      ) : null}
    </View>
  );
}

import { useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { Platform } from 'react-native';

import { api } from '@/api/client';
import { CodingWorkspace } from '@/components/CodingWorkspace';
import { Frame } from '@/components/Frame';
import { isGalleryScheme, type GalleryScheme } from '@/theme/gallery-schemes';
import { GallerySchemeProvider } from '@/theme/Theme';
import { WorkspaceCreating } from '@/workspace/WorkspaceCreating';
import { WorkspaceGone } from '@/workspace/WorkspaceGone';
import { WorkspaceScreen } from '@/workspace/WorkspaceScreen';

type WorkspaceDetail = {
  id: string;
  name: string;
  type: string;
  channel_id: string;
  theme?: string;
  renderer?: string;
  provisioning_state?: string;
};

export default function WorkspaceRoute() {
  const params = useLocalSearchParams<{ id: string }>();
  const workspaceId = Array.isArray(params.id) ? params.id[0] : params.id;
  const [detail, setDetail] = useState<WorkspaceDetail | null>(null);
  const [agentName, setAgentName] = useState('Agent');
  const [missing, setMissing] = useState(false);
  const [gone, setGone] = useState(false);
  const [scheme, setScheme] = useState<GalleryScheme>('rose-pine');

  useEffect(() => {
    if (workspaceId == null || workspaceId === '') {
      setMissing(true);
      return;
    }
    let cancel = false;
    let timer: ReturnType<typeof setInterval> | undefined;
    const load = async () => {
      try {
        const body = await api<{ workspace: WorkspaceDetail }>(`/workspaces/${workspaceId}`);
        if (cancel) {
          return;
        }
        if (isGalleryScheme(body.workspace.theme)) {
          setScheme(body.workspace.theme);
        }
        setDetail(body.workspace);
        const listed = await api<{ channels: { id: string; name: string }[] }>('/channels');
        const channel = listed.channels.find((item) => item.id === body.workspace.channel_id);
        if (!cancel && channel != null) {
          setAgentName(channel.name);
        }
        const state = body.workspace.provisioning_state;
        if (state !== 'new' && state !== 'provisioning' && timer != null) {
          clearInterval(timer);
          timer = undefined;
        }
      } catch {
        if (!cancel) {
          setMissing(true);
          if (timer != null) {
            clearInterval(timer);
            timer = undefined;
          }
        }
      }
    };
    void load();
    timer = setInterval(() => {
      void load();
    }, 1000);
    return () => {
      cancel = true;
      if (timer != null) {
        clearInterval(timer);
      }
    };
  }, [workspaceId]);

  const closed = gone || missing || workspaceId == null;
  const title = closed ? 'This workspace is gone' : (detail?.name ?? 'Workspace');
  const page = (
    <Frame
      brand={false}
      title={title}
      link={{
        label: 'Favlo',
        onPress: () => {
          if (Platform.OS === 'web') {
            window.open('/', '_blank', 'noopener');
          }
        },
      }}
    >
      {closed ? (
        <WorkspaceGone />
      ) : detail == null ? null : detail.provisioning_state === 'new' || detail.provisioning_state === 'provisioning' ? (
        <WorkspaceCreating name={detail.name} />
      ) : detail.renderer === 'chat' ? (
        <CodingWorkspace
          workspaceId={workspaceId}
          agentName={agentName}
          onDeleted={() => setGone(true)}
        />
      ) : (
        <WorkspaceScreen
          workspaceId={workspaceId}
          agentName={agentName}
          scheme={scheme}
          onTheme={setScheme}
          onDeleted={() => setGone(true)}
        />
      )}
    </Frame>
  );
  if (detail?.renderer === 'media' && !gone && !missing) {
    return <GallerySchemeProvider scheme={scheme}>{page}</GallerySchemeProvider>;
  }
  return page;
}

import { Picker } from '@react-native-picker/picker';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Platform, ScrollView, Switch, Text, TextInput, useWindowDimensions, View } from 'react-native';

import { api, idempotencyKey } from '@/api/client';
import { Confirmation } from '@/components/Confirmation';
import { FeatherIcon } from '@/components/FeatherIcon';
import { Frame } from '@/components/Frame';
import { MenuRow } from '@/components/MenuRow';
import { buttonRow, IconButton } from '@/components/IconButton';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { SkillForm, type SkillFields } from '@/components/SkillForm';
import { useTheme } from '@/theme/Theme';
import { radius } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type Model = {
  id: string;
  provider_id: string;
  model_id: string;
  display_name: string;
  enabled: boolean;
  revision: number;
};
export type Provider = { id: string; name: string; base_url: string; models: Model[] };
export type Settings = {
  default_model_id: string | null;
  exchange_budget: number;
  revision: number;
};
export type Agent = {
  id: string;
  name: string;
  persona: string;
  default_model_id: string | null;
  revision: number;
  tool_groups: string[];
};
export type ToolGroup = { id: string; name: string };
type Heartbeat = {
  id: string;
  agent_id: string;
  cron_expression: string;
  timezone: string;
  instructions: string;
  enabled: boolean;
  revision: number;
};
type McpTool = {
  name: string;
  description: string;
  schema: Record<string, unknown>;
  group_id: string;
  group_name: string;
};
type McpGroup = { id: string; name: string; enabled: boolean; tools: string[] };
type McpServer = {
  id: string;
  agent_id: string;
  name: string;
  transport: 'http' | 'stdio';
  command: string;
  url: string;
  enabled: boolean;
  status: string;
  status_message: string;
  revision: number;
  groups: McpGroup[];
};
type Section = 'providers' | 'agents' | 'heartbeats' | 'skills' | 'mcp' | 'orchestration';
type AgentSkill = {
  id: string;
  agent_id: string;
  name: string;
  description: string;
  definition: string;
  revision: number;
};
type GlobalSkill = {
  id: string;
  name: string;
  description: string;
  definition: string;
  add_by_default: boolean;
  revision: number;
};
type WorkspaceOption = { id: string; name: string };
type DrillPane = 'agents' | 'list' | 'detail';

const SECTIONS: { id: Section; label: string; catalog: boolean }[] = [
  { id: 'providers', label: 'Providers', catalog: true },
  { id: 'agents', label: 'Agents', catalog: true },
  { id: 'heartbeats', label: 'Heartbeats', catalog: false },
  { id: 'skills', label: 'Skills', catalog: false },
  { id: 'mcp', label: 'MCP', catalog: false },
  { id: 'orchestration', label: 'Orchestration', catalog: false },
];

export function SettingsScreen() {
  const router = useRouter();
  const { palette } = useTheme();
  const { width } = useWindowDimensions();
  const wide = width >= 1024;
  const drawerWidth = width < 600 ? width : 360;
  const [providers, setProviders] = useState<Provider[]>([]);
  const [providerId, setProviderId] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [agentId, setAgentId] = useState<string | null>(null);
  const [section, setSection] = useState<Section>('agents');
  const [budget, setBudget] = useState('10');
  const [status, setStatus] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelFocused, setModelFocused] = useState(false);
  const [drawer, setDrawer] = useState<'sections' | null>(null);
  const [showDetail, setShowDetail] = useState(false);
  const [creating, setCreating] = useState(false);
  const [heartbeats, setHeartbeats] = useState<Heartbeat[]>([]);
  const [heartbeatId, setHeartbeatId] = useState<string | null>(null);
  const [creatingHeartbeat, setCreatingHeartbeat] = useState(false);
  const [createToken, setCreateToken] = useState(0);
  const [heartbeatPane, setHeartbeatPane] = useState<DrillPane>('agents');
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [mcpId, setMcpId] = useState<string | null>(null);
  const [creatingMcp, setCreatingMcp] = useState(false);
  const [mcpToken, setMcpToken] = useState(0);
  const [mcpPane, setMcpPane] = useState<DrillPane>('agents');
  const [agentSkills, setAgentSkills] = useState<AgentSkill[]>([]);
  const [globalSkills, setGlobalSkills] = useState<GlobalSkill[]>([]);
  const [workspaces, setWorkspaces] = useState<WorkspaceOption[]>([]);
  const [globalSkillId, setGlobalSkillId] = useState<string | null>(null);
  const [creatingGlobal, setCreatingGlobal] = useState(false);
  const [globalToken, setGlobalToken] = useState(0);
  const [globalPane, setGlobalPane] = useState<'list' | 'detail'>('list');
  const [agentSkillId, setAgentSkillId] = useState<string | null>(null);
  const [creatingAgentSkill, setCreatingAgentSkill] = useState(false);
  const [agentSkillToken, setAgentSkillToken] = useState(0);

  async function reload(
    keepHeartbeat?: string,
    keepMcp?: string,
    keepAgentSkill?: string,
    keepGlobal?: string,
  ) {
    const [
      listedProviders,
      nextSettings,
      nextAgents,
      listedGroups,
      listedHeartbeats,
      listedMcp,
      listedSkills,
      listedGlobal,
      listedWorkspaces,
    ] = await Promise.all([
      api<{ providers: Provider[] }>('/providers'),
      api<Settings>('/settings'),
      api<{ agents: Agent[] }>('/agents'),
      api<{ groups: ToolGroup[] }>('/tool-groups'),
      api<{ heartbeats: Heartbeat[] }>('/heartbeats'),
      api<{ servers: McpServer[] }>('/mcp-servers'),
      api<{ skills: AgentSkill[] }>('/agent-skills'),
      api<{ skills: GlobalSkill[] }>('/global-skills'),
      api<{ workspaces: WorkspaceOption[] }>('/workspaces'),
    ]);
    const providerRows = listedProviders.providers;
    const agentRows = nextAgents.agents;
    setProviders(providerRows);
    setProviderId((current) => {
      if (current != null && providerRows.some((row) => row.id === current)) {
        return current;
      }
      return providerRows.find((row) => row.id === 'provider_local')?.id ?? providerRows[0]?.id ?? null;
    });
    setAgents(agentRows);
    setGroups(listedGroups.groups);
    setHeartbeats(listedHeartbeats.heartbeats);
    setAgentSkills(listedSkills.skills);
    setGlobalSkills(listedGlobal.skills);
    setWorkspaces(listedWorkspaces.workspaces);
    setMcpServers(listedMcp.servers);
    setAgentSkillId((current) => {
      const chosen = keepAgentSkill ?? current;
      const row = listedSkills.skills.find((item) => item.id === chosen);
      return row == null ? null : row.id;
    });
    setGlobalSkillId((current) => {
      const chosen = keepGlobal ?? current;
      const row = listedGlobal.skills.find((item) => item.id === chosen);
      return row == null ? null : row.id;
    });
    setMcpId((current) => {
      const chosen = keepMcp ?? current;
      const row = listedMcp.servers.find((item) => item.id === chosen);
      return row == null ? null : row.id;
    });
    setHeartbeatId((current) => {
      const chosen = keepHeartbeat ?? current;
      const row = listedHeartbeats.heartbeats.find((item) => item.id === chosen);
      return row == null ? null : row.id;
    });
    setAgentId((current) => {
      if (current != null && agentRows.some((row) => row.id === current)) {
        return current;
      }
      return agentRows.find((row) => row.id === 'agent_favlo')?.id ?? agentRows[0]?.id ?? null;
    });
    setSettings(nextSettings);
    setBudget(String(nextSettings.exchange_budget));
  }

  useEffect(() => {
    void reload().catch((caught: Error) => setError(caught.message));
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDrawer(null);
        setShowDetail(false);
        setCreating(false);
        setCreatingHeartbeat(false);
        setCreatingMcp(false);
        setCreatingGlobal(false);
        setCreatingAgentSkill(false);
        setHeartbeatPane('agents');
        setMcpPane('agents');
        setGlobalPane('list');
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const provider = providers.find((item) => item.id === providerId) ?? null;
  const agent = agents.find((item) => item.id === agentId) ?? null;
  const hosted = (provider?.models ?? []).filter((model) => model.enabled);
  const selectedModel = hosted.find((model) => model.id === settings?.default_model_id)?.model_id ?? '';
  const current = SECTIONS.find((item) => item.id === section) ?? SECTIONS[0];
  const catalog = current.catalog;
  const showSections = wide || drawer === 'sections';
  const special = section === 'heartbeats' || section === 'mcp' || section === 'skills';
  const showList = !special && catalog && (wide || !showDetail);
  const showDetails = !special && (!catalog || wide || showDetail);
  const heartbeatAgents = section === 'heartbeats' && (wide || heartbeatPane === 'agents');
  const heartbeatList = section === 'heartbeats' && (wide || heartbeatPane === 'list');
  const heartbeatDetail = section === 'heartbeats' && (wide || heartbeatPane === 'detail');
  const agentHeartbeats = heartbeats.filter((row) => row.agent_id === agentId);
  const heartbeat = agentHeartbeats.find((row) => row.id === heartbeatId) ?? null;
  const mcpAgents = section === 'mcp' && (wide || mcpPane === 'agents');
  const mcpList = section === 'mcp' && (wide || mcpPane === 'list');
  const mcpDetail = section === 'mcp' && (wide || mcpPane === 'detail');
  const agentServers = mcpServers.filter((row) => row.agent_id === agentId);
  const mcpServer = agentServers.find((row) => row.id === mcpId) ?? null;
  const skillList = section === 'skills' && (wide || globalPane === 'list');
  const skillDetail = section === 'skills' && (wide || globalPane === 'detail');
  const skillsForAgent = agentSkills.filter((row) => row.agent_id === agentId);
  const agentSkill = skillsForAgent.find((row) => row.id === agentSkillId) ?? null;
  const globalSkill = globalSkills.find((row) => row.id === globalSkillId) ?? null;

  function chooseSection(next: Section) {
    setSection(next);
    setCreating(false);
    setCreatingHeartbeat(false);
    setCreatingMcp(false);
    setCreatingGlobal(false);
    setCreatingAgentSkill(false);
    setShowDetail(false);
    setHeartbeatPane('agents');
    setMcpPane('agents');
    setGlobalPane('list');
    setDrawer(null);
  }

  function startGlobalSkill() {
    setCreatingGlobal(true);
    setGlobalSkillId(null);
    setGlobalToken((current) => current + 1);
    setGlobalPane('detail');
    setDrawer(null);
  }

  function startAgentSkill() {
    setCreatingAgentSkill(true);
    setAgentSkillId(null);
    setAgentSkillToken((current) => current + 1);
  }

  async function saveAgentSkill(fields: SkillFields) {
    if (agent == null) {
      return;
    }
    if (creatingAgentSkill || agentSkill == null) {
      const saved = await api<AgentSkill>('/agent-skills', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify({ ...fields, agent_id: agent.id }),
      });
      setCreatingAgentSkill(false);
      setAgentSkillId(saved.id);
      setSaved('The skill was created.');
      await reload(undefined, undefined, saved.id);
      return;
    }
    await api(`/agent-skills/${agentSkill.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ ...fields, expected_revision: agentSkill.revision }),
    });
    setSaved('The skill was saved.');
    await reload(undefined, undefined, agentSkill.id);
  }

  async function deleteAgentSkill() {
    if (agentSkill == null) {
      return;
    }
    await api(`/agent-skills/${agentSkill.id}`, { method: 'DELETE' });
    setCreatingAgentSkill(false);
    setAgentSkillId(null);
    setSaved('The skill was deleted.');
    await reload();
  }

  async function saveGlobalSkill(fields: SkillFields) {
    if (creatingGlobal || globalSkill == null) {
      const saved = await api<GlobalSkill>('/global-skills', {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify(fields),
      });
      setCreatingGlobal(false);
      setGlobalSkillId(saved.id);
      setGlobalPane('detail');
      setSaved('The skill was created.');
      await reload(undefined, undefined, undefined, saved.id);
      return;
    }
    await api(`/global-skills/${globalSkill.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ ...fields, expected_revision: globalSkill.revision }),
    });
    setSaved('The skill was saved.');
    await reload(undefined, undefined, undefined, globalSkill.id);
  }

  async function deleteGlobalSkill() {
    if (globalSkill == null) {
      return;
    }
    await api(`/global-skills/${globalSkill.id}`, { method: 'DELETE' });
    setCreatingGlobal(false);
    setGlobalSkillId(null);
    setGlobalPane('list');
    setSaved('The skill was deleted.');
    await reload();
  }

  function startMcp() {
    setCreatingMcp(true);
    setMcpId(null);
    setMcpPane('detail');
    setMcpToken((value) => value + 1);
    setDrawer(null);
  }

  function startHeartbeat() {
    setCreatingHeartbeat(true);
    setHeartbeatId(null);
    setHeartbeatPane('detail');
    setCreateToken((value) => value + 1);
    setDrawer(null);
  }

  async function refreshModels() {
    if (provider == null) {
      return;
    }
    try {
      const listed = await api<{ models: Model[]; message?: string }>(`/providers/${provider.id}/refresh`, {
        method: 'POST',
      });
      setProviders((rows) => rows.map((item) => (item.id === provider.id ? { ...item, models: listed.models } : item)));
      setStatus(listed.message ?? null);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The models were not refreshed.');
    }
  }

  async function testConnection() {
    if (provider == null) {
      return;
    }
    const result = await api<{ message: string }>(`/providers/${provider.id}/test`, { method: 'POST' });
    setStatus(result.message);
  }

  async function chooseDefault(modelId: string) {
    if (settings == null || provider == null || modelId === '' || modelId === selectedModel) {
      return;
    }
    try {
      const next = await api<Settings>(`/providers/${provider.id}/models`, {
        method: 'POST',
        body: JSON.stringify({ model_id: modelId, expected_revision: settings.revision }),
      });
      setSettings(next);
      setError(null);
      await reload();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The model was not selected.');
    }
  }

  async function saveBudget() {
    if (settings == null) {
      return;
    }
    const value = Number(budget);
    if (!Number.isInteger(value) || value < 1) {
      setError('The exchange budget is a positive integer.');
      return;
    }
    try {
      const next = await api<Settings>('/settings', {
        method: 'PATCH',
        body: JSON.stringify({
          expected_revision: settings.revision,
          default_model_id: settings.default_model_id,
          exchange_budget: value,
        }),
      });
      setSettings(next);
      setSaved('Maximum turns saved.');
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Settings were not saved.');
    }
  }

  function rememberAgent(row: Agent) {
    setCreating(false);
    setAgentId(row.id);
    setShowDetail(true);
    void reload();
  }

  function forgetAgent(row: Agent) {
    setCreating(false);
    setSaved(`${row.name} was deleted.`);
    setError(null);
    setAgentId((current) => (current === row.id ? null : current));
    void reload();
  }

  const field = {
    ...textStyle('body', palette.textPrimary),
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: radius.control,
    paddingHorizontal: 12,
    backgroundColor: palette.surface,
  };

  return (
    <Frame title="Configuration · Favlo" link={{ label: 'Chat', onPress: () => router.push('/') }}>
      <View style={{ flex: 1, minWidth: 0, maxWidth: '100%' }}>
        {!wide ? (
          <View style={[buttonRow, { padding: 8 }]}>
            <IconButton icon="layout" label="Sections" onPress={() => setDrawer('sections')} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
          </View>
        ) : null}
        <View style={{ flex: 1, flexDirection: 'row', minWidth: 0, maxWidth: '100%' }}>
          {showSections ? (
            <View
              accessibilityLabel="Configuration sections"
              style={{
                width: wide ? 220 : drawerWidth,
                borderRightWidth: 1,
                borderRightColor: palette.border,
                backgroundColor: palette.surface,
                padding: 12,
                gap: 8,
                ...(wide ? {} : { position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: 2 }),
              }}
            >
              <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>Configuration</Text>
              <View accessibilityRole="tablist" accessibilityLabel="Configuration" style={menuRows}>
                {SECTIONS.map((item) => (
                  <MenuRow
                    key={item.id}
                    role="tab"
                    label={item.label}
                    selected={item.id === section}
                    onPress={() => chooseSection(item.id)}
                  />
                ))}
              </View>
              {!wide ? (
                <IconButton icon="x" label="Close" accessibilityLabel="Close sections" onPress={() => setDrawer(null)} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
            </View>
          ) : null}
          {showList ? (
            <View
              accessibilityLabel={`${current.label} list`}
              style={{
                width: wide ? 280 : undefined,
                flex: wide ? undefined : 1,
                borderRightWidth: wide ? 1 : 0,
                borderRightColor: palette.border,
                backgroundColor: palette.surface,
                padding: 12,
                gap: 8,
              }}
            >
              <ScrollView contentContainerStyle={menuRows}>
                {section === 'agents' ? (
                  <MenuRow icon="user-plus" label="New agent" selected={creating} onPress={() => {
                    setCreating(true);
                    setAgentSkillId(null);
                    setCreatingAgentSkill(false);
                    setShowDetail(true);
                    setDrawer(null);
                  }} />
                ) : null}
                {(section === 'providers' ? providers : agents).map((item) => {
                  const selected = section === 'providers' ? item.id === providerId : item.id === agentId && !creating;
                  return (
                    <MenuRow
                      key={item.id}
                      label={item.name}
                      selected={selected}
                      onPress={() => {
                        if (section === 'providers') {
                          setProviderId(item.id);
                        } else {
                          setAgentId(item.id);
                          setCreating(false);
                          setAgentSkillId(null);
                          setCreatingAgentSkill(false);
                        }
                        setShowDetail(true);
                        setDrawer(null);
                      }}
                    />
                  );
                })}
              </ScrollView>
            </View>
          ) : null}
          {showDetails ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide && catalog ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to list" onPress={() => setShowDetail(false)} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
              {section === 'providers' && provider != null ? (
                <View style={{ gap: 12 }}>
                  <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>{provider.name}</Text>
                  <Text style={label(palette.textPrimary)}>Address</Text>
                  <Text style={{ color: palette.textSecondary, fontSize: 14 }}>{provider.base_url}</Text>
                  <View style={buttonRow}>
                    <IconButton icon="zap" label="Test connection" onPress={() => void testConnection()} background={palette.accent} color={palette.onAccent} />
                    <IconButton icon="refresh-cw" label="Refresh models" onPress={() => void refreshModels()} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
                  </View>
                  {status ? <Text accessibilityLiveRegion="polite" style={{ color: palette.textPrimary, fontSize: 14 }}>{status}</Text> : null}
                  <Text style={label(palette.textPrimary)}>Default model</Text>
                  <Picker
                    accessibilityLabel="Default model"
                    selectedValue={selectedModel}
                    onValueChange={(value) => void chooseDefault(String(value))}
                    onFocus={() => setModelFocused(true)}
                    onBlur={() => setModelFocused(false)}
                    style={pickerStyle(palette, modelFocused)}
                  >
                    <Picker.Item label="Choose a model" value="" color={palette.textSecondary} />
                    {hosted.map((model) => (
                      <Picker.Item key={model.id} label={model.display_name} value={model.model_id} color={palette.textPrimary} />
                    ))}
                  </Picker>
                </View>
              ) : null}
              {section === 'agents' ? (
                <View style={{ gap: 12 }}>
                  {creating ? (
                    <AgentForm
                      agent={null}
                      groups={groups}
                      providers={providers}
                      settings={settings}
                      onSuccess={setSaved}
                      onSaved={rememberAgent}
                    />
                  ) : agent != null ? (
                    <AgentForm
                      key={`${agent.id}:${agent.revision}`}
                      agent={agent}
                      groups={groups}
                      providers={providers}
                      settings={settings}
                      onSuccess={setSaved}
                      onSaved={rememberAgent}
                      onDeleted={forgetAgent}
                    />
                  ) : (
                    <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an agent.</Text>
                  )}
                  {!creating && agent != null ? (
                    <View accessibilityLabel="Agent skills" style={{ gap: 8 }}>
                      <Text style={label(palette.textPrimary)}>Skills</Text>
                      <MenuRow icon="plus" label="New skill" selected={creatingAgentSkill} onPress={startAgentSkill} />
                      {skillsForAgent.map((row) => (
                        <MenuRow
                          key={row.id}
                          label={row.name}
                          selected={row.id === agentSkillId && !creatingAgentSkill}
                          onPress={() => {
                            setAgentSkillId(row.id);
                            setCreatingAgentSkill(false);
                          }}
                        />
                      ))}
                      {skillsForAgent.length === 0 ? (
                        <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No skills</Text>
                      ) : null}
                      {creatingAgentSkill || agentSkill != null ? (
                        <SkillForm
                          key={creatingAgentSkill ? `new-${agentSkillToken}` : `${agentSkill?.id}:${agentSkill?.revision}`}
                          skill={creatingAgentSkill ? null : agentSkill}
                          onSave={saveAgentSkill}
                          onDelete={creatingAgentSkill ? undefined : deleteAgentSkill}
                        />
                      ) : null}
                    </View>
                  ) : null}
                </View>
              ) : null}
              {section === 'orchestration' ? (
                <View style={{ gap: 12 }}>
                  <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>Orchestration</Text>
                  <View style={{ gap: 8 }}>
                    <Text style={label(palette.textPrimary)}>Maximum turns</Text>
                    <TextInput accessibilityLabel="Maximum turns" value={budget} onChangeText={setBudget} keyboardType="number-pad" style={field} />
                  </View>
                  <IconButton icon="check" label="Save maximum turns" onPress={() => void saveBudget()} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
                </View>
              ) : null}
              {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
            </ScrollView>
          ) : null}
          {heartbeatAgents ? (
            <View accessibilityLabel="Agents list" style={columnStyle(palette.border, palette.surface, wide, wide ? 200 : undefined)}>
              {!wide ? null : <Text style={label(palette.textPrimary)}>Agents</Text>}
              <ScrollView contentContainerStyle={menuRows}>
                {agents.map((item) => (
                  <MenuRow
                    key={item.id}
                    label={item.name}
                    selected={item.id === agentId}
                    onPress={() => {
                      setAgentId(item.id);
                      setHeartbeatId(null);
                      setCreatingHeartbeat(false);
                      setHeartbeatPane('list');
                      setDrawer(null);
                    }}
                  />
                ))}
              </ScrollView>
            </View>
          ) : null}
          {heartbeatList ? (
            <View accessibilityLabel="Heartbeats list" style={columnStyle(palette.border, palette.surface, wide, wide ? 220 : undefined)}>
              {!wide ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to agents" onPress={() => setHeartbeatPane('agents')} background="transparent" border="transparent" color={palette.accent} />
              ) : (
                <Text style={label(palette.textPrimary)}>Heartbeats</Text>
              )}
              <ScrollView contentContainerStyle={menuRows}>
                {agent != null ? (
                  <MenuRow icon="plus" label="New heartbeat" selected={creatingHeartbeat} onPress={startHeartbeat} />
                ) : null}
                {agentHeartbeats.map((row) => (
                  <MenuRow
                    key={row.id}
                    label={row.cron_expression}
                    selected={row.id === heartbeatId && !creatingHeartbeat}
                    onPress={() => {
                      setHeartbeatId(row.id);
                      setCreatingHeartbeat(false);
                      setHeartbeatPane('detail');
                      setDrawer(null);
                    }}
                  />
                ))}
                {agentHeartbeats.length === 0 ? <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No heartbeats</Text> : null}
              </ScrollView>
            </View>
          ) : null}
          {heartbeatDetail ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to heartbeats" onPress={() => setHeartbeatPane('list')} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
              {agent == null ? (
                <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an agent.</Text>
              ) : (
                <View style={{ gap: 12 }}>
                  {creatingHeartbeat ? (
                    <HeartbeatForm
                      key={`new-${createToken}`}
                      agent={agent}
                      heartbeat={null}
                      onSuccess={setSaved}
                      onSaved={(row) => {
                        setCreatingHeartbeat(false);
                        void reload(row.id);
                      }}
                    />
                  ) : heartbeat != null ? (
                    <HeartbeatForm
                      key={`${heartbeat.id}:${heartbeat.revision}`}
                      agent={agent}
                      heartbeat={heartbeat}
                      onSuccess={setSaved}
                      onSaved={() => void reload()}
                    />
                  ) : (
                    <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose a heartbeat.</Text>
                  )}
                </View>
              )}
              {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
            </ScrollView>
          ) : null}
          {skillList ? (
            <View accessibilityLabel="Skills list" style={columnStyle(palette.border, palette.surface, wide, wide ? 220 : undefined)}>
              <Text style={label(palette.textPrimary)}>Skills</Text>
              <ScrollView contentContainerStyle={menuRows}>
                <MenuRow icon="plus" label="New skill" selected={creatingGlobal} onPress={startGlobalSkill} />
                {globalSkills.map((row) => (
                  <MenuRow
                    key={row.id}
                    label={row.name}
                    selected={row.id === globalSkillId && !creatingGlobal}
                    onPress={() => {
                      setGlobalSkillId(row.id);
                      setCreatingGlobal(false);
                      setGlobalPane('detail');
                      setDrawer(null);
                    }}
                  />
                ))}
                {globalSkills.length === 0 ? <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No skills</Text> : null}
              </ScrollView>
            </View>
          ) : null}
          {skillDetail ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to skills" onPress={() => setGlobalPane('list')} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
              <View style={{ gap: 12 }}>
                {creatingGlobal || globalSkill != null ? (
                  <SkillForm
                    key={creatingGlobal ? `new-${globalToken}` : `${globalSkill?.id}:${globalSkill?.revision}`}
                    skill={creatingGlobal ? null : globalSkill}
                    showDefault
                    onSave={saveGlobalSkill}
                    onDelete={creatingGlobal ? undefined : deleteGlobalSkill}
                  />
                ) : (
                  <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose a skill.</Text>
                )}
                {!creatingGlobal && globalSkill != null ? (
                  <GlobalCopy
                    key={globalSkill.id}
                    skillId={globalSkill.id}
                    agents={agents}
                    workspaces={workspaces}
                    onCopied={(message) => {
                      setSaved(message);
                      void reload(undefined, undefined, undefined, globalSkill.id);
                    }}
                  />
                ) : null}
              </View>
              {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
            </ScrollView>
          ) : null}
          {mcpAgents ? (
            <View accessibilityLabel="Agents list" style={columnStyle(palette.border, palette.surface, wide, wide ? 200 : undefined)}>
              {wide ? <Text style={label(palette.textPrimary)}>Agents</Text> : null}
              <ScrollView contentContainerStyle={menuRows}>
                {agents.map((item) => (
                  <MenuRow
                    key={item.id}
                    label={item.name}
                    selected={item.id === agentId}
                    onPress={() => {
                      setAgentId(item.id);
                      setMcpId(null);
                      setCreatingMcp(false);
                      setMcpPane('list');
                      setDrawer(null);
                    }}
                  />
                ))}
              </ScrollView>
            </View>
          ) : null}
          {mcpList ? (
            <View accessibilityLabel="MCP list" style={columnStyle(palette.border, palette.surface, wide, wide ? 220 : undefined)}>
              {!wide ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to agents" onPress={() => setMcpPane('agents')} background="transparent" border="transparent" color={palette.accent} />
              ) : (
                <Text style={label(palette.textPrimary)}>MCP</Text>
              )}
              <ScrollView contentContainerStyle={menuRows}>
                {agent != null ? (
                  <MenuRow icon="plus" label="New MCP" selected={creatingMcp} onPress={startMcp} />
                ) : null}
                {agentServers.map((row) => (
                  <MenuRow
                    key={row.id}
                    label={row.name}
                    selected={row.id === mcpId && !creatingMcp}
                    onPress={() => {
                      setMcpId(row.id);
                      setCreatingMcp(false);
                      setMcpPane('detail');
                      setDrawer(null);
                    }}
                  />
                ))}
                {agentServers.length === 0 ? <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No MCP servers</Text> : null}
              </ScrollView>
            </View>
          ) : null}
          {mcpDetail ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide ? (
                <IconButton icon="arrow-left" label="Back" accessibilityLabel="Back to MCP" onPress={() => setMcpPane('list')} background="transparent" border="transparent" color={palette.accent} />
              ) : null}
              {agent == null ? (
                <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an agent.</Text>
              ) : (
                <View style={{ gap: 12 }}>
                  {creatingMcp ? (
                    <McpForm
                      key={`new-mcp-${mcpToken}`}
                      agent={agent}
                      server={null}
                      onSuccess={setSaved}
                      onSaved={(row) => {
                        setCreatingMcp(false);
                        void reload(undefined, row.id);
                      }}
                    />
                  ) : mcpServer != null ? (
                    <McpForm
                      key={`${mcpServer.id}:${mcpServer.revision}`}
                      agent={agent}
                      server={mcpServer}
                      onSuccess={setSaved}
                      onSaved={(row) => void reload(undefined, row.id)}
                    />
                  ) : (
                    <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an MCP server.</Text>
                  )}
                </View>
              )}
            </ScrollView>
          ) : null}
        </View>
        {saved ? <Confirmation message={saved} onDismiss={() => setSaved(null)} /> : null}
      </View>
    </Frame>
  );
}

function McpForm({
  agent,
  server,
  onSuccess,
  onSaved,
}: {
  agent: Agent;
  server: McpServer | null;
  onSuccess: (message: string) => void;
  onSaved: (row: McpServer) => void;
}) {
  const { palette } = useTheme();
  const [name, setName] = useState(server?.name ?? '');
  const [transport, setTransport] = useState<'http' | 'stdio'>(server?.transport ?? 'http');
  const [url, setUrl] = useState(server?.url || (server == null ? 'http://127.0.0.1:8129/mcp' : ''));
  const [command, setCommand] = useState(server?.command ?? '');
  const [enabled, setEnabled] = useState(server?.enabled ?? true);
  const [groups, setGroups] = useState(server?.groups ?? []);
  const [notice, setNotice] = useState<string | null>(server?.status_message || null);
  const [error, setError] = useState<string | null>(null);
  const enabledBox = `Enabled for ${server?.name ?? 'MCP'}`;

  async function save() {
    const payload = {
      name,
      transport,
      command,
      url,
      enabled,
      groups: groups.map((group) => ({ id: group.id, name: group.name, enabled: group.enabled })),
    };
    try {
      const saved =
        server == null
          ? await api<McpServer>('/mcp-servers', {
              method: 'POST',
              headers: { 'Idempotency-Key': idempotencyKey() },
              body: JSON.stringify({ agent_id: agent.id, ...payload }),
            })
          : await api<McpServer>(`/mcp-servers/${server.id}`, {
              method: 'PATCH',
              body: JSON.stringify({ expected_revision: server.revision, ...payload }),
            });
      const label = saved.name.trim() || 'MCP server';
      onSuccess(server == null ? `${label} was created.` : `${label} was saved.`);
      onSaved(saved);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The MCP server was not saved.');
    }
  }

  async function test() {
    try {
      const tested = await api<{ ok: boolean; message: string; tools: McpTool[]; groups: McpGroup[]; url?: string }>(
        '/mcp-servers/test',
        { method: 'POST', body: JSON.stringify({ transport, command, url }) },
      );
      if (!tested.ok) {
        setError(tested.message);
        return;
      }
      setGroups((current) => {
        const prior = new Map(current.map((group) => [group.id, group.enabled]));
        return tested.groups.map((group) => ({ ...group, enabled: prior.get(group.id) ?? group.enabled }));
      });
      if (tested.url) {
        setUrl(tested.url);
      }
      setNotice(tested.message);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Connection failed');
    }
  }

  return (
    <View style={{ gap: 8 }}>
      <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>{server?.name ?? 'New MCP'}</Text>
      <Text style={label(palette.textPrimary)}>Name</Text>
      <TextInput accessibilityLabel="MCP name" value={name} onChangeText={setName} style={fieldStyle(palette)} />
      <Text style={label(palette.textPrimary)}>Transport</Text>
      <View accessibilityRole="radiogroup" accessibilityLabel="Transport" style={{ flexDirection: 'row', gap: 8 }}>
        {(['http', 'stdio'] as const).map((choice) => {
          const on = transport === choice;
          const box = choice === 'http' ? 'HTTP' : 'Stdio';
          return (
            <Pressable
              key={choice}
              accessibilityRole="radio"
              accessibilityLabel={box}
              accessibilityState={{ selected: on }}
              aria-checked={on}
              onPress={() => setTransport(choice)}
              style={[button(on ? palette.accentSubtle : palette.surface, palette.borderStrong), { flexDirection: 'row', alignItems: 'center', gap: 8 }]}
            >
              <FeatherIcon name={choice === 'http' ? 'globe' : 'terminal'} size={18} color={palette.textPrimary} />
              <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>{box}</Text>
            </Pressable>
          );
        })}
      </View>
      {transport === 'http' ? (
        <>
          <Text style={label(palette.textPrimary)}>URL</Text>
          <TextInput accessibilityLabel="URL" value={url} onChangeText={setUrl} autoCapitalize="none" style={fieldStyle(palette)} />
        </>
      ) : (
        <>
          <Text style={label(palette.textPrimary)}>Command</Text>
          <TextInput accessibilityLabel="Command" value={command} onChangeText={setCommand} autoCapitalize="none" style={fieldStyle(palette)} />
        </>
      )}
      <SettingSwitch label={enabledBox} value={enabled} onValueChange={setEnabled} />
      <IconButton icon="zap" label="Test connection" onPress={() => void test()} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
      {notice ? <Text accessibilityLiveRegion="polite" style={{ color: palette.textPrimary, fontSize: 14 }}>{notice}</Text> : null}
      {groups.length > 0 ? <Text style={label(palette.textPrimary)}>Tools</Text> : null}
      {groups.map((group) => {
        const box = `${group.name} for ${server?.name ?? name}`;
        return (
          <View key={group.id} style={{ gap: 4 }}>
            <SettingSwitch
              label={box}
              value={group.enabled}
              onValueChange={(next) =>
                setGroups((current) => current.map((item) => (item.id === group.id ? { ...item, enabled: next } : item)))
              }
            />
            {group.tools.map((tool) => (
              <Text key={tool} style={{ color: palette.textSecondary, fontSize: 14 }}>{tool}</Text>
            ))}
          </View>
        );
      })}
      <IconButton
        icon={server == null ? 'plus' : 'check'}
        label={server == null ? 'Create MCP' : `Save ${server.name}`}
        onPress={() => void save()}
        background={palette.accent}
        color={palette.onAccent}
      />
      {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
    </View>
  );
}

function modelRow(providers: Provider[], rowId: string | null): Model | undefined {
  if (rowId == null) {
    return undefined;
  }
  return providers.flatMap((provider) => provider.models).find((model) => model.id === rowId);
}

export function AgentForm({
  agent,
  groups,
  providers,
  settings,
  onSuccess,
  onSaved,
  onDeleted,
}: {
  agent: Agent | null;
  groups: ToolGroup[];
  providers: Provider[];
  settings: Settings | null;
  onSuccess: (message: string) => void;
  onSaved: (agent: Agent) => void;
  onDeleted?: (agent: Agent) => void;
}) {
  const { palette } = useTheme();
  const creating = agent == null;
  const current = modelRow(providers, agent?.default_model_id ?? settings?.default_model_id ?? null);
  const [name, setName] = useState(agent?.name ?? '');
  const [persona, setPersona] = useState(agent?.persona ?? '');
  const [providerId, setProviderId] = useState(
    current?.provider_id ?? providers.find((item) => item.id === 'provider_local')?.id ?? providers[0]?.id ?? '',
  );
  const hosted = (providers.find((item) => item.id === providerId)?.models ?? []).filter((model) => model.enabled);
  const [modelId, setModelId] = useState(
    current != null && current.provider_id === providerId ? current.model_id : hosted[0]?.model_id ?? '',
  );
  const [enabled, setEnabled] = useState(agent?.tool_groups ?? []);
  const [providerFocused, setProviderFocused] = useState(false);
  const [modelFocused, setModelFocused] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function chooseProvider(next: string) {
    setProviderId(next);
    const models = (providers.find((item) => item.id === next)?.models ?? []).filter((model) => model.enabled);
    setModelId(models.some((model) => model.model_id === modelId) ? modelId : models[0]?.model_id ?? '');
  }

  async function remove() {
    if (agent == null) {
      return;
    }
    try {
      await api(`/agents/${agent.id}`, { method: 'DELETE' });
      onSuccess(`${agent.name} was deleted.`);
      onDeleted?.(agent);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The agent was not deleted.');
    }
  }

  async function save() {
    const row = hosted.find((model) => model.model_id === modelId);
    const modelRowId = row?.id ?? null;
    try {
      if (agent == null) {
        const created = await api<{ agent: Agent }>('/agents', {
          method: 'POST',
          headers: { 'Idempotency-Key': idempotencyKey() },
          body: JSON.stringify({
            name,
            persona,
            default_model_id: modelRowId,
            tool_groups: enabled,
          }),
        });
        onSuccess(`${created.agent.name} was created.`);
        onSaved(created.agent);
        return;
      }
      const updated = await api<Agent>(`/agents/${agent.id}`, {
        method: 'PATCH',
        body: JSON.stringify({
          expected_revision: agent.revision,
          name,
          persona,
          default_model_id: modelRowId,
          tool_groups: enabled,
        }),
      });
      onSuccess(`${name.trim() || agent.name} was saved.`);
      onSaved(updated);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The agent was not saved.');
    }
  }

  const nameLabel = creating ? 'Agent name' : `Name for ${agent.name}`;
  const personaLabel = creating ? 'Persona' : `Persona for ${agent.name}`;
  const providerLabel = creating ? 'Provider' : `Provider for ${agent.name}`;
  const modelLabel = creating ? 'Model' : `Model for ${agent.name}`;

  return (
    <View style={{ gap: 8 }}>
      <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>{creating ? 'New agent' : agent.name}</Text>
      <Text style={label(palette.textPrimary)}>{creating ? 'Name' : nameLabel}</Text>
      <TextInput accessibilityLabel={nameLabel} value={name} onChangeText={setName} style={fieldStyle(palette)} />
      <Text style={label(palette.textPrimary)}>{creating ? 'Persona' : personaLabel}</Text>
      <TextInput accessibilityLabel={personaLabel} multiline value={persona} onChangeText={setPersona} style={[fieldStyle(palette), { minHeight: 88 }]} />
      <Text style={label(palette.textPrimary)}>{providerLabel}</Text>
      <Picker
        accessibilityLabel={providerLabel}
        selectedValue={providerId}
        onValueChange={(value) => chooseProvider(String(value))}
        onFocus={() => setProviderFocused(true)}
        onBlur={() => setProviderFocused(false)}
        style={pickerStyle(palette, providerFocused)}
      >
        {providers.map((item) => (
          <Picker.Item key={item.id} label={item.name} value={item.id} color={palette.textPrimary} />
        ))}
      </Picker>
      <Text style={label(palette.textPrimary)}>{modelLabel}</Text>
      <Picker
        accessibilityLabel={modelLabel}
        selectedValue={modelId}
        onValueChange={(value) => setModelId(String(value))}
        onFocus={() => setModelFocused(true)}
        onBlur={() => setModelFocused(false)}
        style={pickerStyle(palette, modelFocused)}
      >
        <Picker.Item label="Choose a model" value="" color={palette.textSecondary} />
        {hosted.map((model) => (
          <Picker.Item key={model.id} label={model.display_name} value={model.model_id} color={palette.textPrimary} />
        ))}
      </Picker>
      <Text style={label(palette.textPrimary)}>Tools</Text>
      {groups.map((group) => {
        const on = enabled.includes(group.id);
        const box = creating ? group.name : `${group.name} for ${agent.name}`;
        return (
          <SettingSwitch
            key={group.id}
            label={box}
            value={on}
            onValueChange={(next) =>
              setEnabled((current) => (next ? [...current, group.id] : current.filter((id) => id !== group.id)))
            }
          />
        );
      })}
      <View style={buttonRow}>
        <IconButton
          icon={creating ? 'plus' : 'check'}
          label={creating ? 'Create agent' : `Save ${agent.name}`}
          onPress={() => void save()}
          background={creating ? palette.accent : palette.surface}
          border={creating ? palette.accent : palette.borderStrong}
          color={creating ? palette.onAccent : palette.textPrimary}
        />
        {!creating && agent.id !== 'agent_favlo' ? (
          <IconButton
            icon="trash-2"
            label={`Delete ${agent.name}`}
            onPress={() => void remove()}
            background={palette.danger}
            color={palette.onAccent}
          />
        ) : null}
      </View>
      {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
    </View>
  );
}

function HeartbeatForm({
  agent,
  heartbeat,
  onSuccess,
  onSaved,
}: {
  agent: Agent;
  heartbeat: Heartbeat | null;
  onSuccess: (message: string) => void;
  onSaved: (row: Heartbeat) => void;
}) {
  const { palette } = useTheme();
  const [cron, setCron] = useState(heartbeat?.cron_expression ?? '');
  const [timezone, setTimezone] = useState(heartbeat?.timezone ?? machineTimezone());
  const [instructions, setInstructions] = useState(heartbeat?.instructions ?? '');
  const [enabled, setEnabled] = useState(heartbeat?.enabled ?? true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const box = `Enabled for ${agent.name}`;

  async function save() {
    const payload = {
      cron_expression: cron,
      timezone,
      instructions,
      enabled,
    };
    try {
      const saved =
        heartbeat == null
          ? await api<Heartbeat>('/heartbeats', {
              method: 'POST',
              headers: { 'Idempotency-Key': idempotencyKey() },
              body: JSON.stringify({ agent_id: agent.id, ...payload }),
            })
          : await api<Heartbeat>(`/heartbeats/${heartbeat.id}`, {
              method: 'PATCH',
              body: JSON.stringify({ expected_revision: heartbeat.revision, ...payload }),
            });
      onSuccess(heartbeat == null ? 'Heartbeat created.' : 'Heartbeat saved.');
      onSaved(saved);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The heartbeat was not saved.');
    }
  }

  async function runNow() {
    if (heartbeat == null) {
      return;
    }
    try {
      await api(`/heartbeats/${heartbeat.id}/run`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
      });
      setNotice('The heartbeat is running.');
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The heartbeat did not start.');
    }
  }

  return (
    <View style={{ gap: 8 }}>
      <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>
        {heartbeat?.cron_expression ?? 'New heartbeat'}
      </Text>
      <Text style={label(palette.textPrimary)}>Cron for {agent.name}</Text>
      <TextInput accessibilityLabel={`Cron for ${agent.name}`} value={cron} onChangeText={setCron} style={fieldStyle(palette)} />
      <Text style={label(palette.textPrimary)}>Timezone for {agent.name}</Text>
      <TextInput accessibilityLabel={`Timezone for ${agent.name}`} value={timezone} onChangeText={setTimezone} style={fieldStyle(palette)} />
      <Text style={label(palette.textPrimary)}>Instructions for {agent.name}</Text>
      <TextInput
        accessibilityLabel={`Instructions for ${agent.name}`}
        multiline
        value={instructions}
        onChangeText={setInstructions}
        style={[fieldStyle(palette), { minHeight: 88 }]}
      />
      <SettingSwitch label={box} value={enabled} onValueChange={setEnabled} />
      <View style={buttonRow}>
        <IconButton
          icon={heartbeat == null ? 'plus' : 'check'}
          label={heartbeat == null ? 'Create heartbeat' : 'Save heartbeat'}
          onPress={() => void save()}
          background={palette.accent}
          color={palette.onAccent}
        />
        {heartbeat != null ? (
          <IconButton icon="play" label="Run heartbeat" onPress={() => void runNow()} background={palette.surface} border={palette.borderStrong} color={palette.textPrimary} />
        ) : null}
      </View>
      {notice ? <Text accessibilityLiveRegion="polite" style={{ color: palette.textPrimary, fontSize: 14 }}>{notice}</Text> : null}
      {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
    </View>
  );
}

function GlobalCopy({
  skillId,
  agents,
  workspaces,
  onCopied,
}: {
  skillId: string;
  agents: Agent[];
  workspaces: WorkspaceOption[];
  onCopied: (message: string) => void;
}) {
  const { palette } = useTheme();
  const [agentId, setAgentId] = useState(
    agents.find((item) => item.id === 'agent_favlo')?.id ?? agents[0]?.id ?? '',
  );
  const [workspaceId, setWorkspaceId] = useState(workspaces[0]?.id ?? '');
  const [agentFocused, setAgentFocused] = useState(false);
  const [workspaceFocused, setWorkspaceFocused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const agent = agents.find((item) => item.id === agentId) ?? null;
  const workspace = workspaces.find((item) => item.id === workspaceId) ?? null;

  async function copyTo(body: { agent_id?: string; workspace_id?: string }, target: string) {
    try {
      await api(`/global-skills/${skillId}/copy`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey() },
        body: JSON.stringify(body),
      });
      setError(null);
      onCopied(`Copied to ${target}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The skill was not copied.');
    }
  }

  return (
    <View style={{ gap: 8 }}>
      <Text style={label(palette.textPrimary)}>Copy to an agent</Text>
      <Picker
        accessibilityLabel="Agent for copy"
        selectedValue={agentId}
        onValueChange={(value) => setAgentId(String(value))}
        onFocus={() => setAgentFocused(true)}
        onBlur={() => setAgentFocused(false)}
        style={pickerStyle(palette, agentFocused)}
      >
        {agents.map((item) => (
          <Picker.Item key={item.id} label={item.name} value={item.id} color={palette.textPrimary} />
        ))}
      </Picker>
      <IconButton
        icon="copy"
        label="Copy to agent"
        disabled={agent == null}
        onPress={() => void copyTo({ agent_id: agentId }, agent?.name ?? 'the agent')}
        background={agent == null ? palette.disabledSurface : palette.accent}
        color={agent == null ? palette.disabledText : palette.onAccent}
      />
      <Text style={label(palette.textPrimary)}>Copy to a workspace</Text>
      {workspaces.length === 0 ? (
        <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No workspaces</Text>
      ) : (
        <Picker
          accessibilityLabel="Workspace for copy"
          selectedValue={workspaceId}
          onValueChange={(value) => setWorkspaceId(String(value))}
          onFocus={() => setWorkspaceFocused(true)}
          onBlur={() => setWorkspaceFocused(false)}
          style={pickerStyle(palette, workspaceFocused)}
        >
          {workspaces.map((item) => (
            <Picker.Item key={item.id} label={item.name} value={item.id} color={palette.textPrimary} />
          ))}
        </Picker>
      )}
      <IconButton
        icon="copy"
        label="Copy to workspace"
        disabled={workspace == null}
        onPress={() => void copyTo({ workspace_id: workspaceId }, workspace?.name ?? 'the workspace')}
        background={workspace == null ? palette.disabledSurface : palette.surface}
        border={palette.borderStrong}
        color={workspace == null ? palette.disabledText : palette.textPrimary}
      />
      {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
    </View>
  );
}

function SettingSwitch({
  label,
  value,
  onValueChange,
}: {
  label: string;
  value: boolean;
  onValueChange: (value: boolean) => void;
}) {
  const { palette } = useTheme();
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12, minHeight: 44 }}>
      <Text onPress={() => onValueChange(!value)} style={{ color: palette.textPrimary, fontSize: 16, flex: 1 }}>
        {label}
      </Text>
      <Switch
        accessibilityLabel={label}
        accessibilityState={{ checked: value }}
        value={value}
        onValueChange={onValueChange}
        trackColor={{ false: palette.borderStrong, true: palette.accent }}
        thumbColor={palette.onAccent}
        ios_backgroundColor={palette.borderStrong}
      />
    </View>
  );
}

function machineTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  } catch {
    return 'UTC';
  }
}

function columnStyle(border: string, surface: string, wide: boolean, width: number | undefined) {
  return {
    width,
    flex: wide ? undefined : 1,
    borderRightWidth: wide ? 1 : 0,
    borderRightColor: border,
    backgroundColor: surface,
    padding: 12,
    gap: 8,
  };
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

function label(color: string) {
  return textStyle('label', color);
}

const menuRows = { gap: focusRingInset * 2, padding: focusRingInset };

function heading(color: string) {
  return textStyle('title', color);
}

function button(background: string, border?: string) {
  return {
    minHeight: 44,
    alignSelf: 'flex-start' as const,
    borderRadius: radius.control,
    paddingHorizontal: 16,
    justifyContent: 'center' as const,
    backgroundColor: background,
    borderWidth: 1,
    borderColor: border ?? background,
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

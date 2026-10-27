import { Picker } from '@react-native-picker/picker';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Platform, ScrollView, Switch, Text, TextInput, useWindowDimensions, View } from 'react-native';

import { api, idempotencyKey } from '@/api/client';
import { Confirmation } from '@/components/Confirmation';
import { Frame } from '@/components/Frame';
import { focusRingInset, Pressable } from '@/components/Pressable';
import { useTheme } from '@/theme/Theme';

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
type Section = 'providers' | 'agents' | 'heartbeats' | 'mcp' | 'orchestration';
type DrillPane = 'agents' | 'list' | 'detail';

const SECTIONS: { id: Section; label: string; catalog: boolean }[] = [
  { id: 'providers', label: 'Providers', catalog: true },
  { id: 'agents', label: 'Agents', catalog: true },
  { id: 'heartbeats', label: 'Heartbeats', catalog: false },
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

  async function reload(keepHeartbeat?: string, keepMcp?: string) {
    const [listedProviders, nextSettings, nextAgents, listedGroups, listedHeartbeats, listedMcp] = await Promise.all([
      api<{ providers: Provider[] }>('/providers'),
      api<Settings>('/settings'),
      api<{ agents: Agent[] }>('/agents'),
      api<{ groups: ToolGroup[] }>('/tool-groups'),
      api<{ heartbeats: Heartbeat[] }>('/heartbeats'),
      api<{ servers: McpServer[] }>('/mcp-servers'),
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
    setMcpServers(listedMcp.servers);
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
        setHeartbeatPane('agents');
        setMcpPane('agents');
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
  const special = section === 'heartbeats' || section === 'mcp';
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

  function chooseSection(next: Section) {
    setSection(next);
    setCreating(false);
    setCreatingHeartbeat(false);
    setCreatingMcp(false);
    setShowDetail(false);
    setHeartbeatPane('agents');
    setMcpPane('agents');
    setDrawer(null);
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
    minHeight: 44,
    borderWidth: 1,
    borderColor: palette.borderStrong,
    borderRadius: 6,
    paddingHorizontal: 12,
    color: palette.textPrimary,
    backgroundColor: palette.surface,
    fontSize: 16,
  };

  return (
    <Frame link={{ label: 'Chat', onPress: () => router.push('/') }}>
      <View style={{ flex: 1, minWidth: 0, maxWidth: '100%' }}>
        {!wide ? (
          <View style={{ flexDirection: 'row', gap: 8, padding: 8 }}>
            <Pressable accessibilityRole="button" accessibilityLabel="Sections" onPress={() => setDrawer('sections')} style={button(palette.surface)}>
              <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>Sections</Text>
            </Pressable>
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
                {SECTIONS.map((item) => {
                  const selected = item.id === section;
                  return (
                    <Pressable
                      key={item.id}
                      accessibilityRole="tab"
                      accessibilityLabel={item.label}
                      accessibilityState={{ selected }}
                      onPress={() => chooseSection(item.id)}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{item.label}</Text>
                    </Pressable>
                  );
                })}
              </View>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Close sections" onPress={() => setDrawer(null)} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Close</Text>
                </Pressable>
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
                {(section === 'providers' ? providers : agents).map((item) => {
                  const selected = section === 'providers' ? item.id === providerId : item.id === agentId;
                  return (
                    <Pressable
                      key={item.id}
                      accessibilityRole="button"
                      accessibilityLabel={item.name}
                      onPress={() => {
                        if (section === 'providers') {
                          setProviderId(item.id);
                        } else {
                          setAgentId(item.id);
                          setCreating(false);
                        }
                        setShowDetail(true);
                        setDrawer(null);
                      }}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{item.name}</Text>
                    </Pressable>
                  );
                })}
              </ScrollView>
            </View>
          ) : null}
          {showDetails ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide && catalog ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Back to list" onPress={() => setShowDetail(false)} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Back</Text>
                </Pressable>
              ) : null}
              {section === 'providers' && provider != null ? (
                <View style={{ gap: 12 }}>
                  <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>{provider.name}</Text>
                  <Text style={label(palette.textPrimary)}>Address</Text>
                  <Text style={{ color: palette.textSecondary, fontSize: 14 }}>{provider.base_url}</Text>
                  <Pressable accessibilityRole="button" accessibilityLabel="Test connection" onPress={() => void testConnection()} style={button(palette.accent)}>
                    <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>Test connection</Text>
                  </Pressable>
                  <Pressable accessibilityRole="button" accessibilityLabel="Refresh models" onPress={() => void refreshModels()} style={button(palette.surface)}>
                    <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>Refresh models</Text>
                  </Pressable>
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
                  <Pressable
                    accessibilityRole="button"
                    accessibilityLabel="New agent"
                    onPress={() => setCreating(true)}
                    style={button(palette.accent)}
                  >
                    <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New agent</Text>
                  </Pressable>
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
                </View>
              ) : null}
              {section === 'orchestration' ? (
                <View style={{ gap: 12 }}>
                  <Text accessibilityRole="header" aria-level={2} style={heading(palette.textPrimary)}>Orchestration</Text>
                  <View style={{ gap: 8 }}>
                    <Text style={label(palette.textPrimary)}>Maximum turns</Text>
                    <TextInput accessibilityLabel="Maximum turns" value={budget} onChangeText={setBudget} keyboardType="number-pad" style={field} />
                  </View>
                  <Pressable accessibilityRole="button" accessibilityLabel="Save maximum turns" onPress={() => void saveBudget()} style={button(palette.surface)}>
                    <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>Save maximum turns</Text>
                  </Pressable>
                </View>
              ) : null}
              {error ? <Text accessibilityRole="alert" style={{ color: palette.danger, fontSize: 14 }}>{error}</Text> : null}
            </ScrollView>
          ) : null}
          {heartbeatAgents ? (
            <View accessibilityLabel="Agents list" style={columnStyle(palette.border, palette.surface, wide, wide ? 200 : undefined)}>
              {!wide ? null : <Text style={label(palette.textPrimary)}>Agents</Text>}
              <ScrollView contentContainerStyle={menuRows}>
                {agents.map((item) => {
                  const selected = item.id === agentId;
                  return (
                    <Pressable
                      key={item.id}
                      accessibilityRole="button"
                      accessibilityLabel={item.name}
                      onPress={() => {
                        setAgentId(item.id);
                        setHeartbeatId(null);
                        setCreatingHeartbeat(false);
                        setHeartbeatPane('list');
                        setDrawer(null);
                      }}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{item.name}</Text>
                    </Pressable>
                  );
                })}
              </ScrollView>
            </View>
          ) : null}
          {heartbeatList ? (
            <View accessibilityLabel="Heartbeats list" style={columnStyle(palette.border, palette.surface, wide, wide ? 220 : undefined)}>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Back to agents" onPress={() => setHeartbeatPane('agents')} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Back</Text>
                </Pressable>
              ) : (
                <Text style={label(palette.textPrimary)}>Heartbeats</Text>
              )}
              <ScrollView contentContainerStyle={menuRows}>
                {agentHeartbeats.map((row) => {
                  const selected = row.id === heartbeatId;
                  return (
                    <Pressable
                      key={row.id}
                      accessibilityRole="button"
                      accessibilityLabel={row.cron_expression}
                      onPress={() => {
                        setHeartbeatId(row.id);
                        setCreatingHeartbeat(false);
                        setHeartbeatPane('detail');
                        setDrawer(null);
                      }}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{row.cron_expression}</Text>
                    </Pressable>
                  );
                })}
                {agentHeartbeats.length === 0 ? <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No heartbeats</Text> : null}
              </ScrollView>
              {!wide && agent != null ? (
                <Pressable accessibilityRole="button" accessibilityLabel="New heartbeat" onPress={startHeartbeat} style={button(palette.accent)}>
                  <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New heartbeat</Text>
                </Pressable>
              ) : null}
            </View>
          ) : null}
          {heartbeatDetail ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Back to heartbeats" onPress={() => setHeartbeatPane('list')} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Back</Text>
                </Pressable>
              ) : null}
              {agent == null ? (
                <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an agent.</Text>
              ) : (
                <View style={{ gap: 12 }}>
                  {wide ? (
                    <Pressable accessibilityRole="button" accessibilityLabel="New heartbeat" onPress={startHeartbeat} style={button(palette.accent)}>
                      <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New heartbeat</Text>
                    </Pressable>
                  ) : null}
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
          {mcpAgents ? (
            <View accessibilityLabel="Agents list" style={columnStyle(palette.border, palette.surface, wide, wide ? 200 : undefined)}>
              {wide ? <Text style={label(palette.textPrimary)}>Agents</Text> : null}
              <ScrollView contentContainerStyle={menuRows}>
                {agents.map((item) => {
                  const selected = item.id === agentId;
                  return (
                    <Pressable
                      key={item.id}
                      accessibilityRole="button"
                      accessibilityLabel={item.name}
                      onPress={() => {
                        setAgentId(item.id);
                        setMcpId(null);
                        setCreatingMcp(false);
                        setMcpPane('list');
                        setDrawer(null);
                      }}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{item.name}</Text>
                    </Pressable>
                  );
                })}
              </ScrollView>
            </View>
          ) : null}
          {mcpList ? (
            <View accessibilityLabel="MCP list" style={columnStyle(palette.border, palette.surface, wide, wide ? 220 : undefined)}>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Back to agents" onPress={() => setMcpPane('agents')} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Back</Text>
                </Pressable>
              ) : (
                <Text style={label(palette.textPrimary)}>MCP</Text>
              )}
              <ScrollView contentContainerStyle={menuRows}>
                {agentServers.map((row) => {
                  const selected = row.id === mcpId;
                  return (
                    <Pressable
                      key={row.id}
                      accessibilityRole="button"
                      accessibilityLabel={row.name}
                      onPress={() => {
                        setMcpId(row.id);
                        setCreatingMcp(false);
                        setMcpPane('detail');
                        setDrawer(null);
                      }}
                      style={{ minHeight: 44, justifyContent: 'center', paddingHorizontal: 8, borderRadius: 6, backgroundColor: selected ? palette.accentSubtle : palette.surface }}
                    >
                      <Text style={{ color: palette.textPrimary, fontSize: 16 }}>{row.name}</Text>
                    </Pressable>
                  );
                })}
                {agentServers.length === 0 ? <Text style={{ color: palette.textSecondary, fontSize: 14 }}>No MCP servers</Text> : null}
              </ScrollView>
              {!wide && agent != null ? (
                <Pressable accessibilityRole="button" accessibilityLabel="New MCP" onPress={startMcp} style={button(palette.accent)}>
                  <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New MCP</Text>
                </Pressable>
              ) : null}
            </View>
          ) : null}
          {mcpDetail ? (
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16, gap: 16, maxWidth: 720 }}>
              {!wide ? (
                <Pressable accessibilityRole="button" accessibilityLabel="Back to MCP" onPress={() => setMcpPane('list')} style={{ minHeight: 44, justifyContent: 'center' }}>
                  <Text style={{ color: palette.accent, fontSize: 14 }}>Back</Text>
                </Pressable>
              ) : null}
              {agent == null ? (
                <Text style={{ color: palette.textPrimary, fontSize: 16 }}>Choose an agent.</Text>
              ) : (
                <View style={{ gap: 12 }}>
                  {wide ? (
                    <Pressable accessibilityRole="button" accessibilityLabel="New MCP" onPress={startMcp} style={button(palette.accent)}>
                      <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>New MCP</Text>
                    </Pressable>
                  ) : null}
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
  const [catalog, setCatalog] = useState<McpTool[] | null>(null);
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
      groups: groups.map((group) => ({ id: group.id, enabled: group.enabled })),
      ...(catalog == null ? {} : { tools: catalog }),
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
      setCatalog(tested.tools);
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
              style={button(on ? palette.accentSubtle : palette.surface)}
            >
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
      <Pressable accessibilityRole="button" accessibilityLabel="Test connection" onPress={() => void test()} style={button(palette.surface)}>
        <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>Test connection</Text>
      </Pressable>
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
      <Pressable accessibilityRole="button" accessibilityLabel={server == null ? 'Create MCP' : `Save ${server.name}`} onPress={() => void save()} style={button(palette.accent)}>
        <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>{server == null ? 'Create MCP' : `Save ${server.name}`}</Text>
      </Pressable>
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
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={creating ? 'Create agent' : `Save ${agent.name}`}
        onPress={() => void save()}
        style={button(creating ? palette.accent : palette.surface)}
      >
        <Text style={{ color: creating ? palette.onAccent : palette.textPrimary, fontSize: 14, fontWeight: '600' }}>
          {creating ? 'Create agent' : `Save ${agent.name}`}
        </Text>
      </Pressable>
      {!creating && agent.id !== 'agent_favlo' ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Delete ${agent.name}`}
          onPress={() => void remove()}
          style={button(palette.danger)}
        >
          <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>{`Delete ${agent.name}`}</Text>
        </Pressable>
      ) : null}
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
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={heartbeat == null ? 'Create heartbeat' : 'Save heartbeat'}
        onPress={() => void save()}
        style={button(palette.accent)}
      >
        <Text style={{ color: palette.onAccent, fontSize: 14, fontWeight: '600' }}>
          {heartbeat == null ? 'Create heartbeat' : 'Save heartbeat'}
        </Text>
      </Pressable>
      {heartbeat != null ? (
        <Pressable accessibilityRole="button" accessibilityLabel="Run heartbeat" onPress={() => void runNow()} style={button(palette.surface)}>
          <Text style={{ color: palette.textPrimary, fontSize: 14, fontWeight: '600' }}>Run heartbeat</Text>
        </Pressable>
      ) : null}
      {notice ? <Text accessibilityLiveRegion="polite" style={{ color: palette.textPrimary, fontSize: 14 }}>{notice}</Text> : null}
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
        thumbColor="#FFFFFF"
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
    borderRadius: 6,
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
  return { color, fontSize: 16, lineHeight: 24 };
}

const menuRows = { gap: focusRingInset * 2, padding: focusRingInset };

function heading(color: string) {
  return { color, fontSize: 18, lineHeight: 24, fontWeight: '600' as const };
}

function button(background: string) {
  return {
    minHeight: 44,
    alignSelf: 'flex-start' as const,
    borderRadius: 6,
    paddingHorizontal: 16,
    justifyContent: 'center' as const,
    backgroundColor: background,
  };
}

function fieldStyle(palette: { textPrimary: string; surface: string; borderStrong: string }) {
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

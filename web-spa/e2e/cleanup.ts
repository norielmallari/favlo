const API = 'http://127.0.0.1:8130';
const FAVLO = 'agent_favlo';

type Agent = { id: string; name: string };
type Channel = { id: string };
type Workspace = { id: string; name: string };
type AgentSkill = { id: string; name: string };

export async function deleteAgentsNamed(...names: string[]): Promise<void> {
  if (names.length === 0) {
    return;
  }
  const wanted = new Set(names);
  const body = await read<{ agents: Agent[] }>('/agents');
  for (const agent of body.agents) {
    if (agent.id === FAVLO || !wanted.has(agent.name)) {
      continue;
    }
    await send('DELETE', `/agents/${agent.id}`);
  }
}

export async function deleteWorkspacesNamed(...names: string[]): Promise<void> {
  if (names.length === 0) {
    return;
  }
  const wanted = new Set(names);
  const channels = await read<{ channels: Channel[] }>('/channels');
  for (const channel of channels.channels) {
    const listed = await read<{ workspaces: Workspace[] }>(
      `/workspaces?channel_id=${encodeURIComponent(channel.id)}`,
    );
    for (const workspace of listed.workspaces) {
      if (!wanted.has(workspace.name)) {
        continue;
      }
      await send('DELETE', `/workspaces/${workspace.id}`);
    }
  }
}

export async function deleteAgentSkillsNamed(...names: string[]): Promise<void> {
  if (names.length === 0) {
    return;
  }
  const wanted = new Set(names);
  const body = await read<{ skills: AgentSkill[] }>('/agent-skills');
  for (const skill of body.skills) {
    if (!wanted.has(skill.name)) {
      continue;
    }
    await send('DELETE', `/agent-skills/${skill.id}`);
  }
}

async function read<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return (await response.json()) as T;
}

async function send(method: string, path: string): Promise<void> {
  const response = await fetch(`${API}${path}`, { method });
  if (!response.ok && response.status !== 404) {
    throw new Error(`${method} ${path} returned ${response.status}`);
  }
}

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { expect, test } from '@playwright/test';

const API = 'http://127.0.0.1:8130';
const PROMPT = 'Reply with the single word ready.';

test('a new install seeds Favlo and answers the first chat', async ({ page }) => {
  const models = await liveModels();
  test.skip(models.length === 0, 'favlo serve has no language model');
  const home = fs.readFileSync(path.join(process.cwd(), 'e2e/web-new-user-e2e-setup/.home'), 'utf8').trim();
  const database = path.join(home, 'app.db');
  const installed = path.join(os.homedir(), '.favlo', 'web', 'app.db');
  const installedStamp = fs.existsSync(installed) ? fs.statSync(installed).mtimeMs : null;
  expect(path.resolve(home)).not.toBe(path.dirname(installed));
  expect(fs.existsSync(database)).toBe(true);
  expect(fs.statSync(database).size).toBeGreaterThan(0);
  expect(fs.existsSync(path.join(home, 'secrets'))).toBe(true);

  const agents = await read<{ agents: Agent[] }>('/agents');
  expect(agents.agents.map((agent) => agent.name)).toEqual(['Favlo']);
  expect(agents.agents[0].id).toBe('agent_favlo');
  expect(agents.agents[0].persona.length).toBeGreaterThan(0);
  expect(agents.agents[0].tool_groups).toEqual(expect.arrayContaining(['favlo-config', 'session']));

  const channels = await read<{ channels: Channel[] }>('/channels');
  expect(channels.channels).toEqual([
    expect.objectContaining({ id: 'channel_favlo', name: 'Favlo', agent_id: 'agent_favlo', session_id: 'session_favlo' }),
  ]);
  const session = await read<{ messages: unknown[] }>('/sessions/session_favlo');
  expect(session.messages).toEqual([]);

  const settings = await read<{ default_model_id: string | null }>('/settings');
  expect(settings.default_model_id).toBeTruthy();
  const providers = await read<{ providers: Provider[] }>('/providers');
  const local = providers.providers.find((provider) => provider.id === 'provider_local');
  expect(local?.base_url).toBe('http://127.0.0.1:8129/v1');

  const catalog = await read<{ skills: Skill[] }>('/global-skills');
  const names = catalog.skills.map((skill) => skill.name).sort();
  expect(names).toEqual(['session-clear', 'session-remember']);
  expect(catalog.skills.every((skill) => skill.add_by_default)).toBe(true);
  const copied = await read<{ skills: Skill[] }>('/agent-skills');
  expect(copied.skills.map((skill) => skill.name).sort()).toEqual(['session-clear', 'session-remember']);
  expect(copied.skills.every((skill) => skill.agent_id === 'agent_favlo')).toBe(true);

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await expect(page.getByRole('dialog', { name: 'New agent' })).toHaveCount(0);
  await expect(page.getByText('Chat, images, music, and voice on this machine.')).toHaveCount(0);
  await expect(page.getByLabel('Conversation').getByRole('heading', { name: 'Favlo', exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill(PROMPT);
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByLabel(`You said: ${PROMPT}`)).toBeVisible();
  await expect.poll(async () => page.getByLabel(/^Favlo said: /).count(), { timeout: 360_000 }).toBeGreaterThan(0);

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Channels' })).toBeVisible();
  await expect(page.getByLabel(`You said: ${PROMPT}`)).toBeVisible();
  await expect(page.getByLabel(/^Favlo said: /).first()).toBeVisible();

  if (installedStamp != null) {
    expect(fs.statSync(installed).mtimeMs).toBe(installedStamp);
  }
});

async function read<T>(pathname: string): Promise<T> {
  const response = await fetch(`${API}${pathname}`);
  expect(response.ok).toBe(true);
  return (await response.json()) as T;
}

async function liveModels(): Promise<string[]> {
  let response: Response;
  try {
    response = await fetch('http://127.0.0.1:8129/v1/models');
  } catch {
    return [];
  }
  if (!response.ok) {
    return [];
  }
  const body = (await response.json()) as { data?: { id?: string }[] };
  return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
}

type Agent = { id: string; name: string; persona: string; tool_groups: string[] };
type Channel = { id: string; name: string; agent_id: string; session_id: string };
type Provider = { id: string; base_url: string };
type Skill = { name: string; add_by_default?: boolean; agent_id?: string };

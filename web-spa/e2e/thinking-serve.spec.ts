import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

const MODEL = 'qwen36-35b-a3b:thinking';

let created = '';

test.afterAll(async () => {
  if (created) {
    await deleteAgentsNamed(created);
  }
});

test('an agent chats with the 35B thinking model through favlo serve', async ({ page }) => {
  const refreshed = await fetch('http://127.0.0.1:8130/providers/provider_local/refresh', { method: 'POST' });
  expect(refreshed.ok).toBe(true);
  const listed = (await refreshed.json()) as { models?: { model_id?: string; enabled?: boolean }[] };
  const enabled = (listed.models ?? []).some((row) => row.model_id === MODEL && row.enabled !== false);
  expect(enabled).toBe(true);

  const name = `Smoke ${Date.now()}`;
  created = name;
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('Reply in one short sentence.');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(MODEL);
  await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('medium');
  await createAgent(page, name);
  await openAgentChat(page, name);
  await page.getByRole('textbox', { name: 'Message' }).fill('Reply with the single word ready.');
  const send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();

  const turn = page.getByLabel(new RegExp(`^${name} said:`)).first();
  await expect(turn).toBeVisible({ timeout: 180_000 });
  const thinking = turn.getByRole('button', { name: 'Thinking' });
  await expect(thinking).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  await expect(thinking).toHaveAttribute('aria-expanded', 'false', { timeout: 180_000 });
  await page.setViewportSize({ width: 390, height: 844 });
  await onScreen(thinking);
  await thinking.click();
  await expect(turn.getByTestId('thinking-text')).toBeVisible();
  await expect(turn.getByTestId('thinking-text')).not.toHaveText('');
});

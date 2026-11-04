import { expect, test, type Page } from '@playwright/test';

import { LOCAL_PROVIDER_ID } from '../src/api/seed-ids';
import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

const MODEL = 'qwen35-4b:thinking';
const PROMPT = 'Reply with the single word ready.';

test('qwen35 reasoning effort none hides thinking and medium shows it', async ({ page }) => {
  const models = await liveModels();
  test.skip(!models.includes(MODEL) && !models.includes('qwen35-4b'), 'qwen35-4b is not registered');
  const model = models.includes(MODEL) ? MODEL : 'qwen35-4b';
  const quiet = `Quiet ${Date.now()}`;
  const loud = `Loud ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);

    await page.getByRole('button', { name: 'New agent' }).click();
    await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(quiet);
    await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
    await chooseModel(page, model);
    await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('none');
    await createAgent(page, quiet);
    await openAgentChat(page, quiet);
    await sendPrompt(page, PROMPT);
    const quietTurn = page.getByLabel(new RegExp(`^${quiet} said:`)).first();
    await expect(quietTurn).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
    await expect(quietTurn.getByRole('button', { name: 'Thinking' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Thinking' })).toHaveCount(0);

    await page.getByRole('button', { name: 'New agent' }).click();
    await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(loud);
    await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
    await chooseModel(page, model);
    await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('medium');
    await createAgent(page, loud);
    await openAgentChat(page, loud);
    await sendPrompt(page, PROMPT);
    const thinking = page.getByTestId('transcript').getByRole('button', { name: 'Thinking', exact: true });
    await expect(thinking).toBeVisible({ timeout: 180_000 });
    await expect(thinking).toHaveAttribute('aria-expanded', 'false');
    await onScreen(thinking);
    await thinking.click();
    await expect(thinking).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByTestId('thinking-text')).toBeVisible();
    await expect(page.getByTestId('thinking-text')).not.toHaveText('');
    await page.setViewportSize({ width: 390, height: 844 });
    await onScreen(thinking);
    await expect(page.getByTestId('thinking-text')).toBeVisible();
    const stop = page.getByRole('button', { name: 'Stop' });
    await onScreen(stop);
    await stop.click();
    await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 60_000 });
  } finally {
    await deleteAgentsNamed(quiet, loud);
  }
});

async function chooseModel(page: Page, model: string): Promise<void> {
  await page.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(LOCAL_PROVIDER_ID);
  const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(refresh);
  await refresh.click();
  await onScreen(page.getByText('Models refreshed'));
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model);
}

async function sendPrompt(page: Page, prompt: string): Promise<void> {
  await page.getByRole('textbox', { name: 'Message' }).fill(prompt);
  const send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();
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

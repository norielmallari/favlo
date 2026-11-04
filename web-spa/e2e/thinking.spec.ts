import { expect, test } from '@playwright/test';

import { LOCAL_PROVIDER_ID } from '../src/api/seed-ids';
import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

test('thinking opens during a reply and closes when the message is stored', async ({ page }) => {
  const name = `Wren ${Date.now()}`;
  try {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(LOCAL_PROVIDER_ID);
  const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(refresh);
  await refresh.click();
  await onScreen(page.getByText('Models refreshed'));
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:thinking');
  await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('medium');
  await createAgent(page, name);
  await openAgentChat(page, name);
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi.');
  let send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();

  await expect(page.getByLabel(new RegExp(`^${name} said:`)).first()).toBeVisible({ timeout: 180_000 });
  const firstTurn = page.getByLabel(new RegExp(`^${name} said:`)).first();
  const thinking = firstTurn.getByRole('button', { name: 'Thinking' });
  await expect(thinking).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  await expect(thinking).toHaveAttribute('aria-expanded', 'false', { timeout: 180_000 });
  await thinking.click();
  await expect(thinking).toHaveAttribute('aria-expanded', 'true');
  await expect(firstTurn.getByTestId('thinking-text')).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi again.');
  send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();
  const secondTurn = page.getByLabel(new RegExp(`^${name} said:`)).last();
  await expect(secondTurn).toBeVisible({ timeout: 180_000 });
  const narrowThinking = secondTurn.getByRole('button', { name: 'Thinking' });
  await expect(narrowThinking).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  await expect(narrowThinking).toHaveAttribute('aria-expanded', 'false', { timeout: 180_000 });
  await onScreen(narrowThinking);
  await narrowThinking.click();
  await expect(secondTurn.getByTestId('thinking-text')).toBeVisible();
  } finally {
    await deleteAgentsNamed(name);
  }
});

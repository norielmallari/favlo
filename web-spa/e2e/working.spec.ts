import { expect, test } from '@playwright/test';

import { LOCAL_PROVIDER_ID } from '../src/api/seed-ids';
import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

test('working pulses the activity dot during a reply', async ({ page }) => {
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
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:instruct');
  await createAgent(page, name);
  await openAgentChat(page, name);
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi.');
  let send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();

  const working = page.getByTestId('working');
  await expect(working).toBeVisible({ timeout: 180_000 });
  await expect(working).toHaveText(/^(Working|Thinking|Tool call)/);
  await expect(page.getByTestId('working-dot')).toHaveCSS('animation-name', 'favlo-working-pulse');
  await expect(working).not.toHaveCSS('-webkit-background-clip', 'text');
  await expect(working).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('textbox', { name: 'Message' }).fill('Count slowly to five.');
  send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();
  await expect(working).toBeVisible({ timeout: 180_000 });
  await onScreen(working);
  const stop = page.getByRole('button', { name: 'Stop' });
  await onScreen(stop);
  await stop.click();
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible();
  await expect(page.getByText('The reply stopped.')).toBeAttached();
  } finally {
    await deleteAgentsNamed(name);
  }
});

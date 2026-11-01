import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

test('working sweeps a light band across the word during a reply', async ({ page }) => {
  const name = `Wren ${Date.now()}`;
  try {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:instruct');
  await createAgent(page, name);
  await openAgentChat(page, name);
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi.');
  let send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();

  const working = page.getByTestId('working');
  await expect(working).toBeVisible({ timeout: 180_000 });
  await expect(working).toHaveText('Working');
  await expect(working).toHaveCSS('animation-name', 'favlo-working-shimmer');
  await expect(working).toHaveCSS('-webkit-background-clip', 'text');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('textbox', { name: 'Message' }).fill('Count slowly to five.');
  send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();
  await expect(working).toBeVisible({ timeout: 180_000 });
  await onScreen(working);
  const stop = page.getByRole('button', { name: 'Stop' });
  await onScreen(stop);
  await stop.click();
  await expect(page.getByRole('button', { name: 'Send' })).toBeVisible();
  await expect(page.getByText('The reply stopped.')).toBeAttached();
  } finally {
    await deleteAgentsNamed(name);
  }
});

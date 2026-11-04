import { execFileSync } from 'node:child_process';
import path from 'node:path';

import { expect, test } from '@playwright/test';

import { onScreen } from './on-screen';

const root = path.resolve(__dirname, '../..');
const prompt = 'Send again check.';

function smoke(action: 'acquire' | 'release') {
  const args = action === 'acquire'
    ? ['run', 'python', '-m', 'favlo.smoke_lock', 'acquire', '--pipeline', 'qwen36-35b-a3b', '--command', 'reply-error']
    : ['run', 'python', '-m', 'favlo.smoke_lock', 'release'];
  execFileSync('uv', args, { cwd: root, timeout: 120_000 });
}

function released(): boolean {
  try {
    smoke('release');
    return true;
  } catch {
    return false;
  }
}

test('a failed reply can be sent again', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models');
  test.skip(!models.ok, 'favlo serve has no language model');
  await expect.poll(() => {
    try {
      smoke('acquire');
      return true;
    } catch {
      return false;
    }
  }, { timeout: 90_000 }).toBe(true);
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    const session = page.waitForResponse(
      (response) => response.url().includes('/sessions/') && response.request().method() === 'GET' && response.ok(),
    );
    await page.goto('http://127.0.0.1:8012/');
    await session;
    await expect(page.getByRole('heading', { name: 'Favlo', level: 2 })).toBeVisible();
    const said = page.getByLabel(`You said: ${prompt}`);
    const before = await said.count();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(prompt);
    const send = page.getByRole('button', { name: 'Send', exact: true });
    await onScreen(send);
    await send.click();
    const alert = page.getByRole('alert').filter({ hasText: 'The GPU is held for a pipeline smoke.' });
    await expect(alert).toBeVisible();
    await expect.poll(async () => said.count()).toBe(before + 1);

    await page.setViewportSize({ width: 1280, height: 640 });
    const again = page.getByRole('button', { name: 'Send again' });
    await onScreen(again);
    await page.setViewportSize({ width: 390, height: 844 });
    await onScreen(again);
    await page.setViewportSize({ width: 1280, height: 800 });
    await onScreen(again);

    smoke('release');
    await again.click();
    await expect.poll(async () => said.count(), { timeout: 30_000 }).toBe(before + 2);
    const stop = page.getByRole('button', { name: 'Stop' });
    await onScreen(stop);
    await stop.click();
    await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible();
  } finally {
    await expect.poll(() => released(), { timeout: 30_000 }).toBe(true);
  }
});

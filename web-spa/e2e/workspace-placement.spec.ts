import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test('the agent can change gallery placement on the shell', async ({ page }) => {
  test.setTimeout(360_000);
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-place-'));
  fs.writeFileSync(path.join(root, 'frame.png'), TINY_PNG);
  const name = `Place ${Date.now()}`;
  try {
    await step('create gallery with folder', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
    });

    await step('opening placement shows the preview and the filmstrip', async () => {
      const frame = page.getByRole('button', { name: 'frame.png', exact: true });
      await expect(frame).toBeVisible({ timeout: 60_000 });
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
      await expect(page.getByText('No media selected')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Download frame.png' })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Add to chat' })).toBeVisible();
    });

    await step('a place call cannot drop the preview', async () => {
      await ask(page, 'Remove the preview. Keep the filmstrip as it is.');
      await expect(page.getByRole('button', { name: 'frame.png', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Download frame.png' })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Add to chat' })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function ask(page: import('@playwright/test').Page, text: string) {
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill(text);
  const send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();
  await expect(page.getByTestId('working')).toBeVisible();
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
}
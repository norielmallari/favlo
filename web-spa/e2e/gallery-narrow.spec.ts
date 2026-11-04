import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test('narrow gallery keeps preview, filmstrip, and chat in one view', async ({ page }) => {
  test.setTimeout(360_000);
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-narrow-'));
  fs.writeFileSync(path.join(root, 'horse.png'), TINY_PNG);
  fs.writeFileSync(path.join(root, 'cat.png'), TINY_PNG);
  const name = `Narrow ${Date.now()}`;
  try {
    await step('create gallery on a wide window', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'cat.png', exact: true })).toBeVisible();
    });

    await step('narrow shows gallery and chat together', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(page.getByRole('button', { name: 'Chat', exact: true })).toHaveCount(0);
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expect(page.getByLabel('Conversation')).toBeVisible();
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      await expect(page.getByText(/^\d+ of \d+$/)).toBeVisible();
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
    });

    await step('select the second image and read its place', async () => {
      const horse = page.getByRole('button', { name: 'horse.png', exact: true });
      await onScreen(horse);
      await horse.click();
      await expect(page.getByRole('button', { name: 'Download horse.png' })).toBeVisible();
      await expect(page.getByText('2 of 2')).toBeVisible();
    });

    await step('add to chat lands in the visible draft', async () => {
      const add = page.getByRole('button', { name: 'Add to chat', exact: true });
      await onScreen(add);
      await add.click();
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue(
        /(?:^|\n)horse\.png$/,
      );
    });

    await step('composer focus expands the dock with Send on screen', async () => {
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await box.click();
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
    });

    await step('full screen preview and back', async () => {
      const enter = page.getByRole('button', { name: 'Full screen preview', exact: true });
      await onScreen(enter);
      await enter.click();
      await expect(page.getByLabel('Filmstrip')).toHaveCount(0);
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible();
      const exit = page.getByRole('button', { name: 'Exit full screen', exact: true });
      await onScreen(exit);
      await exit.click();
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      await expect(page.getByRole('button', { name: 'Download horse.png' })).toBeVisible();
      await expect(page.getByText('2 of 2')).toBeVisible();
    });

    await step('send a short line and see the reply in the dock', async () => {
      const line = `Narrow hello ${Date.now()}`;
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await box.fill(line);
      const send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await expect(page.getByLabel(`You said: ${line}`)).toBeVisible();
      await expect(page.getByTestId('working')).toBeVisible();
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
      await expect(page.getByLabel(/said:/).filter({ hasNotText: 'You said:' }).first()).toBeVisible();
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
    });

    await step('WCAG AA in light and dark at the narrow size', async () => {
      await page.emulateMedia({ colorScheme: 'light' });
      await assertWcagAa(page);
      await page.emulateMedia({ colorScheme: 'dark' });
      await assertWcagAa(page);
      await page.emulateMedia({ colorScheme: 'light' });
    });

    await step('wide window still shows the side-by-side chat', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expect(page.getByLabel('Conversation')).toBeVisible();
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
      await onScreen(page.getByRole('slider', { name: 'Resize chat' }));
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

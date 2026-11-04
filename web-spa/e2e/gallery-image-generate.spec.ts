import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test.describe.configure({ mode: 'serial' });

async function requireQwen35b() {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve is not listening');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  const served = (listed.data ?? []).some((row) => row.id === 'qwen36-35b-a3b');
  test.skip(!served, 'qwen36-35b-a3b is not served');
}

test('gallery chat keeps the request while an image generates, then shows it', async ({ page }) => {
  await requireQwen35b();

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-draw-'));
  const name = `Stills ${Date.now()}`;
  try {
    await step('open a gallery workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
    });

    await step('the reply arrives before the image, then both stay with the request', async () => {
      const imagesBefore = await page.getByRole('img', { name: 'Generated image' }).count();
      const saidBefore = await page.getByLabel(/^Favlo said:/).count();
      await send(page, 'Draw a red square.');

      const said = page.getByLabel(/^Favlo said:/);
      const images = page.getByRole('img', { name: 'Generated image' });
      await expect
        .poll(
          async () =>
            (await said.count()) > saidBefore &&
            (await images.count()) === imagesBefore &&
            (await page.getByLabel(/^You said:/).filter({ hasText: 'Draw a red square.' }).count()) > 0,
          { timeout: 45_000 },
        )
        .toBe(true);

      await expect.poll(async () => images.count(), { timeout: 45_000 }).toBeGreaterThan(imagesBefore);
      await expect(page.getByLabel(/^You said:/).filter({ hasText: 'Draw a red square.' })).toBeVisible();
      await expect(said.nth(saidBefore)).toBeVisible();
      const image = images.last();
      await expect(image).toBeVisible();
      const src = await image.getAttribute('src');
      expect(src).toContain('/media/images/');

      await expect
        .poll(async () => fs.readdirSync(root).some((file) => file.endsWith('.png')), {
          timeout: 15_000,
        })
        .toBe(true);
      // Gallery outcome on screen: the filmstrip tile (chat image may sit in the scrolled transcript).
      const stripImage = page.getByLabel('Filmstrip').getByRole('button', { name: /\.png$/i }).last();
      await onScreen(stripImage);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('gallery chat keeps both turns and the image after a second message and a reload', async ({
  page,
}) => {
  await requireQwen35b();

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-draw2-'));
  const name = `Stills ${Date.now()}`;
  try {
    await step('open a gallery workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await createGalleryWorkspace(page, name, root);
    });

    await step('send while the file is generating, then reload', async () => {
      const imagesBefore = await page.getByRole('img', { name: 'Generated image' }).count();
      const saidBefore = await page.getByLabel(/^Favlo said:/).count();
      await send(page, 'Draw a blue circle.');
      // Wait for the draw reply (not the greeting), then Working clears so Send returns.
      await expect
        .poll(async () => page.getByLabel(/^Favlo said:/).count(), { timeout: 45_000 })
        .toBeGreaterThan(saidBefore);
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 15_000 });
      await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible();

      // Second turn while the image file may still be generating in the background.
      const saidAfterDraw = await page.getByLabel(/^Favlo said:/).count();
      await send(page, 'Say ready.');
      await expect
        .poll(
          async () =>
            (await page.getByLabel(/^You said:/).filter({ hasText: 'Draw a blue circle.' }).count()) >
              0 &&
            (await page.getByLabel(/^You said:/).filter({ hasText: 'Say ready.' }).count()) > 0 &&
            (await page.getByLabel(/^Favlo said:/).count()) > saidAfterDraw,
          { timeout: 45_000 },
        )
        .toBe(true);

      await expect
        .poll(async () => page.getByRole('img', { name: 'Generated image' }).count(), {
          timeout: 45_000,
        })
        .toBeGreaterThan(imagesBefore);

      await page.reload();
      await expect(page.getByLabel('Gallery workspace', { exact: true })).toBeVisible();
      await expect(page.getByLabel(/^You said:/).filter({ hasText: 'Draw a blue circle.' })).toBeVisible();
      await expect(page.getByLabel(/^You said:/).filter({ hasText: 'Say ready.' })).toBeVisible();
      await expect(page.getByRole('img', { name: 'Generated image' }).first()).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('gallery chat clears Working after a reload during a generate turn', async ({ page }) => {
  await requireQwen35b();

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-reload-'));
  const name = `Stills ${Date.now()}`;
  try {
    await step('open a gallery workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await createGalleryWorkspace(page, name, root);
    });

    await step('reload mid-turn, then send again', async () => {
      await send(page, 'Draw a green triangle.');
      await expect(page.getByTestId('working')).toBeVisible();
      await page.reload();
      await expect(page.getByLabel('Gallery workspace', { exact: true })).toBeVisible();
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 15_000 });
      // Empty composer keeps Send disabled; Stop must be gone so a new turn can start.
      await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0);
      const saidBefore = await page.getByLabel(/^Favlo said:/).count();
      await send(page, 'Say hello.');
      await expect(page.getByLabel(/^You said:/).filter({ hasText: 'Say hello.' })).toBeVisible();
      await expect
        .poll(async () => page.getByLabel(/^Favlo said:/).count(), { timeout: 45_000 })
        .toBeGreaterThan(saidBefore);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function send(page: Page, text: string) {
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 15_000 });
  const box = page.getByRole('textbox', { name: 'Message', exact: true });
  await box.fill(text);
  const sendButton = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(sendButton);
  await sendButton.click();
}

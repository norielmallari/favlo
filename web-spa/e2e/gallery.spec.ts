import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test('a gallery guide binds a folder, places a strip, and browses without AG-UI', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-'));
  const albums = path.join(root, 'Albums');
  fs.mkdirSync(albums);
  fs.writeFileSync(path.join(root, 'horse.png'), TINY_PNG);
  fs.writeFileSync(path.join(root, 'cat.png'), TINY_PNG);
  fs.writeFileSync(path.join(albums, 'zebra.png'), TINY_PNG);
  const name = `Stills ${Date.now()}`;
  let aguiCount = 0;
  page.on('request', (request) => {
    if (request.method() === 'POST' && request.url().includes('/ag-ui')) {
      aguiCount += 1;
    }
  });
  try {
    await step('open app and create gallery with folder (no AG-UI)', async () => {
      const beforeBind = aguiCount;
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await expect(page.getByRole('button', { name: 'New workspace' })).toBeVisible();
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByText('Ask the agent to fill the filmstrip')).toBeVisible();
      await expect(page.getByText('No media selected')).toBeVisible();
      // Index wake writes per-folder sidecars (Albums owns its .favlo_metadata).
      await expect
        .poll(() => fs.existsSync(path.join(albums, '.favlo_metadata', 'zebra.png.json')))
        .toBe(true);
      await expect
        .poll(() => fs.existsSync(path.join(root, '.favlo_metadata', 'horse.png.json')))
        .toBe(true);
      expect(aguiCount).toBe(beforeBind);
    });

    await step('agent fills filmstrip', async () => {
      const zebra = page.getByRole('button', { name: 'zebra.png', exact: true });
      await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
        'Put every image from this folder on the filmstrip.',
      );
      const send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await expect(page.getByTestId('working')).toBeVisible();
      try {
        await expect(zebra).toBeVisible();
      } catch {
        await expect(page.getByTestId('working')).toHaveCount(0);
        await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
          'Please put every image on the filmstrip now, including Albums/zebra.png.',
        );
        await onScreen(send);
        await send.click();
        await expect(page.getByTestId('working')).toBeVisible();
        await expect(zebra).toBeVisible();
      }
      await expect(page.getByTestId('working')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'cat.png', exact: true })).toBeVisible();
      await expect(page.getByText('No media selected')).toHaveCount(0);
      await expect(page.getByRole('button', { name: /Download .+\.png/ })).toBeVisible();
      await expectChatRestsOnComposer(page);
    });

    await step('browse strip without AG-UI', async () => {
      const beforeBrowse = aguiCount;
      await page.getByRole('button', { name: 'zebra.png', exact: true }).click();
      await expect(page.getByRole('button', { name: 'Download zebra.png' })).toBeVisible();
      await page.keyboard.press('ArrowRight');
      await page.keyboard.press('ArrowLeft');
      await expect(page.getByRole('button', { name: 'Download zebra.png' })).toBeVisible();
      const download = page.waitForEvent('download', { timeout: 15_000 });
      await page.getByRole('button', { name: 'Download zebra.png' }).click();
      expect((await download).suggestedFilename()).toBe('zebra.png');
      await page.getByRole('button', { name: 'Add to chat' }).click();
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue(
        /(?:^|\n)Albums\/zebra\.png$/,
      );
      expect(aguiCount).toBe(beforeBrowse);
    });

    await step('reload restores strip without AG-UI', async () => {
      const beforeReload = aguiCount;
      await page.reload();
      await expect(page.getByRole('heading', { name })).toBeVisible();
      await expect(page.getByRole('button', { name: 'zebra.png', exact: true })).toBeVisible();
      expect(aguiCount).toBe(beforeReload);
    });

    await step('narrow chat / back to workspace', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      const chatTab = page.getByRole('button', { name: 'Chat', exact: true });
      await expect(chatTab).toBeVisible();
      await chatTab.click();
      await expect(page.getByLabel('Conversation')).toBeVisible();
      await expectChatRestsOnComposer(page);
      await page.getByRole('button', { name: 'Back to workspace' }).click();
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expectFits(page);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function expectChatRestsOnComposer(page: Page) {
  const place = await page.getByTestId('transcript').evaluate((node) => {
    const scroller = node as HTMLElement;
    const conversation = scroller.closest('[aria-label="Conversation"]') as HTMLElement | null;
    const composer = conversation?.querySelector('[aria-label="Message"]') as HTMLElement | null;
    const rows = [...scroller.querySelectorAll('[aria-label]')].filter((el) =>
      / said: /.test(el.getAttribute('aria-label') ?? ''),
    );
    const first = rows[0]?.getBoundingClientRect();
    const last = rows[rows.length - 1]?.getBoundingClientRect();
    const box = scroller.getBoundingClientRect();
    const shell = conversation?.getBoundingClientRect();
    const input = composer?.getBoundingClientRect();
    return {
      count: rows.length,
      overflow: scroller.scrollHeight > scroller.clientHeight + 2,
      fromBottom: scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop,
      firstTop: first == null ? 0 : first.top - box.top,
      lastBottom: last == null ? box.height : box.bottom - last.bottom,
      height: box.height,
      // Composer should sit on the bottom of the conversation column (home chat).
      composerGap:
        shell == null || input == null ? Number.POSITIVE_INFINITY : shell.bottom - input.bottom,
    };
  });
  expect(place.count).toBeGreaterThan(0);
  expect(place.fromBottom).toBeLessThan(8);
  expect(place.lastBottom).toBeLessThan(place.height * 0.25);
  expect(place.composerGap).toBeLessThan(64);
  // A single short reply leaves empty space above the first bubble, like home chat.
  if (!place.overflow && place.count === 1) {
    expect(place.firstTop).toBeGreaterThan(place.height * 0.25);
  }
}

async function expectFits(page: Page) {
  await onScreen(page.getByRole('button', { name: 'Configuration' }));
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

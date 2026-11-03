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
  test.setTimeout(360_000);
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
  let gallery: Page;
  page.context().on('request', (request) => {
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
      await expect(page.getByRole('button', { name: 'New workspace for Favlo' })).toBeVisible();
      gallery = await createGalleryWorkspace(page, name, root);
      await expect(gallery.getByLabel(/said:/).first()).toBeVisible();
      assertCaption(path.join(root, '.favlo_metadata', 'horse.png.json'));
      assertCaption(path.join(root, '.favlo_metadata', 'cat.png.json'));
      assertCaption(path.join(albums, '.favlo_metadata', 'zebra.png.json'));
      expect(aguiCount).toBe(beforeBind);
    });

    await step('opening filmstrip shows the images', async () => {
      const zebra = gallery.getByRole('button', { name: 'zebra.png', exact: true });
      await expect(zebra).toBeVisible();
      await expect(zebra).toHaveText('');
      await expect(gallery.locator('img[alt="zebra.png"]').first()).toHaveJSProperty('naturalWidth', 1);
      await expect(gallery.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
      await expect(gallery.getByRole('button', { name: 'cat.png', exact: true })).toBeVisible();
      await expect(gallery.getByText('No media selected')).toHaveCount(0);
      await expect(gallery.getByRole('button', { name: /Download .+\.png/ })).toBeVisible();
      await expectChatRestsOnComposer(gallery);
    });

    await step('browse strip without AG-UI', async () => {
      const beforeBrowse = aguiCount;
      await gallery.getByRole('button', { name: 'zebra.png', exact: true }).click();
      await expect(gallery.getByRole('button', { name: 'Download zebra.png' })).toBeVisible();
      await gallery.keyboard.press('ArrowRight');
      await gallery.keyboard.press('ArrowLeft');
      await expect(gallery.getByRole('button', { name: 'Download zebra.png' })).toBeVisible();
      const download = gallery.waitForEvent('download', { timeout: 15_000 });
      await gallery.getByRole('button', { name: 'Download zebra.png' }).click();
      expect((await download).suggestedFilename()).toBe('zebra.png');
      await gallery.getByRole('button', { name: 'Add to chat' }).click();
      await expect(gallery.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue(
        /(?:^|\n)Albums\/zebra\.png$/,
      );
      expect(aguiCount).toBe(beforeBrowse);
    });

    await step('reload restores strip without AG-UI', async () => {
      const beforeReload = aguiCount;
      await gallery.reload();
      await expect(gallery.getByRole('heading', { name })).toBeVisible();
      await expect(gallery.getByRole('button', { name: 'zebra.png', exact: true })).toBeVisible();
      expect(aguiCount).toBe(beforeReload);
    });

    await step('narrow chat / back to workspace', async () => {
      await gallery.setViewportSize({ width: 390, height: 844 });
      const chatTab = gallery.getByRole('button', { name: 'Chat', exact: true });
      await expect(chatTab).toBeVisible();
      await chatTab.click();
      await expect(gallery.getByLabel('Conversation')).toBeVisible();
      await expectChatRestsOnComposer(gallery);
      await gallery.getByRole('button', { name: 'Back to workspace' }).click();
      await expect(gallery.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expectFits(gallery);
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

function assertCaption(file: string) {
  const side = JSON.parse(fs.readFileSync(file, 'utf8')) as { description?: string };
  expect(side.description ?? '').not.toMatch(/^A \w+ file at /);
}

async function expectFits(page: Page) {
  await onScreen(page.getByRole('button', { name: 'Favlo', exact: true }));
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

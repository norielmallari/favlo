import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const logFile = path.join(os.homedir(), '.favlo', 'logs', 'favlo.log');
const patches = 'vision patches 256';
const cached = 'vision cache 64';

test('an attached strip image is embedded once across later messages', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  const served = (listed.data ?? []).some((row) => row.id === 'qwen36-35b-a3b');
  test.skip(!served, 'qwen36-35b-a3b is not served');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-'));
  const stamp = Date.now() % 200 + 20;
  fs.writeFileSync(path.join(root, 'frame.png'), tinyPng(stamp, 40, 90));
  const name = `Frame ${Date.now()}`;
  try {
    await step('create gallery with folder', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
      await expect
        .poll(() => fs.existsSync(path.join(root, '.favlo_metadata', 'frame.png.json')))
        .toBe(true);
    });

    await step('agent places frame.png on filmstrip', async () => {
      const frame = page.getByRole('button', { name: 'frame.png', exact: true });
      if (!(await frame.isVisible().catch(() => false))) {
        await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
          'Put every image from this folder on the filmstrip, and keep the preview so I can open one.',
        );
        const send = page.getByRole('button', { name: 'Send', exact: true });
        await onScreen(send);
        await send.click();
        await expect(page.getByTestId('working')).toBeVisible();
        try {
          await expect(frame).toBeVisible({ timeout: 180_000 });
        } catch {
          await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
          await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
            'Please put frame.png on the filmstrip now.',
          );
          await onScreen(send);
          await send.click();
          await expect(page.getByTestId('working')).toBeVisible();
          await expect(frame).toBeVisible({ timeout: 180_000 });
        }
      }
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
    });

    await step('add frame.png to chat', async () => {
      await page.getByRole('button', { name: 'frame.png', exact: true }).click();
      const download = page.getByRole('button', { name: 'Download frame.png' });
      if ((await download.count()) === 0) {
        await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
          'Bring back the preview pane next to the filmstrip.',
        );
        const send = page.getByRole('button', { name: 'Send', exact: true });
        await onScreen(send);
        await send.click();
        await expect(page.getByTestId('working')).toBeVisible();
        await expect(page.getByTestId('working')).toHaveCount(0);
        await page.getByRole('button', { name: 'frame.png', exact: true }).click();
      }
      await expect(download).toBeVisible();
      await page.getByRole('button', { name: 'Add to chat' }).click();
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue('frame.png');
    });

    await step('describe image (first vision embed)', async () => {
      const beforePatches = countTail(patches);
      const beforeCached = countTail(cached);
      await sendLine(page, 'Describe this image.\nframe.png', 'Describe this image.');
      // Fresh tower → patches; same bytes already seen this process → cache.
      await expect
        .poll(
          () =>
            countTail(patches) - beforePatches + (countTail(cached) - beforeCached),
        )
        .toBeGreaterThanOrEqual(1);
    });

    await step('follow-ups reuse vision cache', async () => {
      const beforePatches = countTail(patches);
      const beforeCached = countTail(cached);
      await sendLine(page, 'What color is it?', 'What color is it?');
      await sendLine(page, 'Name one shape.', 'Name one shape.', true);
      await expect.poll(() => countTail(patches) - beforePatches).toBe(0);
      await expect.poll(() => countTail(cached) - beforeCached).toBeGreaterThanOrEqual(2);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function sendLine(page: Page, text: string, shown: string, finish = false) {
  const box = page.getByRole('textbox', { name: 'Message', exact: true });
  await box.fill(text);
  if (finish) {
    await page.setViewportSize({ width: 1280, height: 640 });
    await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Chat', exact: true }).click();
    await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
    await page.setViewportSize({ width: 1280, height: 800 });
  }
  const send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await send.click();
  await expect(page.getByLabel(`You said: ${shown}`)).toBeVisible();
  await expect(page.getByTestId('working')).toBeVisible();
  await expect(page.getByTestId('working')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Send again' })).toHaveCount(0);
}

/** Count needle hits in the trailing window of the capped favlo.log. */
function countTail(needle: string, windowBytes = 512_000): number {
  if (!fs.existsSync(logFile)) {
    return 0;
  }
  const size = fs.statSync(logFile).size;
  const start = Math.max(0, size - windowBytes);
  const length = size - start;
  if (length <= 0) {
    return 0;
  }
  const fd = fs.openSync(logFile, 'r');
  try {
    const body = Buffer.alloc(length);
    fs.readSync(fd, body, 0, length, start);
    return body.toString('utf8').split(needle).length - 1;
  } finally {
    fs.closeSync(fd);
  }
}

function tinyPng(red: number, green: number, blue: number): Buffer {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = chunk('IHDR', Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]));
  const idat = chunk('IDAT', zlib.deflateSync(Buffer.from([0, red, green, blue])));
  return Buffer.concat([signature, ihdr, idat, chunk('IEND', Buffer.alloc(0))]);
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(zlib.crc32(body) >>> 0);
  return Buffer.concat([length, body, crc]);
}

import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';

const logFile = path.join(os.homedir(), '.favlo', 'logs', 'favlo.log');
const patches = 'vision patches 256';
const cached = 'vision cache 64';

test('an attached image is embedded once across later messages', async ({ page }) => {
  test.setTimeout(720_000);
  const models = await fetch('http://127.0.0.1:8129/v1/models');
  test.skip(!models.ok, 'favlo serve has no language model');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  const served = (listed.data ?? []).some((row) => row.id === 'qwen36-35b-a3b');
  test.skip(!served, 'qwen36-35b-a3b is not served');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-'));
  const stamp = Date.now() % 200 + 20;
  fs.writeFileSync(path.join(root, 'frame.png'), tinyPng(stamp, 40, 90));
  const name = `Frame ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://127.0.0.1:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await page.getByRole('button', { name: 'New workspace' }).click();
    await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
    const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
    await onScreen(createWorkspace);
    await createWorkspace.click();
    await expect(page.getByRole('heading', { name })).toBeVisible();

    await page.getByRole('button', { name: 'New folder' }).click();
    await page.getByRole('textbox', { name: 'Folder name' }).fill('Camera');
    await page.route('**/workspaces/*/folders/pick', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ cancelled: false, location: root }),
      });
    });
    await page.getByRole('button', { name: 'Choose folder' }).click();
    const saveFolder = page.getByRole('button', { name: 'Save folder' });
    await onScreen(saveFolder);
    await saveFolder.click();
    await page.getByRole('button', { name: 'frame.png', exact: true }).click();
    await expect(page.getByRole('img', { name: 'frame.png' })).toBeVisible();
    await page.getByRole('button', { name: 'Add to chat' }).click();
    await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue('frame.png');
    const anchor = logEnding();

    await sendLine(page, 'Describe this image.\nframe.png', 'Describe this image.');
    await expect.poll(() => countAdded(anchor, patches), { timeout: 30_000 }).toBe(1);

    await sendLine(page, 'What color is it?', 'What color is it?');
    await sendLine(page, 'Name one shape.', 'Name one shape.', true);
    await expect.poll(() => countAdded(anchor, patches), { timeout: 30_000 }).toBe(1);
    await expect.poll(() => countAdded(anchor, cached), { timeout: 30_000 }).toBeGreaterThanOrEqual(2);
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
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  await expect(page.getByRole('button', { name: 'Send again' })).toHaveCount(0);
}

function countAdded(anchor: string, needle: string): number {
  const body = logTail();
  const at = body.lastIndexOf(anchor);
  const added = at < 0 ? '' : body.slice(at + anchor.length);
  return added.split(needle).length - 1;
}

function logEnding(): string {
  return logTail().slice(-2000);
}

function logTail(): string {
  const size = fs.statSync(logFile).size;
  const start = Math.max(0, size - 1024 * 1024);
  const fd = fs.openSync(logFile, 'r');
  try {
    const body = Buffer.alloc(size - start);
    fs.readSync(fd, body, 0, body.length, start);
    return body.toString('utf8');
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

import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';

import { expect, test, type Page, type Request } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const logFile = path.join(os.homedir(), '.favlo', 'logs', 'favlo.log');

test('the image selected before send is the one the agent receives', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  const served = (listed.data ?? []).some((row) => row.id === 'qwen36-35b-a3b');
  test.skip(!served, 'qwen36-35b-a3b is not served');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-select-'));
  fs.writeFileSync(path.join(root, 'horse.png'), tinyPng(200, 30, 30));
  fs.writeFileSync(path.join(root, 'zebra.png'), tinyPng(30, 30, 200));
  const name = `Select ${Date.now()}`;
  let posted = '';
  const onRequest = (request: Request) => {
    if (request.method() !== 'POST' || !request.url().includes('/ag-ui')) {
      return;
    }
    const body = request.postDataJSON() as {
      state?: { favlo_workspace?: { selection?: { path?: string } } };
    } | null;
    const pathName = body?.state?.favlo_workspace?.selection?.path;
    if (pathName) {
      posted = pathName;
    }
  };
  page.on('request', onRequest);
  try {
    await step('create gallery with two images', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'zebra.png', exact: true })).toBeVisible();
    });

    await step('select zebra after horse, without a model run', async () => {
      const before = posted;
      const horse = page.getByRole('button', { name: 'horse.png', exact: true });
      await onScreen(horse);
      await horse.click();
      await expect(page.getByRole('button', { name: 'Download horse.png' })).toBeVisible();
      const zebra = page.getByRole('button', { name: 'zebra.png', exact: true });
      await onScreen(zebra);
      await zebra.click();
      await expect(page.getByRole('button', { name: 'Download zebra.png' })).toBeVisible();
      expect(posted).toBe(before);
    });

    await step('send carries zebra to the model', async () => {
      const beforePatches = countTail('vision patches');
      const beforeCached = countTail('vision cache');
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await box.fill('What is this?');
      const send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await expect(page.getByLabel('You said: What is this?')).toBeVisible();
      expect(posted).toBe('zebra.png');
      await expect
        .poll(() => countTail('vision patches') - beforePatches + (countTail('vision cache') - beforeCached))
        .toBeGreaterThanOrEqual(1);
      await expect(page.getByLabel('You said: What is this?')).toBeVisible();
      await expect(page.getByText('system_reminder')).toHaveCount(0);
    });
  } finally {
    page.off('request', onRequest);
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

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

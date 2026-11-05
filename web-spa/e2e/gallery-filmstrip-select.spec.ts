import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('a filmstrip click highlights that folder file', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  const served = (listed.data ?? []).some((row) => row.id === 'qwen36-35b-a3b');
  test.skip(!served, 'qwen36-35b-a3b is not served');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-strip-'));
  fs.writeFileSync(path.join(root, 'horse.png'), tinyPng(200, 30, 30));
  const name = `Strip ${Date.now()}`;
  try {
    await step('open a gallery that already shows horse', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toHaveAttribute(
        'aria-current',
        'true',
      );
    });

    await step('click a file that arrived after the gallery opened', async () => {
      fs.writeFileSync(path.join(root, 'late.png'), tinyPng(30, 30, 200));
      await page.reload();
      const late = page.getByRole('button', { name: 'late.png', exact: true });
      await expect(late).toBeVisible();
      await onScreen(late);
      await late.click();
      await expect(late).toHaveAttribute('aria-current', 'true');
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).not.toHaveAttribute(
        'aria-current',
        'true',
      );
      const download = page.getByRole('button', { name: 'Download late.png' });
      await expect(download).toBeVisible();
      await onScreen(download);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

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

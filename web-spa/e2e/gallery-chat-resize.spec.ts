import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

let workspaceName = '';
let folderRoot = '';

test.afterAll(async () => {
  if (workspaceName) {
    await deleteWorkspacesNamed(workspaceName);
  }
  if (folderRoot) {
    fs.rmSync(folderRoot, { recursive: true, force: true });
  }
});

test('wide gallery chat width resizes within limits', async ({ page }) => {
  folderRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-resize-'));
  fs.writeFileSync(
    path.join(folderRoot, 'a.png'),
    Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
      'base64',
    ),
  );
  workspaceName = `Resize ${Date.now()}`;
  await step('open wide gallery', async () => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://127.0.0.1:8012/');
    await createGalleryWorkspace(page, workspaceName, folderRoot);
    await expect(page.getByLabel('Conversation')).toBeVisible();
  });

  await step('drag chat wider then clamp at the max', async () => {
    const handle = page.getByRole('slider', { name: 'Resize chat' });
    await onScreen(handle);
    const before = await page.getByLabel('Conversation').evaluate((node) => node.getBoundingClientRect().width);
    const box = await handle.boundingBox();
    expect(box).not.toBeNull();
    if (box == null) {
      return;
    }
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x - 200, box.y + box.height / 2, { steps: 8 });
    await page.mouse.up();
    const wider = await page.getByLabel('Conversation').evaluate((node) => node.getBoundingClientRect().width);
    expect(wider).toBeGreaterThan(before + 40);
    expect(wider).toBeLessThanOrEqual(560);

    const again = await handle.boundingBox();
    expect(again).not.toBeNull();
    if (again == null) {
      return;
    }
    await page.mouse.move(again.x + again.width / 2, again.y + again.height / 2);
    await page.mouse.down();
    await page.mouse.move(again.x - 800, again.y + again.height / 2, { steps: 12 });
    await page.mouse.up();
    const maxed = await page.getByLabel('Conversation').evaluate((node) => node.getBoundingClientRect().width);
    expect(maxed).toBeLessThanOrEqual(560);
    expect(maxed).toBeGreaterThanOrEqual(320);
  });

  await step('drag chat narrower then clamp at the min', async () => {
    const handle = page.getByRole('slider', { name: 'Resize chat' });
    await onScreen(handle);
    const box = await handle.boundingBox();
    expect(box).not.toBeNull();
    if (box == null) {
      return;
    }
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + 900, box.y + box.height / 2, { steps: 12 });
    await page.mouse.up();
    const minned = await page.getByLabel('Conversation').evaluate((node) => node.getBoundingClientRect().width);
    expect(minned).toBe(320);
  });

  await step('reload keeps the clamped width', async () => {
    await page.reload();
    await expect(page.getByRole('heading', { name: workspaceName })).toBeVisible();
    const kept = await page.getByLabel('Conversation').evaluate((node) => node.getBoundingClientRect().width);
    expect(kept).toBe(320);
    const handle = page.getByRole('slider', { name: 'Resize chat' });
    await onScreen(handle);
  });
});

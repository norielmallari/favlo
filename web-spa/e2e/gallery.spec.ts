import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test('a gallery preview and filmstrip browse local folders', async ({ page }) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-'));
  const albums = path.join(root, 'Albums');
  fs.mkdirSync(albums);
  fs.writeFileSync(path.join(root, 'horse.png'), TINY_PNG);
  fs.writeFileSync(path.join(root, 'cat.png'), TINY_PNG);
  fs.writeFileSync(path.join(albums, 'zebra.png'), TINY_PNG);
  const name = `Stills ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await expect(page.getByRole('button', { name: 'New workspace' })).toBeVisible();

    await page.getByRole('button', { name: 'New workspace' }).click();
    await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
    const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
    await onScreen(createWorkspace);
    await createWorkspace.click();
    await expect(page.getByRole('heading', { name })).toBeVisible();
    await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
    await expect(page.getByText('No folders')).toBeVisible();

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
    await expect(page.getByRole('button', { name: 'Camera' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Albums' })).toBeVisible();
    await page.getByRole('button', { name: 'horse.png', exact: true }).click();
    await expect(page.getByRole('img', { name: 'horse.png' })).toBeVisible();

    await page.getByRole('button', { name: 'Albums' }).click();
    await expect(page.getByLabel('Breadcrumb').getByText('Albums')).toBeVisible();
    await expect(page.getByRole('img', { name: 'zebra.png' })).toBeVisible();

    await page.getByLabel('Breadcrumb').getByRole('button', { name }).click();
    await page.getByRole('button', { name: 'cat.png', exact: true }).click();
    await expect(page.getByRole('img', { name: 'cat.png' })).toBeVisible();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('img', { name: 'horse.png' })).toBeVisible();
    await page.keyboard.press('ArrowLeft');
    await expect(page.getByRole('img', { name: 'cat.png' })).toBeVisible();

    await page.setViewportSize({ width: 1280, height: 640 });
    await page.getByRole('button', { name: 'New folder' }).click();
    const sheetSave = page.getByRole('button', { name: 'Save folder' });
    await onScreen(sheetSave);
    await page.getByRole('button', { name: 'Close new folder' }).click();

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.getByRole('button', { name: 'Move horse.png' }).click();
    const moved = page.waitForResponse(
      (response) => response.url().includes('/images/move') && response.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Albums' }).click();
    await moved;
    await expect(page.getByRole('button', { name: 'horse.png' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Albums' }).click();
    await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();

    await page.getByLabel('Breadcrumb').getByRole('button', { name }).click();
    await page.getByRole('button', { name: 'cat.png', exact: true }).click();
    await page.getByRole('button', { name: 'Delete cat.png' }).click();
    await page.getByRole('button', { name: 'Delete cat.png' }).click();
    await expect(page.getByRole('button', { name: 'cat.png', exact: true })).toHaveCount(0);

    await page.getByRole('button', { name: 'Albums' }).click();
    await expect(page.getByRole('button', { name: 'Delete Albums' })).toBeDisabled();
    await page.getByRole('button', { name: 'horse.png', exact: true }).click();
    await page.getByRole('button', { name: 'Delete horse.png' }).click();
    await page.getByRole('button', { name: 'Delete horse.png' }).click();
    await expect(page.getByRole('button', { name: 'horse.png' })).toHaveCount(0);
    const removeFolder = page.getByRole('button', { name: 'Delete Albums' });
    await expect(removeFolder).toBeEnabled();
    await removeFolder.click();
    await page.getByRole('button', { name: 'Delete Albums' }).click();
    await expect(page.getByRole('button', { name: 'Albums' })).toHaveCount(0);

    await page.getByRole('button', { name: 'zebra.png', exact: true }).click();
    await page.getByRole('button', { name: 'Add to chat' }).click();
    await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue('zebra.png');
    const chatLine = `Gallery path ${Date.now()}`;
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(`${chatLine}\nzebra.png`);
    let send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    const posted = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().includes('/session/messages'),
    );
    await send.click();
    const body = (await posted).postDataJSON() as { image?: string; attachment?: string; text?: string };
    expect(body.image).toBe('zebra.png');
    expect(body.attachment).toBe('zebra.png');
    await expect(page.getByLabel(`You said: ${chatLine}`)).toBeVisible();

    await page.getByRole('button', { name: 'New folder' }).click();
    await page.getByRole('combobox', { name: 'Folder type' }).selectOption('cloud');
    await page.getByRole('combobox', { name: 'Cloud provider' }).selectOption('onedrive');
    await page.getByRole('textbox', { name: 'Folder name' }).fill('OneDrive');
    await page.getByRole('button', { name: 'Save folder' }).click();
    await expect(page.getByRole('button', { name: 'OneDrive' })).toBeVisible();

    await expect(page.getByRole('button', { name: 'Back to channel' })).toBeVisible();
    await expect(page.getByLabel('Conversation')).toBeVisible();
    await expect(page.getByLabel('Breadcrumb')).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByRole('button', { name: 'Chat' })).toBeVisible();
    await page.getByRole('button', { name: 'Chat' }).click();
    await expect(page.getByLabel('Conversation')).toBeVisible();
    await page.getByRole('button', { name: 'Back to gallery' }).click();
    await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Breadcrumb').getByText(name)).toBeVisible();
    await expectFits(page);
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

async function expectFits(page: Page) {
  await onScreen(page.getByRole('button', { name: 'Configuration' }));
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

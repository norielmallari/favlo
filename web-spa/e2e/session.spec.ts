import { expect, test } from '@playwright/test';

import fs from 'fs';
import os from 'os';
import path from 'path';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';

test('session clear and archive stay on their session', async ({ page }) => {
  const models = await liveModels();
  const channelLine = `Channel note ${Date.now()}`;
  const workspaceLine = `Workspace note ${Date.now()}`;
  const workspaceName = `Session ${Date.now()}`;
  const folderRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-session-'));
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);

    const channel = page.getByLabel('Conversation').filter({ has: page.getByRole('heading', { name: 'Favlo', exact: true }) });
    const gallery = () => page.getByLabel('Gallery workspace');

    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(channelLine);
    let send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(channel.getByLabel(`You said: ${channelLine}`)).toBeVisible();

    await page.getByRole('button', { name: 'New workspace' }).click();
    await page.getByRole('textbox', { name: 'Workspace name' }).fill(workspaceName);
    const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
    await onScreen(createWorkspace);
    await createWorkspace.click();
    await expect(page.getByRole('heading', { name: workspaceName })).toBeVisible();

    await page.getByRole('button', { name: 'New folder' }).click();
    await page.getByRole('textbox', { name: 'Folder name' }).fill('Stills');
    await page.route('**/workspaces/*/folders/pick', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ cancelled: false, location: folderRoot }),
      });
    });
    await page.getByRole('button', { name: 'Choose folder' }).click();
    const saveFolder = page.getByRole('button', { name: 'Save folder' });
    await onScreen(saveFolder);
    await saveFolder.click();
    await expect(page.getByText('No images in this folder')).toBeVisible();

    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(workspaceLine);
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toBeVisible();

    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('/session clear');
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toHaveCount(0);
    await expect(page.getByRole('heading', { name: workspaceName })).toBeVisible();

    await page.getByRole('button', { name: 'Back to channel' }).click();
    await expect(channel.getByLabel(`You said: ${channelLine}`)).toBeVisible();

    await page.getByRole('button', { name: workspaceName, exact: true }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(workspaceLine);
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toBeVisible();

    await page.getByRole('button', { name: 'Home', exact: true }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('/session clear');
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(channel.getByLabel(`You said: ${channelLine}`)).toHaveCount(0);

    await page.getByRole('button', { name: workspaceName, exact: true }).click();
    await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toBeVisible();

    const ordinary = `Please /session clear later ${Date.now()}`;
    await page.getByRole('button', { name: 'Home', exact: true }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(ordinary);
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(channel.getByLabel(`You said: ${ordinary}`)).toBeVisible();

    test.skip(models.length === 0, 'favlo serve has no language model');
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('/session archive');
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(channel.getByLabel(`You said: ${ordinary}`)).toHaveCount(0);
    await expect(page.getByTestId('working')).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 360_000 });
    await expect(channel.getByLabel(/^You said: /)).toHaveCount(0);
    await page.getByRole('button', { name: workspaceName, exact: true }).click();
    await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: workspaceName, exact: true }).click();
    await page.getByRole('button', { name: 'Chat' }).click();
    const narrowLine = `Narrow workspace ${Date.now()}`;
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(narrowLine);
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(gallery().getByLabel(`You said: ${narrowLine}`)).toBeVisible();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill('/session clear');
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    await expect(gallery().getByLabel(`You said: ${narrowLine}`)).toHaveCount(0);
    await expect(page.getByRole('heading', { name: workspaceName })).toBeVisible();
  } finally {
    await deleteWorkspacesNamed(workspaceName);
    fs.rmSync(folderRoot, { recursive: true, force: true });
  }
});

async function liveModels(): Promise<string[]> {
  try {
    const response = await fetch('http://127.0.0.1:8129/v1/models');
    if (!response.ok) {
      return [];
    }
    const body = (await response.json()) as { data?: { id?: string }[] };
    return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
  } catch {
    return [];
  }
}

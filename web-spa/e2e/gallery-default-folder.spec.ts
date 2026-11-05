import fs from 'fs';

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';
import { step } from './progress';

test('a gallery opens when no folder is chosen', async ({ page, request }) => {
  const name = `Default ${Date.now()}`;
  let location = '';
  try {
    await step('create a gallery with the default folder', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
      await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      const create = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(create);
      const popupPromise = page.waitForEvent('popup');
      await create.click();
      const workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      await expect(
        workspace.getByRole('button', { name: `${name}, Gallery workspace menu` }),
      ).toBeVisible();
      const id = new URL(workspace.url()).pathname.split('/').pop();
      const listed = await request.get(`http://127.0.0.1:8130/workspaces/${id}/folders`);
      expect(listed.ok()).toBeTruthy();
      location = String((await listed.json()).folders[0].location);
      expect(location.endsWith(`/workspaces/${name}`)).toBeTruthy();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    if (location.endsWith(`/workspaces/${name}`)) {
      fs.rmSync(location, { recursive: true, force: true });
    }
  }
});

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { chooseFolder, makeWorkspaceRoot, removeWorkspaceRoot } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('each workspace on the channel menu can be deleted', async ({ page }) => {
  const name = `Menu ${Date.now()}`;
  const root = makeWorkspaceRoot();
  try {
    await step('create a workspace from the channel menu', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      const open = page.getByRole('button', { name: 'New workspace for Favlo' });
      await page.setViewportSize({ width: 1280, height: 640 });
      await onScreen(open);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(open);
      await open.click();
      const form = page.getByRole('dialog', { name: 'New workspace for Favlo' });
      await expect(form.getByText('New workspace for Favlo')).toBeVisible();
      await form.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      await page.getByRole('combobox', { name: 'Workspace type' }).selectOption('coding');
      await chooseFolder(page, root);
      const create = page.getByRole('button', { name: 'Create workspace' });
      await page.setViewportSize({ width: 1280, height: 640 });
      await onScreen(create);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(create);
      const popupPromise = page.waitForEvent('popup');
      await create.click();
      const workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      await workspace.close();
    });

    await step('delete asks first, then the row is gone', async () => {
      const menu = page.getByLabel('Workspaces');
      const remove = menu.getByRole('button', { name: `Delete ${name}`, exact: true });
      await page.setViewportSize({ width: 1280, height: 640 });
      await onScreen(remove);
      await page.setViewportSize({ width: 390, height: 844 });
      await page.getByRole('button', { name: 'Channels' }).click();
      await onScreen(remove);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(remove);
      await remove.click();
      const dialog = page.getByRole('dialog', { name: 'Delete workspace' });
      await expect(dialog.getByText(`Delete ${name}?`)).toBeVisible();
      await expect(dialog.getByText('Folders and files on disk stay.')).toBeVisible();
      const confirm = dialog.getByRole('button', { name: `Delete ${name}`, exact: true });
      await page.setViewportSize({ width: 1280, height: 640 });
      await onScreen(confirm);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(confirm);
      await confirm.click();
      await expect(menu.getByRole('button', { name, exact: true })).toHaveCount(0);
      await expect(menu.getByRole('button', { name: `Delete ${name}`, exact: true })).toHaveCount(0);
      await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    removeWorkspaceRoot(root);
  }
});

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';
import { step } from './progress';

test('creating a coding workspace picks the type and delete asks first', async ({ page }) => {
  const name = `Repos ${Date.now()}`;
  const line = `Coding note ${Date.now()}`;
  try {
    await step('create coding workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await page.getByRole('button', { name: 'New workspace' }).click();
      await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      await page.getByRole('combobox', { name: 'Workspace type' }).selectOption('coding');
      await page.setViewportSize({ width: 1280, height: 640 });
      const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(createWorkspace);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(createWorkspace);
      await createWorkspace.click();
      await expect(page.getByRole('heading', { name })).toBeVisible();
      await expect(page.getByLabel('Coding workspace')).toBeVisible();
      await expect(page.getByText('Coding', { exact: true })).toBeVisible();
    });

    await step('chat once then delete with confirm', async () => {
      await page.getByRole('textbox', { name: 'Message', exact: true }).fill(line);
      const send = page.getByRole('button', { name: 'Send' });
      await onScreen(send);
      await send.click();
      await expect(page.getByLabel(`You said: ${line}`)).toBeVisible();
      await page.getByRole('button', { name: 'Delete workspace' }).click();
      await expect(page.getByRole('dialog', { name: 'Delete workspace' })).toBeVisible();
      await expect(page.getByText('Folders and files on disk stay.')).toBeVisible();
      const confirm = page.getByRole('button', { name: `Delete ${name}`, exact: true });
      await onScreen(confirm);
      await confirm.click();
      await expect(page.getByLabel('Coding workspace')).toHaveCount(0);
      await expect(page.getByLabel('Workspaces').getByRole('button', { name, exact: true })).toHaveCount(0);
      await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

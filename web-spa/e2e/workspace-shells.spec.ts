import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { chooseFolder, makeWorkspaceRoot, removeWorkspaceRoot, waitUntilProvisioned } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

type Shell = { id: string; label: string; setup: string };

test('new workspace lists the shells from the API', async ({ page, request }) => {
  test.setTimeout(360_000);
  const listed = await request.get('http://127.0.0.1:8130/shells');
  expect(listed.ok()).toBeTruthy();
  const shells = (await listed.json()).shells as Shell[];
  expect(shells.length).toBeGreaterThan(0);
  const chat = shells.find((item) => item.id === 'coding');
  expect(chat?.setup).toBe('local_folder');
  const name = `Shell ${Date.now()}`;
  const root = makeWorkspaceRoot();
  try {
    await step('the picker shows the shell labels', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await page.goto('http://127.0.0.1:8012/');
      await page.getByRole('button', { name: 'Channels' }).click();
      await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
      const picker = page.getByRole('combobox', { name: 'Workspace type' });
      await onScreen(picker);
      await expect(picker.locator('option')).toHaveText(shells.map((item) => item.label));
      await page.getByRole('button', { name: 'Close new workspace' }).click();
    });

    await step('creating uses a shell from that list', async () => {
      await page.setViewportSize({ width: 1280, height: 640 });
      await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
      await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      const picker = page.getByRole('combobox', { name: 'Workspace type' });
      await onScreen(picker);
      await picker.selectOption(chat!.id);
      await chooseFolder(page, root);
      const create = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(create);
      const popupPromise = page.waitForEvent('popup');
      await create.click();
      const workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      await waitUntilProvisioned(workspace);
      await expect(workspace.getByRole('button', { name: `${name}, Coding workspace menu` })).toBeVisible();
      await expect(workspace.getByLabel('Coding workspace', { exact: true })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    removeWorkspaceRoot(root);
  }
});

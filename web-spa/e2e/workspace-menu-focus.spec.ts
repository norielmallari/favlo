import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { chooseFolder, makeWorkspaceRoot, removeWorkspaceRoot } from './gallery-create';
import { expectFullFocusRing } from './focus-ring';
import { onScreen } from './on-screen';
import { step } from './progress';

test('wide workspace menu focus ring runs along every side', async ({ page }) => {
  const name = `Ring ${Date.now()}`;
  const root = makeWorkspaceRoot();
  let workspace: Page | undefined;
  try {
    await step('open a workspace on a wide window', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      const open = page.getByRole('button', { name: 'New workspace for Favlo' });
      await onScreen(open);
      await open.click();
      const form = page.getByRole('dialog', { name: 'New workspace for Favlo' });
      await form.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      await page.getByRole('combobox', { name: 'Workspace type' }).selectOption('coding');
      await chooseFolder(page, root);
      const create = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(create);
      const popupPromise = page.waitForEvent('popup');
      await create.click();
      workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      const url = workspace.url();
      await workspace.close();
      await page.goto(url);
      await page.setViewportSize({ width: 1280, height: 800 });
    });

    const menu = page.getByRole('button', { name: `${name}, Coding workspace menu` });

    await step('keyboard focus on the workspace menu draws a full ring', async () => {
      await onScreen(menu);
      await menu.focus();
      await expectFullFocusRing(page, menu);
    });

    await step('the same ring holds on a narrow window', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await onScreen(menu);
      await menu.focus();
      await expectFullFocusRing(page, menu);
    });

    await step('a menu row keeps the ring on every side', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(menu);
      await menu.click();
      const skills = page.getByRole('menuitem', { name: 'Skills' });
      await onScreen(skills);
      await skills.focus();
      await expectFullFocusRing(page, skills);
    });
  } finally {
    await workspace?.close().catch(() => undefined);
    await deleteWorkspacesNamed(name);
    removeWorkspaceRoot(root);
  }
});

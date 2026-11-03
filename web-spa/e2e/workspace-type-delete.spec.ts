import { expect, test, type Locator, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { waitUntilProvisioned } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('creating a coding workspace picks the type and delete asks first', async ({ page }) => {
  test.setTimeout(360_000);
  const name = `Repos ${Date.now()}`;
  const line = `Coding note ${Date.now()}`;
  let workspace: import('@playwright/test').Page;
  try {
    await step('create coding workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
      await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      await page.getByRole('combobox', { name: 'Workspace type' }).selectOption('coding');
      await page.setViewportSize({ width: 1280, height: 640 });
      const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(createWorkspace);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(createWorkspace);
      const popupPromise = page.waitForEvent('popup');
      await createWorkspace.click();
      workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      await waitUntilProvisioned(workspace);
      await expect(workspace.getByRole('heading', { name })).toBeVisible();
      await expect(workspace.getByLabel('Coding workspace')).toBeVisible();
      await expect(workspace.getByText('Coding', { exact: true })).toBeVisible();
    });

    await step('chat once then delete with confirm', async () => {
      await workspace.getByRole('textbox', { name: 'Message', exact: true }).fill(line);
      const send = workspace.getByRole('button', { name: 'Send' });
      await onScreen(send);
      await send.click();
      await expect(workspace.getByLabel(`You said: ${line}`)).toBeVisible();
      await workspace.getByRole('button', { name: 'Delete workspace' }).click();
      await expect(workspace.getByRole('dialog', { name: 'Delete workspace' })).toBeVisible();
      await expect(workspace.getByText('Folders and files on disk stay.')).toBeVisible();
      const confirm = workspace.getByRole('button', { name: `Delete ${name}`, exact: true });
      await onScreen(confirm);
      await confirm.click();
      const gone = workspace.getByTestId('workspace-gone');
      const heading = workspace.getByRole('heading', { name: 'This workspace is gone.' });
      await onScreen(heading);
      await expect(workspace.getByText('This tab can be closed. Files on disk stay where they were.')).toBeVisible();
      await expectCentered(workspace, gone);
      await workspace.setViewportSize({ width: 390, height: 844 });
      await onScreen(heading);
      await expectCentered(workspace, gone);
      await workspace.setViewportSize({ width: 1280, height: 800 });
      await expect(page.getByLabel('Workspaces').getByRole('button', { name, exact: true })).toHaveCount(0);
      await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

async function expectCentered(page: Page, card: Locator) {
  const box = await card.boundingBox();
  const size = page.viewportSize();
  expect(box).toBeTruthy();
  expect(size).toBeTruthy();
  if (box == null || size == null) {
    return;
  }
  const midX = box.x + box.width / 2;
  const midY = box.y + box.height / 2;
  expect(Math.abs(midX - size.width / 2)).toBeLessThan(24);
  expect(Math.abs(midY - size.height / 2)).toBeLessThan(80);
}

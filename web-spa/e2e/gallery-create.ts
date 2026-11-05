import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { expect, test, type Page } from '@playwright/test';

import { onScreen } from './on-screen';

export function makeWorkspaceRoot(): string {
  return mkdtempSync(join(tmpdir(), 'favlo-workspace-'));
}

export function removeWorkspaceRoot(root: string) {
  rmSync(root, { recursive: true, force: true });
}

export async function chooseFolder(page: Page, root: string) {
  await page.route('**/folders/pick', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ cancelled: false, location: root }),
    });
  });
  const choose = page.getByRole('button', { name: 'Choose folder' });
  await onScreen(choose);
  await choose.click();
  await expect(page.getByLabel('Folder location')).toHaveValue(root);
}

/**
 * Create a gallery workspace. Create opens `/w/<id>` in a new tab.
 * By default the original page then follows that URL, so later steps keep using `page`.
 * Pass `keepChannel` when the channel document must stay open. The return value is the workspace page.
 */
export async function waitUntilProvisioned(workspace: Page) {
  const card = workspace.getByTestId('creating-workspace');
  await expect(card).toBeVisible();
  await onScreen(card);
  const box = await card.boundingBox();
  const size = workspace.viewportSize();
  expect(box).toBeTruthy();
  expect(size).toBeTruthy();
  if (box != null && size != null) {
    expect(Math.abs(box.x + box.width / 2 - size.width / 2)).toBeLessThan(24);
    expect(Math.abs(box.y + box.height / 2 - size.height / 2)).toBeLessThan(80);
  }
  await expect(card).toHaveCount(0, { timeout: 180_000 });
}

export async function createGalleryWorkspace(
  page: Page,
  name: string,
  root: string,
  options?: { keepChannel?: boolean },
): Promise<Page> {
  test.setTimeout(360_000);
  await page.route('**/folders/pick', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ cancelled: false, location: root }),
    });
  });
  await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
  await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
  await page.getByRole('button', { name: 'Choose folder' }).click();
  await expect(page.getByLabel('Folder location')).toHaveValue(root);
  const create = page.getByRole('button', { name: 'Create workspace' });
  await onScreen(create);
  const popupPromise = page.waitForEvent('popup');
  await create.click();
  const workspace = await popupPromise;
  await expect(workspace).toHaveURL(/\/w\/.+/);
  await expect(workspace.getByRole('button', { name: `${name}, Gallery workspace menu` })).toBeVisible();
  const list = page.getByLabel('Workspaces');
  if (await list.isVisible()) {
    await expect(list.getByTestId('workspace-creating').first()).toBeVisible();
  }
  await waitUntilProvisioned(workspace);
  await expect(workspace.getByLabel('Gallery workspace', { exact: true })).toBeVisible();
  await expect(workspace.getByText('Choose a local folder')).toHaveCount(0);
  await expect(workspace.getByLabel(/^Favlo said:/).first()).toBeVisible();
  if (options?.keepChannel) {
    return workspace;
  }
  const url = workspace.url();
  await workspace.close();
  await page.goto(url);
  await expect(page.getByRole('button', { name: `${name}, Gallery workspace menu` })).toBeVisible();
  await expect(page.getByLabel('Gallery workspace', { exact: true })).toBeVisible();
  await expect(page.getByLabel(/^Favlo said:/).first()).toBeVisible();
  return page;
}

import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';

test('a gallery workspace needs a local root before the grid opens', async ({ page }) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-'));
  const name = `Stills ${Date.now()}`;
  try {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await expect(page.getByRole('heading', { name: 'Workspaces' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New workspace' })).toBeVisible();

  await page.getByRole('button', { name: 'New workspace' }).click();
  await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
  await page.getByRole('button', { name: 'Create workspace' }).click();
  await expect(page.getByRole('heading', { name })).toBeVisible();
  await expect(page.getByLabel('Gallery root')).toBeVisible();
  const type = page.getByRole('combobox', { name: 'Folder type' });
  await expect(type).toHaveValue('local');
  await expect(type.locator('option[value="cloud"]')).toBeDisabled();
  await expect(page.getByRole('combobox', { name: 'Cloud provider' })).toHaveCount(0);
  await expect(page.getByRole('option', { name: 'OneDrive' })).toHaveCount(0);
  await expect(page.getByRole('option', { name: 'Dropbox' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Choose folder' })).toBeVisible();
  await page.route('**/workspaces/*/root/pick', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ cancelled: false, location: root }),
    });
  });
  await page.getByRole('button', { name: 'Choose folder' }).click();
  await expect(page.getByRole('textbox', { name: 'Location' })).toHaveValue(root);
  await page.getByRole('button', { name: 'Save root' }).click();
  await expect(page.getByLabel('Gallery grid')).toBeVisible();
  await expect(page.getByText('No images in this folder')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Back to channel' })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Workspaces' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Folders' })).toBeVisible();
  await expectFits(page);
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

async function expectFits(page: Page) {
  await expect(page.getByRole('button', { name: 'Configuration' })).toBeInViewport();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

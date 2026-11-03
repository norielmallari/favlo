import { expect, type Page } from '@playwright/test';

import { onScreen } from './on-screen';

/** Create a gallery workspace with the local folder chosen in the new-workspace modal. */
export async function createGalleryWorkspace(page: Page, name: string, root: string): Promise<void> {
  await page.route('**/folders/pick', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ cancelled: false, location: root }),
    });
  });
  await page.getByRole('button', { name: 'New workspace' }).click();
  await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
  await page.getByRole('button', { name: 'Choose folder' }).click();
  await expect(page.getByLabel('Folder location')).toHaveValue(root);
  const create = page.getByRole('button', { name: 'Create workspace' });
  await onScreen(create);
  await create.click();
  await expect(page.getByRole('heading', { name })).toBeVisible();
  await expect(page.getByLabel('Gallery workspace')).toBeVisible();
  await expect(page.getByText('Choose a local folder')).toHaveCount(0);
}

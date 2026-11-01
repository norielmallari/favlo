import { expect, type Page } from '@playwright/test';

import { onScreen } from './on-screen';

export async function createAgent(page: Page, name: string): Promise<void> {
  const dialog = page.getByRole('dialog', { name: 'New agent' });
  const create = dialog.getByRole('button', { name: 'Create agent' });
  await onScreen(create);
  await create.click();
  await expect(page.getByTestId('save-confirmation')).toContainText(`${name} was created.`);
  await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
}

export async function openAgentChat(page: Page, name: string): Promise<void> {
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  const row = page.getByLabel('Channels').getByRole('button', { name, exact: true });
  await expect(row).toBeVisible({ timeout: 30_000 });
  await onScreen(row);
  await row.click();
  const home = page.getByLabel('Channels').getByRole('button', { name: 'Home', exact: true });
  await onScreen(home);
  await home.click();
  await expect(page.getByLabel('Conversation').getByRole('heading', { name, exact: true })).toBeVisible({
    timeout: 30_000,
  });
}

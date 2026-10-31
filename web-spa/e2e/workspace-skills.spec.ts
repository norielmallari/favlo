import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';

test('a gallery workspace stores a skill for its agents', async ({ page }) => {
  const name = `Captions ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8081/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
    await page.getByRole('button', { name: 'New workspace' }).click();
    await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
    await page.getByRole('button', { name: 'Create workspace' }).click();
    await expect(page.getByRole('heading', { name })).toBeVisible();
    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toBeVisible();
    await expect(page.getByText('No skills')).toBeVisible();
    await page.getByRole('textbox', { name: 'Skill name' }).fill('Photo captions');
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
    await page.getByRole('button', { name: 'Add skill' }).click();
    await expect(page.getByText('Photo captions')).toBeVisible();
    await expect(page.getByText('Write a one-line caption.')).toBeVisible();
    await expect(page.getByText('No skills')).toHaveCount(0);
    await page.getByRole('button', { name: 'Close skills' }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toHaveCount(0);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toBeVisible();
    await expect(page.getByText('Photo captions')).toBeVisible();
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

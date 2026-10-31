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
    await expect(page.getByRole('button', { name: 'manage-gallery', exact: true })).toBeVisible();
    await expect(page.getByText('No skills')).toHaveCount(0);
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name' }).fill('Photo captions');
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
    await page.getByRole('button', { name: 'Create skill' }).click();
    await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue('Write a one-line caption.');
    await page.getByRole('textbox', { name: 'Skill description' }).click();
    await page.keyboard.press('ControlOrMeta+A');
    await page.keyboard.insertText('Prefer the folder palette.');
    await page.getByRole('button', { name: 'Save skill' }).click();
    await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue('Prefer the folder palette.');
    await page.getByRole('button', { name: 'Back to skills' }).click();
    await expect(page.getByRole('button', { name: 'Photo captions', exact: true })).toBeVisible();
    await expect(page.getByText('No skills')).toHaveCount(0);
    await page.getByRole('button', { name: 'Close skills' }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toHaveCount(0);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Photo captions', exact: true })).toBeVisible();
  } finally {
    await deleteWorkspacesNamed(name);
  }
});

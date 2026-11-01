import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { onScreen } from './on-screen';

test('a gallery workspace stores a skill for its agents', async ({ page }) => {
  const name = `Captions ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await page.getByRole('button', { name: 'New workspace' }).click();
    await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
    const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
    await onScreen(createWorkspace);
    await createWorkspace.click();
    await expect(page.getByRole('heading', { name })).toBeVisible();
    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Skills' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'manage-gallery', exact: true })).toBeVisible();
    await expect(page.getByText('No skills')).toHaveCount(0);
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name' }).fill('Photo captions');
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
    const createSkill = page.getByRole('button', { name: 'Create skill' });
    await onScreen(createSkill);
    await createSkill.click();
    await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue('Write a one-line caption.');
    await page.getByRole('textbox', { name: 'Skill description' }).click();
    await page.keyboard.press('ControlOrMeta+A');
    await page.keyboard.insertText('Prefer the folder palette.');
    await page.getByRole('button', { name: 'Save skill' }).click();
    await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue('Prefer the folder palette.');
    await page.getByRole('button', { name: 'Back to skills' }).click();
    await expect(page.getByRole('button', { name: 'Photo captions', exact: true })).toBeVisible();

    await page.setViewportSize({ width: 1280, height: 640 });
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name' }).fill('Short captions');
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Short window caption.');
    const shortCreate = page.getByRole('button', { name: 'Create skill' });
    await onScreen(shortCreate);
    await shortCreate.click();
    await page.getByRole('button', { name: 'Back to skills' }).click();
    await expect(page.getByRole('button', { name: 'Short captions', exact: true })).toBeVisible();

    await page.setViewportSize({ width: 1280, height: 800 });
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

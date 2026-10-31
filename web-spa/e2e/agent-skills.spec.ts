import { expect, test } from '@playwright/test';

import { deleteAgentSkillsNamed } from './cleanup';

test('configuration stores a skill on an agent', async ({ page }) => {
  const name = `Captions ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8081/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
    await page.getByRole('button', { name: 'Configuration' }).click();
    await page.getByRole('tab', { name: 'Skills' }).click();
    await expect(page.getByLabel('Agents list')).toBeVisible();
    await expect(page.getByLabel('Skills list')).toBeVisible();
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name' }).fill(name);
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
    await page.getByRole('button', { name: 'Create skill' }).click();
    await expect(page.getByLabel('Skills list').getByRole('button', { name, exact: true })).toBeVisible();
    await expect(page.getByText('Write a one-line caption.')).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Sections' }).click();
    await page.getByRole('tab', { name: 'Skills' }).click();
    await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
    await expect(page.getByLabel('Skills list').getByRole('button', { name, exact: true })).toBeVisible();
  } finally {
    await deleteAgentSkillsNamed(name);
  }
});

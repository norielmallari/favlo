import { expect, test } from '@playwright/test';

import { deleteAgentSkillsNamed, deleteGlobalSkillsNamed } from './cleanup';
import { onScreen } from './on-screen';

test('configuration stores a global skill and copies it to an agent', async ({ page }) => {
  const name = `Captions ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await page.getByRole('button', { name: 'Configuration' }).click();
    await page.getByRole('tab', { name: 'Skills' }).click();
    await expect(page.getByLabel('Skills list')).toBeVisible();
    await expect(page.getByLabel('Agents list')).toHaveCount(0);
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name' }).fill(name);
    await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
    const create = page.getByRole('button', { name: 'Create skill' });
    await onScreen(create);
    await create.click();
    await expect(page.getByTestId('save-confirmation')).toContainText('The skill was created.');
    await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
    await expect(page.getByLabel('Skills list').getByRole('button', { name, exact: true })).toBeVisible();
    await expect(page.getByRole('textbox', { name: 'Skill description', exact: true })).toHaveValue('Write a one-line caption.');

    await page.setViewportSize({ width: 1280, height: 640 });
    await page.getByRole('button', { name: 'New skill' }).click();
    await page.getByRole('textbox', { name: 'Skill name', exact: true }).fill(`${name} short`);
    await page.getByRole('textbox', { name: 'Skill description', exact: true }).fill('Short window skill.');
    const shortCreate = page.getByRole('button', { name: 'Create skill', exact: true });
    await onScreen(shortCreate);
    await shortCreate.click();
    await expect(page.getByTestId('save-confirmation')).toContainText('The skill was created.');
    await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
    await expect(page.getByLabel('Skills list').getByRole('button', { name: `${name} short`, exact: true })).toBeVisible();

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.getByLabel('Skills list').getByRole('button', { name, exact: true }).click();
    await page.getByRole('button', { name: 'Copy to agent', exact: true }).click();
    await expect(page.getByText('Copied to Favlo.')).toBeVisible();
    await page.getByRole('tab', { name: 'Agents' }).click();
    await page.getByLabel('Agent details').getByRole('tab', { name: 'Skills' }).click();
    await expect(page.getByLabel('Agent skills').getByRole('button', { name, exact: true })).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Sections' }).click();
    await page.getByRole('tab', { name: 'Skills' }).click();
    await expect(page.getByLabel('Skills list').getByRole('button', { name, exact: true })).toBeVisible();
  } finally {
    await deleteGlobalSkillsNamed(name, `${name} short`);
    await deleteAgentSkillsNamed(name);
  }
});

import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';

test('working sweeps a light band across the word during a reply', async ({ page }) => {
  const name = `Wren ${Date.now()}`;
  try {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:instruct');
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('status')).toContainText(`${name} was created.`);
  await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
  await page.getByLabel('Channels').getByRole('button', { name, exact: true }).click();
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi.');
  await page.getByRole('button', { name: 'Send' }).click();

  const working = page.getByTestId('working');
  await expect(working).toBeVisible({ timeout: 180_000 });
  await expect(working).toHaveText('Working');
  await expect(working).toHaveCSS('animation-name', 'favlo-working-shimmer');
  await expect(working).toHaveCSS('-webkit-background-clip', 'text');
  } finally {
    await deleteAgentsNamed(name);
  }
});

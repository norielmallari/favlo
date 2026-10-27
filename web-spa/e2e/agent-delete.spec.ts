import { expect, test } from '@playwright/test';

test('deleting an agent removes it from configuration and from chat', async ({ page }) => {
  test.setTimeout(90000);
  const name = `Reed ${Date.now()}`;
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await page.getByRole('button', { name: 'Configuration' }).click();
  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('status')).toContainText(`${name} was created.`);
  await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
  const remove = page.getByRole('button', { name: `Delete ${name}`, exact: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await remove.scrollIntoViewIfNeeded();
  await expect(remove).toBeVisible();
  await remove.click();
  await expect(page.getByRole('status')).toContainText(`${name} was deleted.`);
  await page.getByRole('button', { name: 'Back to list' }).click();
  await expect(page.getByLabel('Agents list').getByRole('button', { name, exact: true })).toHaveCount(0);
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Delete Favlo' })).toHaveCount(0);

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByRole('button', { name: 'Chat' }).click();
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await expect(page.getByRole('button', { name, exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Favlo', exact: true })).toBeVisible();
});

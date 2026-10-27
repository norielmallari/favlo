import { expect, test } from '@playwright/test';

test('thinking opens during a reply and closes when the message is stored', async ({ page }) => {
  const name = `Wren ${Date.now()}`;
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:thinking');
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('status')).toContainText(`${name} was created.`);
  await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
  await page.getByLabel('Channels').getByRole('button', { name, exact: true }).click();
  await page.getByRole('textbox', { name: 'Message' }).fill('Say hi.');
  await page.getByRole('button', { name: 'Send' }).click();

  const thinking = page.getByRole('button', { name: 'Thinking' });
  await expect(thinking).toBeVisible({ timeout: 180_000 });
  if ((await thinking.getAttribute('aria-expanded')) === 'true') {
    await expect(thinking).toHaveAttribute('aria-expanded', 'false', { timeout: 180_000 });
  }
  await expect(page.getByText('Working')).toHaveCount(0);
  await expect(thinking).toHaveAttribute('aria-expanded', 'false');
  await thinking.click();
  await expect(thinking).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByTestId('thinking-text')).toBeVisible();
});

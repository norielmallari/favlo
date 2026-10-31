import { expect, test, type Page } from '@playwright/test';

test('an empty conversation shows starters and hides workspaces', async ({ page }) => {
  const name = `Grove ${Date.now()}`;
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('button', { name, exact: true })).toBeVisible();
  await page.getByRole('button', { name, exact: true }).click();
  await expect(page.getByLabel('Conversation').getByRole('heading', { name })).toBeVisible();
  await expect(page.getByText('Chat, images, music, and voice on this machine.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate an image' })).toBeVisible();
  await expect(page.getByLabel('Workspaces')).toHaveCount(0);
  await page.getByRole('button', { name: 'List registered models' }).click();
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue('Which models are registered?');
  await expect(page.getByRole('link', { name: 'Skip to message' })).not.toBeInViewport();
  await expectComposerInColumn(page);
  const wideColumn = await page.getByTestId('reading-column').boundingBox();
  const wideConversation = await page.getByLabel('Conversation').boundingBox();
  expect(wideColumn).not.toBeNull();
  expect(wideConversation).not.toBeNull();
  expect(wideColumn!.width).toBeLessThan(wideConversation!.width);

  await page.setViewportSize({ width: 390, height: 844 });
  await expectFits(page);
  await expect(page.getByRole('button', { name: 'Channels' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New agent' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Generate an image' })).toBeInViewport();
  await expect(page.getByRole('button', { name: 'Set up a model' })).toBeInViewport();
  await expectComposerInColumn(page);
  await page.getByRole('button', { name: /^Appearance, / }).click();
  await expect(page.getByRole('menuitem', { name: 'Light' })).toBeInViewport();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('menuitem', { name: 'Light' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Channels' }).click();
  await expect(page.getByRole('button', { name: 'New agent' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Close channels' })).toHaveCount(0);
  await expectFits(page);
});

async function expectComposerInColumn(page: Page) {
  const column = await page.getByTestId('reading-column').boundingBox();
  const input = await page.getByRole('textbox', { name: 'Message', exact: true }).boundingBox();
  const send = await page.getByRole('button', { name: 'Send' }).boundingBox();
  expect(column).not.toBeNull();
  expect(input).not.toBeNull();
  expect(send).not.toBeNull();
  expect(input!.x).toBeGreaterThanOrEqual(column!.x - 1);
  expect(send!.x + send!.width).toBeLessThanOrEqual(column!.x + column!.width + 1);
}

async function expectFits(page: Page) {
  await expect(page.getByRole('button', { name: 'Configuration' })).toBeInViewport();
  await expect(page.getByRole('button', { name: /^Appearance, / })).toBeInViewport();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

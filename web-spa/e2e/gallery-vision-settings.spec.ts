import { expect, test } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { onScreen } from './on-screen';

test('orchestration says gallery captions use the agent model', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await assertWcagAa(page);

  await page.getByRole('button', { name: 'Configuration' }).click();
  await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
  await page.getByRole('tab', { name: 'Orchestration' }).click();
  const note = page.getByText(
    "Gallery captions use the workspace agent's model. When that agent has no model, they use the default model from Providers.",
  );
  await onScreen(note);
  await expect(note).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Gallery vision model' })).toHaveCount(0);
});

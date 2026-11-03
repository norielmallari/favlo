import { expect, test } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { onScreen } from './on-screen';

test('orchestration saves gallery vision model', async ({ page }) => {
  const models = await liveModels();
  test.skip(models.length === 0, 'favlo serve has no language model');
  const modelId = models.includes('qwen36-35b-a3b') ? 'qwen36-35b-a3b' : models[0];

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await assertWcagAa(page);

  await page.getByRole('button', { name: 'Configuration' }).click();
  await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
  await page.getByRole('tab', { name: 'Orchestration' }).click();
  const visionPicker = page.getByRole('combobox', { name: 'Gallery vision model' });
  await onScreen(visionPicker);
  await expect(visionPicker).toBeVisible();

  const visionValues = await visionPicker.evaluate((el: HTMLSelectElement) =>
    Array.from(el.options)
      .map((option) => option.value)
      .filter((value) => value !== ''),
  );
  expect(visionValues).toContain(modelId);
  const otherVision = visionValues.find((value) => value !== modelId);
  if (otherVision != null) {
    await visionPicker.selectOption(otherVision);
    await expect(page.getByRole('status')).toHaveText(/Gallery vision model saved/);
  }
  await visionPicker.selectOption(modelId);
  await expect(page.getByRole('status')).toHaveText(/Gallery vision model saved/);
  await expect(visionPicker).toHaveValue(modelId);
});

async function liveModels(): Promise<string[]> {
  const response = await fetch('http://127.0.0.1:8129/v1/models');
  if (!response.ok) {
    return [];
  }
  const body = (await response.json()) as { data?: { id?: string }[] };
  return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
}

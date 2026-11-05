import { expect, test } from '@playwright/test';

import { onScreen } from './on-screen';
import { step } from './progress';

type Shell = { id: string; label: string; setup: string };

test('new workspace lists the shells from the API', async ({ page, request }) => {
  test.setTimeout(360_000);
  const listed = await request.get('http://127.0.0.1:8130/shells');
  expect(listed.ok()).toBeTruthy();
  const shells = (await listed.json()).shells as Shell[];
  expect(shells.map((item) => item.id)).toEqual(['gallery']);
  await step('the picker shows Gallery and not Coding', async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('http://127.0.0.1:8012/');
    await page.getByRole('button', { name: 'Channels' }).click();
    await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
    const picker = page.getByRole('combobox', { name: 'Workspace type' });
    await onScreen(picker);
    await expect(picker.locator('option')).toHaveText(['Gallery']);
    await page.getByRole('button', { name: 'Close new workspace' }).click();
  });
});

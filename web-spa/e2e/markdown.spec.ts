import { expect, test } from '@playwright/test';

const message = ['See **bold** and *italic*.', '', '| Name | Note |', '| --- | --- |', '| One | first |'].join('\n');

test('a chat message renders bold, italics, and a table', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('textbox', { name: 'Message' }).fill(message);
  await page.getByRole('button', { name: 'Send' }).click();

  const yours = page.getByLabel(/^You said:/);
  const bold = yours.getByText('bold', { exact: true });
  const italic = yours.getByText('italic', { exact: true });
  await expect(bold).toBeVisible();
  await expect(italic).toBeVisible();
  expect(await bold.evaluate((node) => getComputedStyle(node).fontWeight)).toMatch(/^(700|bold)$/);
  expect(await italic.evaluate((node) => getComputedStyle(node).fontStyle)).toBe('italic');
  await expect(yours.getByRole('columnheader', { name: 'Name' })).toBeVisible();
  await expect(yours.getByRole('cell', { name: 'One' })).toBeVisible();
  await expect(yours.getByText('**bold**')).toHaveCount(0);

  await page.setViewportSize({ width: 390, height: 844 });
  const box = await yours.getByTestId('markdown-table').boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
});

import { expect, test, type Locator } from '@playwright/test';

import { onScreen } from './on-screen';

const message = ['See **bold** and *italic*.', '', '| Name | Note |', '| --- | --- |', '| One | first |'].join('\n');

test('a chat message renders bold, italics, and a table', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://127.0.0.1:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('textbox', { name: 'Message' }).fill('/session clear');
  let send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();
  await expect(page.getByLabel(/^You said: /)).toHaveCount(0);
  await page.getByRole('textbox', { name: 'Message' }).fill(message);
  send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();
  await expect(page.getByTestId('working')).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });

  const yours = page.getByLabel(/^You said:/).last();
  const bold = yours.getByText('bold', { exact: true });
  const italic = yours.getByText('italic', { exact: true });
  await expect(bold).toBeVisible();
  await expect(italic).toBeVisible();
  expect(await bold.evaluate((node: HTMLElement) => node.style.fontWeight || getComputedStyle(node).fontWeight)).toMatch(/^(700|bold)$/);
  expect(await italic.evaluate((node: HTMLElement) => node.style.fontStyle || getComputedStyle(node).fontStyle)).toBe('italic');
  await expect(yours.getByRole('columnheader', { name: 'Name' })).toBeVisible();
  await expect(yours.getByRole('cell', { name: 'One' })).toBeVisible();
  await expect(yours.getByText('**bold**')).toHaveCount(0);

  await page.setViewportSize({ width: 390, height: 844 });
  const box = await yours.getByTestId('markdown-table').evaluate((node) => {
    const rect = node.getBoundingClientRect();
    return { x: rect.x, width: rect.width };
  });
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);

  const listMessage =
    '- Hair:\n  Long, wavy pink, flowing past shoulders with extra detail so the line wraps in a narrow column.';
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByRole('textbox', { name: 'Message' }).fill(listMessage);
  send = page.getByRole('button', { name: 'Send' });
  await onScreen(send);
  await send.click();
  await expect(page.getByTestId('working')).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
  const listTurn = page.getByLabel(/^You said:/).last();
  await expectBulletIndent(listTurn);

  await page.setViewportSize({ width: 390, height: 844 });
  await expectBulletIndent(listTurn);
});

async function expectBulletIndent(turn: Locator) {
  const layout = await turn.evaluate((turnNode) => {
    const bullet = Array.from(turnNode.querySelectorAll('*')).find(
      (node) => node.textContent === '•' && node.childElementCount === 0,
    );
    const hair = Array.from(turnNode.querySelectorAll('*')).find(
      (node) => node.textContent?.includes('Hair:') && node.childElementCount === 0,
    );
    if (bullet == null || hair == null) {
      return null;
    }
    const bulletBox = bullet.getBoundingClientRect();
    const hairBox = hair.getBoundingClientRect();
    return { bulletLeft: bulletBox.left, hairLeft: hairBox.left };
  });
  expect(layout).not.toBeNull();
  expect(layout!.hairLeft).toBeGreaterThan(layout!.bulletLeft);
}

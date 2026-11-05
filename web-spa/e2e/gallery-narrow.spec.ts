import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Locator, type Page } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

test('narrow gallery keeps preview, filmstrip, and chat in one view', async ({ page }) => {
  test.setTimeout(360_000);
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve has no language model');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-gallery-narrow-'));
  fs.writeFileSync(path.join(root, 'horse.png'), TINY_PNG);
  fs.writeFileSync(path.join(root, 'cat.png'), TINY_PNG);
  const name = `Narrow ${Date.now()}`;
  try {
    await step('create gallery on a wide window', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByRole('button', { name: 'horse.png', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'cat.png', exact: true })).toBeVisible();
      const remove = page.getByRole('button', { name: /Delete .+\.png/ });
      await onScreen(remove);
      const place = await actionPlace(remove);
      expect(place.inBanner).toBe(false);
      expect(place.belowBanner).toBe(true);
      expect(place.topGap).toBeLessThan(48);
      expect(place.rightGap).toBeLessThan(48);
      await expectChatFillsColumn(page);
    });

    await step('narrow shows gallery and chat together', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(page.getByRole('button', { name: 'Chat', exact: true })).toHaveCount(0);
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expect(page.getByLabel('Conversation')).toBeVisible();
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      await expect(page.getByText(/^\d+ of \d+$/)).toBeVisible();
      const download = page.getByRole('button', { name: /Download .+\.png/ });
      await onScreen(download);
      expect((await actionPlace(download)).inBanner).toBe(true);
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
    });

    await step('narrow title menu, Appearance, and Open Favlo chat stay on screen', async () => {
      const title = page.getByRole('button', { name: `${name}, Gallery workspace menu` });
      await onScreen(title);
      await title.click();
      await onScreen(page.getByRole('menuitem', { name: 'Skills' }));
      await onScreen(page.getByRole('menuitem', { name: 'Appearance' }));
      const remove = page.getByRole('menuitem', { name: 'Delete workspace' });
      await onScreen(remove);
      await page.getByRole('menuitem', { name: 'Appearance' }).click();
      await expect(page.getByRole('menu', { name: 'Appearance' })).toBeVisible();
      await onScreen(page.getByRole('menuitemradio', { name: 'Rosé Pine', exact: true }));
      const dark = page.getByRole('menuitemradio', { name: 'Dark', exact: true });
      await onScreen(dark);
      const edges = await page.evaluate(() => {
        const leftOf = (name: string) => {
          const row = document.querySelector(`[role="menuitemradio"][aria-label="${name}"]`);
          if (row == null) {
            return -1;
          }
          const text = [...row.querySelectorAll('div,span')].find((el) => el.textContent === name);
          return (text ?? row).getBoundingClientRect().left;
        };
        return { system: leftOf('System'), light: leftOf('Light'), dark: leftOf('Dark') };
      });
      expect(Math.abs(edges.dark - edges.system)).toBeLessThan(2);
      expect(Math.abs(edges.light - edges.system)).toBeLessThan(2);
      await page.keyboard.press('Escape');
      await title.click();
      await onScreen(page.getByRole('menuitem', { name: 'Open Favlo chat' }));
      await page.keyboard.press('Escape');
      await expect(page).toHaveTitle(`${name} — Gallery`);
    });

    await step('select the second image and read its place', async () => {
      const horse = page.getByRole('button', { name: 'horse.png', exact: true });
      await onScreen(horse);
      await horse.click();
      await expect(page.getByRole('button', { name: 'Download horse.png' })).toBeVisible();
      await expect(page.getByText('2 of 2')).toBeVisible();
      await expect
        .poll(async () => (await page.locator('[aria-label="Gallery"] img[alt="horse.png"]').first().boundingBox())?.height ?? 0)
        .toBeGreaterThan(120);
    });

    await step('add to chat lands in the visible draft', async () => {
      const add = page.getByRole('button', { name: 'Add to chat', exact: true });
      await onScreen(add);
      await add.click();
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toHaveValue(
        /(?:^|\n)horse\.png$/,
      );
    });

    await step('peek keeps a one-line composer and three reply lines plus Copy', async () => {
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await onScreen(box);
      await box.click();
      await box.fill('Peek line');
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
      await expect(page.getByRole('slider', { name: 'Expand chat', exact: true })).toHaveAttribute(
        'aria-valuetext',
        'peek',
      );
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      const sizes = await page.evaluate(() => {
        const field = document.querySelector('[aria-label="Message"]')?.getBoundingClientRect();
        const history = document.querySelector('[data-testid="transcript"]')?.getBoundingClientRect();
        return { field: field?.height ?? 0, history: history?.height ?? 0 };
      });
      expect(sizes.field).toBeGreaterThanOrEqual(40);
      expect(sizes.field).toBeLessThanOrEqual(48);
      expect(sizes.history).toBeGreaterThanOrEqual(3 * 26 + 24);
      expect(sizes.history).toBeLessThanOrEqual(4 * 26 + 24 + 16);
    });

    await step('peek, expanded, and full screen', async () => {
      const expandChat = page.getByRole('button', { name: 'Expand chat horse.png', exact: true });
      await onScreen(expandChat);
      await expandChat.click();
      await expect(page.getByRole('slider', { name: 'Collapse chat' })).toBeVisible();
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      const expand = page.getByRole('button', { name: 'Expand horse.png', exact: true });
      await onScreen(expand);
      await expand.click();
      await expect(page.getByLabel('Filmstrip')).toHaveCount(0);
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible();
      const exit = page.getByRole('button', { name: 'Exit full screen horse.png', exact: true });
      await onScreen(exit);
      await exit.click();
      await expect(page.getByLabel('Filmstrip')).toBeVisible();
      await expect(page.getByRole('button', { name: 'Download horse.png' })).toBeVisible();
      await expect(page.getByText('2 of 2')).toBeVisible();
      await onScreen(page.getByRole('button', { name: 'Full screen preview', exact: true }));
      await expect(page.getByRole('slider', { name: 'Expand chat', exact: true })).toHaveAttribute(
        'aria-valuetext',
        'peek',
      );
    });

    await step('send a short line and see the reply in the dock', async () => {
      const line = `Narrow hello ${Date.now()}`;
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await box.fill(line);
      const send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await expect(page.getByLabel(`You said: ${line}`)).toBeVisible();
      await expect(page.getByTestId('working')).toBeVisible();
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 180_000 });
      await expect(page.getByLabel(/said:/).filter({ hasNotText: 'You said:' }).first()).toBeVisible();
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
    });

    await step('WCAG AA in light and dark at the narrow size', async () => {
      await page.emulateMedia({ colorScheme: 'light' });
      await assertWcagAa(page);
      await page.emulateMedia({ colorScheme: 'dark' });
      await assertWcagAa(page);
      await page.emulateMedia({ colorScheme: 'light' });
    });

    await step('wide window still shows the side-by-side chat', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
      await expect(page.getByLabel('Conversation')).toBeVisible();
      await onScreen(page.getByRole('button', { name: 'Send', exact: true }));
      await onScreen(page.getByRole('slider', { name: 'Resize chat' }));
      const remove = page.getByRole('button', { name: /Delete .+\.png/ });
      await onScreen(remove);
      const place = await actionPlace(remove);
      expect(place.inBanner).toBe(false);
      expect(place.topGap).toBeLessThan(48);
      expect(place.rightGap).toBeLessThan(48);
      await expectChatFillsColumn(page);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function expectChatFillsColumn(page: Page) {
  const edges = await page.evaluate(() => {
    const chat = document.querySelector('[aria-label="Conversation"]')?.getBoundingClientRect();
    const banner = document.querySelector('[data-testid="app-banner"]')?.getBoundingClientRect();
    const frame = document.querySelector('[data-testid="app-frame"]')?.getBoundingClientRect();
    if (chat == null || banner == null || frame == null) {
      return null;
    }
    return {
      chatTop: chat.top,
      chatBottom: chat.bottom,
      chatLeft: chat.left,
      bannerTop: banner.top,
      bannerRight: banner.right,
      frameTop: frame.top,
      frameBottom: frame.bottom,
    };
  });
  expect(edges).not.toBeNull();
  if (edges == null) {
    return;
  }
  expect(Math.abs(edges.chatTop - edges.frameTop)).toBeLessThan(2);
  expect(Math.abs(edges.chatBottom - edges.frameBottom)).toBeLessThan(2);
  expect(Math.abs(edges.bannerTop - edges.frameTop)).toBeLessThan(2);
  expect(edges.bannerRight).toBeLessThanOrEqual(edges.chatLeft);
}

async function actionPlace(button: Locator) {
  return button.evaluate((node) => {
    const box = node.getBoundingClientRect();
    const banner = document.querySelector('[data-testid="app-banner"]')?.getBoundingClientRect();
    const gallery = document.querySelector('[aria-label="Gallery"]')?.getBoundingClientRect();
    return {
      inBanner: node.closest('[data-testid="app-banner"]') != null,
      belowBanner: banner == null ? false : box.top >= banner.bottom - 1,
      topGap: gallery == null ? 999 : box.top - gallery.top,
      rightGap: gallery == null ? 999 : gallery.right - box.right,
    };
  });
}

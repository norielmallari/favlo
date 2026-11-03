import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('gallery theme changes the workspace and leaves the Favlo page alone', async ({ page }) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-theme-'));
  const name = `Theme ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://127.0.0.1:8012/');
    await setAppearance(page, 'Light');
    const workspace = await createGalleryWorkspace(page, name, root, { keepChannel: true });
    await workspace.setViewportSize({ width: 1280, height: 800 });

    await step('a new gallery starts on Rosé Pine and the channel stays Favlo', async () => {
      await expect(workspace.getByRole('button', { name: 'Gallery theme, Rosé Pine' })).toBeVisible();
      await expectBackground(workspace, '#faf4ed');
      await expectBackground(page, '#F6F8FC');
    });

    await step('Rosé Pine follows dark appearance', async () => {
      await setAppearance(workspace, 'Dark');
      await expectBackground(workspace, '#232136');
      await page.reload();
      await expectBackground(page, '#0F1117');
    });

    await step('Catppuccin follows light, dark, and system', async () => {
      await setAppearance(workspace, 'Light');
      await chooseTheme(workspace, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
      await setAppearance(workspace, 'Dark');
      await expectBackground(workspace, '#1e1e2e');
      await setAppearance(workspace, 'System');
      await workspace.emulateMedia({ colorScheme: 'dark' });
      await expectBackground(workspace, '#1e1e2e');
      await workspace.emulateMedia({ colorScheme: 'light' });
      await expectBackground(workspace, '#eff1f5');
    });

    await step('the scheme stays after reload', async () => {
      await setAppearance(workspace, 'Light');
      await workspace.reload();
      await expect(workspace.getByRole('button', { name: 'Gallery theme, Catppuccin' })).toBeVisible();
      await expectBackground(workspace, '#eff1f5');
    });

    await step('both schemes meet WCAG AA in light and dark', async () => {
      await setAppearance(workspace, 'Light');
      await assertWcagAa(workspace);
      await setAppearance(workspace, 'Dark');
      await assertWcagAa(workspace);
      await setAppearance(workspace, 'Light');
      await chooseTheme(workspace, 'Rosé Pine');
      await assertWcagAa(workspace);
      await setAppearance(workspace, 'Dark');
      await assertWcagAa(workspace);
    });

    await step('a narrow window can choose the theme', async () => {
      await workspace.setViewportSize({ width: 390, height: 844 });
      await setAppearance(workspace, 'Light');
      await chooseTheme(workspace, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
    });

    await step('the wide gallery shows Rosé Pine', async () => {
      await workspace.setViewportSize({ width: 1280, height: 800 });
      await chooseTheme(workspace, 'Rosé Pine');
      await expect(workspace.getByRole('button', { name: 'Gallery theme, Rosé Pine' })).toBeVisible();
      await expectBackground(workspace, '#faf4ed');
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function chooseTheme(page: Page, label: string) {
  const button = page.getByRole('button', { name: /^Gallery theme, / });
  await onScreen(button);
  const current = await button.getAttribute('aria-label');
  if (current === `Gallery theme, ${label}`) {
    return;
  }
  await button.click();
  const item = page.getByRole('menuitem', { name: label, exact: true });
  await onScreen(item);
  await item.click();
  await expect(page.getByRole('button', { name: `Gallery theme, ${label}` })).toBeVisible();
}

async function setAppearance(page: Page, mode: 'Light' | 'Dark' | 'System') {
  const appearance = page.getByRole('button', { name: /^Appearance, / });
  await onScreen(appearance);
  const current = await appearance.getAttribute('aria-label');
  if (current === `Appearance, ${mode}`) {
    return;
  }
  await appearance.click();
  const item = page.getByRole('menuitem', { name: mode, exact: true });
  await onScreen(item);
  await item.click();
  await expect(page.getByRole('button', { name: `Appearance, ${mode}` })).toBeVisible();
}

async function expectBackground(page: Page, hex: string) {
  const value = hex.replace('#', '');
  const expected = [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
  ];
  await expect
    .poll(async () => {
      const color = await page.getByTestId('app-frame').evaluate((node) => getComputedStyle(node).backgroundColor);
      const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      return match == null ? [] : [Number(match[1]), Number(match[2]), Number(match[3])];
    })
    .toEqual(expected);
}

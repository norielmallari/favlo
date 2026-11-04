import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('gallery appearance changes the workspace and leaves the Favlo page alone', async ({ page }) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-theme-'));
  const name = `Theme ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://127.0.0.1:8012/');
    await setChatAppearance(page, 'Light');
    const workspace = await createGalleryWorkspace(page, name, root, { keepChannel: true });
    await workspace.setViewportSize({ width: 1280, height: 800 });

    await step('a new gallery starts on Rosé Pine and the channel stays Favlo', async () => {
      await expect(workspace).toHaveTitle(`${name} — Gallery`);
      await expectBackground(workspace, '#faf4ed');
      await expectBackground(page, '#F6F8FC');
      await expectSelectedScheme(workspace, name, 'Rosé Pine');
    });

    await step('Rosé Pine follows dark appearance', async () => {
      await setWorkspaceAppearance(workspace, name, 'Dark');
      await expectBackground(workspace, '#232136');
      await page.reload();
      await expectBackground(page, '#0F1117');
    });

    await step('Catppuccin follows light, dark, and system', async () => {
      await setWorkspaceAppearance(workspace, name, 'Light');
      await chooseScheme(workspace, name, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
      await setWorkspaceAppearance(workspace, name, 'Dark');
      await expectBackground(workspace, '#1e1e2e');
      await setWorkspaceAppearance(workspace, name, 'System');
      await workspace.emulateMedia({ colorScheme: 'dark' });
      await expectBackground(workspace, '#1e1e2e');
      await workspace.emulateMedia({ colorScheme: 'light' });
      await expectBackground(workspace, '#eff1f5');
    });

    await step('the scheme stays after reload', async () => {
      await setWorkspaceAppearance(workspace, name, 'Light');
      await workspace.reload();
      await expect(workspace).toHaveTitle(`${name} — Gallery`);
      await expectSelectedScheme(workspace, name, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
    });

    await step('both schemes meet WCAG AA in light and dark', async () => {
      await setWorkspaceAppearance(workspace, name, 'Light');
      await expectBackground(workspace, '#eff1f5');
      await assertWcagAa(workspace);
      await openAppearance(workspace, name);
      await assertWcagAa(workspace);
      await workspace.keyboard.press('Escape');
      await setWorkspaceAppearance(workspace, name, 'Dark');
      await expectBackground(workspace, '#1e1e2e');
      await assertWcagAa(workspace);
      await openAppearance(workspace, name);
      await assertWcagAa(workspace);
      await workspace.keyboard.press('Escape');
      await setWorkspaceAppearance(workspace, name, 'Light');
      await chooseScheme(workspace, name, 'Rosé Pine');
      await expectBackground(workspace, '#faf4ed');
      await assertWcagAa(workspace);
      await setWorkspaceAppearance(workspace, name, 'Dark');
      await expectBackground(workspace, '#232136');
      await assertWcagAa(workspace);
    });

    await step('a narrow window can choose the theme', async () => {
      await workspace.setViewportSize({ width: 390, height: 844 });
      await setWorkspaceAppearance(workspace, name, 'Light');
      await chooseScheme(workspace, name, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
      await onScreen(workspace.getByRole('button', { name: `${name}, Gallery workspace menu` }));
    });

    await step('the wide gallery shows Rosé Pine', async () => {
      await workspace.setViewportSize({ width: 1280, height: 800 });
      await chooseScheme(workspace, name, 'Rosé Pine');
      await expectSelectedScheme(workspace, name, 'Rosé Pine');
      await expectBackground(workspace, '#faf4ed');
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function openTitleMenu(page: Page, name: string) {
  const button = page.getByRole('button', { name: `${name}, Gallery workspace menu` });
  await onScreen(button);
  if ((await button.getAttribute('aria-expanded')) !== 'true') {
    await button.click();
  }
  await expect(page.getByRole('menuitem', { name: 'Skills' })).toBeVisible();
}

async function openAppearance(page: Page, name: string) {
  await openTitleMenu(page, name);
  const appearance = page.getByRole('menuitem', { name: 'Appearance' });
  await onScreen(appearance);
  await appearance.click();
  await expect(page.getByRole('menu', { name: 'Appearance' })).toBeVisible();
}

async function chooseScheme(page: Page, name: string, label: string) {
  await openAppearance(page, name);
  const item = page.getByRole('menuitemradio', { name: label, exact: true });
  await onScreen(item);
  await item.click();
  await expect(page.getByRole('menu', { name: 'Appearance' })).toHaveCount(0);
}

async function setWorkspaceAppearance(page: Page, name: string, mode: 'Light' | 'Dark' | 'System') {
  await openAppearance(page, name);
  const item = page.getByRole('menuitemradio', { name: mode, exact: true });
  const selected = await item.getAttribute('aria-checked');
  if (selected === 'true') {
    await page.keyboard.press('Escape');
    return;
  }
  await onScreen(item);
  await item.click();
  await expect(page.getByRole('menu', { name: 'Appearance' })).toHaveCount(0);
}

async function expectSelectedScheme(page: Page, name: string, label: string) {
  await openAppearance(page, name);
  await expect(page.getByRole('menuitemradio', { name: label, exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await page.keyboard.press('Escape');
}

async function setChatAppearance(page: Page, mode: 'Light' | 'Dark' | 'System') {
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

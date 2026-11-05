import { expect, test, type Page } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { chooseFolder, makeWorkspaceRoot, removeWorkspaceRoot, waitUntilProvisioned } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test.skip('Coding is not offered when creating a workspace.', async ({ page }) => {
  const name = `CodeHead ${Date.now()}`;
  const root = makeWorkspaceRoot();
  let workspace: Page;
  try {
    await step('create coding workspace', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      const appearance = page.getByRole('button', { name: /^Appearance, / });
      await onScreen(appearance);
      if ((await appearance.getAttribute('aria-label')) !== 'Appearance, Light') {
        await appearance.click();
        await page.getByRole('menuitem', { name: 'Light', exact: true }).click();
      }
      await page.getByRole('button', { name: 'New workspace for Favlo' }).click();
      await page.getByRole('textbox', { name: 'Workspace name' }).fill(name);
      await page.getByRole('combobox', { name: 'Workspace type' }).selectOption('coding');
      await chooseFolder(page, root);
      await page.setViewportSize({ width: 1280, height: 640 });
      const createWorkspace = page.getByRole('button', { name: 'Create workspace' });
      await onScreen(createWorkspace);
      await page.setViewportSize({ width: 1280, height: 800 });
      await onScreen(createWorkspace);
      const popupPromise = page.waitForEvent('popup');
      await createWorkspace.click();
      workspace = await popupPromise;
      await expect(workspace).toHaveURL(/\/w\/.+/);
      await waitUntilProvisioned(workspace);
      await expect(workspace.getByLabel('Coding workspace', { exact: true })).toBeVisible();
    });

    await step('title menu, Catppuccin default, and tab identity', async () => {
      await expect(workspace).toHaveTitle(`${name} — Coding`);
      await expectBackground(workspace, '#eff1f5');
      const title = workspace.getByRole('button', { name: `${name}, Coding workspace menu` });
      await onScreen(title);
      await title.click();
      await onScreen(workspace.getByRole('menuitem', { name: 'Skills' }));
      await onScreen(workspace.getByRole('menuitem', { name: 'Appearance' }));
      await onScreen(workspace.getByRole('menuitem', { name: 'Open Favlo chat' }));
      await onScreen(workspace.getByRole('menuitem', { name: 'Delete workspace' }));
      await workspace.getByRole('menuitem', { name: 'Appearance' }).click();
      await expect(workspace.getByRole('menuitemradio', { name: 'Catppuccin', exact: true })).toHaveAttribute(
        'aria-checked',
        'true',
      );
      await workspace.keyboard.press('Escape');
      const icon = await workspace.evaluate(() => {
        const link = document.querySelector<HTMLLinkElement>("link[rel='icon']");
        return link?.getAttribute('href') ?? '';
      });
      expect(icon.startsWith('data:image/svg+xml')).toBe(true);
      const themeColor = await workspace.evaluate(() => {
        const meta = document.querySelector<HTMLMetaElement>("meta[name='theme-color']");
        return meta?.getAttribute('content') ?? '';
      });
      expect(themeColor.toLowerCase()).toBe('#e6e9ef');
    });

    await step('Appearance can switch scheme and mode', async () => {
      await chooseScheme(workspace, name, 'Rosé Pine');
      await expectBackground(workspace, '#faf4ed');
      await setWorkspaceAppearance(workspace, name, 'Dark');
      await expectBackground(workspace, '#232136');
      await assertWcagAa(workspace);
      await setWorkspaceAppearance(workspace, name, 'Light');
      await chooseScheme(workspace, name, 'Catppuccin');
      await expectBackground(workspace, '#eff1f5');
      await assertWcagAa(workspace);
    });

    await step('narrow keeps title menu and Open Favlo chat on screen', async () => {
      await workspace.setViewportSize({ width: 390, height: 844 });
      const title = workspace.getByRole('button', { name: `${name}, Coding workspace menu` });
      await onScreen(title);
      await title.click();
      await onScreen(workspace.getByRole('menuitem', { name: 'Open Favlo chat' }));
      await workspace.keyboard.press('Escape');
      await assertWcagAa(workspace);
    });

    await step('leaving restores the Favlo tab title', async () => {
      await workspace.close();
      await expect(page).toHaveTitle(/Favlo/);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    removeWorkspaceRoot(root);
  }
});

async function openAppearance(page: Page, name: string) {
  const button = page.getByRole('button', { name: `${name}, Coding workspace menu` });
  await onScreen(button);
  if ((await button.getAttribute('aria-expanded')) !== 'true') {
    await button.click();
  }
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
}

async function setWorkspaceAppearance(page: Page, name: string, mode: 'Light' | 'Dark' | 'System') {
  await openAppearance(page, name);
  const item = page.getByRole('menuitemradio', { name: mode, exact: true });
  if ((await item.getAttribute('aria-checked')) === 'true') {
    await page.keyboard.press('Escape');
    return;
  }
  await onScreen(item);
  await item.click();
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

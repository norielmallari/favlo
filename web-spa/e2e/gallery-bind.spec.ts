import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { assertWcagAa } from './a11y';
import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('gallery create binds folder once and opens the shell', async ({ page }) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-bind-'));
  fs.writeFileSync(
    path.join(root, 'a.png'),
    Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
      'base64',
    ),
  );
  const name = `Bind ${Date.now()}`;
  try {
    await step('create gallery with folder in the modal', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await createGalleryWorkspace(page, name, root);
      await expect(page.getByText('Ask the agent to fill the filmstrip')).toBeVisible();
      await expect(page.getByText('No media selected')).toBeVisible();
      await expect(page.getByLabel('Gallery', { exact: true })).toBeVisible();
    });

    await step('bound gallery meets WCAG AA in light and dark', async () => {
      await setAppearance(page, 'Dark');
      await assertWcagAa(page);
      await setAppearance(page, 'Light');
      await assertWcagAa(page);
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

async function setAppearance(page: Page, mode: 'Light' | 'Dark') {
  const appearance = page.getByRole('button', { name: /^Appearance, / });
  await onScreen(appearance);
  const current = await appearance.getAttribute('aria-label');
  if (current === `Appearance, ${mode}`) {
    return;
  }
  await appearance.click();
  const item = page.getByRole('menuitem', { name: mode });
  await onScreen(item);
  await item.click();
  await expect(page.getByRole('button', { name: `Appearance, ${mode}` })).toBeVisible();
}

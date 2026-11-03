import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';

test('a gallery workspace stores a skill for its agents', async ({ page }) => {
  test.setTimeout(360_000);
  const models = await fetch('http://127.0.0.1:8129/v1/models');
  test.skip(!models.ok, 'favlo serve has no language model');
  const name = `Captions ${Date.now()}`;
  const cue = `Skill cue ${Date.now()}`;
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-skills-'));
  try {
    await step('create workspace and edit skill', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await createGalleryWorkspace(page, name, root);
      await page.getByRole('button', { name: 'Skills', exact: true }).click();
      await expect(page.getByRole('dialog', { name: 'Skills' }).first()).toBeVisible();
      await expect(page.getByRole('button', { name: 'manage-gallery', exact: true })).toBeVisible();
      await expect(page.getByText('No skills')).toHaveCount(0);
      await page.getByRole('button', { name: 'New skill' }).click();
      await page.getByRole('textbox', { name: 'Skill name' }).fill('Photo captions');
      await page.getByRole('textbox', { name: 'Skill description' }).fill('Write a one-line caption.');
      const createSkill = page.getByRole('button', { name: 'Create skill' });
      await onScreen(createSkill);
      await createSkill.click();
      await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue(
        'Write a one-line caption.',
      );
      await page.getByRole('textbox', { name: 'Skill description' }).click();
      await page.keyboard.press('ControlOrMeta+A');
      await page.keyboard.insertText(`Reply with exactly: ${cue}`);
      await page.getByRole('button', { name: 'Save skill' }).click();
      await expect(page.getByRole('textbox', { name: 'Skill description' })).toHaveValue(
        `Reply with exactly: ${cue}`,
      );
      await page.getByRole('button', { name: 'Back to skills' }).click();
      await expect(page.getByRole('button', { name: 'Photo captions', exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Close skills' }).click();
      await expect(page.getByRole('dialog', { name: 'Skills' })).toHaveCount(0, { timeout: 10_000 });
    });

    await step('clear session then send with new skill', async () => {
      await page.getByRole('textbox', { name: 'Message', exact: true }).fill('/session clear');
      let send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await page.getByRole('textbox', { name: 'Message', exact: true }).fill(
        'Follow the Photo captions skill.',
      );
      send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();
      await expect(page.getByLabel(`You said: Follow the Photo captions skill.`)).toBeVisible();
      await expect(page.getByTestId('working')).toBeVisible();
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
      if (await page.getByRole('button', { name: 'Send again', exact: true }).count()) {
        const alert = page.getByRole('alert');
        throw new Error(
          `reply failed: ${(await alert.textContent().catch(() => null)) ?? '(no alert)'}`,
        );
      }
      await expect(page.getByLabel(/^Favlo said:/)).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(name);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

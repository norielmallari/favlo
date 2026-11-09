import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';
import { step } from './progress';
import { LOCAL_PROVIDER_ID } from '../src/api/seed-ids';

test('the reply shows while an image generates, then the image appears', async ({ page }) => {
  const models = await fetch('http://127.0.0.1:8129/v1/models', {
    signal: AbortSignal.timeout(15_000),
  });
  test.skip(!models.ok, 'favlo serve is not listening');
  const listed = (await models.json()) as { data?: { id?: string }[] };
  test.skip((listed.data ?? []).length === 0, 'favlo serve has no language model');

  const name = `Draw ${Date.now()}`;
  try {
    await step('open a new agent', async () => {
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:8012/');
      await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
      await page.getByRole('button', { name: 'New agent' }).click();
      await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
      await page
        .getByRole('textbox', { name: 'Persona', exact: true })
        .fill(
          'When asked to draw, call z_image_turbo__generate_image. Omit model. Then reply in one short sentence.',
        );
      await page.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(LOCAL_PROVIDER_ID);
      const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
      await onScreen(refresh);
      await refresh.click();
      await onScreen(page.getByText('Models refreshed'));
      await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b');
      await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('none');
      await createAgent(page, name);
      await openAgentChat(page, name);
    });

    await step('the reply arrives before the image, then the image shows', async () => {
      const imagesBefore = await page.getByRole('img', { name: 'Generated image' }).count();
      const saidBefore = await page.getByLabel(new RegExp(`^${name} said:`)).count();
      const box = page.getByRole('textbox', { name: 'Message', exact: true });
      await box.fill('Draw a red square.');
      const send = page.getByRole('button', { name: 'Send', exact: true });
      await onScreen(send);
      await send.click();

      const said = page.getByLabel(new RegExp(`^${name} said:`));
      const images = page.getByRole('img', { name: 'Generated image' });
      // One model turn writes the sentence (~25s). The image file follows after the GPU resumes.
      await expect
        .poll(
          async () => (await said.count()) > saidBefore && (await images.count()) === imagesBefore,
          { timeout: 45_000 },
        )
        .toBe(true);
      const reply = said.nth(saidBefore);
      await expect(reply).toBeVisible();

      await expect.poll(async () => images.count(), { timeout: 45_000 }).toBeGreaterThan(imagesBefore);
      await expect(reply).toBeVisible();
      const image = images.last();
      await onScreen(image);
      const src = await image.getAttribute('src');
      expect(src).toContain('/media/images/');
      const file = await page.request.get(src ?? '');
      expect(file.ok()).toBeTruthy();
      expect(file.headers()['content-type']).toContain('image/png');
    });
  } finally {
    await deleteAgentsNamed(name);
  }
});

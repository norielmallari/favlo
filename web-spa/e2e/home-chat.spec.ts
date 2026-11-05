import { expect, test, type Page } from '@playwright/test';

import { onScreen } from './on-screen';
import { step } from './progress';

const PROMPT = 'Say hello in one short sentence.';

test('home chat sends a message and shows one Thinking block', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://127.0.0.1:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await openFavloHome(page);

  const transcript = page.getByTestId('transcript');
  const replies = transcript.locator('[aria-label^="Favlo said:"]');
  const before = await replies.count();

  await step('send a short hello', async () => {
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(PROMPT);
    const send = page.getByRole('button', { name: 'Send', exact: true });
    await onScreen(send);
    await send.click();
    await expect(transcript.getByLabel(`You said: ${PROMPT}`).last()).toBeVisible();
  });

  await step('one Thinking block sits on the reply', async () => {
    await expect(replies).toHaveCount(before + 1);
    await expect(page.getByTestId('working')).toHaveCount(0);
    const reply = replies.last();
    const thinking = reply.getByRole('button', { name: 'Thinking', exact: true });
    await expect(thinking).toHaveCount(1);
    await onScreen(thinking);
  });

  await step('narrow home keeps that one Thinking block', async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    const thinking = replies.last().getByRole('button', { name: 'Thinking', exact: true });
    await expect(thinking).toHaveCount(1);
    await onScreen(thinking);
  });
});

async function openFavloHome(page: Page) {
  const channels = page.getByLabel('Channels');
  const favlo = channels.getByRole('button', { name: 'Favlo', exact: true });
  const home = channels.getByRole('button', { name: 'Home', exact: true });
  await expect(favlo).toBeVisible();
  // A click on an expanded channel collapses it. Wait for Home first.
  try {
    await expect(home).toBeVisible({ timeout: 3_000 });
  } catch {
    await favlo.click();
    await expect(home).toBeVisible();
  }
  await home.click();
  await expect(page.getByLabel('Gallery workspace', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible();
}

import { expect, test, type Locator, type Page } from '@playwright/test';

test('the Favlo chat shows a generated image', async ({ page }) => {
  const channels = await page.request.get('http://127.0.0.1:8130/channels');
  expect(channels.ok()).toBeTruthy();
  const listed = (await channels.json()) as { channels: { name: string; agent_id: string; session_id: string }[] };
  const favlo = listed.channels.find((item) => item.agent_id === 'agent_favlo');
  expect(favlo).toBeTruthy();
  const snapshot = await page.request.get(`http://127.0.0.1:8130/sessions/${favlo?.session_id}`);
  expect(snapshot.ok()).toBeTruthy();
  const messages = (await snapshot.json()) as { messages: { body: string }[] };
  const link = messages.messages
    .map((item) => item.body.match(/favlo:\/\/images\/([A-Za-z0-9_-]+)\.png/))
    .find((item) => item != null);
  test.skip(link == null, 'the Favlo chat has no generated image');

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await page.reload();
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await expect(page.getByLabel('Conversation').getByRole('heading', { name: 'Favlo' })).toBeVisible();
  const image = page.getByRole('img', { name: 'Generated image' }).last();
  await expect(image).toBeVisible();
  const src = await image.getAttribute('src');
  expect(src).toContain(`/media/images/${link?.[1]}.png`);
  const file = await page.request.get(src ?? '');
  expect(file.ok()).toBeTruthy();
  expect(file.headers()['content-type']).toContain('image/png');
  await expectLatest(page);
  await expectSameRatio(image);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await expectLatest(page);
  await expectSameRatio(image);
  await page.setViewportSize({ width: 390, height: 844 });
  await expectLatest(page);
  await expectSameRatio(image);
});

async function expectLatest(page: Page) {
  await expect.poll(async () => page.getByTestId('transcript').evaluate((node) => {
    let best = node as HTMLElement;
    let overflow = best.scrollHeight - best.clientHeight;
    best.querySelectorAll('*').forEach((item) => {
      if (!(item instanceof HTMLElement)) {
        return;
      }
      const extra = item.scrollHeight - item.clientHeight;
      if (extra > overflow) {
        overflow = extra;
        best = item;
      }
    });
    return best.scrollHeight - best.clientHeight - best.scrollTop;
  })).toBeLessThan(48);
}

async function expectSameRatio(image: Locator) {
  const drift = await image.evaluate((img: HTMLImageElement) => {
    const natural = img.naturalWidth / img.naturalHeight;
    const shown = img.clientWidth / img.clientHeight;
    return Math.abs(shown - natural) / natural;
  });
  expect(drift).toBeLessThan(0.02);
}

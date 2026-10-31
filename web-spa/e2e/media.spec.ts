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
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.reload();
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await expect(page.getByLabel('Conversation').getByRole('heading', { name: 'Favlo' })).toBeVisible();
  await expect(page.getByLabel('Channels width')).toHaveCount(0);
  const image = page.getByRole('img', { name: 'Generated image' }).last();
  await expect(image).toBeVisible();
  await expect(page.getByLabel('Conversation').getByText('generate_image finished.', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Conversation').getByText(link?.[1] ?? 'image', { exact: true })).toHaveCount(0);
  const src = await image.getAttribute('src');
  expect(src).toContain(`/media/images/${link?.[1]}.png`);
  const file = await page.request.get(src ?? '');
  expect(file.ok()).toBeTruthy();
  expect(file.headers()['content-type']).toContain('image/png');
  const shown = page.getByRole('img', { name: 'Generated image' });
  for (let index = 0; index < await shown.count(); index += 1) {
    const address = await shown.nth(index).getAttribute('src');
    expect((await page.request.get(address ?? '')).ok()).toBeTruthy();
  }
  await expectLatest(page);
  await expectSameRatio(image);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await expectLatest(page);
  await expectSameRatio(image);
  await page.setViewportSize({ width: 390, height: 844 });
  await expectLatest(page);
  await expectSameRatio(image);
  await expectFits(page);
  await page.setViewportSize({ width: 375, height: 667 });
  await expectLatest(page);
  await expectFits(page);
  await page.getByRole('button', { name: 'Appearance, System' }).click();
  await page.getByRole('menuitem', { name: 'Light' }).click();
  await expect(page.getByRole('button', { name: 'Appearance, Light' })).toBeInViewport();
  const css = await page.getByTestId('transcript-scrollbar').evaluate((node) => node.textContent ?? '');
  expect(css).toContain('#718096');
  expect(css).toContain('#F6F8FC');
  const applied = await page.getByTestId('transcript').evaluate((node) => getComputedStyle(node).scrollbarColor);
  expect(applied).toContain('113, 128, 150');
  await page.getByRole('button', { name: 'View generated image' }).click();
  const dialog = page.getByRole('dialog', { name: 'Generated image' });
  await expect(dialog).toBeVisible();
  const preview = dialog.getByRole('img', { name: 'Generated image' });
  await expect.poll(() => preview.evaluate((node: HTMLImageElement) => node.naturalWidth)).toBeGreaterThan(0);
  await expect(dialog.getByRole('button', { name: 'Download image' })).toBeInViewport();
  await expect(dialog.getByRole('button', { name: 'Close' })).toBeInViewport();
  const card = await dialog.boundingBox();
  const picture = await preview.boundingBox();
  const download = await dialog.getByRole('button', { name: 'Download image' }).boundingBox();
  expect(card).not.toBeNull();
  expect(picture).not.toBeNull();
  expect(download).not.toBeNull();
  expect(picture?.x ?? 0).toBeGreaterThanOrEqual((card?.x ?? 0) - 1);
  expect((picture?.x ?? 0) + (picture?.width ?? 0)).toBeLessThanOrEqual((card?.x ?? 0) + (card?.width ?? 0) + 1);
  expect(picture?.height ?? 0).toBeGreaterThan(40);
  expect(picture?.y ?? 0).toBeGreaterThan((download?.y ?? 0) + (download?.height ?? 0));
  expect((picture?.y ?? 0) + (picture?.height ?? 0)).toBeLessThanOrEqual((card?.y ?? 0) + (card?.height ?? 0) + 1);
  const saved = page.waitForEvent('download');
  await dialog.getByRole('button', { name: 'Download image' }).click();
  expect((await saved).suggestedFilename()).toBe('favlo-image.png');
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await page.getByRole('button', { name: 'View generated image' }).click();
  const again = page.getByRole('dialog', { name: 'Generated image' });
  await expect(again).toBeVisible();
  await expect.poll(() => again.getByRole('img', { name: 'Generated image' }).evaluate((node: HTMLImageElement) => node.naturalWidth)).toBeGreaterThan(0);
});

async function expectFits(page: Page) {
  await expect(page.getByRole('button', { name: 'Configuration' })).toBeInViewport();
  await expect(page.getByRole('button', { name: /^Appearance, / })).toBeInViewport();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  const image = page.getByRole('img', { name: 'Generated image' }).last();
  const box = await image.boundingBox();
  const viewport = page.viewportSize();
  expect(box).not.toBeNull();
  expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual((viewport?.width ?? 0) + 1);
}

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

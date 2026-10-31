import { expect, test, type Page } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';

const PROMPT = 'Reply with the single word ready.';

test('a short transcript rests on the composer', async ({ page }) => {
  const models = await liveModels();
  test.skip(models.length === 0, 'favlo serve has no language model');
  const name = `Align ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8081/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
    await page.getByRole('button', { name: 'New agent' }).click();
    await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
    await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('Reply in one short sentence.');
    await page.getByRole('button', { name: 'Create agent' }).click();
    await page.getByRole('button', { name, exact: true }).click();
    await page.getByRole('button', { name: 'Home', exact: true }).click();
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(PROMPT);
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByLabel(`You said: ${PROMPT}`)).toBeVisible();
    await expectShortThread(page);

    await expect.poll(async () => page.getByLabel(new RegExp(`^${name} said: `)).count(), {
      timeout: 360_000,
    }).toBeGreaterThan(0);
    await expectLatest(page);

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByRole('button', { name: 'Channels' })).toBeVisible();
    await expect(page.getByLabel(`You said: ${PROMPT}`)).toBeVisible();
    await expectLatest(page);
  } finally {
    await deleteAgentsNamed(name);
  }
});

async function expectShortThread(page: Page) {
  const place = await threadPlace(page);
  expect(place.count).toBeGreaterThan(0);
  expect(place.overflow).toBe(false);
  expect(place.firstTop).toBeGreaterThan(place.height * 0.35);
  expect(place.lastBottom).toBeLessThan(place.height * 0.25);
}

async function expectLatest(page: Page) {
  const place = await threadPlace(page);
  expect(place.fromBottom).toBeLessThan(8);
  if (!place.overflow) {
    expect(place.firstTop).toBeGreaterThan(place.height * 0.35);
    expect(place.lastBottom).toBeLessThan(place.height * 0.25);
  }
}

async function threadPlace(page: Page) {
  return page.getByTestId('transcript').evaluate((node) => {
    const scroller = node as HTMLElement;
    const rows = [...scroller.querySelectorAll('[aria-label]')].filter((el) =>
      / said: /.test(el.getAttribute('aria-label') ?? ''),
    );
    const first = rows[0]?.getBoundingClientRect();
    const last = rows[rows.length - 1]?.getBoundingClientRect();
    const box = scroller.getBoundingClientRect();
    return {
      count: rows.length,
      overflow: scroller.scrollHeight > scroller.clientHeight + 2,
      fromBottom: scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop,
      firstTop: first == null ? 0 : first.top - box.top,
      lastBottom: last == null ? box.height : box.bottom - last.bottom,
      height: box.height,
    };
  });
}

async function liveModels(): Promise<string[]> {
  const response = await fetch('http://127.0.0.1:8129/v1/models');
  if (!response.ok) {
    return [];
  }
  const body = (await response.json()) as { data?: { id?: string }[] };
  return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
}

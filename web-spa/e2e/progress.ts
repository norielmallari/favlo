import { test } from '@playwright/test';

/** Print and nest a Playwright step so the list reporter shows live progress. */
export async function step<T>(title: string, body: () => Promise<T>): Promise<T> {
  const stamp = new Date().toISOString().slice(11, 19);
  console.log(`[e2e ${stamp}] ${title}`);
  return await test.step(title, body);
}

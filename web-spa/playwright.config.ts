import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  testIgnore: '**/web-new-user-e2e-setup/**',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  // Whole test budget. Individual expects / actions fail at 15s (see use + expect).
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [
    ['list', { printSteps: true }],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
  ],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:8012',
    headless: true,
    actionTimeout: 15_000,
    navigationTimeout: 15_000,
    trace: 'on-first-retry',
    screenshot: 'on',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});

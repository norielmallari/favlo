import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { defineConfig, devices } from '@playwright/test';

const repo = path.resolve(process.cwd(), '..');
const suite = path.join(process.cwd(), 'e2e/web-new-user-e2e-setup');
const home = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-web-new-user-'));
fs.writeFileSync(path.join(suite, '.home'), home);

export default defineConfig({
  testDir: suite,
  testMatch: '**/*.spec.ts',
  outputDir: path.join(process.cwd(), 'test-results'),
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 420_000,
  expect: { timeout: 15_000 },
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:8012',
    headless: true,
    screenshot: 'on',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: 'sh web-spa/e2e/web-new-user-e2e-setup/serve.sh',
    cwd: repo,
    env: {
      ...process.env,
      FAVLO_WEB_HOME: home,
      BROWSER: 'none',
    },
    url: 'http://127.0.0.1:8012',
    reuseExistingServer: false,
    timeout: 120_000,
  },
});

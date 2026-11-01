import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';

test('a new agent shows the same options as edit', async ({ page }) => {
  const name = `Reed ${Date.now()}`;
  try {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'Configuration' }).click();
  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByRole('button', { name: 'New agent' }).click();
  await expect(page.getByRole('textbox', { name: 'Agent name', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Persona', exact: true })).toBeVisible();
  const provider = page.getByRole('combobox', { name: 'Provider', exact: true });
  const model = page.getByRole('combobox', { name: 'Model', exact: true });
  await expect(provider).toBeVisible();
  await expect(model).not.toHaveValue('');
  await expect(page.getByRole('switch', { name: 'Configuration tools', exact: true })).toBeVisible();
  await expect(page.getByRole('switch', { name: 'Conversation tools', exact: true })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: 'Heartbeat tools', exact: true })).toBeVisible();
  const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(refresh);
  await refresh.click();
  const refreshed = page.getByText('Models refreshed');
  await onScreen(refreshed);
  await expect(refreshed).toBeVisible();
  const chosen = await model.inputValue();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Thinking', exact: true }).selectOption('true');
  await page.getByRole('switch', { name: 'Conversation tools', exact: true }).click();
  const wideCreate = page.getByRole('button', { name: 'Create agent' });
  await onScreen(wideCreate);
  await wideCreate.click();
  await expect(page.getByTestId('save-confirmation')).toContainText(`${name} was created.`);
  await expect(page.getByLabel('Agents list').getByRole('button', { name, exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: `Name for ${name}`, exact: true })).toHaveValue(name);
  await expect(page.getByRole('combobox', { name: `Provider for ${name}`, exact: true })).toHaveValue('provider_local');
  await expect(page.getByRole('combobox', { name: `Model for ${name}`, exact: true })).toHaveValue(chosen);
  await expect(page.getByRole('combobox', { name: `Thinking for ${name}`, exact: true })).toHaveValue('true');
  await expect(page.getByRole('combobox', { name: `Preserve thinking for ${name}`, exact: true })).toHaveValue('');
  await expect(page.getByRole('combobox', { name: `Reasoning effort for ${name}`, exact: true })).toHaveValue('');
  await expect(page.getByRole('switch', { name: `Configuration tools for ${name}` })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: `Conversation tools for ${name}` })).toBeChecked();
  await expect(page.getByRole('switch', { name: `Heartbeat tools for ${name}` })).not.toBeChecked();

  await page.getByRole('button', { name: 'Chat' }).click();
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.getByRole('button', { name: 'New agent' }).click();
  const shortRefresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(page.getByRole('button', { name: 'Create agent' }));
  await onScreen(shortRefresh);
  await shortRefresh.click();
  const shortRefreshed = page.getByText('Models refreshed');
  await onScreen(shortRefreshed);
  await expect(page.getByRole('combobox', { name: 'Provider', exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toBeVisible();
  await expect(page.getByRole('switch', { name: 'Heartbeat tools', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close new agent' }).click();
  await expect(page.getByRole('textbox', { name: 'Agent name', exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Configuration' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await page.getByRole('button', { name: 'Back to list' }).click();
  await page.getByRole('button', { name: 'New agent' }).click();
  const narrowRefresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(page.getByRole('button', { name: 'Create agent' }));
  await onScreen(narrowRefresh);
  await narrowRefresh.click();
  await onScreen(page.getByText('Models refreshed'));
  const heartbeat = page.getByRole('switch', { name: 'Heartbeat tools', exact: true });
  await expect(page.getByRole('combobox', { name: 'Provider', exact: true })).toBeVisible();
  await expect(heartbeat).toBeVisible();
  } finally {
    await deleteAgentsNamed(name);
  }
});

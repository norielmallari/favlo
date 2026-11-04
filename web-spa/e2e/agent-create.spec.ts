import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { LOCAL_PROVIDER_ID } from '../src/api/seed-ids';

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
  await expect(provider).toHaveValue('');
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Refresh models', exact: true })).toHaveCount(0);
  await expect(
    page.getByText('Uses the default provider and model from Configuration → Providers.'),
  ).toBeVisible();
  await expect(page.getByRole('switch', { name: 'Configuration tools', exact: true })).toBeVisible();
  await expect(page.getByRole('switch', { name: 'Conversation tools', exact: true })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: 'Heartbeat tools', exact: true })).toBeVisible();

  await provider.selectOption(LOCAL_PROVIDER_ID);
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toBeVisible();
  const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(refresh);
  await refresh.click();
  const refreshed = page.getByText('Models refreshed');
  await onScreen(refreshed);
  await expect(refreshed).toBeVisible();
  await provider.selectOption('');
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Refresh models', exact: true })).toHaveCount(0);

  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('medium');
  const thinking = page.getByRole('combobox', { name: 'Expand thinking', exact: true });
  const toolCalls = page.getByRole('combobox', { name: 'Expand tool calls', exact: true });
  await expect(thinking.locator('option:checked')).toHaveText('Collapsed (default)');
  await expect(toolCalls.locator('option:checked')).toHaveText('Not shown (default)');
  await thinking.selectOption('hidden');
  await toolCalls.selectOption('collapsed');
  await page.getByRole('switch', { name: 'Conversation tools', exact: true }).click();
  const wideCreate = page.getByRole('button', { name: 'Create agent' });
  await onScreen(wideCreate);
  const created = page.waitForResponse(
    (response) =>
      response.url().includes('/agents') &&
      response.request().method() === 'POST' &&
      response.ok(),
  );
  await wideCreate.click();
  const createdBody = (await (await created).json()) as {
    agent: { default_model_id: string | null };
  };
  expect(createdBody.agent.default_model_id).toBeNull();
  await expect(page.getByTestId('save-confirmation')).toContainText(`${name} was created.`);
  await expect(page.getByLabel('Agents list').getByRole('button', { name, exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: `Name for ${name}`, exact: true })).toHaveValue(name);
  await expect(page.getByRole('combobox', { name: `Provider for ${name}`, exact: true })).toHaveValue('');
  await expect(page.getByRole('combobox', { name: `Model for ${name}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: `Reasoning effort for ${name}`, exact: true })).toHaveValue('medium');
  await expect(page.getByRole('combobox', { name: `Expand thinking for ${name}`, exact: true })).toHaveValue('hidden');
  await expect(page.getByRole('combobox', { name: `Expand tool calls for ${name}`, exact: true })).toHaveValue('collapsed');
  await page.getByRole('combobox', { name: `Expand thinking for ${name}`, exact: true }).selectOption('always');
  await page.getByRole('combobox', { name: `Expand tool calls for ${name}`, exact: true }).selectOption('running');
  const save = page.getByRole('button', { name: `Save ${name}` });
  await onScreen(save);
  await save.click();
  await expect(page.getByRole('combobox', { name: `Expand thinking for ${name}`, exact: true })).toHaveValue('always');
  await expect(page.getByRole('combobox', { name: `Expand tool calls for ${name}`, exact: true })).toHaveValue('running');
  await expect(page.getByRole('combobox', { name: `Preserve thinking for ${name}`, exact: true })).toHaveValue('');
  await expect(page.getByRole('combobox', { name: `Thinking for ${name}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole('switch', { name: `Configuration tools for ${name}` })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: `Conversation tools for ${name}` })).toBeChecked();
  await expect(page.getByRole('switch', { name: `Heartbeat tools for ${name}` })).not.toBeChecked();

  await page.getByLabel('Agent details').getByRole('tab', { name: 'MCP' }).click();
  const media = page.getByLabel('Agent MCP').getByRole('button', { name: 'Favlo media', exact: true });
  await onScreen(media);
  await media.click();
  const url = page.getByRole('textbox', { name: 'URL' });
  await onScreen(url);
  await expect(url).toHaveValue('http://127.0.0.1:8129/mcp');

  await page.getByRole('button', { name: 'Chat' }).click();
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.getByRole('button', { name: 'New agent' }).click();
  await expect(page.getByRole('combobox', { name: 'Provider', exact: true })).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Refresh models', exact: true })).toHaveCount(0);
  await page.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(LOCAL_PROVIDER_ID);
  const shortRefresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await onScreen(page.getByRole('button', { name: 'Create agent' }));
  await onScreen(shortRefresh);
  await shortRefresh.click();
  const shortRefreshed = page.getByText('Models refreshed');
  await onScreen(shortRefreshed);
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toBeVisible();
  await expect(page.getByRole('switch', { name: 'Heartbeat tools', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close new agent' }).click();
  await expect(page.getByRole('textbox', { name: 'Agent name', exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Configuration' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Provider for Favlo', exact: true })).toHaveValue('');
  await expect(page.getByRole('combobox', { name: 'Model for Favlo', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Back to list' }).click();
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByRole('combobox', { name: 'Provider', exact: true }).selectOption(LOCAL_PROVIDER_ID);
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

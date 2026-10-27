import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

test('two agent channels keep separate conversations', async ({ page }) => {
  const models = await liveModels();
  test.skip(models.length === 0, 'favlo serve has no language model');
  const modelId = models[0];

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8081/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8081\/?$/);
  await expect(page.getByRole('heading', { name: 'Channels' })).toBeVisible();
  await assertShell(page);

  await page.getByRole('button', { name: 'Configuration' }).click();
  await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
  await page.getByRole('tab', { name: 'Orchestration' }).click();
  await expect(page.getByText('Maximum turns', { exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Server heartbeat' })).toHaveValue('5');
  const saveTurns = page.getByRole('button', { name: 'Save maximum turns' });
  const box = await saveTurns.boundingBox();
  if (box == null) {
    throw new Error('Save maximum turns has no box');
  }
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect(saveTurns.getByTestId('press-feedback')).toBeVisible();
  await page.mouse.up();
  await expect(page.getByRole('status')).toHaveText(/Maximum turns saved/);
  await expect(page.getByLabel('Agents list')).toHaveCount(0);
  await expect(page.getByLabel('Providers list')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'New agent' }).filter({ visible: true })).toHaveCount(0);

  await page.getByRole('tab', { name: 'Agents' }).click();
  const sections = page.getByLabel('Configuration sections');
  const mcpTab = page.getByRole('tab', { name: 'MCP' });
  const orchestrationTab = page.getByRole('tab', { name: 'Orchestration' });
  const sectionBox = await sections.boundingBox();
  const mcpBox = await mcpTab.boundingBox();
  const orchestrationBox = await orchestrationTab.boundingBox();
  if (sectionBox == null || mcpBox == null || orchestrationBox == null) {
    throw new Error('Configuration menu has no box');
  }
  expect(mcpBox.x - sectionBox.x).toBeGreaterThanOrEqual(16);
  expect(sectionBox.x + sectionBox.width - mcpBox.x - mcpBox.width).toBeGreaterThanOrEqual(16);
  expect(orchestrationBox.y - (mcpBox.y + mcpBox.height)).toBeGreaterThanOrEqual(10);
  const agents = page.getByLabel('Agents list');
  await expect(agents.getByRole('button', { name: 'Favlo', exact: true })).toBeVisible();
  await expect(page.getByLabel('Providers list')).toHaveCount(0);
  await expectColumns(page, 'Agents list', page.getByRole('textbox', { name: 'Name for Favlo' }));
  await expect(page.getByText('Persona for Favlo', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Provider for Favlo' })).toHaveValue('provider_local');
  await expect(page.getByRole('combobox', { name: 'Model for Favlo' })).not.toHaveValue('');
  await expect(page.getByRole('switch', { name: 'Configuration tools for Favlo' })).toBeChecked();
  const conversation = page.getByRole('switch', { name: 'Conversation tools for Favlo' });
  await expect(conversation).not.toBeChecked();
  await expect(page.getByRole('switch', { name: 'Heartbeat tools for Favlo' })).not.toBeChecked();
  await conversation.click();
  await page.getByRole('button', { name: 'Save Favlo' }).click();
  await expect(page.getByRole('status')).toHaveText(/Favlo was saved/);
  await expect(conversation).toBeChecked();
  await expect(page.getByRole('switch', { name: 'Configuration tools for Favlo' })).toBeChecked();

  await page.getByRole('tab', { name: 'MCP' }).click();
  await expect(page.getByLabel('MCP list')).toBeVisible();
  await expect(page.getByText('No MCP servers')).toBeVisible();
  await page.getByRole('button', { name: 'New MCP' }).click();
  await page.getByRole('textbox', { name: 'MCP name' }).fill('Favlo media');
  await page.getByRole('radio', { name: 'HTTP' }).click();
  await page.getByRole('textbox', { name: 'URL' }).fill('http://127.0.0.1:8129/mcp');
  await page.getByRole('button', { name: 'Create MCP' }).click();
  await expect(page.getByLabel('MCP list').getByRole('button', { name: 'Favlo media' })).toBeVisible();
  await page.getByRole('button', { name: 'Test connection' }).click();
  await expect(page.getByText('Connection succeeded')).toBeVisible();
  await expect(page.getByText('generate_image', { exact: true })).toBeVisible();
  const imageTools = page.getByRole('switch', { name: 'Image for Favlo media' });
  await expect(imageTools).toBeChecked();
  await imageTools.click();
  await page.getByRole('button', { name: 'Save Favlo media' }).click();
  await expect(page.getByRole('switch', { name: 'Image for Favlo media' })).not.toBeChecked();
  await page.getByRole('tab', { name: 'Agents' }).click();

  const brook = `Brook ${Date.now()}`;
  await page.getByRole('button', { name: 'New agent' }).filter({ visible: true }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(brook);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill('A short persona.');
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(agents.getByRole('button', { name: brook, exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: `Name for ${brook}` })).toHaveValue(brook);
  await expect(page.getByRole('switch', { name: `Configuration tools for ${brook}` })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: `Conversation tools for ${brook}` })).not.toBeChecked();
  await expect(page.getByRole('switch', { name: `Heartbeat tools for ${brook}` })).not.toBeChecked();
  await agents.getByRole('button', { name: 'Favlo', exact: true }).click();
  await expect(page.getByRole('switch', { name: 'Configuration tools for Favlo' })).toBeChecked();
  await expect(page.getByRole('switch', { name: 'Conversation tools for Favlo' })).toBeChecked();

  await page.getByRole('tab', { name: 'Heartbeats' }).click();
  await expect(page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true })).toBeVisible();
  await expect(page.getByLabel('Heartbeats list')).toBeVisible();
  await expectHeartbeatColumns(page);
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await page.getByRole('button', { name: 'New heartbeat' }).click();
  await page.getByRole('textbox', { name: 'Cron for Favlo' }).fill('nope');
  await page.getByRole('button', { name: 'Create heartbeat' }).click();
  await expect(page.getByRole('alert')).toContainText('five fields');
  await page.getByRole('textbox', { name: 'Cron for Favlo' }).fill('15 * * * *');
  await page.getByRole('textbox', { name: 'Instructions for Favlo' }).fill('Check the garden.');
  await page.getByRole('button', { name: 'Create heartbeat' }).click();
  const beat = page.getByLabel('Heartbeats list').getByRole('button', { name: '15 * * * *' });
  await expect(beat).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Instructions for Favlo' })).toHaveValue('Check the garden.');
  await page.getByLabel('Agents list').getByRole('button', { name: brook, exact: true }).click();
  await expect(page.getByLabel('Heartbeats list').getByRole('button', { name: '15 * * * *' })).toHaveCount(0);
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await beat.click();
  await expect(page.getByRole('textbox', { name: 'Cron for Favlo' })).toHaveValue('15 * * * *');
  await page.getByRole('button', { name: 'Run heartbeat' }).click();
  await expect(page.getByText('The heartbeat is running.')).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Back to heartbeats' }).click();
  await expect(page.getByLabel('Heartbeats list')).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Cron for Favlo' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Back to agents' }).click();
  await expect(page.getByLabel('Agents list')).toBeVisible();
  await expect(page.getByLabel('Heartbeats list')).toHaveCount(0);
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await page.getByLabel('Heartbeats list').getByRole('button', { name: '15 * * * *' }).click();
  await expect(page.getByRole('textbox', { name: 'Cron for Favlo' })).toHaveValue('15 * * * *');

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByRole('tab', { name: 'Agents' }).click();
  await agents.getByRole('button', { name: 'Favlo', exact: true }).click();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Back to list' }).click();
  await expect(agents).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Name for Favlo' })).toHaveCount(0);
  await agents.getByRole('button', { name: 'Favlo', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Name for Favlo' })).toBeVisible();
  await page.getByRole('button', { name: 'Sections' }).click();
  await page.getByRole('tab', { name: 'Providers' }).click();
  await expect(page.getByLabel('Providers list')).toBeVisible();
  await page.getByLabel('Providers list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Test connection' })).toBeVisible();
  await page.getByRole('button', { name: 'Back to list' }).click();
  await expect(page.getByLabel('Providers list').getByRole('button', { name: 'Favlo', exact: true })).toBeVisible();

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByRole('tab', { name: 'Providers' }).click();
  await expect(page.getByLabel('Agents list')).toHaveCount(0);
  await page.getByLabel('Providers list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await expectColumns(page, 'Providers list', page.getByRole('button', { name: 'Test connection' }));
  await page.getByRole('button', { name: 'Test connection' }).click();
  await expect(page.getByText('Connection succeeded')).toBeVisible();
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByText('Models refreshed')).toBeVisible();
  const modelsList = page.getByRole('combobox', { name: 'Default model' });
  await expect(modelsList).toBeVisible();
  await assertShell(page);
  await modelsList.selectOption(modelId);
  await expect(modelsList).toHaveValue(modelId);

  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByLabel('Agents list').getByRole('button', { name: brook, exact: true }).click();
  const brookModel = page.getByRole('combobox', { name: `Model for ${brook}` });
  await expect(page.getByRole('combobox', { name: `Provider for ${brook}` })).toHaveValue('provider_local');
  const modelChoices = await brookModel.locator('option').evaluateAll((nodes) =>
    nodes.map((node) => (node as HTMLOptionElement).value).filter((value) => value !== ''),
  );
  const chosen = modelChoices.find((id) => id !== modelId) ?? modelId;
  await brookModel.selectOption(chosen);
  await page.getByRole('button', { name: `Save ${brook}` }).click();
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await page.getByLabel('Agents list').getByRole('button', { name: brook, exact: true }).click();
  await expect(page.getByRole('combobox', { name: `Model for ${brook}` })).toHaveValue(chosen);

  const listed = page.waitForResponse(
    (response) => response.url().includes('/channels') && response.request().method() === 'GET',
    { timeout: 10_000 },
  );
  await page.getByRole('button', { name: 'Chat' }).click();
  await listed;
  await expect(page.getByLabel('Conversation').getByRole('heading', { name: 'Favlo' })).toBeVisible();
  await createAgent(page, 'River', 'Reply in one short sentence.');
  await createAgent(page, 'Cedar', 'Reply in one short sentence.');
  const emptyName = `Glen ${Date.now()}`;
  await createAgent(page, emptyName, 'Reply in one short sentence.');
  await page.getByRole('button', { name: emptyName, exact: true }).click();
  await expect(page.getByRole('heading', { name: emptyName })).toBeVisible();
  await expectComposerCentered(page, emptyName);
  await page.getByRole('button', { name: 'River', exact: true }).click();
  const prompt = `Say hello. ${Date.now()}`;
  const replies = page.getByLabel(/^River said: .+/);
  const before = await replies.count();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill(prompt);
  await page.getByRole('button', { name: 'Send' }).click();
  await expect.poll(async () => replies.count(), { timeout: 360_000 }).toBeGreaterThan(before);
  await expectComposerDocked(page, 'River');

  await page.getByRole('button', { name: 'Cedar', exact: true }).click();
  const cedar = page.getByLabel('Conversation').filter({ has: page.getByRole('heading', { name: 'Cedar' }) });
  await expect(cedar).toBeVisible();
  await expect(cedar.getByLabel(`You said: ${prompt}`)).toHaveCount(0);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Channels' }).click();
  await expect(page.getByRole('button', { name: 'River' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Cedar' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Close channels' })).toHaveCount(0);
});

async function expectHeartbeatColumns(page: Page) {
  const sections = await page.getByLabel('Configuration sections').boundingBox();
  const agents = await page.getByLabel('Agents list').boundingBox();
  const beats = await page.getByLabel('Heartbeats list').boundingBox();
  const detail = await page.getByRole('button', { name: 'New heartbeat' }).boundingBox();
  expect(sections).not.toBeNull();
  expect(agents).not.toBeNull();
  expect(beats).not.toBeNull();
  expect(detail).not.toBeNull();
  expect(agents!.x).toBeGreaterThanOrEqual(sections!.x + sections!.width - 2);
  expect(beats!.x).toBeGreaterThanOrEqual(agents!.x + agents!.width - 2);
  expect(detail!.x).toBeGreaterThanOrEqual(beats!.x + beats!.width - 2);
}

async function expectColumns(page: Page, listLabel: string, detail: ReturnType<Page['getByRole']>) {
  const sections = await page.getByLabel('Configuration sections').boundingBox();
  const list = await page.getByLabel(listLabel).boundingBox();
  const field = await detail.boundingBox();
  expect(sections).not.toBeNull();
  expect(list).not.toBeNull();
  expect(field).not.toBeNull();
  expect(list!.x).toBeGreaterThanOrEqual(sections!.x + sections!.width - 2);
  expect(field!.x).toBeGreaterThanOrEqual(list!.x + list!.width - 2);
}

async function expectComposerCentered(page: Page, name: string) {
  const place = await composerPlace(page, name);
  expect(Math.abs(place.composer - place.middle)).toBeLessThan(place.height * 0.2);
}

async function expectComposerDocked(page: Page, name: string) {
  const place = await composerPlace(page, name);
  expect(place.bottom).toBeGreaterThan(place.top + place.height * 0.75);
}

async function composerPlace(page: Page, name: string) {
  const region = page.getByLabel('Conversation').filter({ has: page.getByRole('heading', { name }) });
  const area = await region.boundingBox();
  const box = await page.getByRole('textbox', { name: 'Message', exact: true }).boundingBox();
  expect(area).not.toBeNull();
  expect(box).not.toBeNull();
  return {
    top: area!.y,
    height: area!.height,
    middle: area!.y + area!.height / 2,
    composer: box!.y + box!.height / 2,
    bottom: box!.y + box!.height,
  };
}

async function createAgent(page: Page, name: string, persona: string) {
  const existing = page.getByRole('button', { name, exact: true });
  if ((await existing.count()) > 0) {
    return;
  }
  await page.getByRole('button', { name: 'New agent' }).filter({ visible: true }).click();
  await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
  await page.getByRole('textbox', { name: 'Persona', exact: true }).fill(persona);
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('button', { name, exact: true })).toBeVisible();
}

async function assertShell(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations).toEqual([]);
}

async function liveModels(): Promise<string[]> {
  const response = await fetch('http://127.0.0.1:8129/v1/models');
  if (!response.ok) {
    return [];
  }
  const body = (await response.json()) as { data?: { id?: string }[] };
  return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
}

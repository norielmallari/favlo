import { expect, test } from '@playwright/test';

import { onScreen } from './on-screen';

test('test connection lists MCP tools before save', async ({ page }) => {
  const tools = await liveMcpTools();
  test.skip(tools.length === 0, 'favlo serve MCP is not answering');
  const name = `Media ${Date.now()}`;

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('http://localhost:8012/');
  await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
  await page.getByRole('button', { name: 'Configuration' }).click();
  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByLabel('Agent details').getByRole('tab', { name: 'MCP' }).click();
  await page.getByLabel('Agent MCP').getByRole('button', { name: 'New MCP' }).click();
  await expect(page.getByRole('button', { name: 'Test connection' })).toBeVisible();
  await expect(page.getByText('generate_image', { exact: true })).toHaveCount(0);

  await page.getByRole('textbox', { name: 'MCP name' }).fill(name);
  await page.getByRole('radio', { name: 'HTTP' }).click();
  await page.getByRole('textbox', { name: 'URL' }).fill('http://127.0.0.1:8129');
  const testConnection = page.getByRole('button', { name: 'Test connection' });
  await onScreen(testConnection);
  await testConnection.click();
  await expect(page.getByText('Connection succeeded')).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'URL' })).toHaveValue('http://127.0.0.1:8129/mcp');
  for (const tool of tools) {
    await expect(page.getByText(tool, { exact: true })).toBeVisible();
  }
  const image = page.getByRole('switch', { name: `Image for ${name}` });
  await expect(image).toBeChecked();
  await image.click();
  await expect(image).not.toBeChecked();

  const createMcp = page.getByRole('button', { name: 'Create MCP' });
  await createMcp.scrollIntoViewIfNeeded();
  await onScreen(createMcp);
  await createMcp.click();
  await expect(page.getByLabel('Agent MCP').getByRole('button', { name, exact: true })).toBeVisible();
  await expect(page.getByText('generate_image', { exact: true })).toBeVisible();
  await expect(page.getByRole('switch', { name: `Image for ${name}` })).not.toBeChecked();

  await page.setViewportSize({ width: 1280, height: 640 });
  await page.getByLabel('Agent MCP').getByRole('button', { name: 'New MCP' }).click();
  await page.getByRole('textbox', { name: 'MCP name' }).fill(`${name} narrow`);
  await page.getByRole('radio', { name: 'HTTP' }).click();
  await page.getByRole('textbox', { name: 'URL' }).fill('http://127.0.0.1:8129');
  const shortTest = page.getByRole('button', { name: 'Test connection' });
  await shortTest.scrollIntoViewIfNeeded();
  await onScreen(shortTest);
  await shortTest.click();
  await expect(page.getByText('Connection succeeded')).toBeVisible();
  const shortCreate = page.getByRole('button', { name: 'Create MCP' });
  await shortCreate.scrollIntoViewIfNeeded();
  await onScreen(shortCreate);
  await shortCreate.click();
  await expect(page.getByLabel('Agent MCP').getByRole('button', { name: `${name} narrow`, exact: true })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Sections' }).click();
  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByLabel('Agents list').getByRole('button', { name: 'Favlo', exact: true }).click();
  await page.getByLabel('Agent details').getByRole('tab', { name: 'MCP' }).click();
  await page.getByLabel('Agent MCP').getByRole('button', { name: 'New MCP' }).click();
  await page.getByRole('textbox', { name: 'MCP name' }).fill(`${name} mobile`);
  await page.getByRole('radio', { name: 'HTTP' }).click();
  await page.getByRole('textbox', { name: 'URL' }).fill('http://127.0.0.1:8129');
  const mobileCreate = page.getByRole('button', { name: 'Create MCP' });
  await mobileCreate.scrollIntoViewIfNeeded();
  await onScreen(mobileCreate);
  await mobileCreate.click();
  await expect(page.getByTestId('save-confirmation')).toContainText(`${name} mobile was created.`);
  await page.getByRole('button', { name: 'Dismiss confirmation' }).click();
  await expect(page.getByLabel('Agent MCP').getByRole('button', { name: `${name} mobile`, exact: true })).toBeVisible();

  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByRole('button', { name: 'Sections' }).click().catch(() => undefined);
  await page.getByRole('tab', { name: 'Agents' }).click();
  await page.getByLabel('Agent details').getByRole('tab', { name: 'MCP' }).click();
  await page.getByLabel('Agent MCP').getByRole('button', { name, exact: true }).click();
  for (const tool of tools) {
    await expect(page.getByText(tool, { exact: true })).toBeVisible();
  }
  await expect(page.getByRole('switch', { name: `Image for ${name}` })).not.toBeChecked();
});

async function liveMcpTools(): Promise<string[]> {
  let response: Response;
  try {
    response = await fetch('http://127.0.0.1:8129/mcp', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'application/json, text/event-stream',
      'mcp-protocol-version': '2026-07-28',
      'mcp-method': 'tools/list',
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
      params: {
        _meta: {
          'io.modelcontextprotocol/protocolVersion': '2026-07-28',
          'io.modelcontextprotocol/clientCapabilities': {
            extensions: { 'io.modelcontextprotocol/tasks': {} },
          },
          'io.modelcontextprotocol/clientInfo': { name: 'favlo-web', version: '0' },
        },
      },
    }),
  });
  } catch {
    return [];
  }
  if (!response.ok) {
    return [];
  }
  const body = (await response.json()) as { result?: { tools?: { name?: string }[] } };
  return (body.result?.tools ?? []).map((tool) => tool.name).filter((name): name is string => Boolean(name));
}

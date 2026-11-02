import { expect, test } from '@playwright/test';

import { deleteAgentsNamed } from './cleanup';
import { onScreen } from './on-screen';
import { createAgent, openAgentChat } from './open-agent-chat';

test('a tool call opens and closes apart from thinking', async ({ page }) => {
  const name = `Wren ${Date.now()}`;
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('http://localhost:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await page.getByRole('button', { name: 'New agent' }).click();
    await page.getByRole('textbox', { name: 'Agent name', exact: true }).fill(name);
    await page.getByRole('textbox', { name: 'Persona', exact: true }).fill(
      'Call conversation_list_conversations before you answer. Use kind all and limit 1. Then reply with the word done.',
    );
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('qwen36-35b-a3b:thinking');
    await page.getByRole('combobox', { name: 'Reasoning effort', exact: true }).selectOption('medium');
    await page.getByRole('switch', { name: 'Conversation tools', exact: true }).click();
    await createAgent(page, name);
    await openAgentChat(page, name);
    await page.getByRole('textbox', { name: 'Message' }).fill('List the conversations.');
    let send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();

    const transcript = page.getByTestId('transcript');
    const tool = transcript.getByRole('button', { name: 'Tool call conversation_list_conversations' }).last();
    await expect(tool).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
    await expect(tool).toHaveAttribute('aria-expanded', 'false');
    await onScreen(tool);
    await tool.click();
    await expect(tool).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByTestId('tool-call-text')).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('textbox', { name: 'Message' }).fill('List the conversations again.');
    send = page.getByRole('button', { name: 'Send' });
    await onScreen(send);
    await send.click();
    const narrow = transcript.getByRole('button', { name: 'Tool call conversation_list_conversations' }).last();
    await expect(narrow).toBeVisible({ timeout: 180_000 });
    await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 240_000 });
    await expect(narrow).toHaveAttribute('aria-expanded', 'false');
    await onScreen(narrow);
    await narrow.click();
    await expect(page.getByTestId('tool-call-text').last()).toBeVisible();
  } finally {
    await deleteAgentsNamed(name);
  }
});

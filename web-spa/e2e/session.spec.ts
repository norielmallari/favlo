import fs from 'fs';
import os from 'os';
import path from 'path';

import { expect, test, type Page } from '@playwright/test';

import { deleteWorkspacesNamed } from './cleanup';
import { createGalleryWorkspace } from './gallery-create';
import { onScreen } from './on-screen';
import { step } from './progress';
import { FAVLO_AGENT_ID } from '../src/api/seed-ids';

test('session clear and archive stay on their session', async ({ page }) => {
  const models = await liveModels();
  const channelLine = `Channel note ${Date.now()}`;
  const workspaceLine = `Workspace note ${Date.now()}`;
  const workspaceName = `Session ${Date.now()}`;
  const folderRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'favlo-session-'));
  try {
    await page.setViewportSize({ width: 1280, height: 800 });
    await clearFavloSession();
    await page.goto('http://127.0.0.1:8012/');
    await expect(page).toHaveURL(/^http:\/\/127\.0\.0\.1:8012\/?$/);
    await openFavloHome(page);
    await expect(page.getByLabel('Gallery workspace', { exact: true })).toHaveCount(0);
    await waitIdle(page);

    const channel = page
      .getByLabel('Conversation')
      .filter({ has: page.getByRole('heading', { name: 'Favlo', exact: true }) });
    let galleryPage = page;
    const gallery = () => galleryPage.getByLabel('Gallery workspace', { exact: true });

    await step('send channel note', async () => {
      await sendLine(page, channelLine);
      await expect(channel.getByLabel(`You said: ${channelLine}`).first()).toBeVisible();
    });

    await step('create gallery with folder, send workspace note', async () => {
      galleryPage = await createGalleryWorkspace(page, workspaceName, folderRoot, { keepChannel: true });
      await sendLine(galleryPage, workspaceLine);
      await expect(gallery().getByLabel(`You said: ${workspaceLine}`).first()).toBeVisible();
      await waitIdle(galleryPage);
    });

    await step('clear workspace chat; strip resets', async () => {
      await waitIdle(galleryPage);
      await sendLine(galleryPage, '/session clear');
      await expect(gallery().getByLabel(`You said: ${workspaceLine}`)).toHaveCount(0);
      await expect(galleryPage.getByText('Ask the agent to fill the filmstrip')).toBeVisible();
      await expect(galleryPage.getByRole('heading', { name: workspaceName })).toBeVisible();
    });

    await step('channel note survives; workspace note can be resent', async () => {
      await expect(channel.getByLabel(`You said: ${channelLine}`).first()).toBeVisible();
      galleryPage = await openWorkspaceNamed(page, workspaceName);
      await expect(galleryPage.getByRole('heading', { name: workspaceName })).toBeVisible();
      await waitIdle(galleryPage);
      await sendLine(galleryPage, workspaceLine);
      await expect(gallery().getByLabel(`You said: ${workspaceLine}`).first()).toBeVisible();
      await waitIdle(galleryPage);
    });

    await step('clear home; workspace chat remains', async () => {
      await openFavloHome(page);
      await waitIdle(page);
      await sendLine(page, '/session clear');
      await expect(channel.getByLabel(`You said: ${channelLine}`)).toHaveCount(0);
      galleryPage = await openWorkspaceNamed(page, workspaceName);
      await expect(galleryPage.getByRole('heading', { name: workspaceName })).toBeVisible();
      const bodies = await workspaceMessageBodies(workspaceName);
      expect(bodies, `server messages for ${workspaceName}`).toContain(workspaceLine);
      await expect(gallery().getByLabel(`You said: ${workspaceLine}`).first()).toBeVisible();
    });

    await step('archive home chat', async () => {
      const ordinary = `Please clear this chat later ${Date.now()}`;
      await openFavloHome(page);
      await waitIdle(page);
      await sendLine(page, ordinary);
      await expect(channel.getByLabel(`You said: ${ordinary}`).first()).toBeVisible();
      test.skip(models.length === 0, 'favlo serve has no language model');
      // Let the reply finish. Stop aborts SSE and can leave a zombie running run,
      // which blocks /session archive with "already in progress".
      await expect(page.getByTestId('working')).toBeVisible({ timeout: 60_000 });
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 360_000 });
      await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible();
      await sendLine(page, '/session archive');
      await expect(page.getByTestId('working')).toBeVisible({ timeout: 180_000 });
      await expect(channel.getByLabel(`You said: ${ordinary}`)).toHaveCount(0, { timeout: 360_000 });
      await expect(page.getByTestId('working')).toHaveCount(0, { timeout: 360_000 });
      await expect(channel.getByLabel(/^You said: /)).toHaveCount(0);
      galleryPage = await openWorkspaceNamed(page, workspaceName);
      await expect(gallery().getByLabel(`You said: ${workspaceLine}`).first()).toBeVisible();
    });

    await step('narrow workspace clear', async () => {
      await galleryPage.setViewportSize({ width: 390, height: 844 });
      await expect(galleryPage.getByRole('heading', { name: workspaceName })).toBeVisible();
      await galleryPage.getByRole('button', { name: 'Chat' }).click();
      await waitIdle(galleryPage);
      const narrowLine = `Narrow workspace ${Date.now()}`;
      await sendLine(galleryPage, narrowLine);
      await expect(gallery().getByLabel(`You said: ${narrowLine}`).first()).toBeVisible();
      await waitIdle(galleryPage);
      await sendLine(galleryPage, '/session clear');
      await expect(gallery().getByLabel(`You said: ${narrowLine}`)).toHaveCount(0);
      await expect(galleryPage.getByRole('heading', { name: workspaceName })).toBeVisible();
    });
  } finally {
    await deleteWorkspacesNamed(workspaceName);
    fs.rmSync(folderRoot, { recursive: true, force: true });
  }
});

async function waitIdle(page: Page) {
  const stop = page.getByRole('button', { name: 'Stop', exact: true });
  const working = page.getByTestId('working');
  const send = page.getByRole('button', { name: 'Send', exact: true });
  if (await stop.count()) {
    await onScreen(stop);
    await stop.click();
  }
  await expect(working).toHaveCount(0, { timeout: 30_000 });
  await expect(stop).toHaveCount(0);
  await expect(send).toBeVisible();
}

async function typeMessage(page: Page, text: string) {
  const box = page.getByRole('textbox', { name: 'Message', exact: true });
  await box.click();
  await box.fill(text);
  await expect(box).toHaveValue(text);
}

/** Send a line; for chat text wait until /ag-ui starts so Stop cannot drop the user message. */
async function sendLine(page: Page, text: string) {
  await typeMessage(page, text);
  const send = page.getByRole('button', { name: 'Send', exact: true });
  await onScreen(send);
  await expect(send).toBeEnabled();
  const trimmed = text.trim();
  if (trimmed === '/session clear' || trimmed === '/session archive') {
    await send.click();
    return;
  }
  await Promise.all([
    page.waitForResponse((response) => response.url().includes('/ag-ui') && response.ok(), {
      timeout: 60_000,
    }),
    send.click(),
  ]);
}

async function openWorkspaceNamed(page: Page, name: string): Promise<Page> {
  const popupPromise = page.waitForEvent('popup');
  await page.getByRole('button', { name, exact: true }).click();
  const workspace = await popupPromise;
  await expect(workspace).toHaveURL(/\/w\/.+/);
  await expect(workspace.getByRole('heading', { name })).toBeVisible();
  return workspace;
}

async function openFavloHome(page: Page) {
  const channels = page.getByLabel('Channels');
  const favlo = channels.getByRole('button', { name: 'Favlo', exact: true });
  const home = channels.getByRole('button', { name: 'Home', exact: true });
  await expect(favlo).toBeVisible();
  // Seed expand shows Home; wait before clicking Favlo (a click collapses).
  try {
    await expect(home).toBeVisible({ timeout: 3_000 });
  } catch {
    await favlo.click();
    await expect(home).toBeVisible();
  }
  await home.click();
  await expect(page.getByLabel('Gallery workspace', { exact: true })).toHaveCount(0);
}

async function favloSessionId(): Promise<string | null> {
  const listed = await fetch('http://127.0.0.1:8130/channels');
  if (!listed.ok) {
    return null;
  }
  const body = (await listed.json()) as {
    channels?: { agent_id?: string; session_id?: string }[];
  };
  return (body.channels ?? []).find((row) => row.agent_id === FAVLO_AGENT_ID)?.session_id ?? null;
}

async function clearFavloSession() {
  const sessionId = await favloSessionId();
  if (sessionId == null) {
    return;
  }
  await fetch(`http://127.0.0.1:8130/sessions/${sessionId}/clear`, { method: 'POST' });
}

async function workspaceMessageBodies(name: string): Promise<string[]> {
  const listed = await fetch('http://127.0.0.1:8130/workspaces');
  if (!listed.ok) {
    return [];
  }
  const body = (await listed.json()) as {
    workspaces?: { name?: string; session_id?: string }[];
  };
  const row = (body.workspaces ?? []).find((item) => item.name === name);
  if (row?.session_id == null) {
    return [];
  }
  const snap = await fetch(`http://127.0.0.1:8130/sessions/${row.session_id}`);
  if (!snap.ok) {
    return [];
  }
  const session = (await snap.json()) as { messages?: { body?: string }[] };
  return (session.messages ?? []).map((item) => item.body ?? '');
}

async function liveModels(): Promise<string[]> {
  try {
    const response = await fetch('http://127.0.0.1:8129/v1/models');
    if (!response.ok) {
      return [];
    }
    const body = (await response.json()) as { data?: { id?: string }[] };
    return (body.data ?? []).map((row) => row.id).filter((id): id is string => Boolean(id));
  } catch {
    return [];
  }
}

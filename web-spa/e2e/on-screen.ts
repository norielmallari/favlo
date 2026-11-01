import { expect, type Locator } from '@playwright/test';

/**
 * The control is already inside the viewport.
 * locator.evaluate does not scroll. toBeVisible, toBeInViewport, and boundingBox do.
 * The read is repeated until the layout settles. A control that stays outside still fails.
 */
export async function onScreen(control: Locator): Promise<void> {
  await expect.poll(async () => {
    const box = await control.evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        right: rect.right,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      };
    });
    if (box.width <= 0 || box.height <= 0) {
      return `empty ${box.width}x${box.height}`;
    }
    if (box.x < -1 || box.y < -1) {
      return `origin ${Math.round(box.x)},${Math.round(box.y)}`;
    }
    if (box.right > box.viewportWidth + 1) {
      return `right ${Math.round(box.right)} > ${box.viewportWidth}`;
    }
    if (box.bottom > box.viewportHeight + 1) {
      return `bottom ${Math.round(box.bottom)} > ${box.viewportHeight}`;
    }
    return 'ok';
  }).toBe('ok');
}

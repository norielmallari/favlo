import zlib from 'zlib';

import { expect, type Locator, type Page } from '@playwright/test';

export async function expectFullFocusRing(page: Page, control: Locator) {
  const ring = await control.evaluate((element) => {
    const shadow = getComputedStyle(element).boxShadow;
    const match = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(shadow);
    const rect = element.getBoundingClientRect();
    return {
      shadow,
      color: match == null ? null : [Number(match[1]), Number(match[2]), Number(match[3])],
      box: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    };
  });
  expect(ring.shadow, 'focused control has no focus ring').not.toBe('none');
  expect(ring.color, `focus ring color missing from ${ring.shadow}`).toBeTruthy();

  const pad = 6;
  const clipX = Math.max(0, ring.box.x - pad);
  const clipY = Math.max(0, ring.box.y - pad);
  const clip = {
    x: clipX,
    y: clipY,
    width: ring.box.width + (ring.box.x - clipX) + pad,
    height: ring.box.height + (ring.box.y - clipY) + pad,
  };
  const png = decodePng(await page.screenshot({ clip }));
  const scale = png.width / clip.width;
  const left = (ring.box.x - clip.x) * scale;
  const top = (ring.box.y - clip.y) * scale;
  const width = ring.box.width * scale;
  const height = ring.box.height * scale;
  const spread = 2 * scale;
  const color = ring.color as number[];
  const sides = {
    top: countRing(png, color, left + width * 0.25, top - spread - 1, width * 0.5, spread + 1),
    bottom: countRing(png, color, left + width * 0.25, top + height, width * 0.5, spread + 1),
    left: countRing(png, color, left - spread - 1, top + height * 0.25, spread + 1, height * 0.5),
    right: countRing(png, color, left + width, top + height * 0.25, spread + 1, height * 0.5),
  };
  const missing = Object.entries(sides)
    .filter(([, count]) => count < 4)
    .map(([side, count]) => `${side}=${count}`);
  expect(missing, `focus ring is missing a side (${ring.shadow})`).toEqual([]);
}

function countRing(
  image: Png,
  color: number[],
  x: number,
  y: number,
  width: number,
  height: number,
): number {
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(image.width, Math.ceil(x + width));
  const y1 = Math.min(image.height, Math.ceil(y + height));
  let hits = 0;
  for (let py = y0; py < y1; py += 1) {
    for (let px = x0; px < x1; px += 1) {
      const index = (py * image.width + px) * 4;
      const red = image.data[index] ?? 0;
      const green = image.data[index + 1] ?? 0;
      const blue = image.data[index + 2] ?? 0;
      if (
        Math.abs(red - color[0]) <= 72 &&
        Math.abs(green - color[1]) <= 72 &&
        Math.abs(blue - color[2]) <= 72
      ) {
        hits += 1;
      }
    }
  }
  return hits;
}

type Png = { width: number; height: number; data: Uint8Array };

function decodePng(buffer: Buffer): Png {
  const raw = zlib.inflateSync(idat(buffer));
  const width = buffer.readUInt32BE(16);
  const height = buffer.readUInt32BE(20);
  const color = buffer[25];
  const channels = color === 6 ? 4 : color === 2 ? 3 : 0;
  if (channels === 0 || buffer[24] !== 8) {
    throw new Error(`unsupported png color ${color}`);
  }
  const stride = width * channels;
  const data = new Uint8Array(width * height * 4);
  let offset = 0;
  let previous = new Uint8Array(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[offset];
    offset += 1;
    const row = Uint8Array.from(raw.subarray(offset, offset + stride));
    offset += stride;
    unfilter(filter, row, previous, channels);
    for (let x = 0; x < width; x += 1) {
      const source = x * channels;
      const target = (y * width + x) * 4;
      data[target] = row[source] ?? 0;
      data[target + 1] = row[source + 1] ?? 0;
      data[target + 2] = row[source + 2] ?? 0;
      data[target + 3] = channels === 4 ? (row[source + 3] ?? 255) : 255;
    }
    previous = row;
  }
  return { width, height, data };
}

function idat(buffer: Buffer): Buffer {
  const chunks: Buffer[] = [];
  let position = 8;
  while (position < buffer.length) {
    const length = buffer.readUInt32BE(position);
    const type = buffer.toString('ascii', position + 4, position + 8);
    const data = buffer.subarray(position + 8, position + 8 + length);
    position += 12 + length;
    if (type === 'IDAT') {
      chunks.push(data);
    }
    if (type === 'IEND') {
      break;
    }
  }
  return Buffer.concat(chunks);
}

function unfilter(filter: number, row: Uint8Array, previous: Uint8Array, channels: number) {
  for (let index = 0; index < row.length; index += 1) {
    const left = index >= channels ? (row[index - channels] ?? 0) : 0;
    const up = previous[index] ?? 0;
    const upLeft = index >= channels ? (previous[index - channels] ?? 0) : 0;
    if (filter === 1) {
      row[index] = (row[index] + left) & 255;
    } else if (filter === 2) {
      row[index] = (row[index] + up) & 255;
    } else if (filter === 3) {
      row[index] = (row[index] + Math.floor((left + up) / 2)) & 255;
    } else if (filter === 4) {
      const estimate = left + up - upLeft;
      const leftDistance = Math.abs(estimate - left);
      const upDistance = Math.abs(estimate - up);
      const diagonal = Math.abs(estimate - upLeft);
      const predictor =
        leftDistance <= upDistance && leftDistance <= diagonal ? left : upDistance <= diagonal ? up : upLeft;
      row[index] = (row[index] + predictor) & 255;
    } else if (filter !== 0) {
      throw new Error(`unsupported png filter ${filter}`);
    }
  }
}

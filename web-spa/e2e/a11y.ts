import AxeBuilder from '@axe-core/playwright';
import { expect, type Page } from '@playwright/test';

/** Fail the test when axe finds WCAG A/AA text or UI contrast issues. */
export async function assertWcagAa(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
    .analyze();
  expect(results.violations, formatViolations(results.violations)).toEqual([]);
}

function formatViolations(
  violations: { id: string; help: string; nodes: { target: unknown[] }[] }[],
): string {
  if (violations.length === 0) {
    return '';
  }
  return violations
    .map((item) => {
      const targets = item.nodes
        .map((node) => node.target.map(String).join(' '))
        .join('; ');
      return `${item.id}: ${item.help} (${targets})`;
    })
    .join('\n');
}

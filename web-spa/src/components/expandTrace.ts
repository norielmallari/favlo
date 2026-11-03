/** How one transcript block opens. Thinking and tool calls each store their own value. */
export type ExpandTrace = 'hidden' | 'collapsed' | 'running' | 'always';

export function readExpandTrace(value: unknown, fallback: ExpandTrace = 'collapsed'): ExpandTrace {
  if (value === 'hidden' || value === 'collapsed' || value === 'running' || value === 'always') {
    return value;
  }
  return fallback;
}

export function traceStartsOpen(settled: boolean, mode: ExpandTrace, forceOpen: boolean): boolean {
  if (mode === 'hidden') {
    return false;
  }
  if (forceOpen || mode === 'always') {
    return true;
  }
  if (mode === 'running') {
    return !settled;
  }
  return false;
}

export function workingLabel(
  steps: { kind: string; settled?: boolean; call?: { name?: string } }[],
  thinking: ExpandTrace = 'collapsed',
  tools: ExpandTrace = 'hidden',
): string {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step == null || step.settled) {
      continue;
    }
    if (step.kind === 'thinking') {
      return thinking === 'hidden' ? 'Working' : 'Thinking';
    }
    if (step.kind === 'tool') {
      if (tools === 'hidden') {
        return 'Working';
      }
      const name = step.call?.name?.trim();
      return name ? `Tool call ${name}` : 'Tool call';
    }
  }
  return 'Working';
}

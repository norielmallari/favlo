/** How thinking and tool-call blocks open in the transcript. */
export type ExpandTrace = 'collapsed' | 'running' | 'always';

export function readExpandTrace(value: unknown): ExpandTrace {
  if (value === 'running' || value === 'always') {
    return value;
  }
  return 'collapsed';
}

export function traceStartsOpen(settled: boolean, mode: ExpandTrace, forceOpen: boolean): boolean {
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
): string {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step == null || step.settled) {
      continue;
    }
    if (step.kind === 'thinking') {
      return 'Thinking';
    }
    if (step.kind === 'tool') {
      const name = step.call?.name?.trim();
      return name ? `Tool call ${name}` : 'Tool call';
    }
  }
  return 'Working';
}

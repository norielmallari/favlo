export function composerCommand(key: string, shift: boolean, composing: boolean): 'send' | 'break' {
  if (composing || key !== 'Enter' || shift) {
    return 'break';
  }
  return 'send';
}

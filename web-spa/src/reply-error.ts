/** Plain language for a failed chat reply. A class name stays out of the alert. */

const detailPattern = /['"]message['"]\s*:\s*['"]([^'"]+)['"]/;
const smoke = 'the gpu is held for a pipeline smoke';

export function spokenError(raw: string): string {
  const text = raw.trim();
  if (!text) {
    return 'The listener did not complete the reply. Send again.';
  }
  if (!dumped(text)) {
    return text;
  }
  const found = detailPattern.exec(text);
  const detail = found?.[1]?.trim() ?? '';
  if (detail === smoke) {
    return 'The GPU is held for a pipeline smoke. Send again when it is free.';
  }
  if (!detail) {
    return 'The listener did not complete the reply. Send again.';
  }
  const sentence = detail.charAt(0).toUpperCase() + detail.slice(1);
  const ended = /[.!?]$/.test(sentence) ? sentence : `${sentence}.`;
  return `${ended} Send again.`;
}

function dumped(text: string): boolean {
  return (
    text.includes('service failed')
    || text.includes('<class ')
    || text.startsWith('(')
    || text.includes('Error code:')
  );
}

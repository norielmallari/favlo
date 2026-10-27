import { composerCommand } from '@/components/composer-keys';

describe('composer keys', () => {
  test('Enter sends', () => {
    expect(composerCommand('Enter', false, false)).toBe('send');
  });

  test('Shift+Enter inserts a newline', () => {
    expect(composerCommand('Enter', true, false)).toBe('break');
  });

  test('Enter during composition does not send', () => {
    expect(composerCommand('Enter', false, true)).toBe('break');
  });
});

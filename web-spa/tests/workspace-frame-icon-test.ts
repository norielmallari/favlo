import { appMarkIcon } from '@/workspace/WorkspaceFrame';

test('the gallery tab icon is the aperture shutter', () => {
  const icon = decodeURIComponent(appMarkIcon('aperture', '#c4a7e7', '#232136').slice('data:image/svg+xml,'.length));
  expect(icon).toContain('cx="12" cy="12" r="10"');
  expect(icon).toContain('M14.31 8L20.05 17.94');
  expect(icon).not.toContain('M12 2v2');
});

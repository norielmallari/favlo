import { act, render } from '@testing-library/react-native';

import { downloadName, formatActionNewlines, MessageBody, mediaFrame, messageParts } from '@/components/MessageBody';
import { ThemeProvider } from '@/theme/Theme';

test('a closed action gets one newline and a model newline is dropped', () => {
  expect(formatActionNewlines('*smiles*Hello')).toBe('*smiles*\nHello');
  expect(formatActionNewlines('*smiles* Hello')).toBe('*smiles*\nHello');
  expect(formatActionNewlines('*smiles*\nHello')).toBe('*smiles*\nHello');
  expect(formatActionNewlines('*smiles*\r\nHello')).toBe('*smiles*\nHello');
  expect(formatActionNewlines('*smiles*\n\nHello')).toBe('*smiles*\n\nHello');
  expect(formatActionNewlines('*a* *b* next')).toBe('*a*\n*b*\nnext');
  expect(formatActionNewlines('**bold** stays')).toBe('**bold** stays');
  expect(formatActionNewlines('*open')).toBe('*open');
  expect(formatActionNewlines('*smiles*')).toBe('*smiles*');
  expect(formatActionNewlines('> *smiles* Hello')).toBe('> *smiles* Hello');
  expect(formatActionNewlines('> *smiles*\n> more')).toBe('> *smiles*\n> more');
  expect(formatActionNewlines('> *a* quoted\n*b* outside')).toBe('> *a* quoted\n*b*\noutside');
  expect(formatActionNewlines('```xml\n*smiles* Hello\n```')).toBe('```xml\n*smiles* Hello\n```');
  expect(formatActionNewlines('```json\n*a* x\n```\n*b* y')).toBe('```json\n*a* x\n```\n*b*\ny');
});

test('a favlo image link becomes a generated image', async () => {
  const parts = messageParts('generate_image finished.\nimage_abc\nfavlo://images/image_abc.png\n');
  expect(parts.map((part) => part.type)).toEqual(['image']);
  expect(parts[0]).toMatchObject({
    type: 'image',
    label: 'Generated image',
    src: 'http://127.0.0.1:8130/media/images/image_abc.png',
  });
  const view = await render(
    <ThemeProvider>
      <MessageBody body={'Here it is.\ngenerate_image finished.\nimage_abc\nfavlo://images/image_abc.png'} />
    </ThemeProvider>,
  );
  expect(view.getByText('Here it is.')).toBeTruthy();
  expect(view.queryByText(/generate_image finished/)).toBeNull();
  expect(view.queryByText('image_abc')).toBeNull();
  expect(view.getByLabelText('Generated image')).toBeTruthy();
  expect(mediaFrame.height).toBe('auto');
  expect(mediaFrame.width).toBe('auto');
  expect(mediaFrame.objectFit).toBe('contain');
  expect(mediaFrame.alignSelf).toBe('flex-start');
});

test('a missing image stays off the screen', async () => {
  const empty = jest.fn();
  const view = await render(
    <ThemeProvider>
      <MessageBody body={'favlo://images/image_missing.png'} onEmpty={empty} />
    </ThemeProvider>,
  );
  await act(async () => {
    view.getByLabelText('Generated image').props.onError();
  });
  expect(view.queryByLabelText('Generated image')).toBeNull();
  expect(view.queryByLabelText('View generated image')).toBeNull();
  expect(empty).toHaveBeenCalled();
});

test('a download uses a plain file name', () => {
  expect(downloadName('http://127.0.0.1:8130/media/images/image_abc.png', 'image')).toBe('favlo-image.png');
  expect(downloadName('http://127.0.0.1:8130/media/videos/video_abc.mp4', 'video')).toBe('favlo-video.mp4');
  expect(downloadName('http://127.0.0.1:8130/media/music/music_abc.flac', 'audio')).toBe('favlo-audio.flac');
});

import { act, render } from '@testing-library/react-native';

import {
  downloadName,
  formatActionNewlines,
  MessageBody,
  mediaFrame,
  messageParts,
  stripHtmlTags,
} from '@/components/MessageBody';
import { ThemeProvider } from '@/theme/Theme';

test('assistant display drops HTML tags outside quotes and fences', () => {
  expect(stripHtmlTags('Hello <b>there</b>')).toBe('Hello there');
  expect(stripHtmlTags('A <br/> break')).toBe('A  break');
  expect(stripHtmlTags('<!-- note -->kept')).toBe('kept');
  expect(stripHtmlTags('> <b>quoted</b>')).toBe('> <b>quoted</b>');
  expect(stripHtmlTags('```html\n<div>x</div>\n```')).toBe('```html\n<div>x</div>\n```');
  expect(stripHtmlTags('out <i>x</i>\n```html\n<span>y</span>\n```\nafter <b>z</b>')).toBe(
    'out x\n```html\n<span>y</span>\n```\nafter z',
  );
  expect(messageParts('Say <em>hi</em>', false, true)[0]).toMatchObject({ type: 'text', text: 'Say hi' });
  expect(messageParts('Keep <em>hi</em>', false, false)[0]).toMatchObject({
    type: 'text',
    text: 'Keep <em>hi</em>',
  });
});

test('a closed action gets one newline and a model newline is dropped', () => {
  expect(formatActionNewlines('*smiles*Hello')).toBe('*smiles*  \nHello');
  expect(formatActionNewlines('*Nicker*! Making one')).toBe('*Nicker*!  \nMaking one');
  expect(formatActionNewlines('*smiles*!? Hello')).toBe('*smiles*!?  \nHello');
  expect(formatActionNewlines('*smiles*... next')).toBe('*smiles*...  \nnext');
  expect(formatActionNewlines('*smiles* Hello')).toBe('*smiles*  \nHello');
  expect(formatActionNewlines('*smiles*\nHello')).toBe('*smiles*  \nHello');
  expect(formatActionNewlines('*smiles*\r\nHello')).toBe('*smiles*  \nHello');
  expect(formatActionNewlines('*smiles*\n\nHello')).toBe('*smiles*\n\nHello');
  expect(formatActionNewlines('*a* *b* next')).toBe('*a*  \n*b*  \nnext');
  expect(formatActionNewlines('Thank you, friend! *Nicker*!')).toBe('Thank you, friend!  \n*Nicker*!');
  expect(formatActionNewlines('Example *this should not* result to a new line.')).toBe(
    'Example *this should not* result to a new line.',
  );
  expect(formatActionNewlines('Hello *smiles* there')).toBe('Hello *smiles* there');
  expect(formatActionNewlines('*while this does*')).toBe('*while this does*');
  expect(formatActionNewlines('*while this does* And more')).toBe('*while this does*  \nAnd more');
  expect(formatActionNewlines('Hello\n*smiles*')).toBe('Hello  \n*smiles*');
  expect(
    formatActionNewlines('*Chomp chomp*!\nOh wow, these are crunching delicious! Thank you, friend! *Nicker*!'),
  ).toBe('*Chomp chomp*!  \nOh wow, these are crunching delicious! Thank you, friend!  \n*Nicker*!');
  expect(formatActionNewlines('**bold** stays')).toBe('**bold** stays');
  expect(formatActionNewlines('**see *smiles* now**')).toBe('**see *smiles* now**');
  expect(formatActionNewlines('*smiles* **bold**')).toBe('*smiles*  \n**bold**');
  expect(formatActionNewlines('**me.png** stays')).toBe('**me.png** stays');
  expect(formatActionNewlines('*open')).toBe('*open');
  expect(formatActionNewlines('*smiles*')).toBe('*smiles*');
  expect(formatActionNewlines('> *smiles* Hello')).toBe('> *smiles* Hello');
  expect(formatActionNewlines('> *smiles*\n> more')).toBe('> *smiles*\n> more');
  expect(formatActionNewlines('> *a* quoted\n*b* outside')).toBe('> *a* quoted  \n*b*  \noutside');
  expect(formatActionNewlines('```xml\n*smiles* Hello\n```')).toBe('```xml\n*smiles* Hello\n```');
  expect(formatActionNewlines('```json\n*a* x\n```\n*b* y')).toBe('```json\n*a* x\n```  \n*b*  \ny');
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

test('a gallery chat shows the image file name', () => {
  const parts = messageParts('Here: **favlo://images/image_abc.png** now', true);
  expect(parts).toEqual([{ type: 'text', text: 'Here: **image_abc.png** now' }]);
});

test('bold markers around an image link stay off their own line', () => {
  const parts = messageParts('See **favlo://images/image_abc.png** now');
  expect(parts.map((part) => part.type)).toEqual(['text', 'image', 'text']);
  expect(parts[0]).toMatchObject({ text: 'See' });
  expect(parts[2]).toMatchObject({ text: 'now' });
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

test('status and task id lines stay off the screen', async () => {
  const task = 's4O1fkWCRA49jY3nUVOBg_IbtlcUwR_z9NxXkCSFE8k';
  const parts = messageParts(`status:\n${task}\nHere is the drawing.`);
  expect(parts.map((part) => part.type)).toEqual(['text']);
  expect(parts[0]).toMatchObject({ type: 'text', text: 'Here is the drawing.' });
  const view = await render(
    <ThemeProvider>
      <MessageBody body={`status:\n${task}\nHere is the drawing.`} />
    </ThemeProvider>,
  );
  expect(view.getByText('Here is the drawing.')).toBeTruthy();
  expect(view.queryByText(/^status:?$/i)).toBeNull();
  expect(view.queryByText(task)).toBeNull();
});

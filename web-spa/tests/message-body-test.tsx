import { render } from '@testing-library/react-native';

import { MessageBody, mediaFrame, messageParts } from '@/components/MessageBody';
import { ThemeProvider } from '@/theme/Theme';

test('a favlo image link becomes a generated image', async () => {
  const parts = messageParts('generate_image finished.\nfavlo://images/image_abc.png\n');
  expect(parts.map((part) => part.type)).toEqual(['text', 'image', 'text']);
  const image = parts[1];
  expect(image).toMatchObject({
    type: 'image',
    label: 'Generated image',
    src: 'http://127.0.0.1:8130/media/images/image_abc.png',
  });
  const view = await render(
    <ThemeProvider>
      <MessageBody body={'generate_image finished.\nfavlo://images/image_abc.png'} />
    </ThemeProvider>,
  );
  expect(view.getByText(/generate_image finished/)).toBeTruthy();
  expect(view.getByLabelText('Generated image')).toBeTruthy();
  expect(mediaFrame.height).toBe('auto');
  expect(mediaFrame.width).toBe('auto');
  expect(mediaFrame.objectFit).toBe('contain');
  expect(mediaFrame.alignSelf).toBe('flex-start');
});

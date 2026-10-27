import { render } from '@testing-library/react-native';

import { MarkdownText } from '@/components/MarkdownText';
import { MessageBody } from '@/components/MessageBody';
import { ThemeProvider } from '@/theme/Theme';

const sample = ['See **bold** and *italic*.', '', '| Name | Note |', '| --- | --- |', '| One | first |'].join('\n');

function types(node: unknown, found: string[] = []): string[] {
  if (node == null || typeof node !== 'object') {
    return found;
  }
  if (Array.isArray(node)) {
    for (const item of node) {
      types(item, found);
    }
    return found;
  }
  const record = node as { type?: string; children?: unknown; props?: { href?: string; src?: string } };
  if (typeof record.type === 'string') {
    found.push(record.type);
  }
  if (record.props?.href) {
    found.push(record.props.href);
  }
  if (record.props?.src) {
    found.push(record.props.src);
  }
  types(record.children, found);
  return found;
}

test('a message renders bold, italics, and a table', async () => {
  const view = await render(
    <ThemeProvider>
      <MessageBody body={`${sample}\n\nfavlo://images/image_abc.png`} />
    </ThemeProvider>,
  );
  expect(view.getByText('bold').props.style).toMatchObject({ fontWeight: '700' });
  expect(view.getByText('italic').props.style).toMatchObject({ fontStyle: 'italic' });
  expect(view.queryByText('**bold**')).toBeNull();
  expect(view.getByText('Name')).toBeTruthy();
  expect(view.getByText('One')).toBeTruthy();
  expect(view.getByTestId('markdown-table')).toBeTruthy();
  expect(view.getByLabelText('Generated image')).toBeTruthy();
});

test('raw html and unsafe links stay inert', async () => {
  const view = await render(
    <ThemeProvider>
      <MarkdownText
        text={'See <script>alert(1)</script> now\n\n[go](javascript:alert(1))\n\n![photo](https://example.com/a.png)'}
        color="#182230"
      />
    </ThemeProvider>,
  );
  const found = types(view.toJSON());
  expect(found).not.toContain('script');
  expect(found.join(' ')).not.toContain('javascript:');
  expect(found).not.toContain('https://example.com/a.png');
  expect(view.getByText('photo')).toBeTruthy();
  expect(view.getByText(/alert\(1\)/)).toBeTruthy();
});

import { type ReactNode } from 'react';
import { Platform, ScrollView, Text, View, type TextStyle, type ViewStyle } from 'react-native';
import Markdown, { defaultUrlTransform, type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { useTheme } from '@/theme/Theme';

type Props = { text: string; color: string; fontSize?: number; lineHeight?: number };

export function MarkdownText({ text, color, fontSize = 16, lineHeight = 24 }: Props) {
  const { palette } = useTheme();
  const block = bodyStyle(color, fontSize, lineHeight);
  const cell = cellStyle(palette.border);
  const components: Components = {
    p: ({ children }) => <Text style={block}>{children}</Text>,
    strong: ({ children }) => <Text style={strong}>{children}</Text>,
    em: ({ children }) => <Text style={emphasis}>{children}</Text>,
    del: ({ children }) => <Text style={strike}>{children}</Text>,
    a: ({ href, children }) => <Link href={href} color={palette.accent}>{children}</Link>,
    img: ({ alt }) => (alt ? <Text style={block}>{alt}</Text> : null),
    h1: ({ children }) => <Text style={headingStyle(block, 20)}>{children}</Text>,
    h2: ({ children }) => <Text style={headingStyle(block, 18)}>{children}</Text>,
    h3: ({ children }) => <Text style={headingStyle(block, 18)}>{children}</Text>,
    h4: ({ children }) => <Text style={headingStyle(block, 18)}>{children}</Text>,
    h5: ({ children }) => <Text style={headingStyle(block, 18)}>{children}</Text>,
    h6: ({ children }) => <Text style={headingStyle(block, 18)}>{children}</Text>,
    ul: ({ children }) => <View style={list}>{children}</View>,
    ol: ({ children }) => <View style={list}>{children}</View>,
    li: ({ children }) => (
      <View style={item}>
        <Text style={block}>•</Text>
        <View style={itemBody}>{children}</View>
      </View>
    ),
    blockquote: ({ children }) => (
      <View style={{ paddingLeft: 12, borderLeftWidth: 2, borderLeftColor: palette.border }}>{children}</View>
    ),
    pre: ({ children }) => (
      <View style={{ padding: 12, borderRadius: 8, backgroundColor: palette.surfaceHover, maxWidth: '100%' }}>{children}</View>
    ),
    code: ({ children }) => <Text style={codeStyle(block)}>{children}</Text>,
    table: ({ children }) => (
      <ScrollView horizontal testID="markdown-table" style={tableScroll}>
        <View role="table" style={table}>{children}</View>
      </ScrollView>
    ),
    thead: ({ children }) => <View role="rowgroup">{children}</View>,
    tbody: ({ children }) => <View role="rowgroup">{children}</View>,
    tr: ({ children }) => <View role="row" style={row}>{children}</View>,
    th: ({ children, style }) => (
      <View role="columnheader" style={[cell, headerCell(palette.surfaceHover), { alignItems: alignItems(style) }]}>
        <Text style={[block, strong]}>{children}</Text>
      </View>
    ),
    td: ({ children, style }) => (
      <View role="cell" style={[cell, { alignItems: alignItems(style) }]}>
        <Text style={block}>{children}</Text>
      </View>
    ),
    hr: () => <View style={{ height: 1, backgroundColor: palette.border, maxWidth: '100%' }} />,
    br: () => <Text>{'\n'}</Text>,
    input: () => null,
  };
  return (
    <View style={{ maxWidth: '100%', minWidth: 0, gap: 8 }}>
      <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[dropBlank]} urlTransform={defaultUrlTransform} components={components}>
        {text}
      </Markdown>
    </View>
  );
}

function Link({ href, color, children }: { href?: string; color: string; children?: ReactNode }) {
  if (!href || (!href.startsWith('https://') && !href.startsWith('http://') && !href.startsWith('mailto:'))) {
    return <Text>{children}</Text>;
  }
  return (
    <Text
      role="link"
      style={{ color, textDecorationLine: 'underline' }}
      onPress={() => {
        if (Platform.OS === 'web') {
          window.open(href, '_blank', 'noopener,noreferrer');
        }
      }}
    >
      {children}
    </Text>
  );
}

function bodyStyle(color: string, fontSize: number, lineHeight: number): TextStyle {
  return { color, fontSize, lineHeight, maxWidth: '100%' };
}

function headingStyle(block: TextStyle, fontSize: number): TextStyle {
  return { ...block, fontSize, fontWeight: '600' };
}

function codeStyle(block: TextStyle): TextStyle {
  return { ...block, fontFamily: 'monospace' };
}

function cellStyle(border: string): ViewStyle {
  return {
    borderWidth: 1,
    borderColor: border,
    paddingVertical: 8,
    paddingHorizontal: 12,
    minWidth: 96,
    maxWidth: 280,
  };
}

function headerCell(background: string): ViewStyle {
  return { backgroundColor: background };
}

function alignItems(style: { textAlign?: string } | undefined): ViewStyle['alignItems'] {
  if (style?.textAlign === 'center') {
    return 'center';
  }
  if (style?.textAlign === 'right') {
    return 'flex-end';
  }
  return 'flex-start';
}

type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

const blockTags = new Set([
  'root',
  'table',
  'thead',
  'tbody',
  'tr',
  'ul',
  'ol',
  'blockquote',
  'pre',
  'li',
  'td',
  'th',
  'div',
]);

function dropBlank() {
  return (tree: HastNode) => {
    clean(tree, 'root');
  };
}

function clean(node: HastNode, tag: string) {
  if (!node.children) {
    return;
  }
  const next: HastNode[] = [];
  for (const child of node.children) {
    if (child.type === 'text' || child.type === 'raw') {
      if (!blockTags.has(tag)) {
        next.push(child);
        continue;
      }
      if ((child.value ?? '').trim() !== '') {
        next.push({ type: 'element', tagName: 'p', properties: {}, children: [{ type: 'text', value: child.value }] });
      }
      continue;
    }
    if (child.type === 'element' && child.tagName) {
      child.properties = child.properties ?? {};
      clean(child, child.tagName);
    }
    next.push(child);
  }
  node.children = next;
}

const strong: TextStyle = { fontWeight: '700' };
const emphasis: TextStyle = { fontStyle: 'italic' };
const strike: TextStyle = { textDecorationLine: 'line-through' };
const list: ViewStyle = { gap: 4, maxWidth: '100%' };
const item: ViewStyle = { flexDirection: 'row', gap: 8, maxWidth: '100%' };
const itemBody: ViewStyle = { flex: 1, minWidth: 0 };
const row: ViewStyle = { flexDirection: 'row', alignItems: 'stretch' };
const table: ViewStyle = { minWidth: '100%' };
const tableScroll: ViewStyle = { maxWidth: '100%' };

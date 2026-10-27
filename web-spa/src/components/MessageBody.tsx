import { createElement, useState, type CSSProperties } from 'react';
import { Image, Platform, Text, View } from 'react-native';

import { API_URL } from '@/api/client';
import { useTheme } from '@/theme/Theme';

const MEDIA = /favlo:\/\/(images|videos|voice|music)\/([A-Za-z0-9_-]+)\.(png|mp4|mp3|wav|flac)/g;
const MEDIA_MAX_HEIGHT = 480;

export const mediaFrame: CSSProperties = {
  display: 'block',
  width: 'auto',
  height: 'auto',
  maxWidth: '100%',
  maxHeight: MEDIA_MAX_HEIGHT,
  objectFit: 'contain',
  alignSelf: 'flex-start',
  borderRadius: 8,
};

export type MessagePart =
  | { type: 'text'; text: string }
  | { type: 'image' | 'video' | 'audio'; src: string; label: string };

export function messageParts(body: string): MessagePart[] {
  const parts: MessagePart[] = [];
  let cursor = 0;
  for (const match of body.matchAll(MEDIA)) {
    const kind = match[1];
    const id = match[2];
    const ext = match[3];
    if (!allowed(kind, ext) || match.index == null) {
      continue;
    }
    if (match.index > cursor) {
      parts.push({ type: 'text', text: body.slice(cursor, match.index) });
    }
    parts.push({
      type: kind === 'images' ? 'image' : kind === 'videos' ? 'video' : 'audio',
      src: `${API_URL}/media/${kind}/${id}.${ext}`,
      label: kind === 'images' ? 'Generated image' : kind === 'videos' ? 'Generated video' : 'Generated audio',
    });
    cursor = match.index + match[0].length;
  }
  if (cursor < body.length) {
    parts.push({ type: 'text', text: body.slice(cursor) });
  }
  return parts;
}

export function MessageBody({ body }: { body: string }) {
  const { palette } = useTheme();
  const parts = messageParts(body);
  return (
    <View style={{ gap: 8 }}>
      {parts.map((part, index) =>
        part.type === 'text' ? (
          <Text key={index} style={{ color: palette.textPrimary, fontSize: 16, lineHeight: 24 }}>
            {part.text}
          </Text>
        ) : (
          <Media key={index} part={part} />
        ),
      )}
    </View>
  );
}

function Media({ part }: { part: Extract<MessagePart, { type: 'image' | 'video' | 'audio' }> }) {
  if (Platform.OS === 'web') {
    if (part.type === 'image') {
      return createElement('img', { src: part.src, alt: part.label, style: mediaFrame });
    }
    if (part.type === 'video') {
      return createElement('video', { src: part.src, controls: true, style: mediaFrame });
    }
    return createElement('audio', { src: part.src, controls: true });
  }
  if (part.type === 'image') {
    return <FittedImage part={part} />;
  }
  return null;
}

function FittedImage({ part }: { part: { src: string; label: string } }) {
  const [ratio, setRatio] = useState(16 / 9);
  return (
    <Image
      source={{ uri: part.src }}
      accessibilityLabel={part.label}
      resizeMode="contain"
      onLoad={(event) => {
        const source = event.nativeEvent.source;
        if (source.width > 0 && source.height > 0) {
          setRatio(source.width / source.height);
        }
      }}
      style={{ width: '100%', maxWidth: MEDIA_MAX_HEIGHT * ratio, aspectRatio: ratio, borderRadius: 8 }}
    />
  );
}

function allowed(kind: string, ext: string): boolean {
  if (kind === 'images') {
    return ext === 'png';
  }
  if (kind === 'videos') {
    return ext === 'mp4';
  }
  return ext === 'mp3' || ext === 'wav' || ext === 'flac';
}

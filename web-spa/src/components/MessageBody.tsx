import { createElement, useEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { Image, Platform, Text, View } from 'react-native';

import { API_URL } from '@/api/client';
import { MarkdownText } from '@/components/MarkdownText';
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
      pushText(parts, body.slice(cursor, match.index));
    }
    parts.push({
      type: kind === 'images' ? 'image' : kind === 'videos' ? 'video' : 'audio',
      src: `${API_URL}/media/${kind}/${id}.${ext}`,
      label: kind === 'images' ? 'Generated image' : kind === 'videos' ? 'Generated video' : 'Generated audio',
    });
    cursor = match.index + match[0].length;
  }
  if (cursor < body.length) {
    pushText(parts, body.slice(cursor));
  }
  return parts;
}

function pushText(parts: MessagePart[], text: string) {
  const kept = text
    .split('\n')
    .filter((line) => !machineLine(line))
    .join('\n')
    .trim();
  if (kept) {
    parts.push({ type: 'text', text: kept });
  }
}

const MEDIA_ID = /^(image|video|music|voice)_[A-Za-z0-9_-]+$/;
const MEDIA_SIZE = /^\d+x\d+$/;
const MEDIA_DURATION = /^\d+(\.\d+)?s?$/;

export function humanText(body: string): string {
  const parts = messageParts(body);
  const words = parts
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
    .trim();
  if (words) {
    return words;
  }
  if (parts.some((part) => part.type === 'image')) {
    return 'an image';
  }
  if (parts.some((part) => part.type === 'video')) {
    return 'a video';
  }
  if (parts.some((part) => part.type === 'audio')) {
    return 'audio';
  }
  return '';
}

function machineLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) {
    return false;
  }
  if (/^[a-z0-9_]+ (finished|failed|was cancelled)\.?$/i.test(trimmed)) {
    return true;
  }
  const tokens = trimmed.split(/\s+/);
  const named = tokens.some((token) => MEDIA_ID.test(token));
  const onlyFile = tokens.every((token) => MEDIA_ID.test(token) || MEDIA_SIZE.test(token) || MEDIA_DURATION.test(token));
  return named && onlyFile;
}

export function MessageBody({ body, onEmpty }: { body: string; onEmpty?: () => void }) {
  const { palette } = useTheme();
  const parts = messageParts(body);
  const media = parts.filter((part) => part.type !== 'text').length;
  const hasText = parts.some((part) => part.type === 'text');
  const [gone, setGone] = useState(0);
  useEffect(() => {
    if (!hasText && media > 0 && gone >= media) {
      onEmpty?.();
    }
  }, [gone, hasText, media, onEmpty]);
  return (
    <View style={{ gap: 8, maxWidth: '100%', minWidth: 0 }}>
      {parts.map((part, index) =>
        part.type === 'text' ? (
          <MarkdownText key={index} text={part.text} color={palette.textPrimary} />
        ) : (
          <Media key={index} part={part} onMissing={() => setGone((count) => count + 1)} />
        ),
      )}
    </View>
  );
}

export function downloadName(src: string, kind: 'image' | 'video' | 'audio'): string {
  const ext = src.split('?')[0]?.split('.').pop()?.toLowerCase() ?? '';
  const safe = ext === 'png' || ext === 'mp4' || ext === 'mp3' || ext === 'wav' || ext === 'flac' ? ext : fallbackExt(kind);
  const stem = kind === 'image' ? 'favlo-image' : kind === 'video' ? 'favlo-video' : 'favlo-audio';
  return `${stem}.${safe}`;
}

export async function saveMedia(src: string, filename: string): Promise<void> {
  const response = await fetch(src);
  if (!response.ok) {
    throw new Error('missing');
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fallbackExt(kind: 'image' | 'video' | 'audio'): string {
  if (kind === 'image') {
    return 'png';
  }
  if (kind === 'video') {
    return 'mp4';
  }
  return 'mp3';
}

function Media({
  part,
  onMissing,
}: {
  part: Extract<MessagePart, { type: 'image' | 'video' | 'audio' }>;
  onMissing: () => void;
}) {
  if (Platform.OS === 'web') {
    return <WebMedia part={part} onMissing={onMissing} />;
  }
  if (part.type === 'image') {
    return <FittedImage part={part} onMissing={onMissing} />;
  }
  return null;
}

function WebMedia({
  part,
  onMissing,
}: {
  part: Extract<MessagePart, { type: 'image' | 'video' | 'audio' }>;
  onMissing: () => void;
}) {
  const { palette } = useTheme();
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const [missing, setMissing] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const opened = useRef(false);

  useEffect(() => {
    if (open) {
      opened.current = true;
      return;
    }
    if (opened.current) {
      trigger.current?.focus();
    }
  }, [open]);

  async function save() {
    setFailed(false);
    try {
      await saveMedia(part.src, downloadName(part.src, part.type));
    } catch {
      setFailed(true);
    }
  }

  const downloadLabel = part.type === 'image' ? 'Download image' : part.type === 'video' ? 'Download video' : 'Download audio';
  if (part.type === 'image') {
    if (missing) {
      return null;
    }
    return (
      <>
        {createElement(
          'button',
          {
            type: 'button',
            ref: trigger,
            'aria-label': 'View generated image',
            'data-media-view': 'image',
            onClick: () => setOpen(true),
            style: { ...viewButton, display: ready ? 'block' : 'none' },
          },
          createElement('img', {
            src: part.src,
            alt: part.label,
            style: mediaFrame,
            onLoad: () => setReady(true),
            onError: () => {
              setMissing(true);
              onMissing();
            },
          }),
        )}
        {createElement('style', null, `[data-media-view]:focus-visible { outline: 2px solid ${palette.focusRing}; outline-offset: 2px; }`)}
        {open ? <MediaDialog part={part} label={downloadLabel} failed={failed} onSave={() => void save()} onClose={() => setOpen(false)} /> : null}
      </>
    );
  }
  const player = part.type === 'video'
    ? createElement('video', { src: part.src, controls: true, style: mediaFrame, 'aria-label': part.label })
    : createElement('audio', { src: part.src, controls: true, 'aria-label': part.label });
  return (
    <View style={{ gap: 8, maxWidth: '100%', alignSelf: 'flex-start' }}>
      {player}
      {createElement('button', { type: 'button', onClick: () => void save(), style: actionButton(palette.surface, palette.textPrimary, palette.border) }, downloadLabel)}
      {failed ? <Text style={{ color: palette.danger, fontSize: 14 }}>The file could not be saved.</Text> : null}
    </View>
  );
}

function MediaDialog({
  part,
  label,
  failed,
  onSave,
  onClose,
}: {
  part: Extract<MessagePart, { type: 'image' | 'video' | 'audio' }>;
  label: string;
  failed: boolean;
  onSave: () => void;
  onClose: () => void;
}) {
  const { palette } = useTheme();
  const dialog = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const node = dialog.current;
    const hidden: Element[] = [];
    for (const child of document.body.children) {
      if (node != null && child.contains(node)) {
        continue;
      }
      child.setAttribute('inert', '');
      hidden.push(child);
    }
    node?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close.current();
        return;
      }
      if (event.key !== 'Tab' || node == null) {
        return;
      }
      const items = [...node.querySelectorAll<HTMLElement>('button')];
      const first = items[0];
      const last = items[items.length - 1];
      if (first == null || last == null) {
        return;
      }
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      for (const child of hidden) {
        child.removeAttribute('inert');
      }
    };
  }, []);

  const preview = part.type === 'image'
    ? createElement('img', { src: part.src, alt: part.label, style: previewFrame })
    : part.type === 'video'
      ? createElement('video', { src: part.src, controls: true, style: previewFrame, 'aria-label': part.label })
      : createElement('audio', { src: part.src, controls: true, 'aria-label': part.label });

  return createPortal(
    createElement(
      'div',
      {
        role: 'presentation',
        onClick: onClose,
        style: {
          position: 'fixed',
          inset: 0,
          zIndex: 30,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 16,
          background: palette.overlay,
        },
      },
      createElement(
        'div',
        {
          ref: dialog,
          role: 'dialog',
          'aria-modal': true,
          'aria-label': part.label,
          tabIndex: -1,
          onClick: (event: { stopPropagation: () => void }) => event.stopPropagation(),
          style: {
            display: 'block',
            boxSizing: 'border-box',
            width: 'min(960px, 100%)',
            maxHeight: 'calc(100vh - 32px)',
            overflow: 'hidden',
            padding: 16,
            borderRadius: 12,
            background: palette.surface,
            outline: 'none',
          },
        },
        createElement(
          'div',
          { style: { display: 'flex', flexWrap: 'wrap', gap: 8 } },
          createElement('button', { type: 'button', onClick: onSave, style: actionButton(palette.accent, palette.onAccent, palette.accent) }, label),
          createElement('button', { type: 'button', onClick: onClose, style: actionButton(palette.surface, palette.textPrimary, palette.border) }, 'Close'),
        ),
        failed ? createElement('p', { style: { margin: 0, color: palette.danger, fontSize: 14 } }, 'The file could not be saved.') : null,
        createElement('div', { style: { paddingTop: 12 } }, preview),
      ),
    ),
    document.body,
  );
}

const viewButton: CSSProperties = {
  display: 'block',
  padding: 0,
  border: 'none',
  background: 'transparent',
  maxWidth: '100%',
  cursor: 'zoom-in',
  alignSelf: 'flex-start',
};

const previewFrame: CSSProperties = {
  display: 'block',
  width: '100%',
  height: 'auto',
  maxHeight: 'calc(100vh - 180px)',
  objectFit: 'contain',
  borderRadius: 8,
};

function actionButton(background: string, color: string, border: string): CSSProperties {
  return {
    minHeight: 44,
    padding: '0 16px',
    borderRadius: 8,
    border: `1px solid ${border}`,
    background,
    color,
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  };
}

function FittedImage({ part, onMissing }: { part: { src: string; label: string }; onMissing: () => void }) {
  const [ratio, setRatio] = useState(16 / 9);
  const [missing, setMissing] = useState(false);
  if (missing) {
    return null;
  }
  return (
    <Image
      source={{ uri: part.src }}
      accessibilityLabel={part.label}
      resizeMode="contain"
      onError={() => {
        setMissing(true);
        onMissing();
      }}
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

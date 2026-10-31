import { useCallback, useState } from 'react';
import { Platform, Text, View } from 'react-native';

import { humanText, MessageBody } from '@/components/MessageBody';
import { IconButton } from '@/components/IconButton';
import { Thinking } from '@/components/Thinking';
import { useTheme } from '@/theme/Theme';
import { radius, space } from '@/theme/tokens';
import { textStyle } from '@/theme/type';

export type ChatMessage = { id: string; role: string; body: string; reasoning?: string };
export const LATEST_SLACK = 64;

export function showsMessage(text?: string, reasoning?: string | null): boolean {
  return Boolean(text) || Boolean(reasoning && reasoning.trim());
}

export function clearedTranscript(
  event: { type: string; stream_id?: string },
  sessionId: string | undefined,
): boolean {
  return sessionId != null && event.type === 'session.cleared' && event.stream_id === `session:${sessionId}`;
}

export function keepLatest(following: boolean, placed: boolean, fromBottom: number): boolean {
  if (!placed) {
    return following;
  }
  return fromBottom <= LATEST_SLACK;
}

export function transcriptFrame(gap: number) {
  return {
    flexGrow: 1,
    justifyContent: 'flex-end' as const,
    gap,
    maxWidth: '100%' as const,
  };
}

export function chatMessage(item: { id: string; role: string; body: string; reasoning?: string | null }): ChatMessage {
  const reasoning = item.reasoning?.trim() ?? '';
  return reasoning
    ? { id: item.id, role: item.role, body: item.body, reasoning }
    : { id: item.id, role: item.role, body: item.body };
}

export function readDraft(sessionId: string): string {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return '';
  }
  return localStorage.getItem(`favlo.draft.${sessionId}`) ?? '';
}

export function writeDraft(sessionId: string, value: string) {
  if (Platform.OS !== 'web' || typeof localStorage === 'undefined') {
    return;
  }
  localStorage.setItem(`favlo.draft.${sessionId}`, value);
}

export function TranscriptTurn({
  name,
  role,
  body,
  reasoning,
}: {
  name: string;
  role: string;
  body: string;
  reasoning?: string;
}) {
  const { palette } = useTheme();
  const [empty, setEmpty] = useState(false);
  const [copied, setCopied] = useState(false);
  const reportEmpty = useCallback(() => setEmpty(true), []);
  const mine = role === 'user';
  if (empty) {
    return null;
  }
  return (
    <View
      accessibilityLabel={`${name} said: ${humanText(body)}`}
      style={{ alignItems: mine ? 'flex-end' : 'flex-start', maxWidth: '100%' }}
    >
      <View
        style={{
          maxWidth: mine ? '85%' : '100%',
          gap: space[1],
          padding: mine ? space[3] : 0,
          borderRadius: radius.card,
          backgroundColor: mine ? palette.accentSubtle : 'transparent',
        }}
      >
        <Text style={textStyle('label', palette.textPrimary, '600')}>{name}</Text>
        {reasoning ? <Thinking text={reasoning} settled /> : null}
        <MessageBody body={body} onEmpty={reportEmpty} />
        {!mine && humanText(body) ? (
          <IconButton
            icon={copied ? 'check' : 'copy'}
            label={copied ? 'Copied' : 'Copy'}
            accessibilityLabel="Copy"
            onPress={() => {
              if (Platform.OS === 'web') {
                void navigator.clipboard.writeText(humanText(body)).then(() => setCopied(true));
              }
            }}
            background="transparent"
            border="transparent"
            color={palette.textSecondary}
          />
        ) : null}
      </View>
    </View>
  );
}

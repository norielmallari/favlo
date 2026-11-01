import { spokenError } from '@/reply-error';

const raw = [
  "(\"<class 'agent_framework_openai._chat_completion_client.OpenAIChatCompletionClient'>",
  " service failed to complete the prompt: Error code: 503 - ",
  "{'error': {'message': 'the gpu is held for a pipeline smoke', 'type': 'busy'}}\", ",
  "InternalServerError(\"Error code: 503 - ",
  "{'error': {'message': 'the gpu is held for a pipeline smoke', 'type': 'busy'}}\"))",
].join('');

test('a smoke hold is a sentence', () => {
  expect(spokenError(raw)).toBe(
    'The GPU is held for a pipeline smoke. Send again when it is free.',
  );
});

test('a plain failure stays as written', () => {
  expect(spokenError('The model returned an empty reply.')).toBe(
    'The model returned an empty reply.',
  );
});

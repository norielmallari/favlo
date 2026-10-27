export const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://127.0.0.1:8130';

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body != null && !headers.has('content-type')) {
    headers.set('content-type', 'application/json');
  }
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  const body = (await response.json()) as { message?: string };
  if (!response.ok) {
    throw new ApiError(body.message ?? 'The request failed.');
  }
  return body as T;
}

export function idempotencyKey(): string {
  return globalThis.crypto.randomUUID();
}

export function socketUrl(): string {
  return `${API_URL.replace(/^http/, 'ws')}/events`;
}

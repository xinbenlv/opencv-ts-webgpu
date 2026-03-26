/**
 * Cached model fetcher using the Cache API with download progress.
 *
 * Models are large (64-111 MB) and don't change between page loads.
 * The Cache API persists across sessions, so after the first download
 * subsequent loads are instant (read from disk).
 */

const CACHE_NAME = 'josh-models-v1';

type StatusFn = (id: string, status: string, text: string) => void;

function formatBytes(bytes: number): string {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatSpeed(bytesPerSec: number): string {
  if (bytesPerSec >= 1e6) return `${(bytesPerSec / 1e6).toFixed(1)} MB/s`;
  if (bytesPerSec >= 1e3) return `${(bytesPerSec / 1e3).toFixed(0)} KB/s`;
  return `${bytesPerSec.toFixed(0)} B/s`;
}

/**
 * Fetch with streaming progress — reads body in chunks and reports speed.
 */
async function fetchWithProgress(
  url: string,
  stepId: string,
  label: string,
  statusFn?: StatusFn,
): Promise<{ buffer: ArrayBuffer; response: Response }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const contentLength = Number(response.headers.get('content-length') || 0);
  const reader = response.body?.getReader();

  if (!reader || !contentLength) {
    // No streaming — fall back to simple arrayBuffer()
    const cloned = response.clone();
    const buf = await response.arrayBuffer();
    return { buffer: buf, response: cloned };
  }

  // Stream the response body, reporting progress
  const chunks: Uint8Array[] = [];
  let received = 0;
  const startTime = performance.now();
  let lastUpdateTime = startTime;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    received += value.byteLength;

    const now = performance.now();
    // Update UI at most every 200ms to avoid flooding
    if (now - lastUpdateTime > 200) {
      lastUpdateTime = now;
      const elapsed = (now - startTime) / 1000;
      const speed = received / elapsed;
      const pct = Math.round((received / contentLength) * 100);
      const eta = (contentLength - received) / speed;
      statusFn?.(
        stepId,
        'active',
        `${label} — ${pct}% (${formatBytes(received)}/${formatBytes(contentLength)}) ${formatSpeed(speed)}${eta > 1 ? ` ~${Math.ceil(eta)}s left` : ''}`,
      );
    }
  }

  // Combine chunks into single ArrayBuffer
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Build a synthetic Response for cache storage
  const cacheResponse = new Response(buffer.buffer, {
    status: 200,
    headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(received) },
  });

  return { buffer: buffer.buffer, response: cacheResponse };
}

/**
 * Fetch a model with Cache API persistence, progress reporting, and retry logic.
 */
export async function cachedFetchModel(
  url: string,
  stepId: string,
  label: string,
  _sizeHint: string,
  statusFn?: StatusFn,
  retries = 3,
): Promise<ArrayBuffer> {
  // Try Cache API first
  if ('caches' in globalThis) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        statusFn?.(stepId, 'active', `${label} (cached, reading from disk)...`);
        const buf = await cached.arrayBuffer();
        statusFn?.(stepId, 'done', `${label} ready (${formatBytes(buf.byteLength)}, cached)`);
        console.log(`[CachedFetch] Cache hit: ${url} (${formatBytes(buf.byteLength)})`);
        return buf;
      }
    } catch (e) {
      console.warn('[CachedFetch] Cache read failed, falling back to network:', e);
    }
  }

  // Network fetch with progress + retry
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const retryLabel = attempt > 1 ? ` (retry ${attempt}/${retries})` : '';
      statusFn?.(stepId, 'active', `${label} — connecting${retryLabel}...`);
      console.log(`[CachedFetch] Downloading ${url} (attempt ${attempt}/${retries})`);

      const { buffer, response: cacheResponse } = await fetchWithProgress(url, stepId, label, statusFn);

      statusFn?.(stepId, 'done', `${label} ready (${formatBytes(buffer.byteLength)})`);

      // Store in cache for next time
      if ('caches' in globalThis) {
        try {
          const cache = await caches.open(CACHE_NAME);
          await cache.put(url, cacheResponse);
          console.log(`[CachedFetch] Cached: ${url} (${formatBytes(buffer.byteLength)})`);
        } catch (e) {
          console.warn('[CachedFetch] Cache write failed:', e);
        }
      }

      return buffer;
    } catch (e) {
      if (attempt === retries) {
        statusFn?.(stepId, 'error', `${label} — download failed after ${retries} attempts`);
        throw e;
      }
      statusFn?.(stepId, 'warn', `${label} — failed, retrying in 2s (${attempt + 1}/${retries})...`);
      console.warn(`[CachedFetch] Attempt ${attempt} failed:`, e);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  throw new Error('unreachable');
}

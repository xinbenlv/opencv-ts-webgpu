/**
 * Cached model fetcher using the Cache API.
 *
 * Models are large (64-111 MB) and don't change between page loads.
 * The Cache API persists across sessions, so after the first download
 * subsequent loads are instant (read from disk).
 *
 * Falls back to regular fetch if Cache API is unavailable.
 */

const CACHE_NAME = 'josh-models-v1';

type StatusFn = (id: string, status: string, text: string) => void;

/**
 * Fetch a model with Cache API persistence and retry logic.
 * Reports download progress via the loading status callback.
 */
export async function cachedFetchModel(
  url: string,
  stepId: string,
  label: string,
  sizeHint: string,
  statusFn?: StatusFn,
  retries = 3,
): Promise<ArrayBuffer> {
  // Try Cache API first
  if ('caches' in globalThis) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        statusFn?.(stepId, 'active', `${label} (cached, loading from disk)...`);
        const buf = await cached.arrayBuffer();
        statusFn?.(stepId, 'done', `${label} ready (cached)`);
        console.log(`[CachedFetch] Cache hit: ${url} (${(buf.byteLength / 1e6).toFixed(1)} MB)`);
        return buf;
      }
    } catch (e) {
      console.warn('[CachedFetch] Cache read failed, falling back to network:', e);
    }
  }

  // Network fetch with retry
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      statusFn?.(stepId, 'active', `${label} — downloading ${sizeHint}${attempt > 1 ? ` (retry ${attempt}/${retries})` : ''}...`);
      console.log(`[CachedFetch] Downloading ${url} (attempt ${attempt}/${retries})`);

      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      // Clone response for caching before consuming body
      const responseForCache = response.clone();
      const buf = await response.arrayBuffer();

      // Store in cache for next time
      if ('caches' in globalThis) {
        try {
          const cache = await caches.open(CACHE_NAME);
          await cache.put(url, responseForCache);
          console.log(`[CachedFetch] Cached: ${url} (${(buf.byteLength / 1e6).toFixed(1)} MB)`);
        } catch (e) {
          console.warn('[CachedFetch] Cache write failed:', e);
        }
      }

      return buf;
    } catch (e) {
      if (attempt === retries) {
        statusFn?.(stepId, 'error', `${label} — download failed`);
        throw e;
      }
      statusFn?.(stepId, 'warn', `${label} — retry ${attempt + 1}/${retries} in 2s...`);
      console.warn(`[CachedFetch] Attempt ${attempt} failed:`, e);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  throw new Error('unreachable');
}

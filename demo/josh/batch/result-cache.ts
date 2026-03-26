/**
 * IndexedDB cache for per-frame JOSH results.
 *
 * DB name:   josh-results-<videoHash>
 * Stores:
 *   frames   — keyPath: 'frameIndex'
 *   metadata — keyPath: 'key'
 *
 * Not supported in SSR / Node environments (throws immediately on open()).
 */

export interface PerFrameResult {
  frameIndex: number;
  timestamp: number;
  smplPose: Float32Array;
  smplShape: Float32Array;
  /** Column-major 4×4 camera transform, length 16 */
  cameraPose: Float32Array;
  depthScale: number;
  /** 6890 × 3 SMPL mesh vertices, or null if not stored */
  vertices: Float32Array | null;
  /** 24 × 3 joint world positions */
  jointPositions: Float32Array;
  /** 6 scalar losses */
  losses: Float32Array;
}

const DB_VERSION = 1;
const FRAMES_STORE = 'frames';
const META_STORE = 'metadata';
const META_KEY = 'meta';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function openDb(name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(name, DB_VERSION);

    req.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(FRAMES_STORE)) {
        db.createObjectStore(FRAMES_STORE, { keyPath: 'frameIndex' });
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'key' });
      }
    };

    req.onsuccess = (event) => resolve((event.target as IDBOpenDBRequest).result);
    req.onerror = (event) => reject((event.target as IDBOpenDBRequest).error);
  });
}

function txGet<T>(store: IDBObjectStore, key: IDBValidKey): Promise<T | null> {
  return new Promise((resolve, reject) => {
    const req = store.get(key);
    req.onsuccess = () => resolve((req.result as T) ?? null);
    req.onerror = () => reject(req.error);
  });
}

function txPut(store: IDBObjectStore, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const req = store.put(value);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

function txGetAll<T>(store: IDBObjectStore): Promise<T[]> {
  return new Promise((resolve, reject) => {
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as T[]);
    req.onerror = () => reject(req.error);
  });
}

function txCount(store: IDBObjectStore): Promise<number> {
  return new Promise((resolve, reject) => {
    const req = store.count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txClear(store: IDBObjectStore): Promise<void> {
  return new Promise((resolve, reject) => {
    const req = store.clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ---------------------------------------------------------------------------
// Cache class
// ---------------------------------------------------------------------------

export class JoshResultCache {
  private readonly dbName: string;
  private db: IDBDatabase | null = null;

  constructor(videoHash: string) {
    this.dbName = `josh-results-${videoHash}`;
  }

  /** Open (or create) the IndexedDB database. Must be called before any other method. */
  async open(): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      throw new Error(
        'JoshResultCache: IndexedDB is not available in this environment (SSR / Node not supported).',
      );
    }
    this.db = await openDb(this.dbName);
  }

  /** Close the database connection. */
  async close(): Promise<void> {
    this.db?.close();
    this.db = null;
  }

  private assertOpen(): IDBDatabase {
    if (!this.db) throw new Error('JoshResultCache: database is not open. Call open() first.');
    return this.db;
  }

  /** Persist a single frame result. Overwrites any existing entry for the same frameIndex. */
  async storeFrame(result: PerFrameResult): Promise<void> {
    const db = this.assertOpen();
    const tx = db.transaction(FRAMES_STORE, 'readwrite');
    await txPut(tx.objectStore(FRAMES_STORE), result);
  }

  /** Load a single frame result by index, or null if not present. */
  async loadFrame(frameIndex: number): Promise<PerFrameResult | null> {
    const db = this.assertOpen();
    const tx = db.transaction(FRAMES_STORE, 'readonly');
    return txGet<PerFrameResult>(tx.objectStore(FRAMES_STORE), frameIndex);
  }

  /** Load all stored frame results, sorted by frameIndex ascending. */
  async loadAllFrames(): Promise<PerFrameResult[]> {
    const db = this.assertOpen();
    const tx = db.transaction(FRAMES_STORE, 'readonly');
    const rows = await txGetAll<PerFrameResult>(tx.objectStore(FRAMES_STORE));
    return rows.sort((a, b) => a.frameIndex - b.frameIndex);
  }

  /** Returns true if a result for the given frameIndex has been cached. */
  async hasFrame(frameIndex: number): Promise<boolean> {
    return (await this.loadFrame(frameIndex)) !== null;
  }

  /** Store video-level metadata. */
  async storeMetadata(meta: {
    videoHash: string;
    frameCount: number;
    fps: number;
    processedAt: number;
  }): Promise<void> {
    const db = this.assertOpen();
    const tx = db.transaction(META_STORE, 'readwrite');
    await txPut(tx.objectStore(META_STORE), { key: META_KEY, ...meta });
  }

  /** Load video-level metadata, or null if not stored yet. */
  async loadMetadata(): Promise<{
    videoHash: string;
    frameCount: number;
    fps: number;
    processedAt: number;
  } | null> {
    const db = this.assertOpen();
    const tx = db.transaction(META_STORE, 'readonly');
    const row = await txGet<{ key: string; videoHash: string; frameCount: number; fps: number; processedAt: number }>(
      tx.objectStore(META_STORE),
      META_KEY,
    );
    if (!row) return null;
    const { key: _key, ...meta } = row;
    return meta;
  }

  /** Delete all stored frames and metadata. */
  async clear(): Promise<void> {
    const db = this.assertOpen();
    const tx = db.transaction([FRAMES_STORE, META_STORE], 'readwrite');
    await Promise.all([
      txClear(tx.objectStore(FRAMES_STORE)),
      txClear(tx.objectStore(META_STORE)),
    ]);
  }

  /** Return the number of cached frame results. */
  async frameCount(): Promise<number> {
    const db = this.assertOpen();
    const tx = db.transaction(FRAMES_STORE, 'readonly');
    return txCount(tx.objectStore(FRAMES_STORE));
  }
}

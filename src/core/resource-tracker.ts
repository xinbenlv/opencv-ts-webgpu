import { DisposedError } from './errors.ts';

/**
 * Any object with a deterministic cleanup method.
 */
export interface Disposable {
  dispose(): void;
}

/**
 * Scope-based resource lifetime manager (RAII pattern).
 *
 * Resources are disposed in LIFO order — dependents freed before dependencies.
 * Critical for GPU resources: GPUBuffer.destroy() is NOT called by GC.
 */
export interface IResourceTracker {
  track<T extends Disposable>(resource: T): T;
  createScope(label?: string): IResourceTracker;
  untrack(resource: Disposable): void;
  dispose(): void;
  readonly isDisposed: boolean;
  readonly trackedCount: number;
}

export class ResourceTracker implements IResourceTracker {
  private readonly _resources: Disposable[] = [];
  private readonly _children: ResourceTracker[] = [];
  private _disposed = false;
  private readonly _label: string;

  constructor(label = 'root') {
    this._label = label;
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  get trackedCount(): number {
    return this._resources.length;
  }

  get label(): string {
    return this._label;
  }

  /**
   * Register a resource for cleanup when this tracker is disposed.
   * Returns the resource for inline usage: `const buf = tracker.track(new GpuBuffer())`.
   */
  track<T extends Disposable>(resource: T): T {
    this._assertAlive();
    this._resources.push(resource);
    return resource;
  }

  /**
   * Remove a resource from tracking without disposing it.
   * Use when transferring ownership to another scope.
   */
  untrack(resource: Disposable): void {
    this._assertAlive();
    const idx = this._resources.indexOf(resource);
    if (idx !== -1) {
      this._resources.splice(idx, 1);
    }
  }

  /**
   * Create a child scope. When the parent is disposed, all children are too.
   */
  createScope(label?: string): ResourceTracker {
    this._assertAlive();
    const child = new ResourceTracker(label ?? `${this._label}/child`);
    this._children.push(child);
    return child;
  }

  /**
   * Dispose all tracked resources in LIFO order, then all child scopes.
   */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;

    // Children first (they may depend on parent resources)
    for (let i = this._children.length - 1; i >= 0; i--) {
      this._children[i]!.dispose();
    }
    this._children.length = 0;

    // Resources in LIFO order
    for (let i = this._resources.length - 1; i >= 0; i--) {
      try {
        this._resources[i]!.dispose();
      } catch {
        // Swallow disposal errors to ensure all resources get cleaned up
      }
    }
    this._resources.length = 0;
  }

  private _assertAlive(): void {
    if (this._disposed) {
      throw new DisposedError(`ResourceTracker(${this._label})`);
    }
  }
}

/**
 * Execute a function within a scoped resource tracker.
 * All tracked resources are automatically disposed when the function returns.
 */
export function using<T>(fn: (tracker: ResourceTracker) => T): T {
  const tracker = new ResourceTracker('using');
  try {
    return fn(tracker);
  } finally {
    tracker.dispose();
  }
}

/**
 * Async variant of `using`.
 */
export async function usingAsync<T>(fn: (tracker: ResourceTracker) => Promise<T>): Promise<T> {
  const tracker = new ResourceTracker('usingAsync');
  try {
    return await fn(tracker);
  } finally {
    tracker.dispose();
  }
}

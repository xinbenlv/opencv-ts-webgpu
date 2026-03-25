import { describe, it, expect, vi } from 'vitest';
import { ResourceTracker, using, usingAsync } from '@core/resource-tracker.ts';
import { DisposedError } from '@core/errors.ts';

describe('ResourceTracker', () => {
  it('should track and dispose resources in LIFO order', () => {
    const order: number[] = [];
    const tracker = new ResourceTracker('test');

    tracker.track({ dispose: () => order.push(1) });
    tracker.track({ dispose: () => order.push(2) });
    tracker.track({ dispose: () => order.push(3) });

    tracker.dispose();

    expect(order).toEqual([3, 2, 1]);
    expect(tracker.isDisposed).toBe(true);
    expect(tracker.trackedCount).toBe(0);
  });

  it('should return tracked resource for inline usage', () => {
    const tracker = new ResourceTracker('test');
    const resource = { dispose: vi.fn(), value: 42 };

    const tracked = tracker.track(resource);
    expect(tracked).toBe(resource);
    expect(tracked.value).toBe(42);

    tracker.dispose();
    expect(resource.dispose).toHaveBeenCalledOnce();
  });

  it('should untrack a resource without disposing it', () => {
    const tracker = new ResourceTracker('test');
    const resource = { dispose: vi.fn() };

    tracker.track(resource);
    expect(tracker.trackedCount).toBe(1);

    tracker.untrack(resource);
    expect(tracker.trackedCount).toBe(0);

    tracker.dispose();
    expect(resource.dispose).not.toHaveBeenCalled();
  });

  it('should dispose child scopes when parent is disposed', () => {
    const parent = new ResourceTracker('parent');
    const child = parent.createScope('child');

    const parentResource = { dispose: vi.fn() };
    const childResource = { dispose: vi.fn() };

    parent.track(parentResource);
    child.track(childResource);

    parent.dispose();

    expect(childResource.dispose).toHaveBeenCalledOnce();
    expect(parentResource.dispose).toHaveBeenCalledOnce();
    expect(child.isDisposed).toBe(true);
  });

  it('should be safe to dispose twice', () => {
    const tracker = new ResourceTracker('test');
    const resource = { dispose: vi.fn() };

    tracker.track(resource);
    tracker.dispose();
    tracker.dispose(); // should not throw

    expect(resource.dispose).toHaveBeenCalledOnce();
  });

  it('should throw DisposedError when tracking on disposed tracker', () => {
    const tracker = new ResourceTracker('test');
    tracker.dispose();

    expect(() => tracker.track({ dispose: () => {} })).toThrow(DisposedError);
  });

  it('should swallow individual disposal errors', () => {
    const tracker = new ResourceTracker('test');

    tracker.track({
      dispose: () => {
        throw new Error('boom');
      },
    });
    const afterBoom = { dispose: vi.fn() };
    tracker.track(afterBoom);

    // afterBoom is added second, disposed first (LIFO)
    // Then the throwing resource is disposed — error swallowed
    expect(() => tracker.dispose()).not.toThrow();
    expect(afterBoom.dispose).toHaveBeenCalledOnce();
  });
});

describe('using', () => {
  it('should dispose tracker after synchronous function', () => {
    const resource = { dispose: vi.fn() };

    const result = using((tracker) => {
      tracker.track(resource);
      return 42;
    });

    expect(result).toBe(42);
    expect(resource.dispose).toHaveBeenCalledOnce();
  });

  it('should dispose tracker even if function throws', () => {
    const resource = { dispose: vi.fn() };

    expect(() =>
      using((tracker) => {
        tracker.track(resource);
        throw new Error('test');
      }),
    ).toThrow('test');

    expect(resource.dispose).toHaveBeenCalledOnce();
  });
});

describe('usingAsync', () => {
  it('should dispose tracker after async function', async () => {
    const resource = { dispose: vi.fn() };

    const result = await usingAsync(async (tracker) => {
      tracker.track(resource);
      await Promise.resolve();
      return 'hello';
    });

    expect(result).toBe('hello');
    expect(resource.dispose).toHaveBeenCalledOnce();
  });
});

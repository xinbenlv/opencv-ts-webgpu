import type { BufferId, BufferLayout } from './types.ts';
import { createBufferId } from './types.ts';
import { BufferError, DisposedError } from './errors.ts';
import type { Disposable } from './resource-tracker.ts';

interface BufferEntry {
  readonly id: BufferId;
  readonly layout: BufferLayout;
  gpuBuffer: GPUBuffer | null;
  cpuBuffer: ArrayBuffer | null;
  readonly label: string;
  inUse: boolean;
}

interface DoubleBufferSlot {
  front: BufferId;
  back: BufferId;
}

/**
 * Manages GPU and CPU buffer allocation with pooling and double-buffering.
 *
 * Key responsibilities:
 * - Pool GPUBuffer/ArrayBuffer instances to reduce allocation churn
 * - Provide double-buffering for real-time video pipelines
 * - Transfer data between GPU and CPU address spaces
 */
export class BufferManager implements Disposable {
  private readonly _buffers = new Map<BufferId, BufferEntry>();
  private readonly _pool: BufferEntry[] = [];
  private readonly _doubleBuffers = new Map<string, DoubleBufferSlot>();
  private readonly _device: GPUDevice;
  private _disposed = false;
  private _allocatedBytes = 0;
  private _pooledBytes = 0;

  constructor(device: GPUDevice) {
    this._device = device;
  }

  get allocatedBytes(): number {
    return this._allocatedBytes;
  }

  get pooledBytes(): number {
    return this._pooledBytes;
  }

  /**
   * Acquire a buffer matching the layout. Reuses pooled buffers when possible.
   */
  acquire(layout: BufferLayout, label?: string): BufferId {
    this._assertAlive();

    // Try to find a compatible pooled buffer
    const poolIdx = this._pool.findIndex(
      (e) => e.layout.byteLength >= layout.byteLength && !e.inUse,
    );

    if (poolIdx !== -1) {
      const entry = this._pool.splice(poolIdx, 1)[0]!;
      entry.inUse = true;
      this._pooledBytes -= entry.layout.byteLength;
      this._buffers.set(entry.id, entry);
      return entry.id;
    }

    // Allocate new buffer
    const id = createBufferId(label);
    const gpuBuffer = this._device.createBuffer({
      size: layout.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      label: label ?? id,
    });

    const entry: BufferEntry = {
      id,
      layout,
      gpuBuffer,
      cpuBuffer: null,
      label: label ?? id,
      inUse: true,
    };

    this._buffers.set(id, entry);
    this._allocatedBytes += layout.byteLength;
    return id;
  }

  /**
   * Return a buffer to the pool for reuse. Does not destroy the underlying GPU buffer.
   */
  release(id: BufferId): void {
    this._assertAlive();
    const entry = this._getEntry(id);
    entry.inUse = false;
    this._buffers.delete(id);
    this._pool.push(entry);
    this._pooledBytes += entry.layout.byteLength;
  }

  getGpuBuffer(id: BufferId): GPUBuffer {
    const entry = this._getEntry(id);
    if (!entry.gpuBuffer) {
      throw new BufferError(`Buffer ${id} has no GPU allocation.`);
    }
    return entry.gpuBuffer;
  }

  getCpuBuffer(id: BufferId): ArrayBuffer {
    const entry = this._getEntry(id);
    if (!entry.cpuBuffer) {
      entry.cpuBuffer = new ArrayBuffer(entry.layout.byteLength);
    }
    return entry.cpuBuffer;
  }

  /**
   * Read GPU buffer contents back to CPU.
   */
  async gpuToCpu(id: BufferId): Promise<ArrayBuffer> {
    this._assertAlive();
    const entry = this._getEntry(id);
    const gpuBuf = entry.gpuBuffer;
    if (!gpuBuf) {
      throw new BufferError(`Buffer ${id} has no GPU allocation.`);
    }

    // Create a staging buffer for readback
    const staging = this._device.createBuffer({
      size: entry.layout.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: `staging_${id}`,
    });

    const encoder = this._device.createCommandEncoder();
    encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, entry.layout.byteLength);
    this._device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    entry.cpuBuffer = result;
    return result;
  }

  /**
   * Upload CPU data to a GPU buffer.
   */
  cpuToGpu(data: ArrayBuffer, layout: BufferLayout): BufferId {
    this._assertAlive();
    const id = this.acquire(layout, 'upload');
    const gpuBuf = this.getGpuBuffer(id);
    this._device.queue.writeBuffer(gpuBuf, 0, data);
    const entry = this._getEntry(id);
    entry.cpuBuffer = data;
    return id;
  }

  /**
   * Set up a double-buffer slot for ping-pong rendering.
   */
  createDoubleBuffer(slotName: string, layout: BufferLayout): void {
    this._assertAlive();
    const front = this.acquire(layout, `${slotName}_front`);
    const back = this.acquire(layout, `${slotName}_back`);
    this._doubleBuffers.set(slotName, { front, back });
  }

  swap(slotName: string): void {
    const slot = this._doubleBuffers.get(slotName);
    if (!slot) {
      throw new BufferError(`Double buffer slot "${slotName}" not found.`);
    }
    const tmp = slot.front;
    slot.front = slot.back;
    slot.back = tmp;
  }

  getFront(slotName: string): BufferId {
    const slot = this._doubleBuffers.get(slotName);
    if (!slot) {
      throw new BufferError(`Double buffer slot "${slotName}" not found.`);
    }
    return slot.front;
  }

  getBack(slotName: string): BufferId {
    const slot = this._doubleBuffers.get(slotName);
    if (!slot) {
      throw new BufferError(`Double buffer slot "${slotName}" not found.`);
    }
    return slot.back;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;

    // Destroy all active buffers
    for (const entry of this._buffers.values()) {
      entry.gpuBuffer?.destroy();
    }
    this._buffers.clear();

    // Destroy all pooled buffers
    for (const entry of this._pool) {
      entry.gpuBuffer?.destroy();
    }
    this._pool.length = 0;

    this._doubleBuffers.clear();
    this._allocatedBytes = 0;
    this._pooledBytes = 0;
  }

  private _getEntry(id: BufferId): BufferEntry {
    const entry = this._buffers.get(id);
    if (!entry) {
      throw new BufferError(`Buffer ${id} not found. It may have been released or disposed.`);
    }
    return entry;
  }

  private _assertAlive(): void {
    if (this._disposed) {
      throw new DisposedError('BufferManager');
    }
  }
}

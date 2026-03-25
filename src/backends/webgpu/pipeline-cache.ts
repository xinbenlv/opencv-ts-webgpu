import type { KernelDescriptor } from '../interface.ts';

/**
 * LRU cache for compiled GPUComputePipelines.
 *
 * Pipeline compilation is expensive — caching compiled pipelines across frames
 * is essential for real-time performance.
 */
export class PipelineCache {
  private readonly _cache = new Map<string, GPUComputePipeline>();
  private readonly _accessOrder: string[] = [];
  private readonly _maxSize: number;

  constructor(maxSize = 128) {
    this._maxSize = maxSize;
  }

  /**
   * Get or create a compiled compute pipeline for the given kernel.
   */
  async getOrCreate(
    device: GPUDevice,
    descriptor: KernelDescriptor,
  ): Promise<GPUComputePipeline> {
    const key = `${descriptor.name}:${descriptor.entryPoint}`;

    const cached = this._cache.get(key);
    if (cached) {
      // Move to end of access order (most recently used)
      const idx = this._accessOrder.indexOf(key);
      if (idx !== -1) this._accessOrder.splice(idx, 1);
      this._accessOrder.push(key);
      return cached;
    }

    // Compile new pipeline
    const shaderModule = device.createShaderModule({
      code: descriptor.wgslSource,
      label: descriptor.name,
    });

    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: descriptor.entryPoint,
      },
      label: descriptor.name,
    });

    // Evict LRU if at capacity
    if (this._cache.size >= this._maxSize) {
      const evictKey = this._accessOrder.shift();
      if (evictKey) this._cache.delete(evictKey);
    }

    this._cache.set(key, pipeline);
    this._accessOrder.push(key);
    return pipeline;
  }

  get size(): number {
    return this._cache.size;
  }

  clear(): void {
    this._cache.clear();
    this._accessOrder.length = 0;
  }
}

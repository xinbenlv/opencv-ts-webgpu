import type { KernelDescriptor } from '../interface.ts';
import { PipelineCache } from './pipeline-cache.ts';

/**
 * Dispatches WGSL compute shaders on the GPU.
 *
 * Usage:
 *   const runner = new KernelRunner(device);
 *   runner.dispatch(encoder, descriptor, buffers, inputShape);
 */
export class KernelRunner {
  private readonly _pipelineCache: PipelineCache;
  private readonly _device: GPUDevice;

  constructor(device: GPUDevice, pipelineCache?: PipelineCache) {
    this._device = device;
    this._pipelineCache = pipelineCache ?? new PipelineCache();
  }

  /**
   * Compile (if needed) and dispatch a kernel.
   */
  async dispatch(
    encoder: GPUCommandEncoder,
    descriptor: KernelDescriptor,
    buffers: GPUBuffer[],
    inputShape: readonly number[],
    uniformData?: ArrayBuffer,
  ): Promise<void> {
    const pipeline = await this._pipelineCache.getOrCreate(this._device, descriptor);

    // Build bind group entries
    const entries: GPUBindGroupEntry[] = descriptor.bindings.map((binding, i) => ({
      binding: binding.index,
      resource: { buffer: buffers[i]! },
    }));

    // Add uniform buffer if provided
    if (uniformData && descriptor.uniforms) {
      const uniformBuffer = this._device.createBuffer({
        size: descriptor.uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `${descriptor.name}_uniforms`,
      });
      this._device.queue.writeBuffer(uniformBuffer, 0, uniformData);
      entries.push({
        binding: descriptor.bindings.length,
        resource: { buffer: uniformBuffer },
      });
    }

    const bindGroup = this._device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
      label: `${descriptor.name}_bindgroup`,
    });

    const [dx, dy, dz] = descriptor.dispatchSize(inputShape);
    const pass = encoder.beginComputePass({ label: descriptor.name });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dx!, dy!, dz!);
    pass.end();
  }

  get cacheSize(): number {
    return this._pipelineCache.size;
  }

  clearCache(): void {
    this._pipelineCache.clear();
  }
}

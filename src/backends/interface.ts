import type { BufferLayout, DType } from '../core/types.ts';

/**
 * Describes the uniform buffer structure for a kernel.
 */
export interface UniformField {
  readonly offset: number;
  readonly type: DType;
}

export interface UniformDescriptor {
  readonly byteLength: number;
  readonly fields: ReadonlyMap<string, UniformField>;
}

/**
 * Describes a single buffer binding within a kernel's bind group.
 */
export interface KernelBinding {
  readonly index: number; // @binding(index)
  readonly type: 'read-only-storage' | 'storage' | 'uniform';
  readonly layout: BufferLayout;
}

/**
 * KernelDescriptor — the registration record for a WGSL compute kernel.
 *
 * Contributors create one of these per .wgsl file. The kernel registry
 * auto-discovers all descriptors at build time.
 *
 * Example:
 *   import wgslSource from './gaussian-blur.wgsl?raw';
 *   export const gaussianBlurKernel: KernelDescriptor = { ... };
 */
export interface KernelDescriptor {
  readonly name: string;
  readonly wgslSource: string;
  readonly entryPoint: string;

  readonly bindings: readonly KernelBinding[];

  readonly workgroupSize: readonly [number, number, number];
  dispatchSize(inputShape: readonly number[]): readonly [number, number, number];

  readonly uniforms?: UniformDescriptor;
}

/**
 * Backend-agnostic kernel execution interface.
 * Wraps a KernelDescriptor with compiled pipeline state.
 */
export interface BackendKernel {
  readonly descriptor: KernelDescriptor;
  compile(device: GPUDevice): Promise<GPUComputePipeline>;
  dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    dispatchDims: readonly [number, number, number],
  ): void;
}

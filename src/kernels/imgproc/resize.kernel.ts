import type { KernelDescriptor } from '../../backends/interface.ts';
import { computeBufferLayout } from '../../core/types.ts';
import type { Shape2D } from '../../core/types.ts';
import { KernelRegistry } from '../registry.ts';

import wgslSource from './resize.wgsl?raw';

export const resizeKernel: KernelDescriptor = {
  name: 'resize',
  wgslSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
    { index: 1, type: 'storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
  ],

  workgroupSize: [16, 16, 1],

  // inputShape: [srcH, srcW, dstH, dstW]
  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const dstW = inputShape[3] ?? inputShape[1]!;
    const dstH = inputShape[2] ?? inputShape[0]!;
    return [Math.ceil(dstW / 16), Math.ceil(dstH / 16), 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['src_width', { offset: 0, type: 'u32' }],
      ['src_height', { offset: 4, type: 'u32' }],
      ['dst_width', { offset: 8, type: 'u32' }],
      ['dst_height', { offset: 12, type: 'u32' }],
    ]),
  },
};

KernelRegistry.register(resizeKernel);

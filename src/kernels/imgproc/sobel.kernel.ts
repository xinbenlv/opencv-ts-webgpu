import type { KernelDescriptor } from '../../backends/interface.ts';
import { computeBufferLayout } from '../../core/types.ts';
import type { Shape2D } from '../../core/types.ts';
import { KernelRegistry } from '../registry.ts';

import wgslSource from './sobel.wgsl?raw';

export const sobelKernel: KernelDescriptor = {
  name: 'sobel',
  wgslSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
    { index: 1, type: 'storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
  ],

  workgroupSize: [16, 16, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const [h, w] = inputShape;
    return [Math.ceil(w! / 16), Math.ceil(h! / 16), 1];
  },

  uniforms: {
    byteLength: 8,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
    ]),
  },
};

KernelRegistry.register(sobelKernel);

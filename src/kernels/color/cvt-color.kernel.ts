import type { KernelDescriptor } from '../../backends/interface.ts';
import { computeBufferLayout } from '../../core/types.ts';
import type { Shape2D } from '../../core/types.ts';
import { KernelRegistry } from '../registry.ts';

import wgslSource from './cvt-color.wgsl?raw';

export const cvtColorKernel: KernelDescriptor = {
  name: 'cvtColor',
  wgslSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
    { index: 1, type: 'storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const [h, w] = inputShape;
    const totalPixels = h! * w!;
    return [Math.ceil(totalPixels / 256), 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
      ['mode', { offset: 8, type: 'u32' }],
    ]),
  },
};

KernelRegistry.register(cvtColorKernel);

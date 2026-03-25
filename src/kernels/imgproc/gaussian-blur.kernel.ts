import type { KernelDescriptor } from '../../backends/interface.ts';
import { computeBufferLayout } from '../../core/types.ts';
import type { Shape2D } from '../../core/types.ts';
import { KernelRegistry } from '../registry.ts';

// Import WGSL source as raw string via Vite
import wgslSource from './gaussian-blur.wgsl?raw';

export const gaussianBlurKernel: KernelDescriptor = {
  name: 'gaussianBlur',
  wgslSource,
  entryPoint: 'main',

  bindings: [
    {
      index: 0,
      type: 'read-only-storage',
      layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32'), // placeholder
    },
    {
      index: 1,
      type: 'storage',
      layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32'), // placeholder
    },
  ],

  workgroupSize: [16, 16, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const [h, w] = inputShape;
    return [
      Math.ceil(w! / 16),
      Math.ceil(h! / 16),
      1,
    ];
  },

  uniforms: {
    byteLength: 16, // width(4) + height(4) + sigma(4) + pad(4)
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
      ['sigma', { offset: 8, type: 'f32' }],
    ]),
  },
};

// Self-register
KernelRegistry.register(gaussianBlurKernel);

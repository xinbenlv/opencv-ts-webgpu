import type { KernelDescriptor } from '../../backends/interface.ts';
import { computeBufferLayout } from '../../core/types.ts';
import type { Shape2D } from '../../core/types.ts';
import { KernelRegistry } from '../registry.ts';

import preprocessSource from './midas-preprocess.wgsl?raw';
import postprocessSource from './midas-postprocess.wgsl?raw';

export const midasPreprocessKernel: KernelDescriptor = {
  name: 'midasPreprocess',
  wgslSource: preprocessSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
    { index: 1, type: 'storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const [h, w] = inputShape;
    return [Math.ceil((h! * w!) / 256), 1, 1];
  },

  uniforms: {
    byteLength: 8,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
    ]),
  },
};

export const midasPostprocessKernel: KernelDescriptor = {
  name: 'midasPostprocess',
  wgslSource: postprocessSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
    { index: 1, type: 'storage', layout: computeBufferLayout([1, 1] as unknown as Shape2D, 'f32') },
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const [h, w] = inputShape;
    return [Math.ceil((h! * w!) / 256), 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
      ['scale', { offset: 8, type: 'f32' }],
      ['shift', { offset: 12, type: 'f32' }],
    ]),
  },
};

KernelRegistry.register(midasPreprocessKernel);
KernelRegistry.register(midasPostprocessKernel);

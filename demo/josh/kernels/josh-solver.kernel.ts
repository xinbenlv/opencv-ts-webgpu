import type { KernelDescriptor } from '../../../src/backends/interface.ts';
import { computeBufferLayout } from '../../../src/core/types.ts';
import type { Shape2D } from '../../../src/core/types.ts';
import { SMPL_VERTEX_COUNT } from '../models/smpl.ts';
import { SMPL_CONTACT_VERTICES } from '../models/smpl.ts';

import contactLossSource from './josh-contact-loss.wgsl?raw';
import depthReprojSource from './josh-depth-reproj.wgsl?raw';
import temporalSource from './josh-temporal.wgsl?raw';

const dummyLayout = computeBufferLayout([1, 1] as unknown as Shape2D, 'f32');

const NUM_CONTACT_VERTICES =
  SMPL_CONTACT_VERTICES.leftFootSole.length +
  SMPL_CONTACT_VERTICES.rightFootSole.length +
  SMPL_CONTACT_VERTICES.leftToes.length +
  SMPL_CONTACT_VERTICES.rightToes.length;

export const contactLossKernel: KernelDescriptor = {
  name: 'joshContactLoss',
  wgslSource: contactLossSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // vertices
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // depth_map
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // contact_indices
    { index: 3, type: 'storage', layout: dummyLayout },            // gradient
    { index: 4, type: 'storage', layout: dummyLayout },            // loss
  ],

  workgroupSize: [64, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [Math.ceil(NUM_CONTACT_VERTICES / 64), 1, 1];
  },

  uniforms: {
    byteLength: 32,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
      ['num_contacts', { offset: 8, type: 'u32' }],
      ['fx', { offset: 12, type: 'f32' }],
      ['fy', { offset: 16, type: 'f32' }],
      ['cx', { offset: 20, type: 'f32' }],
      ['cy', { offset: 24, type: 'f32' }],
      ['contact_threshold', { offset: 28, type: 'f32' }],
    ]),
  },
};

export const depthReprojKernel: KernelDescriptor = {
  name: 'joshDepthReproj',
  wgslSource: depthReprojSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // vertices
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // depth_map
    { index: 2, type: 'storage', layout: dummyLayout },            // gradient
    { index: 3, type: 'storage', layout: dummyLayout },            // loss
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [Math.ceil(SMPL_VERTEX_COUNT / 256), 1, 1];
  },

  uniforms: {
    byteLength: 32,
    fields: new Map([
      ['width', { offset: 0, type: 'u32' }],
      ['height', { offset: 4, type: 'u32' }],
      ['vertex_count', { offset: 8, type: 'u32' }],
      ['fx', { offset: 12, type: 'f32' }],
      ['fy', { offset: 16, type: 'f32' }],
      ['cx', { offset: 20, type: 'f32' }],
      ['cy', { offset: 24, type: 'f32' }],
      ['weight', { offset: 28, type: 'f32' }],
    ]),
  },
};

export const temporalSmoothnessKernel: KernelDescriptor = {
  name: 'joshTemporal',
  wgslSource: temporalSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // current_params
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // prev_params
    { index: 2, type: 'storage', layout: dummyLayout },            // gradient
    { index: 3, type: 'storage', layout: dummyLayout },            // loss
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const paramDim = inputShape[0]!;
    return [Math.ceil(paramDim / 256), 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['param_dim', { offset: 0, type: 'u32' }],
      ['weight', { offset: 4, type: 'f32' }],
      ['_pad0', { offset: 8, type: 'u32' }],
      ['_pad1', { offset: 12, type: 'u32' }],
    ]),
  },
};

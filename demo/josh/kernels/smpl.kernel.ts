import type { KernelDescriptor } from '../../../src/backends/interface.ts';
import { computeBufferLayout } from '../../../src/core/types.ts';
import type { Shape2D } from '../../../src/core/types.ts';
import { SMPL_VERTEX_COUNT } from '../models/smpl.ts';

import smplForwardSource from './smpl-forward.wgsl?raw';
import smplJointsSource from './smpl-joints.wgsl?raw';

const dummyLayout = computeBufferLayout([1, 1] as unknown as Shape2D, 'f32');

/**
 * SMPL LBS kernel — transforms shaped vertices using joint transforms and skinning weights.
 * Runs one thread per vertex (6890 vertices).
 */
export const smplForwardKernel: KernelDescriptor = {
  name: 'smplForward',
  wgslSource: smplForwardSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // mean_template
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // shape_blend_shapes
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // skinning_weights
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // skinning_indices
    { index: 4, type: 'read-only-storage', layout: dummyLayout }, // joint_transforms
    { index: 5, type: 'read-only-storage', layout: dummyLayout }, // shape_params
    { index: 6, type: 'storage', layout: dummyLayout },            // output_vertices
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [Math.ceil(SMPL_VERTEX_COUNT / 256), 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['vertex_count', { offset: 0, type: 'u32' }],
      ['joint_count', { offset: 4, type: 'u32' }],
      ['shape_dim', { offset: 8, type: 'u32' }],
      ['_pad', { offset: 12, type: 'u32' }],
    ]),
  },
};

/**
 * SMPL joints kernel — joint regression + forward kinematics.
 * Runs as single workgroup (sequential FK through kinematic tree).
 */
export const smplJointsKernel: KernelDescriptor = {
  name: 'smplJoints',
  wgslSource: smplJointsSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // shaped_vertices
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // joint_regressor
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // local_rotations
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // parent_indices
    { index: 4, type: 'storage', layout: dummyLayout },            // joint_transforms
    { index: 5, type: 'storage', layout: dummyLayout },            // joint_positions
  ],

  workgroupSize: [1, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // Single workgroup — FK is sequential
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['vertex_count', { offset: 0, type: 'u32' }],
      ['joint_count', { offset: 4, type: 'u32' }],
      ['_pad0', { offset: 8, type: 'u32' }],
      ['_pad1', { offset: 12, type: 'u32' }],
    ]),
  },
};

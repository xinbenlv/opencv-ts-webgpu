import type { KernelDescriptor } from '../../../src/backends/interface.ts';
import { computeBufferLayout } from '../../../src/core/types.ts';
import type { Shape2D } from '../../../src/core/types.ts';
import { SMPL_VERTEX_COUNT } from '../models/smpl.ts';

import rodriguesSource from './rodrigues.wgsl?raw';
import rodriguesDerivSource from './rodrigues-deriv.wgsl?raw';
import jvpGradientSource from './jvp-gradient.wgsl?raw';
import jvpJointGradientSource from './jvp-joint-gradient.wgsl?raw';
import smplPriorSource from './josh-smpl-prior.wgsl?raw';
import contactStaticSource from './josh-contact-static.wgsl?raw';
import reproj2DSource from './josh-reproj-2d.wgsl?raw';
import incrementCounterSource from './increment-counter.wgsl?raw';
import adamOptimizerSource from './adam-optimizer.wgsl?raw';

const dummyLayout = computeBufferLayout([1, 1] as unknown as Shape2D, 'f32');

// ---------------------------------------------------------------------------
// rodriguesKernel
// Converts axis-angle pose parameters to local rotation matrices.
// One thread per joint (24 joints × 3 params = 72 pose dims).
// Bindings: [pose (read), local_rots (storage)]
// ---------------------------------------------------------------------------
export const rodriguesKernel: KernelDescriptor = {
  name: 'rodrigues',
  wgslSource: rodriguesSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // pose (72 f32)
    { index: 1, type: 'storage',           layout: dummyLayout }, // local_rots (24×9 f32)
  ],

  workgroupSize: [24, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // 24 threads in one workgroup
  },
};

// ---------------------------------------------------------------------------
// rodriguesDerivKernel
// Computes dR/dw Jacobians for every joint (needed for JVP backprop).
// One thread per joint: produces three 3×3 matrices (dR/dwx, dR/dwy, dR/dwz).
// Bindings: [pose (read), dR (storage — 24×3×9 f32)]
// ---------------------------------------------------------------------------
export const rodriguesDerivKernel: KernelDescriptor = {
  name: 'rodriguesDeriv',
  wgslSource: rodriguesDerivSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // pose (72 f32)
    { index: 1, type: 'storage',           layout: dummyLayout }, // dR (24×27 f32)
  ],

  workgroupSize: [64, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // 24 joints fit in a single 64-thread workgroup
  },
};

// ---------------------------------------------------------------------------
// jvpGradientKernel
// JVP (Jacobian-vector product) for vertex-level losses:
//   grad_pose += dR/dw^T · (dl/dv · dv/dR)
// One thread per vertex.
// ---------------------------------------------------------------------------
export const jvpGradientKernel: KernelDescriptor = {
  name: 'jvpGradient',
  wgslSource: jvpGradientSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // dl_dv           (V×3 f32)
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // dR               (24×27 f32)
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // joint_transforms (24×16 f32)
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // shaped_vertices  (V×3 f32)
    { index: 4, type: 'read-only-storage', layout: dummyLayout }, // skin_weights     (V×24 f32)
    { index: 5, type: 'read-only-storage', layout: dummyLayout }, // skin_indices     (V×24 u32)
    { index: 6, type: 'read-only-storage', layout: dummyLayout }, // parent_indices   (24 i32)
    { index: 7, type: 'storage',           layout: dummyLayout }, // gradient         (72 f32, atomic add)
  ],

  workgroupSize: [64, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [Math.ceil(SMPL_VERTEX_COUNT / 64), 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['vertex_count', { offset: 0,  type: 'u32' }],
      ['joint_count',  { offset: 4,  type: 'u32' }],
      ['_pad0',        { offset: 8,  type: 'u32' }],
      ['_pad1',        { offset: 12, type: 'u32' }],
    ]),
  },
};

// ---------------------------------------------------------------------------
// jvpJointGradientKernel
// JVP for joint-level losses (2D reprojection, contact):
//   grad_pose += dR/dw^T · (dl/djoint · djoint/dR)
// One thread per joint.
// ---------------------------------------------------------------------------
export const jvpJointGradientKernel: KernelDescriptor = {
  name: 'jvpJointGradient',
  wgslSource: jvpJointGradientSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // dl_djoint       (24×3 f32)
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // dR               (24×27 f32)
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // joint_transforms (24×16 f32)
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // parent_indices   (24 i32)
    { index: 4, type: 'storage',           layout: dummyLayout }, // gradient         (72 f32, atomic add)
  ],

  workgroupSize: [64, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // 24 joints fit in a single 64-thread workgroup
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['joint_count', { offset: 0,  type: 'u32' }],
      ['_pad0',       { offset: 4,  type: 'u32' }],
      ['_pad1',       { offset: 8,  type: 'u32' }],
      ['_pad2',       { offset: 12, type: 'u32' }],
    ]),
  },
};

// ---------------------------------------------------------------------------
// smplPriorKernel
// L2 regularisation on pose and shape parameters relative to their initial
// values.  Accumulates scalar loss and writes pose/shape gradients.
// ---------------------------------------------------------------------------
export const smplPriorKernel: KernelDescriptor = {
  name: 'smplPrior',
  wgslSource: smplPriorSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // pose       (72 f32)
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // init_pose  (72 f32)
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // shape      (10 f32)
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // init_shape (10 f32)
    { index: 4, type: 'storage',           layout: dummyLayout }, // gradient   (82 f32)
    { index: 5, type: 'storage',           layout: dummyLayout }, // loss_out   (1 f32)
  ],

  workgroupSize: [82, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // 72 pose + 10 shape = 82 params in one workgroup
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['pose_dim',       { offset: 0,  type: 'u32' }],
      ['shape_dim',      { offset: 4,  type: 'u32' }],
      ['pose_weight',    { offset: 8,  type: 'f32' }],
      ['shape_weight',   { offset: 12, type: 'f32' }],
    ]),
  },
};

// ---------------------------------------------------------------------------
// contactStaticKernel
// Static contact constraint: penalises vertices that move when they should be
// planted (foot contacts).  One thread per contact vertex.
// ---------------------------------------------------------------------------
export const contactStaticKernel: KernelDescriptor = {
  name: 'contactStatic',
  wgslSource: contactStaticSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // vertices_curr   (V×3 f32)
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // vertices_prev   (V×3 f32)
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // contact_indices (C u32)
    { index: 3, type: 'storage',           layout: dummyLayout }, // dl_dv           (V×3 f32)
    { index: 4, type: 'storage',           layout: dummyLayout }, // loss_out        (1 f32)
  ],

  workgroupSize: [64, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    // Dispatched at runtime based on actual contact count — caller overrides
    return [1, 1, 1];
  },

  uniforms: {
    byteLength: 16,
    fields: new Map([
      ['num_contacts', { offset: 0,  type: 'u32' }],
      ['vertex_count', { offset: 4,  type: 'u32' }],
      ['weight',       { offset: 8,  type: 'f32' }],
      ['_pad0',        { offset: 12, type: 'u32' }],
    ]),
  },
};

// ---------------------------------------------------------------------------
// reproj2DKernel
// 2-D keypoint reprojection loss + gradient w.r.t. joint positions.
// One thread per keypoint (up to 33 MediaPipe landmarks or 17 COCO joints).
// ---------------------------------------------------------------------------
export const reproj2DKernel: KernelDescriptor = {
  name: 'reproj2D',
  wgslSource: reproj2DSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'read-only-storage', layout: dummyLayout }, // joint_positions (J×3 f32)
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // keypoints_2d    (K×2 f32)
    { index: 2, type: 'read-only-storage', layout: dummyLayout }, // confidences     (K f32)
    { index: 3, type: 'read-only-storage', layout: dummyLayout }, // joint_to_smpl   (K u32)
    { index: 4, type: 'storage',           layout: dummyLayout }, // dl_djoint       (J×3 f32)
    { index: 5, type: 'storage',           layout: dummyLayout }, // loss_out        (1 f32)
  ],

  workgroupSize: [32, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1]; // ≤33 keypoints in one 32-thread workgroup (caller can override)
  },

  uniforms: {
    byteLength: 32,
    fields: new Map([
      ['num_keypoints', { offset: 0,  type: 'u32' }],
      ['joint_count',   { offset: 4,  type: 'u32' }],
      ['fx',            { offset: 8,  type: 'f32' }],
      ['fy',            { offset: 12, type: 'f32' }],
      ['cx',            { offset: 16, type: 'f32' }],
      ['cy',            { offset: 20, type: 'f32' }],
      ['weight',        { offset: 24, type: 'f32' }],
      ['_pad0',         { offset: 28, type: 'u32' }],
    ]),
  },
};

// ---------------------------------------------------------------------------
// incrementCounterKernel
// Atomically increments a u32 counter in a storage buffer.
// Used to track Adam optimiser step count across dispatches.
// Bindings: [counter (storage — 1 u32)]
// ---------------------------------------------------------------------------
export const incrementCounterKernel: KernelDescriptor = {
  name: 'incrementCounter',
  wgslSource: incrementCounterSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'storage', layout: dummyLayout }, // counter (1 u32)
  ],

  workgroupSize: [1, 1, 1],

  dispatchSize(_inputShape: readonly number[]): readonly [number, number, number] {
    return [1, 1, 1];
  },
};

// ---------------------------------------------------------------------------
// adamOptimizerKernel
// Per-element Adam parameter update.
//   θ ← θ − lr · m̂ / (√v̂ + ε)
// One thread per parameter.  Timestep is read from a storage buffer so it
// can be incremented on the GPU (via incrementCounterKernel) without a CPU
// round-trip.
// ---------------------------------------------------------------------------
export const adamOptimizerKernel: KernelDescriptor = {
  name: 'adamOptimizer',
  wgslSource: adamOptimizerSource,
  entryPoint: 'main',

  bindings: [
    { index: 0, type: 'storage',           layout: dummyLayout }, // params    (N f32) — updated in-place
    { index: 1, type: 'read-only-storage', layout: dummyLayout }, // gradients (N f32)
    { index: 2, type: 'storage',           layout: dummyLayout }, // m         (N f32) — 1st moment
    { index: 3, type: 'storage',           layout: dummyLayout }, // v         (N f32) — 2nd moment
    { index: 4, type: 'read-only-storage', layout: dummyLayout }, // timestep  (1 u32)
  ],

  workgroupSize: [256, 1, 1],

  dispatchSize(inputShape: readonly number[]): readonly [number, number, number] {
    const paramDim = inputShape[0] ?? 1;
    return [Math.ceil(paramDim / 256), 1, 1];
  },

  uniforms: {
    byteLength: 32,
    fields: new Map([
      ['param_dim', { offset: 0,  type: 'u32' }],
      ['lr',        { offset: 4,  type: 'f32' }],
      ['beta1',     { offset: 8,  type: 'f32' }],
      ['beta2',     { offset: 12, type: 'f32' }],
      ['epsilon',   { offset: 16, type: 'f32' }],
      ['_pad0',     { offset: 20, type: 'u32' }],
      ['_pad1',     { offset: 24, type: 'u32' }],
      ['_pad2',     { offset: 28, type: 'u32' }],
    ]),
  },
};

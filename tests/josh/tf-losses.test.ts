/**
 * Tests for the tf.js JOSH loss functions (losses-ref.ts).
 *
 * Uses synthetic tensor inputs — no real SMPL model required.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import { computeLosses } from '../../demo/josh/tf/losses-ref.ts';
import { createSMPLParams } from '../../demo/josh/tf/smpl-optimizer.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Make synthetic [6890, 3] vertices arranged in a grid */
function makeSyntheticVertices(yOffset = 0): tf.Tensor2D {
  const data = new Float32Array(6890 * 3);
  for (let i = 0; i < 6890; i++) {
    data[i * 3]     = (i % 100) * 0.01;
    data[i * 3 + 1] = Math.floor(i / 100) * 0.01 + yOffset;
    data[i * 3 + 2] = 0;
  }
  return tf.tensor2d(data, [6890, 3], 'float32');
}

/** Make synthetic [24, 3] joints */
function makeSyntheticJoints(yOffset = 0): tf.Tensor2D {
  const data = new Float32Array(24 * 3);
  for (let j = 0; j < 24; j++) {
    data[j * 3]     = (j - 12) * 0.1;
    data[j * 3 + 1] = j * 0.07 + yOffset;
    data[j * 3 + 2] = 2.0;  // z=2m in front of camera
  }
  return tf.tensor2d(data, [24, 3], 'float32');
}

const DEFAULT_WEIGHTS = { w3D: 1, w2D: 1, wc1: 1, wc2: 20, wp: 10, ws: 1 };
const FL = 500, CX = 192, CY = 192;

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

// ---------------------------------------------------------------------------
// L_p tests
// ---------------------------------------------------------------------------

describe('L_p — SMPL prior loss', () => {
  it('pose offset = 0.1 on all 72 dims: expected loss = wp * 72 * 0.01', async () => {
    const priorPose  = new Float32Array(72).fill(0);
    const priorBetas = new Float32Array(10).fill(0);

    // Current pose is 0.1 above prior on all dims
    const params = createSMPLParams({
      pose: new Float32Array(72).fill(0.1),
      betas: new Float32Array(10).fill(0),
    });

    const vertices = makeSyntheticVertices();
    const joints   = makeSyntheticJoints();
    const kp2d = new Float32Array(24 * 3);
    // Set conf=1, u/v match projection so L_2D is 0 for isolation
    for (let j = 0; j < 24; j++) kp2d[j * 3 + 2] = 0; // zero confidence

    const result = computeLosses({
      vertices, joints, pts3d: null, keypoints2d: kp2d, depthMap: null,
      prevVertices: null, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wc2: 0, ws: 0 },
      priorPose, priorBetas, deltaC1: 0, deltaC2: 0.1,
    }, params);

    const lpVal = (await result.lp.data())[0]!;
    // Expected: sum over 72 of (0.1 - 0)^2 = 72 * 0.01 = 0.72
    // Note: computeLp does NOT multiply by wp internally — wp is applied in total.
    // The raw lp = 72 * 0.01 = 0.72; total = wp * lp = 10 * 0.72 = 7.2
    expect(lpVal).toBeCloseTo(0.72, 3);

    const totalVal = (await result.total.data())[0]!;
    expect(totalVal).toBeCloseTo(7.2, 2);

    vertices.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });

  it('pose = prior exactly: L_p should be 0', async () => {
    const priorPose  = new Float32Array(72).fill(0.5);
    const priorBetas = new Float32Array(10).fill(0);

    const params = createSMPLParams({
      pose: new Float32Array(72).fill(0.5),
      betas: new Float32Array(10).fill(0),
    });

    const vertices = makeSyntheticVertices();
    const joints   = makeSyntheticJoints();

    const result = computeLosses({
      vertices, joints, pts3d: null,
      keypoints2d: new Float32Array(24 * 3), depthMap: null,
      prevVertices: null, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wc2: 0, ws: 0 },
      priorPose, priorBetas, deltaC1: 0, deltaC2: 0.1,
    }, params);

    const lpVal = (await result.lp.data())[0]!;
    expect(lpVal).toBeCloseTo(0, 5);

    vertices.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });
});

// ---------------------------------------------------------------------------
// L_2D tests
// ---------------------------------------------------------------------------

describe('L_2D — 2D keypoint reprojection loss', () => {
  it('joint projected exactly onto keypoint: L_2D = 0', async () => {
    const joints = makeSyntheticJoints();
    const jointsData = joints.arraySync() as number[][];

    // Build keypoints that exactly match the projections of current joints
    const kp2d = new Float32Array(24 * 3);
    for (let j = 0; j < 24; j++) {
      const jx = jointsData[j]![0]!;
      const jy = jointsData[j]![1]!;
      const jz = jointsData[j]![2]! + 1e-6;
      const u = FL * jx / jz + CX;
      const v = FL * jy / jz + CY;
      kp2d[j * 3]     = u;
      kp2d[j * 3 + 1] = v;
      kp2d[j * 3 + 2] = 1.0;  // full confidence
    }

    const params = createSMPLParams();
    const vertices = makeSyntheticVertices();

    const result = computeLosses({
      vertices, joints, pts3d: null, keypoints2d: kp2d, depthMap: null,
      prevVertices: null, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, wc1: 0, wc2: 0, wp: 0, ws: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2: 0.1,
    }, params);

    const l2dVal = (await result.l2d.data())[0]!;
    expect(l2dVal).toBeCloseTo(0, 3);

    vertices.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });

  it('zero confidence keypoints: L_2D = 0 regardless of position', async () => {
    const joints   = makeSyntheticJoints();
    const vertices = makeSyntheticVertices();

    // All keypoints at origin but zero confidence
    const kp2d = new Float32Array(24 * 3); // all zeros including conf

    const params = createSMPLParams();

    const result = computeLosses({
      vertices, joints, pts3d: null, keypoints2d: kp2d, depthMap: null,
      prevVertices: null, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, wc1: 0, wc2: 0, wp: 0, ws: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2: 0.1,
    }, params);

    const l2dVal = (await result.l2d.data())[0]!;
    expect(l2dVal).toBeCloseTo(0, 5);

    vertices.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });
});

// ---------------------------------------------------------------------------
// L_smooth tests
// ---------------------------------------------------------------------------

describe('L_s — temporal smoothness loss', () => {
  it('identical current and previous joints: L_s = 0', async () => {
    const joints   = makeSyntheticJoints(0);
    // prevVertices shape [6890,3]; first 24 rows used as prev joints
    const prevVerts = makeSyntheticVertices(0);

    // Joints match rows 0..23 of prevVerts
    // Construct joints identical to first 24 rows of prevVerts
    const prevData = prevVerts.arraySync() as number[][];
    const syncedJoints = tf.tensor2d(
      prevData.slice(0, 24) as number[][],
      [24, 3],
      'float32',
    );

    const vertices = makeSyntheticVertices();
    const params   = createSMPLParams();

    const result = computeLosses({
      vertices, joints: syncedJoints, pts3d: null,
      keypoints2d: new Float32Array(24 * 3), depthMap: null,
      prevVertices: prevVerts, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wc2: 0, wp: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2: 0.1,
    }, params);

    const lsVal = (await result.ls.data())[0]!;
    expect(lsVal).toBeCloseTo(0, 3);

    vertices.dispose(); joints.dispose(); prevVerts.dispose(); syncedJoints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });

  it('no prevVertices: L_s = 0', async () => {
    const joints   = makeSyntheticJoints();
    const vertices = makeSyntheticVertices();
    const params   = createSMPLParams();

    const result = computeLosses({
      vertices, joints, pts3d: null,
      keypoints2d: new Float32Array(24 * 3), depthMap: null,
      prevVertices: null, contactMask: new Uint8Array(6890),
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wc2: 0, wp: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2: 0.1,
    }, params);

    const lsVal = (await result.ls.data())[0]!;
    expect(lsVal).toBeCloseTo(0, 5);

    vertices.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });
});

// ---------------------------------------------------------------------------
// L_c2 — contact static loss
// ---------------------------------------------------------------------------

describe('L_c2 — contact static constraint', () => {
  it('contact vertex moved exactly deltaC2 meters: L_c2 should be 0 (at hinge boundary)', async () => {
    const deltaC2 = 0.1;

    // Two identical vertex sets, then shift one contact vertex by exactly deltaC2
    const vCurr = makeSyntheticVertices(0);
    const vPrev = makeSyntheticVertices(0);

    // Vertex 0 is the contact vertex; shift it by exactly deltaC2 in x
    const vCurrData = vCurr.arraySync() as number[][];
    vCurrData[0]![0] = vCurrData[0]![0]! + deltaC2;
    const vCurrMod = tf.tensor2d(vCurrData as number[][], [6890, 3], 'float32');

    // Contact mask: only vertex 0
    const contactMask = new Uint8Array(6890);
    contactMask[0] = 1;

    const joints = makeSyntheticJoints();
    const params = createSMPLParams();

    const result = computeLosses({
      vertices: vCurrMod, joints, pts3d: null,
      keypoints2d: new Float32Array(24 * 3), depthMap: null,
      prevVertices: vPrev, contactMask,
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wp: 0, ws: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2,
    }, params);

    const lc2Val = (await result.lc2.data())[0]!;
    // dist² = deltaC2² = 0.01, hinge = max(0, 0.01 - 0.01) = 0, so L_c2 = 0
    expect(lc2Val).toBeCloseTo(0, 4);

    vCurr.dispose(); vPrev.dispose(); vCurrMod.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });

  it('contact vertex moved beyond deltaC2: L_c2 > 0', async () => {
    const deltaC2 = 0.1;
    const displacement = 0.2; // 20cm > deltaC2

    const vPrev = makeSyntheticVertices(0);
    const vCurrData = vPrev.arraySync() as number[][];
    vCurrData[0]![0] = vCurrData[0]![0]! + displacement;
    const vCurrMod = tf.tensor2d(vCurrData as number[][], [6890, 3], 'float32');

    const contactMask = new Uint8Array(6890);
    contactMask[0] = 1;

    const joints = makeSyntheticJoints();
    const params = createSMPLParams();

    const result = computeLosses({
      vertices: vCurrMod, joints, pts3d: null,
      keypoints2d: new Float32Array(24 * 3), depthMap: null,
      prevVertices: vPrev, contactMask,
      focalLength: FL, cx: CX, cy: CY, imgW: 384, imgH: 384,
      weights: { ...DEFAULT_WEIGHTS, w3D: 0, w2D: 0, wc1: 0, wp: 0, ws: 0 },
      priorPose: new Float32Array(72), priorBetas: new Float32Array(10),
      deltaC1: 0, deltaC2,
    }, params);

    const lc2Val = (await result.lc2.data())[0]!;
    // dist² = 0.04, threshold = 0.01, hinge (raw, pre-wc2) = 0.03
    expect(lc2Val).toBeGreaterThan(0);
    expect(lc2Val).toBeCloseTo(0.03, 4);

    vPrev.dispose(); vCurrMod.dispose(); joints.dispose();
    result.total.dispose(); result.lp.dispose();
    result.l3d.dispose(); result.l2d.dispose();
    result.lc1.dispose(); result.lc2.dispose(); result.ls.dispose();
    params.pose.dispose(); params.betas.dispose();
    params.transl.dispose(); params.scale.dispose();
  });
});

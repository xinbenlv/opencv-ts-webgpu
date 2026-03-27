/**
 * JOSH optimizer — two-stage Adam loop using TensorFlow.js.
 *
 * Stage 1: 500 iterations, lr=0.07, w2D=0  (3-D + contact losses only)
 * Stage 2: 200 iterations, lr=0.014, w2D=1 (all losses)
 *
 * Uses tf.variableGrads to differentiate through the full SMPL forward pass.
 */

import * as tf from '@tensorflow/tfjs';
import type { SMPLModelData } from '../models/smpl-loader-ui.ts';
import { createSMPLParams, smplForward } from './smpl-optimizer.ts';
import { computeLosses } from './losses-ref.ts';
import { JOSH_CONFIG } from '../config.ts';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface JOSHOptimizerOptions {
  smplModel: SMPLModelData;
  initPose?: Float32Array;
  initBetas?: Float32Array;
  pts3d: Float32Array | null;
  /** [24*3] = [u,v,conf] per SMPL joint */
  keypoints2d: Float32Array;
  depthMap: Float32Array | null;
  prevVertices?: Float32Array;
  /** [6890] bitmask: 1 = contact vertex */
  contactMask: Uint8Array;
  focalLength: number;
  cx: number;
  cy: number;
  imgW: number;
  imgH: number;
  onIter?: (iter: number, loss: number) => void;
}

export interface JOSHOptimizerResult {
  pose: Float32Array;
  betas: Float32Array;
  transl: Float32Array;
  /** [6890*3] row-major */
  vertices: Float32Array;
  /** [24*3] row-major */
  joints: Float32Array;
  finalLoss: number;
}

// ---------------------------------------------------------------------------
// runJOSHOptimizer
// ---------------------------------------------------------------------------

export async function runJOSHOptimizer(
  opts: JOSHOptimizerOptions,
): Promise<JOSHOptimizerResult> {
  const {
    smplModel, initPose, initBetas, pts3d, keypoints2d, depthMap,
    prevVertices, contactMask, focalLength, cx, cy, imgW, imgH, onIter,
  } = opts;

  // Build tf.Variables for all parameters
  const params = createSMPLParams({
    ...(initPose !== undefined ? { pose: initPose } : {}),
    ...(initBetas !== undefined ? { betas: initBetas } : {}),
  });

  // Previous-frame vertices tensor (if provided)
  const prevVTensor: tf.Tensor2D | null = prevVertices != null
    ? tf.tensor2d(prevVertices, [6890, 3], 'float32')
    : null;

  // Prior = initial values (so L_p penalises deviation from ROMP init)
  const priorPose = initPose != null ? initPose.slice() : new Float32Array(72);
  const priorBetas = initBetas != null ? initBetas.slice() : new Float32Array(10);

  const contactMaskFixed = contactMask;

  let finalLoss = 0;

  // -------------------------------------------------------------------------
  // Helper: run one iteration inside tf.tidy, return scalar loss value
  // -------------------------------------------------------------------------
  function runStage(
    optimizer: tf.Optimizer,
    weights: { w3D: number; w2D: number; wc1: number; wc2: number; wp: number; ws: number },
    numIters: number,
    iterOffset: number,
  ): Promise<void> {
    return new Promise<void>((resolve) => {
      let i = 0;

      function step() {
        if (i >= numIters) {
          resolve();
          return;
        }

        let lossVal = 0;

        optimizer.minimize(() => {
          const { vertices, joints } = smplForward(params, smplModel);
          const outputs = computeLosses(
            {
              vertices,
              joints,
              pts3d,
              keypoints2d,
              depthMap,
              prevVertices: prevVTensor,
              contactMask: contactMaskFixed,
              focalLength,
              cx,
              cy,
              imgW,
              imgH,
              weights,
              priorPose,
              priorBetas,
              deltaC1: JOSH_CONFIG.deltaC1,
              deltaC2: JOSH_CONFIG.deltaC2,
            },
            params,
          );

          const loss = outputs.total;
          lossVal = loss.dataSync()[0] ?? 0;
          finalLoss = lossVal;
          return loss;
        }, true);

        const globalIter = iterOffset + i;
        if (globalIter % 10 === 0) {
          onIter?.(globalIter, lossVal);
        }

        i++;

        // Yield to event loop every 50 iters to avoid blocking UI
        if (i % 50 === 0) {
          setTimeout(step, 0);
        } else {
          step();
        }
      }

      step();
    });
  }

  // -------------------------------------------------------------------------
  // Stage 1: 500 iters, lr=0.07, w2D=0
  // -------------------------------------------------------------------------
  const stage1Weights = {
    w3D: JOSH_CONFIG.stage1.w3D,
    w2D: JOSH_CONFIG.stage1.w2D,  // 0
    wc1: JOSH_CONFIG.stage1.wc1,
    wc2: JOSH_CONFIG.stage1.wc2,
    wp:  JOSH_CONFIG.stage1.wp,
    ws:  JOSH_CONFIG.stage1.ws,
  };

  const opt1 = tf.train.adam(JOSH_CONFIG.stage1.lr);
  await runStage(opt1, stage1Weights, JOSH_CONFIG.stage1.iters, 0);
  opt1.dispose();

  // -------------------------------------------------------------------------
  // Stage 2: 200 iters, lr=0.014, all losses
  // -------------------------------------------------------------------------
  const stage2Weights = {
    w3D: JOSH_CONFIG.stage2.w3D,
    w2D: JOSH_CONFIG.stage2.w2D,  // 1
    wc1: JOSH_CONFIG.stage2.wc1,
    wc2: JOSH_CONFIG.stage2.wc2,
    wp:  JOSH_CONFIG.stage2.wp,
    ws:  JOSH_CONFIG.stage2.ws,
  };

  const opt2 = tf.train.adam(JOSH_CONFIG.stage2.lr);
  await runStage(opt2, stage2Weights, JOSH_CONFIG.stage2.iters, JOSH_CONFIG.stage1.iters);
  opt2.dispose();

  // -------------------------------------------------------------------------
  // Extract final results
  // -------------------------------------------------------------------------
  const { vertices: vTensor, joints: jTensor } = smplForward(params, smplModel);

  const poseOut   = (await (params.pose   as tf.Tensor1D).data()) as Float32Array;
  const betasOut  = (await (params.betas  as tf.Tensor1D).data()) as Float32Array;
  const translOut = (await (params.transl as tf.Tensor1D).data()) as Float32Array;
  const vertsOut  = (await vTensor.data()) as Float32Array;
  const jointsOut = (await jTensor.data()) as Float32Array;

  // Cleanup
  vTensor.dispose();
  jTensor.dispose();
  params.pose.dispose();
  params.betas.dispose();
  params.transl.dispose();
  params.scale.dispose();
  prevVTensor?.dispose();

  return {
    pose:      poseOut,
    betas:     betasOut,
    transl:    translOut,
    vertices:  vertsOut,
    joints:    jointsOut,
    finalLoss,
  };
}

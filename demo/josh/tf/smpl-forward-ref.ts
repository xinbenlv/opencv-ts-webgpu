/**
 * Reference SMPL forward pass implementation using TensorFlow.js.
 *
 * This is NOT used in the production pipeline (too slow for 700 iterations).
 * It exists solely for gradient validation: we compare tf.grad() output
 * against our hand-written WGSL analytical Jacobians to verify correctness.
 *
 * Every operation uses tf.js ops so that tf.grad() can compute gradients automatically.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Rodrigues formula: axis-angle [3] → rotation matrix [3,3]
 * All tf.js ops for auto-differentiability.
 *
 * Uses the formulation R = I + sin(θ)/θ · [ω]× + (1-cos(θ))/θ² · [ω]×²
 * which avoids the singularity at θ=0 (sinc(θ) and (1-cos(θ))/θ² are smooth at 0).
 */
export function rodrigues(axisAngle: tf.Tensor1D): tf.Tensor2D {
  return tf.tidy(() => {
    const wx = tf.slice(axisAngle, [0], [1]).squeeze();
    const wy = tf.slice(axisAngle, [1], [1]).squeeze();
    const wz = tf.slice(axisAngle, [2], [1]).squeeze();

    // θ² = wx² + wy² + wz²  (avoid sqrt for gradient stability)
    const theta2 = tf.add(tf.add(tf.square(wx), tf.square(wy)), tf.square(wz));

    // Use Taylor expansion coefficients that are smooth at θ=0:
    // sinc(θ) = sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120
    // cosc(θ) = (1-cos(θ))/θ² ≈ 1/2 - θ²/24 + θ⁴/720
    // We compute these using the safe form that works for all θ
    const theta = tf.sqrt(tf.add(theta2, 1e-12));
    const sincTheta = tf.div(tf.sin(theta), tf.add(theta, 1e-12));
    const coscTheta = tf.div(tf.sub(1, tf.cos(theta)), tf.add(theta2, 1e-12));

    // Skew-symmetric matrix [ω]× elements
    // R = I + sinc(θ) · [ω]× + cosc(θ) · [ω]×²
    // [ω]× = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
    // [ω]×² = [[-(wy²+wz²), wx·wy, wx·wz],
    //          [wx·wy, -(wx²+wz²), wy·wz],
    //          [wx·wz, wy·wz, -(wx²+wy²)]]

    const wxwy = tf.mul(wx, wy);
    const wxwz = tf.mul(wx, wz);
    const wywz = tf.mul(wy, wz);

    const r00 = tf.sub(tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wy), tf.square(wz))))), 0);
    const r01 = tf.sub(tf.mul(coscTheta, wxwy), tf.mul(sincTheta, wz));
    const r02 = tf.add(tf.mul(coscTheta, wxwz), tf.mul(sincTheta, wy));
    const r10 = tf.add(tf.mul(coscTheta, wxwy), tf.mul(sincTheta, wz));
    const r11 = tf.sub(tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wx), tf.square(wz))))), 0);
    const r12 = tf.sub(tf.mul(coscTheta, wywz), tf.mul(sincTheta, wx));
    const r20 = tf.sub(tf.mul(coscTheta, wxwz), tf.mul(sincTheta, wy));
    const r21 = tf.add(tf.mul(coscTheta, wywz), tf.mul(sincTheta, wx));
    const r22 = tf.sub(tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wx), tf.square(wy))))), 0);

    const row0 = tf.stack([r00, r01, r02]);
    const row1 = tf.stack([r10, r11, r12]);
    const row2 = tf.stack([r20, r21, r22]);
    return tf.stack([row0, row1, row2]) as tf.Tensor2D;
  });
}

/**
 * Simplified SMPL forward pass for gradient validation.
 *
 * Takes pose (axis-angle per joint) and returns joint positions.
 * Uses a simplified model with T-pose positions and kinematic tree.
 *
 * @param pose - [72] axis-angle rotations (24 joints × 3)
 * @param tposeJoints - [24, 3] T-pose joint positions
 * @param parentIndices - [24] parent joint indices (-1 for root)
 * @returns joints [24, 3] posed joint positions
 */
export function smplForwardFK(
  pose: tf.Tensor1D,
  tposeJoints: tf.Tensor2D,
  parentIndices: number[],
): tf.Tensor2D {
  return tf.tidy(() => {
    const numJoints = 24;

    // Build local rotation matrices for each joint
    const localRots: tf.Tensor2D[] = [];
    for (let j = 0; j < numJoints; j++) {
      const aa = tf.slice(pose, [j * 3], [3]) as tf.Tensor1D;
      localRots.push(rodrigues(aa));
    }

    // Forward kinematics: compose transforms
    const globalTranslations: tf.Tensor1D[] = new Array(numJoints);
    const globalRotations: tf.Tensor2D[] = new Array(numJoints);

    for (let j = 0; j < numJoints; j++) {
      const jointPos = tf.slice(tposeJoints, [j, 0], [1, 3]).squeeze() as tf.Tensor1D;
      const parent = parentIndices[j]!;

      if (parent < 0) {
        // Root joint
        globalRotations[j] = localRots[j]!;
        globalTranslations[j] = jointPos;
      } else {
        // Child joint: compose with parent
        globalRotations[j] = tf.matMul(globalRotations[parent]!, localRots[j]!) as tf.Tensor2D;

        // Relative position in parent frame
        const parentPos = tf.slice(tposeJoints, [parent, 0], [1, 3]).squeeze() as tf.Tensor1D;
        const relPos = tf.sub(jointPos, parentPos) as tf.Tensor1D;

        // Transform relative position by parent rotation + add parent translation
        const rotatedRel = tf.matMul(
          globalRotations[parent]!,
          relPos.reshape([3, 1]),
        ).squeeze() as tf.Tensor1D;
        globalTranslations[j] = tf.add(globalTranslations[parent]!, rotatedRel) as tf.Tensor1D;
      }
    }

    // Stack all joint positions [24, 3]
    return tf.stack(globalTranslations) as tf.Tensor2D;
  });
}

/**
 * Simple L2 loss on joint positions (for gradient validation).
 * L = Σ ‖joints - target‖²
 */
export function jointLoss(
  joints: tf.Tensor2D,
  target: tf.Tensor2D,
): tf.Scalar {
  return tf.tidy(() => {
    const diff = tf.sub(joints, target);
    return tf.sum(tf.square(diff)) as tf.Scalar;
  });
}

/**
 * SMPL prior loss: L_p = wp × ‖θ - θ₀‖²
 */
export function smplPriorLoss(
  pose: tf.Tensor1D,
  initPose: tf.Tensor1D,
  weight: number,
): tf.Scalar {
  return tf.tidy(() => {
    const diff = tf.sub(pose, initPose);
    return tf.mul(weight, tf.sum(tf.square(diff))) as tf.Scalar;
  });
}

/**
 * Compute gradients of a loss function w.r.t. pose parameters.
 * This is the KEY function — it uses tf.grad() for automatic differentiation.
 */
export function computeGradient(
  pose: tf.Tensor1D,
  tposeJoints: tf.Tensor2D,
  parentIndices: number[],
  targetJoints: tf.Tensor2D,
): tf.Tensor1D {
  const gradFn = tf.grad((p: tf.Tensor) => {
    const joints = smplForwardFK(p as tf.Tensor1D, tposeJoints, parentIndices);
    return jointLoss(joints, targetJoints);
  });

  return gradFn(pose) as tf.Tensor1D;
}

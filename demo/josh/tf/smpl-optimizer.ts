/**
 * SMPL forward pass in TensorFlow.js — fully differentiable via tf.variableGrads.
 *
 * Implements the full SMPL model:
 *   shape blend → joint regression → Rodrigues per joint → FK → pose blend → LBS → global
 *
 * Exported for use by josh-optimizer.ts and tests.
 */

import * as tf from '@tensorflow/tfjs';
import type { SMPLModelData } from '../models/smpl-loader-ui.ts';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface SMPLParams {
  /** [72] axis-angle (24 joints × 3) */
  pose: tf.Variable;
  /** [10] shape coefficients */
  betas: tf.Variable;
  /** [3] global translation */
  transl: tf.Variable;
  /** [1] global scale */
  scale: tf.Variable;
}

// ---------------------------------------------------------------------------
// createSMPLParams
// ---------------------------------------------------------------------------

export function createSMPLParams(init?: Partial<{
  pose: Float32Array;
  betas: Float32Array;
  transl: Float32Array;
  scale: number;
}>): SMPLParams {
  const pose = tf.variable(
    tf.tensor1d(init?.pose ?? new Float32Array(72), 'float32'),
    true,
    'smpl_pose',
  );
  const betas = tf.variable(
    tf.tensor1d(init?.betas ?? new Float32Array(10), 'float32'),
    true,
    'smpl_betas',
  );
  const transl = tf.variable(
    tf.tensor1d(init?.transl ?? new Float32Array(3), 'float32'),
    true,
    'smpl_transl',
  );
  const scale = tf.variable(
    tf.tensor1d(init?.scale != null ? [init.scale] : [1.0], 'float32'),
    true,
    'smpl_scale',
  );
  return { pose, betas, transl, scale };
}

// ---------------------------------------------------------------------------
// Rodrigues formula (axis-angle → rotation matrix)
// ---------------------------------------------------------------------------

function rodrigues(axisAngle: tf.Tensor1D): tf.Tensor2D {
  return tf.tidy(() => {
    const wx = tf.slice(axisAngle, [0], [1]).squeeze() as tf.Scalar;
    const wy = tf.slice(axisAngle, [1], [1]).squeeze() as tf.Scalar;
    const wz = tf.slice(axisAngle, [2], [1]).squeeze() as tf.Scalar;

    // θ² = wx² + wy² + wz²
    const theta2 = tf.add(tf.add(tf.square(wx), tf.square(wy)), tf.square(wz)) as tf.Scalar;
    // Add small eps inside sqrt to avoid NaN gradient at 0
    const theta = tf.sqrt(tf.add(theta2, 1e-12)) as tf.Scalar;
    // sinc(θ) = sin(θ)/(θ+ε)
    const sincTheta = tf.div(tf.sin(theta), tf.add(theta, 1e-12)) as tf.Scalar;
    // cosc(θ) = (1-cos(θ))/(θ²+ε)
    const coscTheta = tf.div(tf.sub(1, tf.cos(theta)), tf.add(theta2, 1e-12)) as tf.Scalar;

    // Cross-product terms
    const wxwy = tf.mul(wx, wy) as tf.Scalar;
    const wxwz = tf.mul(wx, wz) as tf.Scalar;
    const wywz = tf.mul(wy, wz) as tf.Scalar;

    // Row 0
    const r00 = tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wy), tf.square(wz))))) as tf.Scalar;
    const r01 = tf.sub(tf.mul(coscTheta, wxwy), tf.mul(sincTheta, wz)) as tf.Scalar;
    const r02 = tf.add(tf.mul(coscTheta, wxwz), tf.mul(sincTheta, wy)) as tf.Scalar;
    // Row 1
    const r10 = tf.add(tf.mul(coscTheta, wxwy), tf.mul(sincTheta, wz)) as tf.Scalar;
    const r11 = tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wx), tf.square(wz))))) as tf.Scalar;
    const r12 = tf.sub(tf.mul(coscTheta, wywz), tf.mul(sincTheta, wx)) as tf.Scalar;
    // Row 2
    const r20 = tf.sub(tf.mul(coscTheta, wxwz), tf.mul(sincTheta, wy)) as tf.Scalar;
    const r21 = tf.add(tf.mul(coscTheta, wywz), tf.mul(sincTheta, wx)) as tf.Scalar;
    const r22 = tf.add(1, tf.mul(coscTheta, tf.neg(tf.add(tf.square(wx), tf.square(wy))))) as tf.Scalar;

    return tf.stack([
      tf.stack([r00, r01, r02]),
      tf.stack([r10, r11, r12]),
      tf.stack([r20, r21, r22]),
    ]) as tf.Tensor2D;
  });
}

// ---------------------------------------------------------------------------
// Build 4×4 homogeneous transform from rotation [3,3] and translation [3]
// ---------------------------------------------------------------------------

function makeTransform(R: tf.Tensor2D, t: tf.Tensor1D): tf.Tensor2D {
  // Returns [4, 4]
  const tCol = t.reshape([3, 1]) as tf.Tensor2D;
  const top = tf.concat([R, tCol], 1) as tf.Tensor2D;          // [3,4]
  const bottom = tf.tensor2d([[0, 0, 0, 1]], [1, 4], 'float32');
  return tf.concat([top, bottom], 0) as tf.Tensor2D;            // [4,4]
}

// ---------------------------------------------------------------------------
// smplForward
// ---------------------------------------------------------------------------

export function smplForward(
  params: SMPLParams,
  model: SMPLModelData,
): { vertices: tf.Tensor2D; joints: tf.Tensor2D } {
  return tf.tidy(() => {
    const { pose, betas, transl, scale } = params;

    // -----------------------------------------------------------------------
    // 1. Shape blend shapes
    //    v_shaped [6890, 3] = v_template [6890,3] + shapedirs[6890,3,10] @ betas[10]
    // -----------------------------------------------------------------------
    const nV = 6890;
    const shapedirsFull = tf.tensor(
      model.shapedirs,
      [nV, 3, model.shapedirs.length / (nV * 3)],
      'float32',
    );
    // Slice to first 10 if more shape components exist
    const shapedirs10 = tf.slice(shapedirsFull, [0, 0, 0], [nV, 3, 10]) as tf.Tensor3D;

    const vTemplate = tf.tensor2d(model.vertices, [nV, 3], 'float32');
    // einsum('ijk,k->ij', shapedirs10, betas) = reshape(shapedirs10,[nV*3,10]) @ betas[10,1] → [nV,3]
    const shapedirsMat = shapedirs10.reshape([nV * 3, 10]) as tf.Tensor2D;
    const betasCol = (betas as tf.Tensor1D).reshape([10, 1]) as tf.Tensor2D;
    const shapeOffset = tf.matMul(shapedirsMat, betasCol).reshape([nV, 3]) as tf.Tensor2D;
    const vShaped = tf.add(vTemplate, shapeOffset) as tf.Tensor2D;     // [6890,3]

    // -----------------------------------------------------------------------
    // 2. Joint regressor
    //    J [24,3] = J_regressor[24,6890] @ v_shaped[6890,3]
    // -----------------------------------------------------------------------
    const nJ = 24;
    const jReg = tf.tensor2d(model.J_regressor, [nJ, nV], 'float32');
    const J = tf.matMul(jReg, vShaped) as tf.Tensor2D;                // [24,3]

    // -----------------------------------------------------------------------
    // 3. Rodrigues per joint → rotation matrices [24, 3, 3]
    // -----------------------------------------------------------------------
    const rotMats: tf.Tensor2D[] = [];
    for (let k = 0; k < nJ; k++) {
      const aa = tf.slice(pose as tf.Tensor1D, [k * 3], [3]) as tf.Tensor1D;
      rotMats.push(rodrigues(aa));
    }

    // -----------------------------------------------------------------------
    // 4. Forward kinematics (FK) — 4×4 global transforms
    //    kintree_table row 0 = parent indices; parent[0]=-1
    // -----------------------------------------------------------------------
    const parents: number[] = [];
    for (let k = 0; k < nJ; k++) {
      parents.push(model.kintree_table[k]!);
    }

    const G: tf.Tensor2D[] = new Array(nJ);
    for (let k = 0; k < nJ; k++) {
      const Jk = tf.slice(J, [k, 0], [1, 3]).squeeze() as tf.Tensor1D;  // [3]
      const Rk = rotMats[k]!;
      const pk = parents[k]!;

      if (pk < 0) {
        // Root
        G[k] = makeTransform(Rk, Jk);
      } else {
        const Jpar = tf.slice(J, [pk, 0], [1, 3]).squeeze() as tf.Tensor1D;
        const tLocal = tf.sub(Jk, Jpar) as tf.Tensor1D;            // relative offset
        // Local 4×4 for this joint
        const Tlocal = makeTransform(Rk, tLocal);
        G[k] = tf.matMul(G[pk]!, Tlocal) as tf.Tensor2D;
      }
    }

    // -----------------------------------------------------------------------
    // 5. Pose blend shapes
    //    pose_feature [207] = stacked rotation matrices of joints 1..23 minus I
    //    v_posed [6890,3] = v_shaped + posedirs[6890,3,207] @ pose_feature
    // -----------------------------------------------------------------------
    const poseFeatures: tf.Scalar[] = [];
    const I3 = tf.eye(3) as tf.Tensor2D;
    for (let k = 1; k < nJ; k++) {
      const diff = tf.sub(rotMats[k]!, I3) as tf.Tensor2D;          // [3,3]
      const flat = diff.reshape([9]);
      for (let e = 0; e < 9; e++) {
        poseFeatures.push(tf.slice(flat, [e], [1]).squeeze() as tf.Scalar);
      }
    }
    const poseFeatureVec = tf.stack(poseFeatures).reshape([207]) as tf.Tensor1D;  // [207]

    const posedirsMat = tf.tensor(model.posedirs, [nV * 3, 207], 'float32');
    const poseCol = poseFeatureVec.reshape([207, 1]) as tf.Tensor2D;
    const poseOffset = tf.matMul(posedirsMat, poseCol).reshape([nV, 3]) as tf.Tensor2D;
    const vPosed = tf.add(vShaped, poseOffset) as tf.Tensor2D;       // [6890,3]

    // -----------------------------------------------------------------------
    // 6. LBS (Linear Blend Skinning)
    //    weights [6890,24], each G[k] is [4,4]
    // -----------------------------------------------------------------------
    const skinWeights = tf.tensor2d(model.weights, [nV, nJ], 'float32');  // [6890,24]

    // Stack all G as [24,4,4], then weighted sum → [6890,4,4]
    const Gstack = tf.stack(G) as tf.Tensor3D;                       // [24,4,4]

    // Compute joint-corrected transforms: subtract rest-pose joint from G
    // blendedTransform[v] = sum_k w[v,k] * G[k]
    // For efficiency: reshape to [24,16], matmul with w[6890,24] → [6890,16] → [6890,4,4]
    const Gflat = Gstack.reshape([nJ, 16]) as tf.Tensor2D;           // [24,16]
    const blendedFlat = tf.matMul(skinWeights, Gflat) as tf.Tensor2D; // [6890,16]
    const blendedT = blendedFlat.reshape([nV, 4, 4]) as tf.Tensor3D; // [6890,4,4]

    // Homogeneous vertices [6890,4,1]
    const ones = tf.ones([nV, 1], 'float32') as tf.Tensor2D;
    const vHomo = tf.concat([vPosed, ones], 1).reshape([nV, 4, 1]) as tf.Tensor3D;

    // Skinned = blendedT @ vHomo → [6890,4,1] → [6890,3]
    const vSkinnedHomo = tf.matMul(
      blendedT.reshape([nV * 4, 4]) as tf.Tensor2D,
      tf.tile(vHomo.reshape([nV, 4]), [1, 1]).reshape([nV * 4, 1]) as tf.Tensor2D,
    );
    // Actually do it properly with batch matmul workaround:
    // blendedT [nV,4,4], vHomo [nV,4,1]
    // tf.js doesn't have batchMatMul with shape [nV,4,4]x[nV,4,1] directly,
    // so we use: (blendedT.reshape[nV,4,4]) × (vHomo[nV,4,1])
    // Use einsum equivalent: sum over j: blendedT[v,i,j] * vHomo[v,j,0]
    // = matMul(blendedFlat[nV,16], ...) — easier to slice per component
    const bT = blendedT;                                              // [nV,4,4]
    const vH = tf.concat([vPosed, ones], 1) as tf.Tensor2D;          // [nV,4]

    // Result[v,i] = sum_j bT[v,i,j] * vH[v,j]
    // = einsum('vij,vj->vi', bT, vH)
    // Use: reshape bT to [nV,4,4], vH to [nV,4,1], batch matmul
    const vHcol = vH.reshape([nV, 4, 1]) as tf.Tensor3D;
    // Flatten batch dimension for matmul: [nV*4, 4] x [nV*4, 1]
    // Better: for each of the 4 rows, multiply
    const rowResults: tf.Tensor2D[] = [];
    for (let row = 0; row < 4; row++) {
      // bT[:,row,:] shape [nV,4]
      const rowSlice = tf.slice(bT, [0, row, 0], [nV, 1, 4]).squeeze([1]) as tf.Tensor2D;
      // sum_j rowSlice[v,j] * vH[v,j]
      const dot = tf.sum(tf.mul(rowSlice, vH), 1, true) as tf.Tensor2D; // [nV,1]
      rowResults.push(dot);
    }
    const vSkinned4 = tf.concat(rowResults, 1) as tf.Tensor2D;       // [nV,4]
    const vSkinned = tf.slice(vSkinned4, [0, 0], [nV, 3]) as tf.Tensor2D; // [nV,3]

    // -----------------------------------------------------------------------
    // 7. Global transform: vertices = vSkinned * scale + transl
    // -----------------------------------------------------------------------
    const scaleVal = tf.slice(scale as tf.Tensor1D, [0], [1]).squeeze() as tf.Scalar;
    const vertices = tf.add(
      tf.mul(vSkinned, scaleVal),
      (transl as tf.Tensor1D).reshape([1, 3]),
    ) as tf.Tensor2D;                                                 // [6890,3]

    // -----------------------------------------------------------------------
    // 8. Joint positions from G[k] last column (translation)
    // -----------------------------------------------------------------------
    const jointPos: tf.Tensor1D[] = [];
    for (let k = 0; k < nJ; k++) {
      // G[k] [4,4]: last col = [:3, 3]
      const col3 = tf.slice(G[k]!, [0, 3], [3, 1]).squeeze() as tf.Tensor1D;
      jointPos.push(col3);
    }
    const joints = tf.stack(jointPos) as tf.Tensor2D;                // [24,3]

    // Dispose intermediate tensors that won't be returned via tf.tidy cleanup
    void vSkinnedHomo; // computed but replaced by manual row loop

    return { vertices, joints };
  });
}

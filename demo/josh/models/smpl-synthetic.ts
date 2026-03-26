/**
 * Synthetic SMPL model data generator.
 *
 * Generates a humanoid mesh approximation without requiring the proprietary
 * SMPL model files (which have restrictive academic licensing).
 * The synthetic data follows the same buffer layout as real SMPL data,
 * enabling the full GPU pipeline (blend shapes, FK, LBS) to run correctly.
 *
 * For production use with real SMPL data, users should download the model from
 * https://smpl.is.tue.mpg.de/ and use loadSmplModel() from model-loader.ts.
 */

import type { SmplModelBuffers } from './smpl.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_FACE_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_SHAPE_DIM,
  SMPL_KINEMATIC_TREE,
} from './smpl.ts';

/**
 * T-pose joint positions for a ~1.7m humanoid (in meters).
 * Y-up, Z-forward convention.
 */
const T_POSE_JOINTS: [number, number, number][] = [
  [0.00, 0.90, 0.00],   // 0: pelvis
  [0.10, 0.85, 0.00],   // 1: left_hip
  [-0.10, 0.85, 0.00],  // 2: right_hip
  [0.00, 1.05, 0.00],   // 3: spine1
  [0.10, 0.45, 0.00],   // 4: left_knee
  [-0.10, 0.45, 0.00],  // 5: right_knee
  [0.00, 1.20, 0.00],   // 6: spine2
  [0.10, 0.05, 0.00],   // 7: left_ankle
  [-0.10, 0.05, 0.00],  // 8: right_ankle
  [0.00, 1.35, 0.00],   // 9: spine3
  [0.10, 0.00, 0.05],   // 10: left_foot
  [-0.10, 0.00, 0.05],  // 11: right_foot
  [0.00, 1.50, 0.00],   // 12: neck
  [0.10, 1.45, 0.00],   // 13: left_collar
  [-0.10, 1.45, 0.00],  // 14: right_collar
  [0.00, 1.65, 0.00],   // 15: head
  [0.22, 1.42, 0.00],   // 16: left_shoulder
  [-0.22, 1.42, 0.00],  // 17: right_shoulder
  [0.45, 1.42, 0.00],   // 18: left_elbow
  [-0.45, 1.42, 0.00],  // 19: right_elbow
  [0.65, 1.42, 0.00],   // 20: left_wrist
  [-0.65, 1.42, 0.00],  // 21: right_wrist
  [0.72, 1.42, 0.00],   // 22: left_hand
  [-0.72, 1.42, 0.00],  // 23: right_hand
];

/**
 * Generate a cylinder segment of vertices around a joint axis.
 */
function generateLimbVertices(
  start: [number, number, number],
  end: [number, number, number],
  radius: number,
  segments: number,
  rings: number,
): [number, number, number][] {
  const vertices: [number, number, number][] = [];
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const dz = end[2] - start[2];

  for (let r = 0; r <= rings; r++) {
    const t = r / rings;
    const cx = start[0] + dx * t;
    const cy = start[1] + dy * t;
    const cz = start[2] + dz * t;

    for (let s = 0; s < segments; s++) {
      const angle = (s / segments) * Math.PI * 2;
      // Generate circle perpendicular to limb axis
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      // Use a simple perpendicular frame
      let ux: number, uy: number, uz: number;
      if (Math.abs(dy / len) < 0.9) {
        // Cross with Y-up
        ux = -dz / len;
        uy = 0;
        uz = dx / len;
      } else {
        // Cross with X-right
        ux = 0;
        uy = dz / len;
        uz = -dy / len;
      }
      const vx = dy * uz - dz * uy;
      const vy = dz * ux - dx * uz;
      const vz = dx * uy - dy * ux;
      const vlen = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;

      const px = cx + radius * (ux * Math.cos(angle) + (vx / vlen) * Math.sin(angle));
      const py = cy + radius * (uy * Math.cos(angle) + (vy / vlen) * Math.sin(angle));
      const pz = cz + radius * (uz * Math.cos(angle) + (vz / vlen) * Math.sin(angle));

      vertices.push([px, py, pz]);
    }
  }
  return vertices;
}

/**
 * Build synthetic SMPL model buffers.
 * Generates a simplified humanoid mesh with proper skinning weights.
 */
export function buildSyntheticSmplModel(): SmplModelBuffers {
  // Generate mesh vertices distributed around the body
  const allVertices: [number, number, number][] = [];
  const vertexJointAssignments: { joint: number; weight: number }[][] = [];

  // Body segments: pairs of joints forming limbs
  const segments: [number, number, number, number][] = [
    // [startJoint, endJoint, radius, verticesPerRing]
    [0, 3, 0.12, 8],    // pelvis → spine1 (torso lower)
    [3, 6, 0.11, 8],    // spine1 → spine2 (torso mid)
    [6, 9, 0.10, 8],    // spine2 → spine3 (torso upper)
    [9, 12, 0.08, 6],   // spine3 → neck
    [12, 15, 0.09, 8],  // neck → head
    [1, 4, 0.06, 6],    // left hip → knee
    [4, 7, 0.05, 6],    // left knee → ankle
    [7, 10, 0.04, 4],   // left ankle → foot
    [2, 5, 0.06, 6],    // right hip → knee
    [5, 8, 0.05, 6],    // right knee → ankle
    [8, 11, 0.04, 4],   // right ankle → foot
    [16, 18, 0.04, 5],  // left shoulder → elbow
    [18, 20, 0.035, 5], // left elbow → wrist
    [20, 22, 0.03, 4],  // left wrist → hand
    [17, 19, 0.04, 5],  // right shoulder → elbow
    [19, 21, 0.035, 5], // right elbow → wrist
    [21, 23, 0.03, 4],  // right wrist → hand
    [9, 13, 0.05, 4],   // spine3 → left collar
    [9, 14, 0.05, 4],   // spine3 → right collar
    [13, 16, 0.045, 4], // left collar → shoulder
    [14, 17, 0.045, 4], // right collar → shoulder
  ];

  for (const [j0, j1, radius, segs] of segments) {
    const start = T_POSE_JOINTS[j0]!;
    const end = T_POSE_JOINTS[j1]!;
    const rings = 4;
    const verts = generateLimbVertices(start, end, radius, segs, rings);

    for (let i = 0; i < verts.length; i++) {
      const t = Math.floor(i / segs) / rings; // 0..1 along limb
      allVertices.push(verts[i]!);
      // Blend between start and end joint
      vertexJointAssignments.push([
        { joint: j0, weight: 1 - t },
        { joint: j1, weight: t },
      ]);
    }
  }

  // Pad or truncate to exactly SMPL_VERTEX_COUNT
  while (allVertices.length < SMPL_VERTEX_COUNT) {
    // Fill with vertices near the pelvis
    const jitter = (allVertices.length % 100) / 1000;
    allVertices.push([jitter, 0.9 + jitter, jitter]);
    vertexJointAssignments.push([{ joint: 0, weight: 1.0 }]);
  }

  // Build output arrays
  const meanTemplate = new Float32Array(SMPL_VERTEX_COUNT * 3);
  for (let i = 0; i < SMPL_VERTEX_COUNT; i++) {
    const v = allVertices[i]!;
    meanTemplate[i * 3] = v[0];
    meanTemplate[i * 3 + 1] = v[1];
    meanTemplate[i * 3 + 2] = v[2];
  }

  // Shape blend shapes — small random perturbations per PCA component
  const shapeBlendShapes = new Float32Array(SMPL_VERTEX_COUNT * 3 * SMPL_SHAPE_DIM);
  const rng = mulberry32(42); // deterministic
  for (let i = 0; i < shapeBlendShapes.length; i++) {
    shapeBlendShapes[i] = (rng() - 0.5) * 0.02; // ±1cm perturbations
  }

  // Pose blend shapes (207 = (24-1)*9 pose correctives) — leave zeros for simplicity
  const poseBlendShapes = new Float32Array(SMPL_VERTEX_COUNT * 3 * 207);

  // Joint regressor — sparse matrix [24, 6890]
  // Each joint is the average of nearby vertices
  const jointRegressor = new Float32Array(SMPL_JOINT_COUNT * SMPL_VERTEX_COUNT);
  for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
    const jpos = T_POSE_JOINTS[j]!;
    let totalWeight = 0;
    const weights: { idx: number; w: number }[] = [];

    for (let v = 0; v < SMPL_VERTEX_COUNT; v++) {
      const vv = allVertices[v]!;
      const dx = vv[0] - jpos[0];
      const dy = vv[1] - jpos[1];
      const dz = vv[2] - jpos[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 0.1) {
        const w = Math.exp(-dist * dist / 0.002);
        weights.push({ idx: v, w });
        totalWeight += w;
      }
    }

    // Normalize so row sums to 1
    if (totalWeight > 0) {
      for (const { idx, w } of weights) {
        jointRegressor[j * SMPL_VERTEX_COUNT + idx] = w / totalWeight;
      }
    }
  }

  // Skinning weights and indices [6890, 4]
  const skinningWeights = new Float32Array(SMPL_VERTEX_COUNT * 4);
  const skinningIndices = new Uint32Array(SMPL_VERTEX_COUNT * 4);

  for (let v = 0; v < SMPL_VERTEX_COUNT; v++) {
    const assignments = vertexJointAssignments[v]!;
    let totalW = 0;
    for (let k = 0; k < Math.min(4, assignments.length); k++) {
      skinningIndices[v * 4 + k] = assignments[k]!.joint;
      skinningWeights[v * 4 + k] = assignments[k]!.weight;
      totalW += assignments[k]!.weight;
    }
    // Normalize weights
    if (totalW > 0) {
      for (let k = 0; k < 4; k++) {
        const idx = v * 4 + k;
        skinningWeights[idx] = skinningWeights[idx]! / totalW;
      }
    } else {
      skinningWeights[v * 4] = 1.0; // default to pelvis
    }
  }

  // Triangle faces — simple strip triangulation
  const faces = new Uint32Array(SMPL_FACE_COUNT * 3);
  let fi = 0;
  for (let i = 0; i < SMPL_VERTEX_COUNT - 2 && fi < SMPL_FACE_COUNT; i++) {
    faces[fi * 3] = i;
    faces[fi * 3 + 1] = i + 1;
    faces[fi * 3 + 2] = i + 2;
    fi++;
  }

  return {
    meanTemplate,
    shapeBlendShapes,
    poseBlendShapes,
    jointRegressor,
    skinningWeights,
    skinningIndices,
    faces,
  };
}

/** Simple seeded PRNG (Mulberry32) */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Convert axis-angle rotation (3 values) to a 3x3 rotation matrix (9 values, row-major).
 * Uses Rodrigues' formula.
 */
export function axisAngleToRotMat(ax: number, ay: number, az: number): Float32Array {
  const angle = Math.sqrt(ax * ax + ay * ay + az * az);
  const result = new Float32Array(9);

  if (angle < 1e-8) {
    // Identity
    result[0] = 1; result[4] = 1; result[8] = 1;
    return result;
  }

  const nx = ax / angle;
  const ny = ay / angle;
  const nz = az / angle;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;

  result[0] = t * nx * nx + c;
  result[1] = t * nx * ny - s * nz;
  result[2] = t * nx * nz + s * ny;
  result[3] = t * ny * nx + s * nz;
  result[4] = t * ny * ny + c;
  result[5] = t * ny * nz - s * nx;
  result[6] = t * nz * nx - s * ny;
  result[7] = t * nz * ny + s * nx;
  result[8] = t * nz * nz + c;

  return result;
}

/**
 * Get T-pose joint positions for visualization.
 */
export function getTposeJoints(): Float32Array {
  const result = new Float32Array(SMPL_JOINT_COUNT * 3);
  for (let j = 0; j < SMPL_JOINT_COUNT; j++) {
    const pos = T_POSE_JOINTS[j]!;
    result[j * 3] = pos[0];
    result[j * 3 + 1] = pos[1];
    result[j * 3 + 2] = pos[2];
  }
  return result;
}

/**
 * Get kinematic tree as Int32Array for GPU upload.
 */
export function getKinematicTreeI32(): Int32Array {
  return new Int32Array(SMPL_KINEMATIC_TREE);
}

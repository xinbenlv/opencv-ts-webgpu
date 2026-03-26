/**
 * SMPL (Skinned Multi-Person Linear) model constants and topology.
 *
 * Reference: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015
 */

/** Number of vertices in the SMPL mesh */
export const SMPL_VERTEX_COUNT = 6890;

/** Number of triangular faces */
export const SMPL_FACE_COUNT = 13776;

/** Number of joints in the kinematic tree */
export const SMPL_JOINT_COUNT = 24;

/** Pose parameter dimension (24 joints × 3 axis-angle) */
export const SMPL_POSE_DIM = 72;

/** Shape parameter dimension (PCA components) */
export const SMPL_SHAPE_DIM = 10;

/** Number of skinning weights per vertex */
export const SMPL_WEIGHTS_PER_VERTEX = 4;

/**
 * SMPL joint names and their indices in the kinematic tree.
 */
export const SMPL_JOINT_NAMES = [
  'pelvis',         // 0
  'left_hip',       // 1
  'right_hip',      // 2
  'spine1',         // 3
  'left_knee',      // 4
  'right_knee',     // 5
  'spine2',         // 6
  'left_ankle',     // 7
  'right_ankle',    // 8
  'spine3',         // 9
  'left_foot',      // 10
  'right_foot',     // 11
  'neck',           // 12
  'left_collar',    // 13
  'right_collar',   // 14
  'head',           // 15
  'left_shoulder',  // 16
  'right_shoulder', // 17
  'left_elbow',     // 18
  'right_elbow',    // 19
  'left_wrist',     // 20
  'right_wrist',    // 21
  'left_hand',      // 22
  'right_hand',     // 23
] as const;

/**
 * SMPL kinematic tree: parent[i] = parent joint index of joint i.
 * -1 for root (pelvis).
 */
export const SMPL_KINEMATIC_TREE: readonly number[] = [
  -1, // 0: pelvis (root)
  0,  // 1: left_hip → pelvis
  0,  // 2: right_hip → pelvis
  0,  // 3: spine1 → pelvis
  1,  // 4: left_knee → left_hip
  2,  // 5: right_knee → right_hip
  3,  // 6: spine2 → spine1
  4,  // 7: left_ankle → left_knee
  5,  // 8: right_ankle → right_knee
  6,  // 9: spine3 → spine2
  7,  // 10: left_foot → left_ankle
  8,  // 11: right_foot → right_ankle
  9,  // 12: neck → spine3
  9,  // 13: left_collar → spine3
  9,  // 14: right_collar → spine3
  12, // 15: head → neck
  13, // 16: left_shoulder → left_collar
  14, // 17: right_shoulder → right_collar
  16, // 18: left_elbow → left_shoulder
  17, // 19: right_elbow → right_shoulder
  18, // 20: left_wrist → left_elbow
  19, // 21: right_wrist → right_elbow
  20, // 22: left_hand → left_wrist
  21, // 23: right_hand → right_wrist
];

/**
 * Contact vertex indices — sole vertices commonly used for ground contact.
 */
export const SMPL_CONTACT_VERTICES = {
  leftFootSole: [3216, 3217, 3218, 3219, 3220],
  rightFootSole: [6617, 6618, 6619, 6620, 6621],
  leftToes: [3327, 3328, 3329],
  rightToes: [6728, 6729, 6730],
} as const;

/**
 * Describes the data buffers needed for SMPL forward pass on GPU.
 */
export interface SmplModelBuffers {
  /** Mean template vertices [6890, 3] */
  readonly meanTemplate: Float32Array;
  /** Shape blend shapes [6890, 3, 10] */
  readonly shapeBlendShapes: Float32Array;
  /** Pose blend shapes [6890, 3, 207] */
  readonly poseBlendShapes: Float32Array;
  /** Joint regressor [24, 6890] */
  readonly jointRegressor: Float32Array;
  /** Skinning weights [6890, 4] */
  readonly skinningWeights: Float32Array;
  /** Skinning weight indices [6890, 4] */
  readonly skinningIndices: Uint32Array;
  /** Triangle face indices [13776, 3] */
  readonly faces: Uint32Array;
}

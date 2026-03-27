/**
 * SE3 rigid body transform interpolation.
 * SLERP for rotation (quaternions), linear interpolation for translation.
 *
 * Conventions:
 *   SE3  — 4×4 column-major transform matrix as Float32Array(16)
 *   Quat — [x, y, z, w]
 *   mat3 arguments are row-major Float32Array(9)
 */

/** 4×4 column-major transform matrix as Float32Array(16) */
export type SE3 = Float32Array;

/** Quaternion [x, y, z, w] */
export type Quat = [number, number, number, number];

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

/**
 * Convert a row-major 3×3 rotation matrix to a quaternion [x, y, z, w].
 * Uses Shepperd's method for numerical stability.
 *
 * Row-major index layout:
 *   m[0] m[1] m[2]
 *   m[3] m[4] m[5]
 *   m[6] m[7] m[8]
 */
export function mat3ToQuat(m: Float32Array): Quat {
  const m0 = m[0]!, m1 = m[1]!, m2 = m[2]!;
  const m3 = m[3]!, m4 = m[4]!, m5 = m[5]!;
  const m6 = m[6]!, m7 = m[7]!, m8 = m[8]!;
  const trace = m0 + m4 + m8;
  let x: number, y: number, z: number, w: number;

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    w = 0.25 / s;
    x = (m7 - m5) * s;
    y = (m2 - m6) * s;
    z = (m3 - m1) * s;
  } else if (m0 > m4 && m0 > m8) {
    const s = 2 * Math.sqrt(1 + m0 - m4 - m8);
    w = (m7 - m5) / s;
    x = s / 4;
    y = (m1 + m3) / s;
    z = (m2 + m6) / s;
  } else if (m4 > m8) {
    const s = 2 * Math.sqrt(1 + m4 - m0 - m8);
    w = (m2 - m6) / s;
    x = (m1 + m3) / s;
    y = s / 4;
    z = (m5 + m7) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m8 - m0 - m4);
    w = (m3 - m1) / s;
    x = (m2 + m6) / s;
    y = (m5 + m7) / s;
    z = s / 4;
  }

  return [x, y, z, w];
}

/**
 * Convert a quaternion [x, y, z, w] to a row-major 3×3 rotation matrix.
 */
export function quatToMat3(q: Quat): Float32Array {
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;

  return new Float32Array([
    1 - (yy + zz),  xy - wz,        xz + wy,
    xy + wz,        1 - (xx + zz),  yz - wx,
    xz - wy,        yz + wx,        1 - (xx + yy),
  ]);
}

/** Normalize a quaternion to unit length. */
export function normalizeQuat(q: Quat): Quat {
  const [x, y, z, w] = q;
  const len = Math.sqrt(x * x + y * y + z * z + w * w);
  if (len < 1e-10) return [0, 0, 0, 1];
  return [x / len, y / len, z / len, w / len];
}

/**
 * Spherical linear interpolation between two unit quaternions.
 * If the angle between them is very small, falls back to lerp + normalize.
 */
export function slerpQuat(q0: Quat, q1: Quat, t: number): Quat {
  let [x1, y1, z1, w1] = q1;
  const dot = q0[0] * x1 + q0[1] * y1 + q0[2] * z1 + q0[3] * w1;

  // Ensure shortest-path interpolation
  if (dot < 0) {
    x1 = -x1; y1 = -y1; z1 = -z1; w1 = -w1;
  }

  const clampedDot = Math.min(1, Math.max(-1, Math.abs(dot)));
  const angle = Math.acos(clampedDot);

  if (angle < 1e-6) {
    // Quaternions nearly identical — lerp and normalize
    return normalizeQuat([
      q0[0] + t * (x1 - q0[0]),
      q0[1] + t * (y1 - q0[1]),
      q0[2] + t * (z1 - q0[2]),
      q0[3] + t * (w1 - q0[3]),
    ]);
  }

  const sinA = Math.sin(angle);
  const s0 = Math.sin((1 - t) * angle) / sinA;
  const s1 = Math.sin(t * angle) / sinA;

  return [
    s0 * q0[0] + s1 * x1,
    s0 * q0[1] + s1 * y1,
    s0 * q0[2] + s1 * z1,
    s0 * q0[3] + s1 * w1,
  ];
}

// ---------------------------------------------------------------------------
// SE3 interpolation
// ---------------------------------------------------------------------------

/**
 * Extract the upper-left 3×3 rotation block from a column-major 4×4 matrix
 * and return it as a row-major Float32Array(9).
 *
 * Column-major 4×4 layout (index = col*4 + row):
 *   col0: [0,1,2,3]  col1: [4,5,6,7]  col2: [8,9,10,11]  col3: [12,13,14,15]
 *
 * Row r, Col c → index c*4+r
 */
function extractRotation(T: SE3): Float32Array {
  // row-major output: [r00,r01,r02, r10,r11,r12, r20,r21,r22]
  return new Float32Array([
    T[0]!, T[4]!, T[8]!,   // row 0
    T[1]!, T[5]!, T[9]!,   // row 1
    T[2]!, T[6]!, T[10]!,  // row 2
  ]);
}

/** Build a column-major 4×4 SE3 from a row-major 3×3 rotation and a translation [tx, ty, tz]. */
function buildSE3(rot: Float32Array, tx: number, ty: number, tz: number): SE3 {
  // col-major 4×4
  const T = new Float32Array(16);
  // col 0
  T[0] = rot[0]!; T[1] = rot[3]!; T[2] = rot[6]!; T[3] = 0;
  // col 1
  T[4] = rot[1]!; T[5] = rot[4]!; T[6] = rot[7]!; T[7] = 0;
  // col 2
  T[8] = rot[2]!; T[9] = rot[5]!; T[10] = rot[8]!; T[11] = 0;
  // col 3 (translation)
  T[12] = tx; T[13] = ty; T[14] = tz; T[15] = 1;
  return T;
}

/**
 * Interpolate between two column-major 4×4 SE3 transforms.
 * Rotation uses SLERP, translation uses linear interpolation.
 */
export function interpolateSE3(T0: SE3, T1: SE3, t: number): SE3 {
  // Extract rotations (row-major 3×3)
  const r0 = extractRotation(T0);
  const r1 = extractRotation(T1);

  // Convert to quaternions
  const q0 = mat3ToQuat(r0);
  const q1 = mat3ToQuat(r1);

  // SLERP
  const qt = slerpQuat(q0, q1, t);

  // Back to rotation matrix
  const rt = quatToMat3(qt);

  // Lerp translation (column 3, rows 0-2 in col-major → indices 12, 13, 14)
  const tx = T0[12]! + t * (T1[12]! - T0[12]!);
  const ty = T0[13]! + t * (T1[13]! - T0[13]!);
  const tz = T0[14]! + t * (T1[14]! - T0[14]!);

  return buildSE3(rt, tx, ty, tz);
}

/**
 * Linearly interpolate every element of two same-length Float32Arrays.
 * Useful for SMPL pose/shape vectors.
 */
export function interpolatePose(
  pose0: Float32Array,
  pose1: Float32Array,
  t: number,
): Float32Array {
  const out = new Float32Array(pose0.length);
  for (let i = 0; i < pose0.length; i++) {
    out[i] = pose0[i]! + t * (pose1[i]! - pose0[i]!);
  }
  return out;
}

// ---------------------------------------------------------------------------
// FrameParams interpolation
// ---------------------------------------------------------------------------

export interface FrameParams {
  /** Column-major 4×4 camera transform */
  cameraPose: SE3;
  /** SMPL joint angles, length 72 */
  smplPose: Float32Array;
  /** SMPL shape coefficients, length 10 */
  smplShape: Float32Array;
  /** Metric depth scale factor */
  depthScale: number;
}

/**
 * Interpolate between two FrameParams structs at parameter t ∈ [0, 1].
 */
export function interpolateFrameParams(
  kf0: FrameParams,
  kf1: FrameParams,
  t: number,
): FrameParams {
  return {
    cameraPose: interpolateSE3(kf0.cameraPose, kf1.cameraPose, t),
    smplPose: interpolatePose(kf0.smplPose, kf1.smplPose, t),
    smplShape: interpolatePose(kf0.smplShape, kf1.smplShape, t),
    depthScale: kf0.depthScale + t * (kf1.depthScale - kf0.depthScale),
  };
}

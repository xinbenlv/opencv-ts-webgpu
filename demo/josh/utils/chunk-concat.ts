/**
 * Chunk trajectory concatenation utilities.
 *
 * When processing video in overlapping chunks, each chunk's poses are expressed
 * in the local coordinate frame of that chunk.  concatChunkTrajectory anchors
 * chunk2 to the last pose of chunk1 so that the resulting trajectory lives in
 * a single world frame.
 */

import type { SE3, FrameParams } from './se3-interpolation.ts';

// ---------------------------------------------------------------------------
// 4×4 column-major matrix multiply
// ---------------------------------------------------------------------------

/**
 * Multiply two column-major 4×4 matrices: result = a × b.
 *
 * Column-major indexing: element at row r, col c → index c*4 + r.
 */
export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row]! * b[col * 4 + k]!;
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chunk concatenation
// ---------------------------------------------------------------------------

/**
 * Concatenate two SE3 trajectory chunks into a single world-frame trajectory.
 *
 * The anchor is the last pose of chunk1.  Every pose in chunk2 is multiplied
 * on the left by the anchor, bringing it from chunk2's local frame into the
 * world frame established by chunk1.
 *
 * Returns [...chunk1, ...transformedChunk2].
 */
export function concatChunkTrajectory(chunk1: SE3[], chunk2: SE3[]): SE3[] {
  if (chunk1.length === 0) return [...chunk2];
  if (chunk2.length === 0) return [...chunk1];

  const anchor = chunk1[chunk1.length - 1]!;
  const transformedChunk2 = chunk2.map((pose) => mat4Multiply(anchor, pose));
  return [...chunk1, ...transformedChunk2];
}

/**
 * Concatenate two FrameParams trajectory chunks.
 *
 * Camera poses follow the same anchor logic as concatChunkTrajectory.
 * SMPL pose/shape and depthScale from chunk2 are preserved as-is (they are
 * already in body-local frame and do not need world-frame anchoring).
 */
export function concatChunkParams(
  chunk1: FrameParams[],
  chunk2: FrameParams[],
): FrameParams[] {
  if (chunk1.length === 0) return [...chunk2];
  if (chunk2.length === 0) return [...chunk1];

  const anchor = chunk1[chunk1.length - 1]!.cameraPose;
  const transformedChunk2: FrameParams[] = chunk2.map((fp) => ({
    ...fp,
    cameraPose: mat4Multiply(anchor, fp.cameraPose),
  }));
  return [...chunk1, ...transformedChunk2];
}

import { CvError } from '../../../src/core/errors.ts';
import type { SmplModelBuffers } from './smpl.ts';
import {
  SMPL_VERTEX_COUNT,
  SMPL_FACE_COUNT,
  SMPL_JOINT_COUNT,
  SMPL_SHAPE_DIM,
} from './smpl.ts';

/**
 * Loads SMPL model data from a binary file.
 *
 * Expected format: sequential Float32/Uint32 arrays in order:
 * 1. meanTemplate [6890×3] f32
 * 2. shapeBlendShapes [6890×3×10] f32
 * 3. poseBlendShapes [6890×3×207] f32
 * 4. jointRegressor [24×6890] f32
 * 5. skinningWeights [6890×4] f32
 * 6. skinningIndices [6890×4] u32
 * 7. faces [13776×3] u32
 */
export async function loadSmplModel(url: string): Promise<SmplModelBuffers> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new CvError(`Failed to load SMPL model from ${url}: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  let offset = 0;

  function readF32(count: number): Float32Array {
    const arr = new Float32Array(buffer, offset, count);
    offset += count * 4;
    return arr;
  }

  function readU32(count: number): Uint32Array {
    const arr = new Uint32Array(buffer, offset, count);
    offset += count * 4;
    return arr;
  }

  return {
    meanTemplate: readF32(SMPL_VERTEX_COUNT * 3),
    shapeBlendShapes: readF32(SMPL_VERTEX_COUNT * 3 * SMPL_SHAPE_DIM),
    poseBlendShapes: readF32(SMPL_VERTEX_COUNT * 3 * 207),
    jointRegressor: readF32(SMPL_JOINT_COUNT * SMPL_VERTEX_COUNT),
    skinningWeights: readF32(SMPL_VERTEX_COUNT * 4),
    skinningIndices: readU32(SMPL_VERTEX_COUNT * 4),
    faces: readU32(SMPL_FACE_COUNT * 3),
  };
}

/**
 * Load an ONNX model for inference via onnxruntime-web.
 * Returns the raw ArrayBuffer for session creation.
 */
export async function loadOnnxModel(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new CvError(`Failed to load ONNX model from ${url}: ${response.status}`);
  }
  return response.arrayBuffer();
}

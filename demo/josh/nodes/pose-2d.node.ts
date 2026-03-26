/**
 * Phase 1E: 2D pose detection using MoveNet Lightning (via onnxruntime-web).
 *
 * Model file expected at: /models/movenet-lightning.onnx
 * If not found, a mock returning synthetic keypoints is used instead.
 */

import { JOSH_CONFIG } from '../config.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** 17 COCO keypoints in canonical order. */
export const COCO_KEYPOINTS = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
] as const;

/** Mapping from COCO keypoint index → SMPL joint index. */
export const COCO_TO_SMPL: Record<number, number> = {
  0: 15, // nose → head
  5: 16, // left_shoulder → left_shoulder
  6: 17, // right_shoulder → right_shoulder
  7: 18, // left_elbow → left_elbow
  8: 19, // right_elbow → right_elbow
  9: 20, // left_wrist → left_wrist
  10: 21, // right_wrist → right_wrist
  11: 1, // left_hip → left_hip
  12: 2, // right_hip → right_hip
  13: 4, // left_knee → left_knee
  14: 5, // right_knee → right_knee
  15: 7, // left_ankle → left_ankle
  16: 8, // right_ankle → right_ankle
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Keypoint2D {
  x: number; // pixel coordinate
  y: number; // pixel coordinate
  confidence: number; // 0–1
  name: string;
  smplJointIdx: number | null; // null if no SMPL mapping
}

// ---------------------------------------------------------------------------
// ORT lazy-loader (avoids top-level import in Node test environments)
// ---------------------------------------------------------------------------

let ortModule: typeof import('onnxruntime-web') | null = null;

async function getOrt() {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
  }
  return ortModule;
}

// ---------------------------------------------------------------------------
// Pose2DNode
// ---------------------------------------------------------------------------

const MODEL_SIZE = JOSH_CONFIG.moveNetInputSize; // 192

/**
 * Synthetic keypoints placed at anatomically plausible positions relative to
 * the image centre.  Used as a stand-in when the ONNX model is absent.
 */
function syntheticKeypoints(width: number, height: number): Keypoint2D[] {
  const cx = width / 2;
  const cy = height / 2;

  // Rough fractions of (cx, cy) offset for a standing-pose silhouette.
  const offsets: [number, number][] = [
    [0, -0.65], // nose
    [-0.06, -0.72], // left_eye
    [0.06, -0.72], // right_eye
    [-0.12, -0.68], // left_ear
    [0.12, -0.68], // right_ear
    [-0.25, -0.35], // left_shoulder
    [0.25, -0.35], // right_shoulder
    [-0.35, 0.0], // left_elbow
    [0.35, 0.0], // right_elbow
    [-0.38, 0.35], // left_wrist
    [0.38, 0.35], // right_wrist
    [-0.15, 0.1], // left_hip
    [0.15, 0.1], // right_hip
    [-0.16, 0.5], // left_knee
    [0.16, 0.5], // right_knee
    [-0.16, 0.85], // left_ankle
    [0.16, 0.85], // right_ankle
  ];

  return COCO_KEYPOINTS.map((name, idx) => {
    const [dx, dy] = offsets[idx]!;
    return {
      name,
      x: cx + dx * cx,
      y: cy + dy * cy,
      confidence: 0.5,
      smplJointIdx: COCO_TO_SMPL[idx] ?? null,
    };
  });
}

/**
 * Bilinear resize of ImageData to a square targetSize × targetSize canvas.
 * Pure CPU implementation — no DOM Canvas required for the resize math, but
 * we do use OffscreenCanvas / Canvas when available for efficiency.
 */
function resizeCpu(src: ImageData, targetSize: number): ImageData {
  const sw = src.width;
  const sh = src.height;
  const tw = targetSize;
  const th = targetSize;
  const dst = new Uint8ClampedArray(tw * th * 4);

  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const srcX = (tx / tw) * sw;
      const srcY = (ty / th) * sh;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, sw - 1);
      const y1 = Math.min(y0 + 1, sh - 1);
      const fx = srcX - x0;
      const fy = srcY - y0;

      for (let c = 0; c < 4; c++) {
        const v00 = src.data[((y0 * sw + x0) * 4 + c) as number]!;
        const v10 = src.data[((y0 * sw + x1) * 4 + c) as number]!;
        const v01 = src.data[((y1 * sw + x0) * 4 + c) as number]!;
        const v11 = src.data[((y1 * sw + x1) * 4 + c) as number]!;
        dst[(ty * tw + tx) * 4 + c] = Math.round(
          v00 * (1 - fx) * (1 - fy) +
            v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy +
            v11 * fx * fy,
        );
      }
    }
  }

  return new ImageData(dst, tw, th);
}

export class Pose2DNode {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _session: any = null;
  private _mock = false;

  /**
   * Initialize the node.
   * @param modelPath URL/path to movenet-lightning.onnx.
   *                  Defaults to '/models/movenet-lightning.onnx'.
   */
  async initialize(modelPath = '/models/movenet-lightning.onnx'): Promise<void> {
    try {
      const ort = await getOrt();
      ort.env.wasm.wasmPaths = './assets/ort/';
      ort.env.wasm.numThreads = 1;

      this._session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all',
      });
      console.log('[Pose2DNode] MoveNet loaded:', this._session.inputNames, this._session.outputNames);
    } catch (err) {
      console.warn('[Pose2DNode] ONNX model unavailable — using synthetic mock:', err);
      this._mock = true;
    }
  }

  /**
   * Detect 17 COCO keypoints in a 192×192 image.
   * If the model is unavailable, returns plausible synthetic keypoints.
   */
  async detect(imageData: ImageData): Promise<Keypoint2D[]> {
    if (this._mock || this._session === null) {
      return syntheticKeypoints(imageData.width, imageData.height);
    }

    // Build float32 NHWC input tensor [1, 192, 192, 3] with values in [0, 255].
    // MoveNet Lightning expects int32 input — we cast below.
    const { width, height, data } = imageData;
    const numPixels = width * height;
    const inputData = new Int32Array(numPixels * 3);

    for (let i = 0; i < numPixels; i++) {
      inputData[i * 3] = data[i * 4]!;
      inputData[i * 3 + 1] = data[i * 4 + 1]!;
      inputData[i * 3 + 2] = data[i * 4 + 2]!;
    }

    const ort = await getOrt();
    const inputTensor = new ort.Tensor('int32', inputData, [1, height, width, 3]);
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {};
    feeds[this._session.inputNames[0]!] = inputTensor;

    const results = await this._session.run(feeds);
    // MoveNet output shape: [1, 1, 17, 3] (y, x, confidence) — normalised to [0,1]
    const outputData = results[this._session.outputNames[0]!]!.data as Float32Array;

    const keypoints: Keypoint2D[] = [];
    for (let k = 0; k < 17; k++) {
      const yNorm = outputData[k * 3]!;
      const xNorm = outputData[k * 3 + 1]!;
      const conf = outputData[k * 3 + 2]!;
      keypoints.push({
        name: COCO_KEYPOINTS[k]!,
        x: xNorm * width,
        y: yNorm * height,
        confidence: conf,
        smplJointIdx: COCO_TO_SMPL[k] ?? null,
      });
    }

    return keypoints;
  }

  /**
   * Resize imageData to 192×192 for MoveNet input.
   * Uses CPU bilinear interpolation to avoid a Canvas dependency.
   */
  preprocessFrame(imageData: ImageData): ImageData {
    if (imageData.width === MODEL_SIZE && imageData.height === MODEL_SIZE) {
      return imageData;
    }
    return resizeCpu(imageData, MODEL_SIZE);
  }

  dispose(): void {
    this._session?.release();
    this._session = null;
    this._mock = false;
  }
}

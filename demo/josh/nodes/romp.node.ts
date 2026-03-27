/**
 * ROMPNode — SMPL parameter estimation using TensorFlow.js BlazePose GHUM 3D.
 *
 * Replaces the previous ONNX-based ROMP implementation.  BlazePose delivers
 * 33 3-D landmarks with per-keypoint confidence scores; we map those to the
 * 24 SMPL joints and compute axis-angle rotations from the detected bone
 * directions relative to a canonical T-pose.
 *
 * Public interface is identical to the original ROMPNode so that
 * batch-pipeline.ts (Phase E) requires zero changes.
 *
 * Output
 * ------
 *   pose        [72]  axis-angle pose (24 joints × 3)
 *   betas       [10]  shape params — always zeros (BlazePose has no shape)
 *   cam         [3]   weak-perspective camera [scale, tx, ty]
 *   confidence  [0,1] average score over key structural joints
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ROMPOutput {
  /** Axis-angle pose parameters for 24 SMPL joints [72] */
  pose: Float32Array;
  /** SMPL shape coefficients [10] — always zeros for BlazePose backend */
  betas: Float32Array;
  /** Weak-perspective camera [scale, tx, ty] */
  cam: Float32Array;
  /** Confidence of the detection in [0, 1] */
  confidence: number;
}

// ---------------------------------------------------------------------------
// BlazePose landmark indices (MediaPipe Pose 33-point model)
// ---------------------------------------------------------------------------

/** Named indices into the 33-landmark BlazePose output. */
const BP = {
  nose:            0,
  left_shoulder:  11,
  right_shoulder: 12,
  left_elbow:     13,
  right_elbow:    14,
  left_wrist:     15,
  right_wrist:    16,
  left_hip:       23,
  right_hip:      24,
  left_knee:      25,
  right_knee:     26,
  left_ankle:     27,
  right_ankle:    28,
} as const;

// ---------------------------------------------------------------------------
// SMPL T-pose joint positions (approximate, metres, Y-up)
// ---------------------------------------------------------------------------
// Used as the reference configuration for computing axis-angle rotations.
// Values are the approximate SMPL neutral-pose joint locations.

export const SMPL_TPOSE: Record<number, [number, number, number]> = {
   0: [ 0.000,  0.900,  0.000], // pelvis
   1: [-0.100,  0.850,  0.000], // L_hip
   2: [ 0.100,  0.850,  0.000], // R_hip
   3: [ 0.000,  1.000,  0.000], // spine1
   4: [-0.100,  0.500,  0.000], // L_knee
   5: [ 0.100,  0.500,  0.000], // R_knee
   6: [ 0.000,  1.100,  0.000], // spine2
   7: [-0.100,  0.100,  0.000], // L_ankle
   8: [ 0.100,  0.100,  0.000], // R_ankle
   9: [ 0.000,  1.200,  0.000], // spine3
  10: [-0.100,  0.000,  0.050], // L_foot
  11: [ 0.100,  0.000,  0.050], // R_foot
  12: [ 0.000,  1.400,  0.000], // neck
  13: [-0.080,  1.350,  0.000], // L_collar
  14: [ 0.080,  1.350,  0.000], // R_collar
  15: [ 0.000,  1.600,  0.000], // head
  16: [-0.180,  1.350,  0.000], // L_shoulder
  17: [ 0.180,  1.350,  0.000], // R_shoulder
  18: [-0.400,  1.350,  0.000], // L_elbow
  19: [ 0.400,  1.350,  0.000], // R_elbow
  20: [-0.600,  1.350,  0.000], // L_wrist
  21: [ 0.600,  1.350,  0.000], // R_wrist
  22: [-0.700,  1.350,  0.020], // L_hand
  23: [ 0.700,  1.350,  0.020], // R_hand
};

/** SMPL kinematic parent table (−1 = root). */
const SMPL_PARENTS: Record<number, number> = {
   0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4,
   8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
  15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19,
  22: 20, 23: 21,
};

// ---------------------------------------------------------------------------
// 3-D vector helpers  (pure, no deps)
// ---------------------------------------------------------------------------

type Vec3 = [number, number, number];

function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vec3Norm(v: Vec3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vec3Normalize(v: Vec3): Vec3 {
  const n = vec3Norm(v);
  if (n < 1e-9) return [0, 0, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

function vec3Lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return vec3Add(vec3Scale(a, 1 - t), vec3Scale(b, t));
}

// ---------------------------------------------------------------------------
// Axis-angle from two bone directions
// ---------------------------------------------------------------------------

/**
 * Compute the axis-angle rotation vector that rotates the T-pose bone
 * direction toward the detected bone direction.
 *
 * Returns a [3] Float32Array containing ω = axis * θ.
 *
 * Edge cases:
 *  • Parallel vectors → zero rotation → [0, 0, 0]
 *  • Anti-parallel vectors → 180° rotation around an arbitrary perpendicular
 */
export function boneDirToAxisAngle(
  parentPos: Vec3,
  childPos: Vec3,
  parentTPose: Vec3,
  childTPose: Vec3,
): Float32Array {
  const dTPose    = vec3Normalize(vec3Sub(childTPose, parentTPose));
  const dDetected = vec3Normalize(vec3Sub(childPos, parentPos));

  const dotVal = Math.min(1, Math.max(-1, vec3Dot(dTPose, dDetected)));
  const theta  = Math.acos(dotVal);

  const out = new Float32Array(3);

  if (theta < 1e-6) {
    // Already aligned
    return out;
  }

  let axis: Vec3;
  if (Math.PI - theta < 1e-4) {
    // Anti-parallel — choose an arbitrary perpendicular axis
    const arb: Vec3 = Math.abs(dTPose[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    axis = vec3Normalize(vec3Cross(dTPose, arb));
  } else {
    axis = vec3Normalize(vec3Cross(dTPose, dDetected));
  }

  out[0] = axis[0] * theta;
  out[1] = axis[1] * theta;
  out[2] = axis[2] * theta;
  return out;
}

// ---------------------------------------------------------------------------
// BlazePose landmark → 3-D position helper
// ---------------------------------------------------------------------------

interface BPLandmark {
  x: number;   // normalised [0,1]
  y: number;   // normalised [0,1]
  z: number;   // depth (relative, same scale as x/y)
  score?: number;
  visibility?: number;
  name?: string;
}

function lmPos(lms: BPLandmark[], idx: number): Vec3 {
  const lm = lms[idx];
  if (!lm) return [0, 0, 0];
  // BlazePose image-space: y=0 at top, y=1 at bottom.
  // SMPL world-space:      y=0 at feet, y increases upward.
  // Flip Y so bone directions match the world-up convention before computing axis-angles.
  return [lm.x, 1.0 - lm.y, lm.z ?? 0];
}

function lmScore(lms: BPLandmark[], idx: number): number {
  const lm = lms[idx];
  if (!lm) return 0;
  return lm.score ?? lm.visibility ?? 0;
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return vec3Lerp(a, b, 0.5);
}

function extrapolate(from: Vec3, to: Vec3, factor = 0.3): Vec3 {
  // Extend `to` by `factor` times the from→to segment
  return vec3Add(to, vec3Scale(vec3Sub(to, from), factor));
}

// ---------------------------------------------------------------------------
// Map BlazePose 33 landmarks → SMPL 24 joint positions
// ---------------------------------------------------------------------------

export interface SMPLJointMap {
  positions: Vec3[];    // [24] detected 3-D positions (normalised image space)
  scores: number[];     // [24] per-joint confidence
}

/**
 * Project 33 BlazePose landmarks to the 24 SMPL joint positions using the
 * anatomical mapping described in the SMPL joint spec.
 *
 * Coordinates are in normalised image space [0, 1] for x/y and a relative
 * depth for z (same scale as BlazePose).  Unknown joints receive score=0
 * and their position is interpolated or set to zero.
 */
export function mapBlazePoseToSMPL(landmarks: BPLandmark[]): SMPLJointMap {
  const positions: Vec3[] = new Array(24).fill([0, 0, 0] as Vec3);
  const scores: number[] = new Array(24).fill(0);

  // Convenience shortcuts
  const lhip  = lmPos(landmarks, BP.left_hip);
  const rhip  = lmPos(landmarks, BP.right_hip);
  const lsho  = lmPos(landmarks, BP.left_shoulder);
  const rsho  = lmPos(landmarks, BP.right_shoulder);
  const lelb  = lmPos(landmarks, BP.left_elbow);
  const relb  = lmPos(landmarks, BP.right_elbow);
  const lwri  = lmPos(landmarks, BP.left_wrist);
  const rwri  = lmPos(landmarks, BP.right_wrist);
  const lkne  = lmPos(landmarks, BP.left_knee);
  const rkne  = lmPos(landmarks, BP.right_knee);
  const lank  = lmPos(landmarks, BP.left_ankle);
  const rank  = lmPos(landmarks, BP.right_ankle);
  const nosep = lmPos(landmarks, BP.nose);

  // Composite positions
  const pelvis  = midpoint(lhip, rhip);
  const spine3  = midpoint(lsho, rsho); // ~shoulders = top-of-torso
  const spine1  = vec3Lerp(pelvis, spine3, 0.3);
  const spine2  = vec3Lerp(pelvis, spine3, 0.6);
  const neck    = midpoint(lsho, rsho);
  const lcollar = midpoint(spine3, lsho);
  const rcollar = midpoint(spine3, rsho);
  const lhand   = extrapolate(lelb, lwri, 0.3);
  const rhand   = extrapolate(relb, rwri, 0.3);
  const lfoot   = extrapolate(lkne, lank, 0.3);
  const rfoot   = extrapolate(rkne, rank, 0.3);

  // Per-joint scores
  const lhipS  = lmScore(landmarks, BP.left_hip);
  const rhipS  = lmScore(landmarks, BP.right_hip);
  const lshoS  = lmScore(landmarks, BP.left_shoulder);
  const rshoS  = lmScore(landmarks, BP.right_shoulder);
  const lelbS  = lmScore(landmarks, BP.left_elbow);
  const relbS  = lmScore(landmarks, BP.right_elbow);
  const lwriS  = lmScore(landmarks, BP.left_wrist);
  const rwriS  = lmScore(landmarks, BP.right_wrist);
  const lkneS  = lmScore(landmarks, BP.left_knee);
  const rkneS  = lmScore(landmarks, BP.right_knee);
  const lankS  = lmScore(landmarks, BP.left_ankle);
  const rankS  = lmScore(landmarks, BP.right_ankle);
  const noseS  = lmScore(landmarks, BP.nose);

  // Assign SMPL joints
  //  0  pelvis
  positions[0] = pelvis;  scores[0] = (lhipS + rhipS) / 2;
  //  1  L_hip
  positions[1] = lhip;    scores[1] = lhipS;
  //  2  R_hip
  positions[2] = rhip;    scores[2] = rhipS;
  //  3  spine1
  positions[3] = spine1;  scores[3] = (lhipS + rhipS + lshoS + rshoS) / 4;
  //  4  L_knee
  positions[4] = lkne;    scores[4] = lkneS;
  //  5  R_knee
  positions[5] = rkne;    scores[5] = rkneS;
  //  6  spine2
  positions[6] = spine2;  scores[6] = (lhipS + rhipS + lshoS + rshoS) / 4;
  //  7  L_ankle
  positions[7] = lank;    scores[7] = lankS;
  //  8  R_ankle
  positions[8] = rank;    scores[8] = rankS;
  //  9  spine3
  positions[9] = spine3;  scores[9] = (lshoS + rshoS) / 2;
  // 10  L_foot  (extrapolated)
  positions[10] = lfoot;  scores[10] = lankS * 0.7;
  // 11  R_foot  (extrapolated)
  positions[11] = rfoot;  scores[11] = rankS * 0.7;
  // 12  neck
  positions[12] = neck;   scores[12] = (lshoS + rshoS) / 2;
  // 13  L_collar
  positions[13] = lcollar; scores[13] = (lshoS + rshoS) / 2;
  // 14  R_collar
  positions[14] = rcollar; scores[14] = (lshoS + rshoS) / 2;
  // 15  head (nose)
  positions[15] = nosep;  scores[15] = noseS;
  // 16  L_shoulder
  positions[16] = lsho;   scores[16] = lshoS;
  // 17  R_shoulder
  positions[17] = rsho;   scores[17] = rshoS;
  // 18  L_elbow
  positions[18] = lelb;   scores[18] = lelbS;
  // 19  R_elbow
  positions[19] = relb;   scores[19] = relbS;
  // 20  L_wrist
  positions[20] = lwri;   scores[20] = lwriS;
  // 21  R_wrist
  positions[21] = rwri;   scores[21] = rwriS;
  // 22  L_hand  (extrapolated)
  positions[22] = lhand;  scores[22] = lwriS * 0.7;
  // 23  R_hand  (extrapolated)
  positions[23] = rhand;  scores[23] = rwriS * 0.7;

  return { positions, scores };
}

// ---------------------------------------------------------------------------
// Pose axis-angles from joint positions
// ---------------------------------------------------------------------------

/**
 * For each of the 24 SMPL joints, compute the axis-angle that rotates the
 * T-pose bone direction toward the detected bone direction.
 *
 * The root joint (pelvis, index 0) gets a zero rotation since BlazePose
 * output is already in a consistent camera-aligned frame.
 *
 * Returns a [72] Float32Array containing the 24×3 axis-angles.
 */
export function computePoseAxisAngles(jointMap: SMPLJointMap): Float32Array {
  const pose = new Float32Array(72);

  for (let j = 1; j < 24; j++) {
    const parentIdx = SMPL_PARENTS[j] ?? -1;
    if (parentIdx < 0) continue;

    const parentScore = jointMap.scores[parentIdx] ?? 0;
    const childScore  = jointMap.scores[j] ?? 0;

    // Skip joints with low confidence — axis-angles stay at zero
    if (parentScore < 0.2 || childScore < 0.2) continue;

    const parentPos  = jointMap.positions[parentIdx] as Vec3;
    const childPos   = jointMap.positions[j] as Vec3;
    const parentTPose = SMPL_TPOSE[parentIdx] as Vec3;
    const childTPose  = SMPL_TPOSE[j] as Vec3;

    const aa = boneDirToAxisAngle(parentPos, childPos, parentTPose, childTPose);
    pose[j * 3]     = aa[0]!;
    pose[j * 3 + 1] = aa[1]!;
    pose[j * 3 + 2] = aa[2]!;
  }

  return pose;
}

// ---------------------------------------------------------------------------
// Weak-perspective camera estimation
// ---------------------------------------------------------------------------

/**
 * Estimate a weak-perspective camera from the pelvis position and hip width.
 *
 * Assumes the real inter-hip distance is ~20 cm.
 *
 *   scale = 0.2 / hip_width_normalised
 *   tx    = (pelvis_x − 0.5) × 2
 *   ty    = (pelvis_y − 0.5) × 2
 */
export function estimateCamera(
  lhip: Vec3,
  rhip: Vec3,
  pelvis: Vec3,
): Float32Array {
  const cam = new Float32Array(3);

  const hipDx = lhip[0] - rhip[0];
  const hipDy = lhip[1] - rhip[1];
  const hipWidthNorm = Math.sqrt(hipDx * hipDx + hipDy * hipDy);

  // Avoid division by zero with a sensible default scale
  cam[0] = hipWidthNorm > 0.01 ? 0.2 / hipWidthNorm : 1.0;
  cam[1] = (pelvis[0] - 0.5) * 2;
  cam[2] = (pelvis[1] - 0.5) * 2;

  return cam;
}

// ---------------------------------------------------------------------------
// Key-joint confidence score
// ---------------------------------------------------------------------------

/** Structural joints used for overall confidence (hips, shoulders, knees). */
const CONFIDENCE_JOINTS = [1, 2, 4, 5, 9, 16, 17] as const;

export function overallConfidence(scores: number[]): number {
  let sum = 0;
  for (const j of CONFIDENCE_JOINTS) {
    sum += scores[j] ?? 0;
  }
  return sum / CONFIDENCE_JOINTS.length;
}

// ---------------------------------------------------------------------------
// Lazy tf.js backend initialisation + pose-detection import
// ---------------------------------------------------------------------------

let _backendReady = false;

/**
 * Ensure tf.js has a usable backend before creating any detector.
 * Tries WebGL first (GPU-accelerated), falls back to CPU.
 */
async function ensureTfBackend(): Promise<void> {
  if (_backendReady) return;
  try {
    const tf = await import('@tensorflow/tfjs-core');
    // Import backends so they register themselves
    await import('@tensorflow/tfjs-backend-webgl');
    const current = tf.getBackend();
    if (!current || current === '') {
      await tf.setBackend('webgl');
    }
    await tf.ready();
    console.log('[ROMPNode] tf.js backend ready:', tf.getBackend());
    _backendReady = true;
  } catch (e) {
    console.warn('[ROMPNode] WebGL backend failed, trying CPU fallback:', e);
    try {
      const tf = await import('@tensorflow/tfjs-core');
      await import('@tensorflow/tfjs-backend-cpu');
      await tf.setBackend('cpu');
      await tf.ready();
      console.log('[ROMPNode] tf.js CPU backend ready');
      _backendReady = true;
    } catch (e2) {
      console.warn('[ROMPNode] tf.js backend init failed entirely:', e2);
      // Proceed anyway — createDetector may still work if a backend was
      // registered elsewhere (e.g. via a bundled tfjs import).
    }
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PoseDetectionModule = any;
let _pdModule: PoseDetectionModule | null = null;
let _pdLoadError = false;

async function getPoseDetection(): Promise<PoseDetectionModule | null> {
  if (_pdLoadError) return null;
  if (_pdModule) return _pdModule;
  try {
    _pdModule = await import('@tensorflow-models/pose-detection');
  } catch {
    _pdLoadError = true;
    console.warn('[ROMPNode] @tensorflow-models/pose-detection not available — will use fallback');
    return null;
  }
  return _pdModule;
}

// ---------------------------------------------------------------------------
// ROMPNode
// ---------------------------------------------------------------------------

export class ROMPNode {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _detector: any | null = null;
  private _loading = false;
  private _loadError: string | null = null;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Load the BlazePose GHUM Full model (~10 MB download on first call).
   * Safe to call multiple times — subsequent calls are no-ops once loaded.
   * If @tensorflow-models/pose-detection is unavailable, load() resolves
   * without throwing so that callers degrade gracefully.
   */
  async load(): Promise<void> {
    if (this._detector !== null) return;
    if (this._loading) return;
    if (this._loadError !== null) throw new Error(this._loadError);

    this._loading = true;
    try {
      await ensureTfBackend();
      const pd = await getPoseDetection();
      if (!pd) {
        // Package not installed — treat as a soft failure; estimate() will
        // return the zeros fallback.
        return;
      }

      this._detector = await pd.createDetector(
        pd.SupportedModels.BlazePose,
        {
          runtime: 'tfjs',
          modelType: 'full',
          enableSmoothing: false,
        },
      );

      console.log('[ROMPNode] BlazePose detector ready');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._loadError = msg;
      console.warn('[ROMPNode] Failed to load BlazePose:', msg);
      throw err;
    } finally {
      this._loading = false;
    }
  }

  /** Returns true once the BlazePose detector is ready to run inference. */
  isLoaded(): boolean {
    return this._detector !== null;
  }

  /**
   * Estimate SMPL parameters for the dominant person in the image using
   * BlazePose 3-D landmarks.
   *
   * @param image  Any-size image source (ImageData or HTMLCanvasElement).
   * @returns      ROMPOutput on success, or null if no person detected
   *               (confidence < 0.2) or the detector is not loaded.
   */
  async estimate(image: ImageData | HTMLCanvasElement): Promise<ROMPOutput | null> {
    // No detector loaded — return zeros-fallback with zero confidence
    if (this._detector === null) {
      return this._zeroFallback();
    }

    let poses: Array<{ keypoints3D?: BPLandmark[]; keypoints?: BPLandmark[] }>;
    try {
      // BlazePose accepts HTMLVideoElement, HTMLImageElement, HTMLCanvasElement,
      // ImageData, or ImageBitmap.  ImageData is supported directly.
      poses = await this._detector.estimatePoses(image, {
        flipHorizontal: false,
      });
    } catch (err) {
      console.warn('[ROMPNode] estimatePoses() failed:', err);
      return null;
    }

    if (!poses || poses.length === 0) {
      console.debug('[ROMPNode] no poses detected');
      return null;
    }

    const pose3D = poses[0]?.keypoints3D ?? poses[0]?.keypoints;
    if (!pose3D || pose3D.length < 29) return null;

    return this._processLandmarks(pose3D as BPLandmark[]);
  }

  /**
   * Release the BlazePose detector and free associated tf.js tensors.
   * The node can be re-loaded after disposal by calling load() again.
   */
  dispose(): void {
    try {
      this._detector?.dispose?.();
    } catch {
      // ignore
    }
    this._detector  = null;
    this._loadError = null;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private _processLandmarks(landmarks: BPLandmark[]): ROMPOutput | null {
    const jointMap = mapBlazePoseToSMPL(landmarks);
    const conf = overallConfidence(jointMap.scores);

    console.debug(`[ROMPNode] detected ${landmarks.length} landmarks, confidence=${conf.toFixed(3)}`);

    if (conf < 0.2) {
      console.debug('[ROMPNode] confidence too low, skipping');
      return null;
    }

    const pose = computePoseAxisAngles(jointMap);

    const lhip   = jointMap.positions[1] as Vec3;
    const rhip   = jointMap.positions[2] as Vec3;
    const pelvis = jointMap.positions[0] as Vec3;
    const cam    = estimateCamera(lhip, rhip, pelvis);

    return {
      pose,
      betas:      new Float32Array(10), // zeros
      cam,
      confidence: conf,
    };
  }

  /** Returns a zero-confidence fallback when the detector is unavailable. */
  private _zeroFallback(): ROMPOutput {
    return {
      pose:       new Float32Array(72),
      betas:      new Float32Array(10),
      cam:        new Float32Array(3),
      confidence: 0,
    };
  }
}

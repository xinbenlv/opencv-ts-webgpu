/**
 * SMPL skeleton projection and canvas rendering for the realtime Node B panel.
 *
 * Pure functions (`projectJoints`, `boneColor`) are exported separately so they
 * can be unit-tested without a DOM.  `renderSkeletonToCanvas` composes them with
 * optional video-frame background drawing.
 */

// ---------------------------------------------------------------------------
// SMPL skeleton constants
// ---------------------------------------------------------------------------

/** Parent joint index for each of the 24 SMPL joints (-1 = root). */
export const SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21] as const;

const JOINT_NAMES = [
  'pelvis','l_hip','r_hip','spine1','l_knee','r_knee','spine2',
  'l_ankle','r_ankle','spine3','l_foot','r_foot','neck','l_collar','r_collar',
  'head','l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist',
  'l_hand','r_hand',
] as const;

const COLOR_TORSO = '#4ade80';
const COLOR_LEFT  = '#60a5fa';
const COLOR_RIGHT = '#f87171';

/** Returns the display colour for a joint/bone by index. */
export function boneColor(i: number): string {
  const name = JOINT_NAMES[i] ?? '';
  if (name.startsWith('l_')) return COLOR_LEFT;
  if (name.startsWith('r_')) return COLOR_RIGHT;
  return COLOR_TORSO;
}

/**
 * T-pose reference positions, normalised [0–1] in canvas space.
 * Used when joints are all-zero (no detection yet).
 */
export const TPOSE: readonly [number, number][] = [
  [0.50, 0.45], // 0  pelvis
  [0.44, 0.48], // 1  l_hip
  [0.56, 0.48], // 2  r_hip
  [0.50, 0.38], // 3  spine1
  [0.44, 0.62], // 4  l_knee
  [0.56, 0.62], // 5  r_knee
  [0.50, 0.32], // 6  spine2
  [0.44, 0.78], // 7  l_ankle
  [0.56, 0.78], // 8  r_ankle
  [0.50, 0.26], // 9  spine3
  [0.44, 0.84], // 10 l_foot
  [0.56, 0.84], // 11 r_foot
  [0.50, 0.20], // 12 neck
  [0.46, 0.22], // 13 l_collar
  [0.54, 0.22], // 14 r_collar
  [0.50, 0.12], // 15 head
  [0.36, 0.22], // 16 l_shoulder
  [0.64, 0.22], // 17 r_shoulder
  [0.28, 0.32], // 18 l_elbow
  [0.72, 0.32], // 19 r_elbow
  [0.22, 0.42], // 20 l_wrist
  [0.78, 0.42], // 21 r_wrist
  [0.20, 0.45], // 22 l_hand
  [0.80, 0.45], // 23 r_hand
];

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

export type JointMode = 'tpose' | 'tracked' | 'abstract';

/**
 * Project 24 SMPL FK joint positions (world-space metres) to 2D canvas pixels.
 *
 * @param joints  Float32Array [72] — 24 joints × (x, y, z) in SMPL world space
 * @param cam     Float32Array [3]  — weak-perspective camera [scale, tx, ty]
 * @param w       Canvas width in pixels
 * @param h       Canvas height in pixels
 * @returns       Array of 24 [px, py] pixel coordinates and the mode used
 */
export function projectJoints(
  joints: Float32Array,
  cam: Float32Array,
  w: number,
  h: number,
): { positions: [number, number][]; mode: JointMode } {
  // Detect all-zero joints → fall back to T-pose reference
  let sumAbs = 0;
  for (let i = 0; i < Math.min(joints.length, 72); i++) sumAbs += Math.abs(joints[i]!);
  const allZero = sumAbs < 0.01;

  const camScale = cam[0] ?? 0;
  const camTx    = cam[1] ?? 0;
  const camTy    = cam[2] ?? 0;
  const haveCamera = camScale > 0.05;

  const positions: [number, number][] = [];

  if (allZero) {
    for (let j = 0; j < 24; j++) {
      positions.push([TPOSE[j]![0] * w, TPOSE[j]![1] * h]);
    }
    return { positions, mode: 'tpose' };
  }

  if (haveCamera) {
    // Weak-perspective projection.
    // FK joints are in absolute SMPL world space (pelvis ≈ Y=0.9 m), so subtract
    // the FK pelvis to make joints pelvis-relative before applying scale.
    //   px = (s*(Jx - pelvisX) + tx) * w/2 + w/2
    //   py = (-s*(Jy - pelvisY) + ty) * h/2 + h/2   (world Y-up → canvas Y-down)
    const pelvisX = joints[0]!;
    const pelvisY = joints[1]!;
    for (let j = 0; j < 24; j++) {
      const Jx = joints[j * 3]! - pelvisX;
      const Jy = joints[j * 3 + 1]! - pelvisY;
      const px = (camScale * Jx + camTx) * (w / 2) + w / 2;
      const py = (-camScale * Jy + camTy) * (h / 2) + h / 2;
      positions.push([px, py]);
    }
    return { positions, mode: 'tracked' };
  }

  // No camera estimate — auto-scale to fill canvas (abstract skeleton view)
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let j = 0; j < 24; j++) {
    const jx = joints[j * 3]!;
    const jy = joints[j * 3 + 1]!;
    if (jx < minX) minX = jx; if (jx > maxX) maxX = jx;
    if (jy < minY) minY = jy; if (jy > maxY) maxY = jy;
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const scale = Math.min(w, h) * 0.7 / (Math.max(maxX - minX, maxY - minY) || 1);
  for (let j = 0; j < 24; j++) {
    const x = (joints[j * 3]! - cx) * scale + w * 0.5;
    const y = -(joints[j * 3 + 1]! - cy) * scale + h * 0.5;
    positions.push([x, y]);
  }
  return { positions, mode: 'abstract' };
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

/** Draw the skeleton (bones + joints) onto a 2D context. */
export function drawSkeleton(
  positions: [number, number][],
  ctx2d: CanvasRenderingContext2D,
  mode: JointMode,
): void {
  // Bones
  ctx2d.lineWidth = 3;
  for (let j = 1; j < 24; j++) {
    const parent = SMPL_PARENTS[j]!;
    ctx2d.strokeStyle = boneColor(j);
    ctx2d.beginPath();
    ctx2d.moveTo(positions[parent]![0], positions[parent]![1]);
    ctx2d.lineTo(positions[j]![0], positions[j]![1]);
    ctx2d.stroke();
  }

  // Joints
  for (let j = 0; j < 24; j++) {
    const [x, y] = positions[j]!;
    ctx2d.fillStyle = j === 15 ? '#fbbf24' : boneColor(j); // head = yellow
    ctx2d.beginPath();
    ctx2d.arc(x, y, j === 15 ? 8 : 4, 0, Math.PI * 2);
    ctx2d.fill();
    ctx2d.strokeStyle = '#000';
    ctx2d.lineWidth = 1;
    ctx2d.stroke();
  }

  // Mode label
  const LABELS: Record<JointMode, string> = {
    tpose:    'SMPL T-pose (no detection)',
    tracked:  'SMPL Joints — tracked',
    abstract: 'SMPL Joints (abstract)',
  };
  const COLORS: Record<JointMode, string> = {
    tpose: '#666', tracked: '#4ade80', abstract: '#fbbf24',
  };
  const h = ctx2d.canvas.height;
  ctx2d.fillStyle = COLORS[mode];
  ctx2d.font = '11px system-ui';
  ctx2d.textAlign = 'left';
  ctx2d.fillText(LABELS[mode], 8, h - 8);
}

/**
 * Full composite: draw optional video-frame background, then overlay skeleton.
 *
 * @param videoEl  Optional video element for background; pass null in tests.
 */
export function renderSkeletonToCanvas(
  joints: Float32Array,
  cam: Float32Array,
  ctx2d: CanvasRenderingContext2D,
  w: number,
  h: number,
  videoEl: HTMLVideoElement | null,
): void {
  // Video background
  if (videoEl && videoEl.readyState >= 2) {
    const vW = videoEl.videoWidth || w;
    const vH = videoEl.videoHeight || h;
    const srcAR = vW / vH;
    const dstAR = w / h;
    let sx = 0, sy = 0, sw = vW, sh = vH;
    if (Math.abs(srcAR - dstAR) > 0.01) {
      if (srcAR > dstAR) { sw = Math.round(vH * dstAR); sx = Math.round((vW - sw) / 2); }
      else               { sh = Math.round(vW / dstAR); sy = Math.round((vH - sh) / 2); }
    }
    ctx2d.drawImage(videoEl, sx, sy, sw, sh, 0, 0, w, h);
    ctx2d.fillStyle = 'rgba(0,0,0,0.35)';
    ctx2d.fillRect(0, 0, w, h);
  }

  const { positions, mode } = projectJoints(joints, cam, w, h);
  drawSkeleton(positions, ctx2d, mode);
}

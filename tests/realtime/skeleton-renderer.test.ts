/**
 * Unit tests for demo/realtime/skeleton-renderer.ts
 *
 * All tests run in Node.js (vitest) — no browser / DOM required.
 * Coverage:
 *  1.  boneColor — left joints return blue
 *  2.  boneColor — right joints return red
 *  3.  boneColor — torso/spine joints return green
 *  4.  projectJoints — all-zero joints → tpose mode
 *  5.  projectJoints — tpose positions are scaled to canvas size
 *  6.  projectJoints — detected joints with camera → tracked mode
 *  7.  projectJoints — pelvis (joint 0) maps to camTx/camTy screen position
 *  8.  projectJoints — head (joint 15) is above pelvis in tracked mode
 *  9.  projectJoints — camScale = 0 (no camera) → abstract mode
 * 10.  projectJoints — abstract mode centres the skeleton
 * 11.  projectJoints — always returns exactly 24 positions
 */

import { describe, it, expect } from 'vitest';
import { boneColor, projectJoints, TPOSE, SMPL_PARENTS } from '../../demo/realtime/skeleton-renderer.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a Float32Array of 24 SMPL T-pose joint positions (world-space metres). */
function tposeJoints(): Float32Array {
  // Approximate SMPL neutral-pose positions (same as SMPL_TPOSE in romp.node.ts)
  const raw: [number, number, number][] = [
    [ 0.000, 0.900, 0.000], // 0  pelvis
    [-0.100, 0.850, 0.000], // 1  l_hip
    [ 0.100, 0.850, 0.000], // 2  r_hip
    [ 0.000, 1.000, 0.000], // 3  spine1
    [-0.100, 0.500, 0.000], // 4  l_knee
    [ 0.100, 0.500, 0.000], // 5  r_knee
    [ 0.000, 1.100, 0.000], // 6  spine2
    [-0.100, 0.100, 0.000], // 7  l_ankle
    [ 0.100, 0.100, 0.000], // 8  r_ankle
    [ 0.000, 1.200, 0.000], // 9  spine3
    [-0.100, 0.000, 0.050], // 10 l_foot
    [ 0.100, 0.000, 0.050], // 11 r_foot
    [ 0.000, 1.400, 0.000], // 12 neck
    [-0.080, 1.350, 0.000], // 13 l_collar
    [ 0.080, 1.350, 0.000], // 14 r_collar
    [ 0.000, 1.600, 0.000], // 15 head
    [-0.180, 1.350, 0.000], // 16 l_shoulder
    [ 0.180, 1.350, 0.000], // 17 r_shoulder
    [-0.400, 1.350, 0.000], // 18 l_elbow
    [ 0.400, 1.350, 0.000], // 19 r_elbow
    [-0.600, 1.350, 0.000], // 20 l_wrist
    [ 0.600, 1.350, 0.000], // 21 r_wrist
    [-0.700, 1.350, 0.020], // 22 l_hand
    [ 0.700, 1.350, 0.020], // 23 r_hand
  ];
  const out = new Float32Array(72);
  raw.forEach(([x, y, z], i) => { out[i*3] = x; out[i*3+1] = y; out[i*3+2] = z!; });
  return out;
}

function zeroCam(): Float32Array { return new Float32Array(3); }
function trackedCam(scale = 1.5, tx = 0, ty = 0): Float32Array {
  return new Float32Array([scale, tx, ty]);
}

// ---------------------------------------------------------------------------
// boneColor
// ---------------------------------------------------------------------------

describe('boneColor', () => {
  it('returns blue for left-side joints', () => {
    // Joint 1 = l_hip, 4 = l_knee, 16 = l_shoulder
    for (const i of [1, 4, 7, 10, 13, 16, 18, 20, 22]) {
      expect(boneColor(i)).toBe('#60a5fa');
    }
  });

  it('returns red for right-side joints', () => {
    for (const i of [2, 5, 8, 11, 14, 17, 19, 21, 23]) {
      expect(boneColor(i)).toBe('#f87171');
    }
  });

  it('returns green for torso/spine joints', () => {
    for (const i of [0, 3, 6, 9, 12, 15]) {
      expect(boneColor(i)).toBe('#4ade80');
    }
  });
});

// ---------------------------------------------------------------------------
// projectJoints — tpose mode
// ---------------------------------------------------------------------------

describe('projectJoints — tpose mode', () => {
  it('returns tpose mode when joints are all zero', () => {
    const { mode } = projectJoints(new Float32Array(72), zeroCam(), 400, 400);
    expect(mode).toBe('tpose');
  });

  it('returns exactly 24 positions', () => {
    const { positions } = projectJoints(new Float32Array(72), zeroCam(), 400, 400);
    expect(positions).toHaveLength(24);
  });

  it('scales tpose positions to canvas size', () => {
    const W = 800, H = 600;
    const { positions } = projectJoints(new Float32Array(72), zeroCam(), W, H);
    // Pelvis (joint 0) should be at TPOSE[0] * canvas size
    expect(positions[0]![0]).toBeCloseTo(TPOSE[0]![0] * W, 1);
    expect(positions[0]![1]).toBeCloseTo(TPOSE[0]![1] * H, 1);
    // Head (joint 15)
    expect(positions[15]![0]).toBeCloseTo(TPOSE[15]![0] * W, 1);
    expect(positions[15]![1]).toBeCloseTo(TPOSE[15]![1] * H, 1);
  });
});

// ---------------------------------------------------------------------------
// projectJoints — tracked mode
// ---------------------------------------------------------------------------

describe('projectJoints — tracked mode', () => {
  it('returns tracked mode when camScale > 0.05', () => {
    const { mode } = projectJoints(tposeJoints(), trackedCam(), 400, 400);
    expect(mode).toBe('tracked');
  });

  it('pelvis (joint 0) maps to camTx/camTy screen position', () => {
    const W = 400, H = 400;
    // tx=0, ty=0 → pelvis should be at canvas centre
    const { positions } = projectJoints(tposeJoints(), trackedCam(1.5, 0, 0), W, H);
    expect(positions[0]![0]).toBeCloseTo(W / 2, 1);
    expect(positions[0]![1]).toBeCloseTo(H / 2, 1);
  });

  it('pelvis maps to the correct offset when tx/ty are non-zero', () => {
    const W = 400, H = 400;
    // tx=0.5 (25% right of centre), ty=-0.5 (25% above centre)
    const { positions } = projectJoints(tposeJoints(), trackedCam(1.5, 0.5, -0.5), W, H);
    expect(positions[0]![0]).toBeCloseTo(W / 2 + 0.5 * W / 2, 1); // 300
    expect(positions[0]![1]).toBeCloseTo(H / 2 - 0.5 * H / 2, 1); // 100
  });

  it('head (joint 15) is above pelvis (lower py) in tracked mode', () => {
    const { positions } = projectJoints(tposeJoints(), trackedCam(1.5, 0, 0), 400, 400);
    // Canvas Y-down: head should have smaller py than pelvis
    expect(positions[15]![1]).toBeLessThan(positions[0]![1]);
  });

  it('left hip (joint 1) is left of right hip (joint 2)', () => {
    const { positions } = projectJoints(tposeJoints(), trackedCam(1.5, 0, 0), 400, 400);
    expect(positions[1]![0]).toBeLessThan(positions[2]![0]);
  });
});

// ---------------------------------------------------------------------------
// projectJoints — abstract mode
// ---------------------------------------------------------------------------

describe('projectJoints — abstract mode', () => {
  it('returns abstract mode when camScale ≤ 0.05', () => {
    const cam = new Float32Array([0.0, 0.0, 0.0]);
    const { mode } = projectJoints(tposeJoints(), cam, 400, 400);
    expect(mode).toBe('abstract');
  });

  it('centres the skeleton on the canvas', () => {
    const W = 400, H = 400;
    const cam = new Float32Array([0.0, 0.0, 0.0]);
    const { positions } = projectJoints(tposeJoints(), cam, W, H);
    const xs = positions.map(p => p[0]);
    const ys = positions.map(p => p[1]);
    const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
    expect(midX).toBeCloseTo(W / 2, 0);
    expect(midY).toBeCloseTo(H / 2, 0);
  });

  it('always returns exactly 24 positions', () => {
    const cam = new Float32Array([0.0, 0.0, 0.0]);
    const { positions } = projectJoints(tposeJoints(), cam, 400, 400);
    expect(positions).toHaveLength(24);
  });
});

// ---------------------------------------------------------------------------
// SMPL_PARENTS sanity
// ---------------------------------------------------------------------------

describe('SMPL_PARENTS', () => {
  it('has exactly 24 entries', () => {
    expect(SMPL_PARENTS).toHaveLength(24);
  });

  it('root joint (index 0) has parent -1', () => {
    expect(SMPL_PARENTS[0]).toBe(-1);
  });

  it('all non-root parents are valid indices', () => {
    for (let i = 1; i < 24; i++) {
      expect(SMPL_PARENTS[i]).toBeGreaterThanOrEqual(0);
      expect(SMPL_PARENTS[i]).toBeLessThan(i); // parent always precedes child
    }
  });
});

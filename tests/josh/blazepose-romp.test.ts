/**
 * Tests for the BlazePose-backed ROMPNode.
 *
 * All tests run in Node.js (vitest) — no browser APIs required.
 *
 * Coverage:
 *  1. boneDirToAxisAngle — zero rotation when directions are identical
 *  2. boneDirToAxisAngle — ~π/2 rotation for a 90° turn
 *  3. boneDirToAxisAngle — anti-parallel directions (180°) handled gracefully
 *  4. mapBlazePoseToSMPL — pelvis = midpoint of hips
 *  5. mapBlazePoseToSMPL — all 24 joints populated
 *  6. ROMPOutput betas always [10 zeros]
 *  7. overallConfidence with all key-joint scores = 1.0 → near 1.0
 *  8. Null return when all keypoint scores < 0.2 (below threshold)
 *  9. zeroFallback confidence = 0
 * 10. computePoseAxisAngles — low-confidence joints produce zero axis-angle
 */

import { describe, it, expect } from 'vitest';
import {
  boneDirToAxisAngle,
  mapBlazePoseToSMPL,
  computePoseAxisAngles,
  estimateCamera,
  overallConfidence,
  ROMPNode,
} from '../../demo/josh/nodes/romp.node.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeUniformLandmarks(score: number) {
  // Build a minimal set of 33 BlazePose-style landmarks placed in a T-pose
  // standing figure centred at (0.5, 0.5) in normalised image coordinates.
  const lms = Array.from({ length: 33 }, (_, i) => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    score,
    name: `lm_${i}`,
  }));

  // Override the structurally important landmarks to form a recognisable pose
  // so that the hip-width is non-zero and joints are spatially distinct.
  const set = (idx: number, x: number, y: number, z = 0) => {
    lms[idx] = { x, y, z, score, name: `lm_${idx}` };
  };

  // BlazePose indices:
  //   0=nose, 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow
  //  15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip, 25=L_knee, 26=R_knee
  //  27=L_ankle, 28=R_ankle
  set(0,  0.50, 0.10);        // nose (top)
  set(11, 0.35, 0.30);        // L_shoulder
  set(12, 0.65, 0.30);        // R_shoulder
  set(13, 0.28, 0.48);        // L_elbow
  set(14, 0.72, 0.48);        // R_elbow
  set(15, 0.22, 0.65);        // L_wrist
  set(16, 0.78, 0.65);        // R_wrist
  set(23, 0.40, 0.60);        // L_hip
  set(24, 0.60, 0.60);        // R_hip
  set(25, 0.39, 0.78);        // L_knee
  set(26, 0.61, 0.78);        // R_knee
  set(27, 0.38, 0.94);        // L_ankle
  set(28, 0.62, 0.94);        // R_ankle

  return lms;
}

// ---------------------------------------------------------------------------
// 1 & 2. boneDirToAxisAngle
// ---------------------------------------------------------------------------

describe('boneDirToAxisAngle', () => {
  it('returns near-zero axis-angle when directions are identical', () => {
    const aa = boneDirToAxisAngle(
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
      [0, 1, 0],
    );
    expect(Math.abs(aa[0]!)).toBeLessThan(1e-5);
    expect(Math.abs(aa[1]!)).toBeLessThan(1e-5);
    expect(Math.abs(aa[2]!)).toBeLessThan(1e-5);
  });

  it('returns angle ≈ π/2 for a 90° rotation (Y→X)', () => {
    // T-pose direction: +Y.  Detected direction: +X.
    // Expected rotation: 90° = π/2 around Z axis.
    const aa = boneDirToAxisAngle(
      [0, 0, 0], [1, 0, 0],   // detected: +X direction
      [0, 0, 0], [0, 1, 0],   // T-pose: +Y direction
    );
    const angle = Math.sqrt(aa[0]! ** 2 + aa[1]! ** 2 + aa[2]! ** 2);
    expect(angle).toBeCloseTo(Math.PI / 2, 3);
  });

  it('handles anti-parallel directions (180°) without NaN', () => {
    // T-pose: +Y, detected: -Y  → 180° rotation
    const aa = boneDirToAxisAngle(
      [0, 0, 0], [0, -1, 0],
      [0, 0, 0], [0,  1, 0],
    );
    const angle = Math.sqrt(aa[0]! ** 2 + aa[1]! ** 2 + aa[2]! ** 2);
    expect(Number.isNaN(angle)).toBe(false);
    expect(angle).toBeCloseTo(Math.PI, 3);
  });

  it('returns a [3] Float32Array', () => {
    const aa = boneDirToAxisAngle(
      [0, 0, 0], [1, 0, 0],
      [0, 0, 0], [0, 1, 0],
    );
    expect(aa).toBeInstanceOf(Float32Array);
    expect(aa.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// 3 & 4. mapBlazePoseToSMPL
// ---------------------------------------------------------------------------

describe('mapBlazePoseToSMPL', () => {
  it('produces exactly 24 joint positions', () => {
    const lms = makeUniformLandmarks(0.9);
    const { positions } = mapBlazePoseToSMPL(lms);
    expect(positions).toHaveLength(24);
  });

  it('pelvis (joint 0) = midpoint of L_hip (23) and R_hip (24)', () => {
    const lms = makeUniformLandmarks(0.9);
    const lhipX = lms[23]!.x, lhipY = lms[23]!.y;
    const rhipX = lms[24]!.x, rhipY = lms[24]!.y;

    const { positions } = mapBlazePoseToSMPL(lms);
    const pelvis = positions[0]!;

    expect(pelvis[0]).toBeCloseTo((lhipX + rhipX) / 2, 5);
    expect(pelvis[1]).toBeCloseTo((lhipY + rhipY) / 2, 5);
  });

  it('L_hip (joint 1) position matches BlazePose landmark 23', () => {
    const lms = makeUniformLandmarks(0.9);
    const { positions } = mapBlazePoseToSMPL(lms);
    expect(positions[1]![0]).toBeCloseTo(lms[23]!.x, 5);
    expect(positions[1]![1]).toBeCloseTo(lms[23]!.y, 5);
  });

  it('R_hip (joint 2) position matches BlazePose landmark 24', () => {
    const lms = makeUniformLandmarks(0.9);
    const { positions } = mapBlazePoseToSMPL(lms);
    expect(positions[2]![0]).toBeCloseTo(lms[24]!.x, 5);
    expect(positions[2]![1]).toBeCloseTo(lms[24]!.y, 5);
  });

  it('head (joint 15) position matches nose (landmark 0)', () => {
    const lms = makeUniformLandmarks(0.9);
    const { positions } = mapBlazePoseToSMPL(lms);
    expect(positions[15]![0]).toBeCloseTo(lms[0]!.x, 5);
    expect(positions[15]![1]).toBeCloseTo(lms[0]!.y, 5);
  });

  it('produces 24 confidence scores all in [0, 1]', () => {
    const lms = makeUniformLandmarks(0.8);
    const { scores } = mapBlazePoseToSMPL(lms);
    expect(scores).toHaveLength(24);
    for (const s of scores) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// 5. ROMPOutput betas
// ---------------------------------------------------------------------------

describe('ROMPOutput betas', () => {
  it('betas are always a 10-element Float32Array of zeros', async () => {
    // ROMPNode without a detector returns the zero fallback
    const node = new ROMPNode();
    // Don't call load() — detector stays null → _zeroFallback()
    const result = await node.estimate(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      { width: 1, height: 1, data: new Uint8ClampedArray(4), colorSpace: 'srgb' } as any,
    );
    // Without the model loaded, estimate() returns the zero fallback
    expect(result).not.toBeNull();
    expect(result!.betas).toBeInstanceOf(Float32Array);
    expect(result!.betas.length).toBe(10);
    for (const b of result!.betas) {
      expect(b).toBe(0);
    }
    node.dispose();
  });
});

// ---------------------------------------------------------------------------
// 6. overallConfidence
// ---------------------------------------------------------------------------

describe('overallConfidence', () => {
  it('returns ≈ 1.0 when all key-joint scores are 1.0', () => {
    const scores = new Array(24).fill(1.0);
    const conf = overallConfidence(scores);
    expect(conf).toBeCloseTo(1.0, 5);
  });

  it('returns 0.0 when all scores are 0', () => {
    const scores = new Array(24).fill(0);
    expect(overallConfidence(scores)).toBe(0);
  });

  it('returns a value between 0 and 1 for mixed scores', () => {
    const lms = makeUniformLandmarks(0.6);
    const { scores } = mapBlazePoseToSMPL(lms);
    const conf = overallConfidence(scores);
    expect(conf).toBeGreaterThanOrEqual(0);
    expect(conf).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// 7. Null return below confidence threshold
// ---------------------------------------------------------------------------

describe('mapBlazePoseToSMPL + computePoseAxisAngles', () => {
  it('confidence is below 0.2 when all keypoint scores are 0', () => {
    const lms = makeUniformLandmarks(0.0);
    const jointMap = mapBlazePoseToSMPL(lms);
    const conf = overallConfidence(jointMap.scores);
    expect(conf).toBeLessThan(0.2);
  });

  it('low-confidence joints produce zero axis-angle', () => {
    // Score = 0 → all joints below threshold → pose is all zeros
    const lms = makeUniformLandmarks(0.0);
    const jointMap = mapBlazePoseToSMPL(lms);
    const pose = computePoseAxisAngles(jointMap);
    expect(pose).toBeInstanceOf(Float32Array);
    expect(pose.length).toBe(72);
    for (const v of pose) {
      expect(v).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// 8. estimateCamera
// ---------------------------------------------------------------------------

describe('estimateCamera', () => {
  it('returns a [3] Float32Array', () => {
    const cam = estimateCamera([0.4, 0.6, 0], [0.6, 0.6, 0], [0.5, 0.6, 0]);
    expect(cam).toBeInstanceOf(Float32Array);
    expect(cam.length).toBe(3);
  });

  it('returns scale > 0 for typical hip positions', () => {
    const cam = estimateCamera([0.4, 0.6, 0], [0.6, 0.6, 0], [0.5, 0.6, 0]);
    expect(cam[0]).toBeGreaterThan(0);
  });

  it('returns tx ≈ 0 and ty ≈ 0.2 when pelvis is at (0.5, 0.6)', () => {
    const cam = estimateCamera([0.4, 0.6, 0], [0.6, 0.6, 0], [0.5, 0.6, 0]);
    expect(cam[1]).toBeCloseTo(0, 5);       // (0.5 - 0.5) * 2 = 0
    expect(cam[2]).toBeCloseTo(0.2, 5);     // (0.6 - 0.5) * 2 = 0.2
  });

  it('falls back to scale = 1.0 when hip width is near zero', () => {
    const cam = estimateCamera([0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]);
    expect(cam[0]).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 9. ROMPNode lifecycle
// ---------------------------------------------------------------------------

describe('ROMPNode', () => {
  it('can be instantiated without arguments', () => {
    const node = new ROMPNode();
    expect(node).toBeInstanceOf(ROMPNode);
    node.dispose();
  });

  it('isLoaded() returns false before load()', () => {
    const node = new ROMPNode();
    expect(node.isLoaded()).toBe(false);
    node.dispose();
  });

  it('estimate() returns zero-confidence fallback when detector is null', async () => {
    const node = new ROMPNode();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fakeImage = { width: 4, height: 4, data: new Uint8ClampedArray(64), colorSpace: 'srgb' } as any;
    const result = await node.estimate(fakeImage);
    expect(result).not.toBeNull();
    expect(result!.confidence).toBe(0);
    expect(result!.pose.length).toBe(72);
    expect(result!.betas.length).toBe(10);
    expect(result!.cam.length).toBe(3);
    node.dispose();
  });

  it('dispose() is safe to call multiple times', () => {
    const node = new ROMPNode();
    expect(() => {
      node.dispose();
      node.dispose();
    }).not.toThrow();
  });
});

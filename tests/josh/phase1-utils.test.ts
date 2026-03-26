/**
 * Phase 1 utility tests — focal length recovery & contact detection.
 *
 * All synthetic data is fully deterministic (no random number generator).
 */

import { describe, it, expect } from 'vitest';
import {
  recoverFocalLength,
  recoverFocalShift,
} from '../../demo/josh/utils/focal-recovery.ts';
import {
  detectContacts,
  type ContactDetectionParams,
} from '../../demo/josh/utils/contact-detection.ts';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Project a single 3-D point to 2-D with a pinhole camera. */
function project(X: number, Y: number, Z: number, f: number, cx: number, cy: number): [number, number] {
  return [f * X / Z + cx, f * Y / Z + cy];
}

/** Build deterministic pseudo-random 3-D points on a hemisphere in front of camera. */
function makeSyntheticPoints(
  N: number,
  f: number,
  cx: number,
  cy: number,
  noisePx = 0,
): { pts3D: Float32Array; pts2D: Float32Array } {
  const pts3D = new Float32Array(N * 3);
  const pts2D = new Float32Array(N * 2);

  for (let i = 0; i < N; i++) {
    // Deterministic spread using sine/cosine of index
    const angle = (i / N) * 2 * Math.PI;
    const radius = 0.3 + 0.2 * Math.sin(i * 1.7);
    const X = radius * Math.cos(angle);
    const Y = radius * Math.sin(angle) * 0.6;
    const Z = 2.0 + 0.5 * Math.abs(Math.sin(i * 0.9)); // always positive

    pts3D[i * 3]     = X;
    pts3D[i * 3 + 1] = Y;
    pts3D[i * 3 + 2] = Z;

    const [u, v] = project(X, Y, Z, f, cx, cy);

    // Deterministic "noise": small sine-based perturbation
    const noiseU = noisePx * Math.sin(i * 2.3 + 0.1);
    const noiseV = noisePx * Math.cos(i * 1.9 + 0.5);

    pts2D[i * 2]     = u + noiseU;
    pts2D[i * 2 + 1] = v + noiseV;
  }

  return { pts3D, pts2D };
}

// ─────────────────────────────────────────────────────────────────────────────
// Focal Recovery Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('recoverFocalLength', () => {
  const TRUE_F  = 300;
  const W = 384;
  const H = 384;
  const CX = W / 2; // 192
  const CY = H / 2; // 192

  it('recovers f=300 exactly from perfect 2-D projections (N=20)', () => {
    const { pts3D, pts2D } = makeSyntheticPoints(20, TRUE_F, CX, CY, 0);
    const result = recoverFocalLength(pts3D, pts2D, W, H);

    const relErr = Math.abs(result.focalLength - TRUE_F) / TRUE_F;
    expect(relErr).toBeLessThan(0.01); // within 1%
    expect(result.reprojectionError).toBeLessThan(0.1); // sub-pixel
  });

  it('recovers f=300 within 5% with 1-pixel deterministic noise (N=20)', () => {
    const { pts3D, pts2D } = makeSyntheticPoints(20, TRUE_F, CX, CY, 1.0);
    const result = recoverFocalLength(pts3D, pts2D, W, H);

    const relErr = Math.abs(result.focalLength - TRUE_F) / TRUE_F;
    expect(relErr).toBeLessThan(0.05); // within 5%
  });

  it('returns a finite focal length with only 3 points', () => {
    const { pts3D, pts2D } = makeSyntheticPoints(3, TRUE_F, CX, CY, 0);
    const result = recoverFocalLength(pts3D, pts2D, W, H);

    expect(Number.isFinite(result.focalLength)).toBe(true);
    expect(result.focalLength).toBeGreaterThan(0);
  });

  it('returns depthShift close to zero when depths are correct', () => {
    const { pts3D, pts2D } = makeSyntheticPoints(20, TRUE_F, CX, CY, 0);
    const result = recoverFocalLength(pts3D, pts2D, W, H);

    // With perfect data the shift should be near zero
    expect(Math.abs(result.depthShift)).toBeLessThan(0.05);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// recoverFocalShift (standalone)
// ─────────────────────────────────────────────────────────────────────────────

describe('recoverFocalShift', () => {
  it('recovers a known depth shift of +0.5 m', () => {
    const TRUE_F = 300;
    const CX = 192;
    const CY = 192;
    const N = 20;
    const SHIFT = 0.5;

    // Build perfect projections using Z+SHIFT as the true depth
    const pts3D = new Float32Array(N * 3);
    const pts2D = new Float32Array(N * 2);

    for (let i = 0; i < N; i++) {
      const angle = (i / N) * 2 * Math.PI;
      const X = 0.3 * Math.cos(angle);
      const Y = 0.2 * Math.sin(angle);
      const Z = 1.5 + 0.3 * Math.abs(Math.sin(i));

      pts3D[i * 3]     = X;
      pts3D[i * 3 + 1] = Y;
      pts3D[i * 3 + 2] = Z - SHIFT; // stored depth is shifted down

      // Project with true Z
      const [u, v] = project(X, Y, Z, TRUE_F, CX, CY);
      pts2D[i * 2]     = u;
      pts2D[i * 2 + 1] = v;
    }

    const recovered = recoverFocalShift(pts3D, pts2D, TRUE_F, CX, CY);
    expect(Math.abs(recovered - SHIFT)).toBeLessThan(0.05);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Contact Detection Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('detectContacts', () => {
  /**
   * Camera at origin looking along +Z, simple pinhole.
   * Depth map is a flat plane at Z = 2.0 m.
   */
  const W = 64;
  const H = 64;
  const FX = 50;
  const FY = 50;
  const CX = 32;
  const CY = 32;
  const FLOOR_Z = 2.0;

  const PARAMS: ContactDetectionParams = {
    contactThresholdMeters: 0.05,
    cameraFx: FX,
    cameraFy: FY,
    cameraCx: CX,
    cameraCy: CY,
  };

  /** Flat depth map at constant depth FLOOR_Z. */
  function makeDepthMap(depth = FLOOR_Z): Float32Array {
    return new Float32Array(W * H).fill(depth);
  }

  it('all foot vertices are contacts on a flat floor (Y=0, Z=2m)', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    // 4 foot vertices in front of camera, all at Z = FLOOR_Z
    const vertices = new Float32Array([
      -0.1,  0, FLOOR_Z,   // vertex 0
       0.1,  0, FLOOR_Z,   // vertex 1
      -0.1, -0.05, FLOOR_Z, // vertex 2
       0.1, -0.05, FLOOR_Z, // vertex 3
    ]);
    const candidateIndices = [0, 1, 2, 3];

    const result = detectContacts(vertices, depthMap, W, H, candidateIndices, PARAMS);

    expect(result.isContact).toEqual([true, true, true, true]);
    expect(result.contactVertexIndices).toEqual([0, 1, 2, 3]);
  });

  it('lifted foot (Z 0.2 m away from floor) is NOT a contact', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    // Vertex is 0.2 m closer than the floor
    const vertices = new Float32Array([0, 0, FLOOR_Z - 0.2]);
    const result = detectContacts(vertices, depthMap, W, H, [0], PARAMS);

    expect(result.isContact).toEqual([false]);
    expect(result.contactVertexIndices).toHaveLength(0);
  });

  it('correctly classifies 3-on-floor, 2-lifted in a mixed set', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    // Vertices 0,1,2 are on floor; 3,4 are lifted 0.3 m
    const vertices = new Float32Array([
       0.0, 0, FLOOR_Z,          // 0: contact
       0.1, 0, FLOOR_Z,          // 1: contact
      -0.1, 0, FLOOR_Z,          // 2: contact
       0.0, 0, FLOOR_Z - 0.3,    // 3: lifted → not contact
       0.1, 0, FLOOR_Z - 0.3,    // 4: lifted → not contact
    ]);
    const candidateIndices = [0, 1, 2, 3, 4];

    const result = detectContacts(vertices, depthMap, W, H, candidateIndices, PARAMS);

    expect(result.isContact[0]).toBe(true);
    expect(result.isContact[1]).toBe(true);
    expect(result.isContact[2]).toBe(true);
    expect(result.isContact[3]).toBe(false);
    expect(result.isContact[4]).toBe(false);
    expect(result.contactVertexIndices).toEqual([0, 1, 2]);
  });

  it('vertex behind camera (Z ≤ 0) is not a contact', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    const vertices = new Float32Array([0, 0, -1.0]); // behind camera
    const result = detectContacts(vertices, depthMap, W, H, [0], PARAMS);

    expect(result.isContact).toEqual([false]);
    expect(result.contactVertexIndices).toHaveLength(0);
  });

  it('vertex that projects outside image bounds is not a contact', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    // Very large X will project far outside the 64×64 image
    const vertices = new Float32Array([1000, 0, FLOOR_Z]);
    const result = detectContacts(vertices, depthMap, W, H, [0], PARAMS);

    expect(result.isContact).toEqual([false]);
  });

  it('threshold boundary: vertex exactly at threshold distance is contact', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    // Place vertex just inside threshold
    const delta = PARAMS.contactThresholdMeters - 0.001;
    const vertices = new Float32Array([0, 0, FLOOR_Z - delta]);
    const result = detectContacts(vertices, depthMap, W, H, [0], PARAMS);

    expect(result.isContact).toEqual([true]);
  });

  it('threshold boundary: vertex just outside threshold is NOT contact', () => {
    const depthMap = makeDepthMap(FLOOR_Z);

    const delta = PARAMS.contactThresholdMeters + 0.001;
    const vertices = new Float32Array([0, 0, FLOOR_Z - delta]);
    const result = detectContacts(vertices, depthMap, W, H, [0], PARAMS);

    expect(result.isContact).toEqual([false]);
  });
});

/**
 * Unit tests for the MiDaS-backed MASt3RNode depth-unprojection logic.
 *
 * All tests are pure math — no browser APIs, no ONNX model, no fetch.
 * The unproject formula is replicated here to keep tests fast and hermetic.
 *
 * Unprojection model:
 *   Z = DEPTH_SCALE / midas_value   (MiDaS outputs relative inverse depth)
 *   X = (u - cx) * Z / f
 *   Y = (v - cy) * Z / f
 *
 * Confidence model:
 *   conf[i] = min(1, Z_i / Z_max)   (closer pixels get lower conf in [0,1])
 *   conf[i] = 0                      when Z_i == 0 (invalid depth)
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Local re-implementation of the unproject helper
// (avoids importing browser-dependent code; mirrors mast3r.node.ts exactly)
// ---------------------------------------------------------------------------

const DEPTH_SCALE = 5.0;

/**
 * Unproject a flat MiDaS depth map (relative inverse depth) to a 3-D point
 * cloud using the pinhole camera model.
 *
 * @param rawDepth  Float32Array [H*W] of relative inverse-depth values.
 * @param W         Image width in pixels.
 * @param H         Image height in pixels.
 * @param f         Focal length in pixels (fx = fy assumed).
 * @returns [pts3d, conf] — flat arrays of length H*W*3 and H*W respectively.
 */
function unprojectDepth(
  rawDepth: Float32Array,
  W: number,
  H: number,
  f: number,
): [Float32Array, Float32Array] {
  const N = H * W;
  const pts3d = new Float32Array(N * 3);
  const conf  = new Float32Array(N);

  const cx = W / 2;
  const cy = H / 2;

  // Convert inverse-depth → metric depth and find max for normalisation
  const metricDepth = new Float32Array(N);
  let maxZ = 0;
  for (let i = 0; i < N; i++) {
    const d = rawDepth[i]!;
    const Z = d > 0 ? DEPTH_SCALE / d : 0;
    metricDepth[i] = Z;
    if (Z > maxZ) maxZ = Z;
  }

  const invMaxZ = maxZ > 0 ? 1.0 / maxZ : 1.0;

  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const idx = v * W + u;
      const Z   = metricDepth[idx]!;

      if (Z <= 0) continue;

      pts3d[idx * 3]     = (u - cx) * Z / f;
      pts3d[idx * 3 + 1] = (v - cy) * Z / f;
      pts3d[idx * 3 + 2] = Z;

      conf[idx] = Math.min(1.0, Z * invMaxZ);
    }
  }

  return [pts3d, conf];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const NEAR = 1e-5; // absolute tolerance for floating-point comparisons

describe('unprojectDepth — known centre pixel', () => {
  /**
   * Pixel (256, 256) in a 512×512 image (cx=256, cy=256).
   * MiDaS inverse-depth d such that Z = DEPTH_SCALE / d = 2.0 → d = 2.5.
   *
   * Expected:
   *   X = (256 - 256) * 2.0 / 500 = 0
   *   Y = (256 - 256) * 2.0 / 500 = 0
   *   Z = 2.0
   */
  it('centre pixel (256,256) depth 2.0 f=500 → [0, 0, 2.0]', () => {
    const W = 512, H = 512;
    const N = H * W;
    const rawDepth = new Float32Array(N);

    // d = DEPTH_SCALE / desiredZ = 5.0 / 2.0 = 2.5
    const targetPixel = 256 * W + 256;
    rawDepth[targetPixel] = 2.5; // all other pixels are 0 (invalid)

    const [pts3d] = unprojectDepth(rawDepth, W, H, 500);

    const X = pts3d[targetPixel * 3];
    const Y = pts3d[targetPixel * 3 + 1];
    const Z = pts3d[targetPixel * 3 + 2];

    expect(X).toBeCloseTo(0, 5);
    expect(Y).toBeCloseTo(0, 5);
    expect(Z).toBeCloseTo(2.0, 5);
  });
});

describe('unprojectDepth — corner pixel (0, 0)', () => {
  /**
   * Pixel (0, 0) in a 512×512 image (cx=256, cy=256), depth Z=1.0, f=500.
   *
   * MiDaS inverse-depth d = DEPTH_SCALE / 1.0 = 5.0.
   *
   * Expected:
   *   X = (0 - 256) * 1.0 / 500 = -0.512
   *   Y = (0 - 256) * 1.0 / 500 = -0.512
   *   Z = 1.0
   */
  it('corner pixel (0,0) depth 1.0 f=500 512×512 → X=-0.512, Y=-0.512, Z=1.0', () => {
    const W = 512, H = 512;
    const N = H * W;
    const rawDepth = new Float32Array(N);

    // d = DEPTH_SCALE / 1.0 = 5.0
    rawDepth[0] = 5.0;

    const [pts3d] = unprojectDepth(rawDepth, W, H, 500);

    const X = pts3d[0];
    const Y = pts3d[1];
    const Z = pts3d[2];

    expect(X).toBeCloseTo(-0.512, 5);
    expect(Y).toBeCloseTo(-0.512, 5);
    expect(Z).toBeCloseTo(1.0, 5);
  });
});

describe('unprojectDepth — confidence map', () => {
  it('all non-zero depth pixels → confidence in (0, 1]', () => {
    const W = 4, H = 4;
    const N = H * W;
    const rawDepth = new Float32Array(N);

    // Fill every pixel with a valid depth value
    for (let i = 0; i < N; i++) {
      rawDepth[i] = 1.0 + i * 0.1; // varying depths
    }

    const [, conf] = unprojectDepth(rawDepth, W, H, 500);

    for (let i = 0; i < N; i++) {
      expect(conf[i]).toBeGreaterThan(0);
      expect(conf[i]).toBeLessThanOrEqual(1.0 + NEAR);
    }
  });

  it('zero-depth pixels → confidence exactly 0.0', () => {
    const W = 4, H = 4;
    const N = H * W;
    const rawDepth = new Float32Array(N); // all zeros

    const [pts3d, conf] = unprojectDepth(rawDepth, W, H, 500);

    for (let i = 0; i < N; i++) {
      expect(conf[i]).toBe(0);
      // pts3d should also be zero for invalid pixels
      expect(pts3d[i * 3]).toBe(0);
      expect(pts3d[i * 3 + 1]).toBe(0);
      expect(pts3d[i * 3 + 2]).toBe(0);
    }
  });

  it('mixed valid/invalid pixels — only valid ones have conf > 0', () => {
    const W = 2, H = 2;
    const rawDepth = new Float32Array([2.0, 0, 0, 5.0]); // pixels 0,3 valid; 1,2 invalid

    const [, conf] = unprojectDepth(rawDepth, W, H, 500);

    expect(conf[0]).toBeGreaterThan(0);
    expect(conf[1]).toBe(0);
    expect(conf[2]).toBe(0);
    expect(conf[3]).toBeGreaterThan(0);
  });
});

describe('unprojectDepth — symmetry for identical images', () => {
  /**
   * When the same MiDaS depth map is used for both image1 and image2,
   * pts3d_1 and pts3d_2 should be identical (element-wise).
   */
  it('same depth map produces identical pts3d_1 and pts3d_2', () => {
    const W = 8, H = 8;
    const N = H * W;
    const rawDepth = new Float32Array(N);

    // Checkerboard of valid/invalid depths
    for (let i = 0; i < N; i++) {
      rawDepth[i] = (i % 2 === 0) ? 3.0 + (i / N) * 2.0 : 0;
    }

    const [pts3d_1, conf_1] = unprojectDepth(rawDepth, W, H, 500);
    const [pts3d_2, conf_2] = unprojectDepth(rawDepth, W, H, 500);

    for (let i = 0; i < N * 3; i++) {
      expect(pts3d_1[i]).toBe(pts3d_2[i]);
    }
    for (let i = 0; i < N; i++) {
      expect(conf_1[i]).toBe(conf_2[i]);
    }
  });
});

describe('unprojectDepth — depth scale conversion', () => {
  /**
   * Verify that the scale factor DEPTH_SCALE = 5.0 is applied correctly.
   * For d = 1.0:  Z = 5.0 / 1.0 = 5.0
   * For d = 2.0:  Z = 5.0 / 2.0 = 2.5
   */
  it('d=1.0 → Z=5.0 (DEPTH_SCALE / d)', () => {
    const W = 1, H = 1;
    const rawDepth = new Float32Array([1.0]);
    const [pts3d] = unprojectDepth(rawDepth, W, H, 500);
    expect(pts3d[2]).toBeCloseTo(5.0, 5);
  });

  it('d=2.0 → Z=2.5 (DEPTH_SCALE / d)', () => {
    const W = 1, H = 1;
    const rawDepth = new Float32Array([2.0]);
    const [pts3d] = unprojectDepth(rawDepth, W, H, 500);
    expect(pts3d[2]).toBeCloseTo(2.5, 5);
  });

  it('d=0.5 → Z=10.0 (DEPTH_SCALE / d)', () => {
    const W = 1, H = 1;
    const rawDepth = new Float32Array([0.5]);
    const [pts3d] = unprojectDepth(rawDepth, W, H, 500);
    expect(pts3d[2]).toBeCloseTo(10.0, 5);
  });
});

describe('unprojectDepth — focal length sensitivity', () => {
  /**
   * X displacement should scale inversely with focal length.
   * Pixel (u=1, v=0) in a 2×1 image (cx=1, cy=0.5), depth Z.
   * X = (1 - 1) * Z / f = 0  (centre column has u=cx only for even W)
   *
   * Use a wider image where off-centre pixel shift is clear:
   *   W=10, H=1, cx=5, pixel u=0, depth Z via d.
   *   X = (0 - 5) * Z / f  → changes with f.
   */
  it('halving focal length doubles X displacement', () => {
    const W = 10, H = 1;
    const rawDepth = new Float32Array(W); // all zero
    // Only pixel u=0 is valid
    rawDepth[0] = DEPTH_SCALE; // d = DEPTH_SCALE → Z = 1.0

    const [pts3d_f500] = unprojectDepth(rawDepth, W, H, 500);
    const [pts3d_f250] = unprojectDepth(rawDepth, W, H, 250);

    // X_f250 should be approximately double X_f500
    const X_500 = pts3d_f500[0];
    const X_250 = pts3d_f250[0];
    expect(Math.abs(X_250)).toBeCloseTo(Math.abs(X_500) * 2, 4);
  });
});

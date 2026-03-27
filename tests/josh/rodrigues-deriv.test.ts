/**
 * Rodrigues Derivative Validation Tests
 *
 * Validates the analytical Rodrigues Jacobian (dR/dw) against numerical
 * finite differences. Runs in Node.js — no WebGPU required.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

// ---------------------------------------------------------------------------
// rodrigues(wx, wy, wz) → 3×3 rotation matrix (row-major flat array)
// ---------------------------------------------------------------------------

function rodrigues(wx: number, wy: number, wz: number): Mat3 {
  const eps = 1e-12;
  const theta2 = wx * wx + wy * wy + wz * wz;
  const theta = Math.sqrt(theta2 + eps);

  const s = Math.sin(theta) / (theta + eps);          // sinc(theta)
  const c = (1 - Math.cos(theta)) / (theta2 + eps);   // cosc(theta)

  // R = I + s*K + c*K²
  // K (skew-symmetric):
  //   [[0, -wz, wy],
  //    [wz,  0,-wx],
  //    [-wy, wx,  0]]
  //
  // K² (precomputed):
  //   [[-(wy²+wz²), wx*wy,      wx*wz    ],
  //    [ wx*wy,    -(wx²+wz²),  wy*wz    ],
  //    [ wx*wz,     wy*wz,     -(wx²+wy²)]]

  const R: Mat3 = [
    1 + c * (-(wy * wy + wz * wz)),   s * (-wz) + c * wx * wy,          s * wy + c * wx * wz,
    s * wz + c * wx * wy,              1 + c * (-(wx * wx + wz * wz)),   s * (-wx) + c * wy * wz,
    s * (-wy) + c * wx * wz,          s * wx + c * wy * wz,              1 + c * (-(wx * wx + wy * wy)),
  ];

  return R;
}

// ---------------------------------------------------------------------------
// rodriguesDeriv(wx, wy, wz) → [dR_dwx, dR_dwy, dR_dwz]  (three Mat3s)
// ---------------------------------------------------------------------------

function rodriguesDeriv(
  wx: number,
  wy: number,
  wz: number,
): [Mat3, Mat3, Mat3] {
  const eps = 1e-12;
  const theta2 = wx * wx + wy * wy + wz * wz;
  const theta = Math.sqrt(theta2 + eps);
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);

  const s = sinT / (theta + eps);
  const c = (1 - cosT) / (theta2 + eps);

  // Scalar derivatives w.r.t. theta
  const ds_dt = (theta * cosT - sinT) / (theta2 + eps);
  const dc_dt = (theta * sinT - 2 * (1 - cosT)) / (theta2 * theta + eps);

  // dtheta/dwi = wi / theta
  const dth_dwx = wx / theta;
  const dth_dwy = wy / theta;
  const dth_dwz = wz / theta;

  // Chain-rule scalars: d(s)/dwi, d(c)/dwi
  const cs_x = ds_dt * dth_dwx;
  const cs_y = ds_dt * dth_dwy;
  const cs_z = ds_dt * dth_dwz;

  const cc_x = dc_dt * dth_dwx;
  const cc_y = dc_dt * dth_dwy;
  const cc_z = dc_dt * dth_dwz;

  // K (flat row-major):
  // [0, -wz, wy, wz, 0, -wx, -wy, wx, 0]
  const K: Mat3 = [
    0,   -wz,  wy,
    wz,   0,  -wx,
    -wy,  wx,   0,
  ];

  // K² (flat row-major):
  const Ksq: Mat3 = [
    -(wy * wy + wz * wz),  wx * wy,               wx * wz,
     wx * wy,              -(wx * wx + wz * wz),   wy * wz,
     wx * wz,               wy * wz,              -(wx * wx + wy * wy),
  ];

  // dK/dwx:
  // [[0,0,0],[0,0,-1],[0,1,0]]
  const dK_dwx: Mat3 = [
    0, 0,  0,
    0, 0, -1,
    0, 1,  0,
  ];

  // dK/dwy:
  // [[0,0,1],[0,0,0],[-1,0,0]]
  const dK_dwy: Mat3 = [
    0,  0, 1,
    0,  0, 0,
    -1, 0, 0,
  ];

  // dK/dwz:
  // [[0,-1,0],[1,0,0],[0,0,0]]
  const dK_dwz: Mat3 = [
     0, -1, 0,
     1,  0, 0,
     0,  0, 0,
  ];

  // dK²/dwx:
  // [[0,wy,wz],[wy,-2wx,0],[wz,0,-2wx]]
  const dKsq_dwx: Mat3 = [
      0,     wy,      wz,
     wy,  -2*wx,       0,
     wz,      0,   -2*wx,
  ];

  // dK²/dwy:
  // [[-2wy,wx,0],[wx,0,wz],[0,wz,-2wy]]  (note: d/dwy of -(wx²+wz²) = 0, d/dwy of -(wy²+wz²) = -2wy)
  const dKsq_dwy: Mat3 = [
    -2*wy,  wx,    0,
     wx,     0,   wz,
     0,     wz,  -2*wy,
  ];

  // dK²/dwz:
  // [[-2wz,0,wx],[0,-2wz,wy],[wx,wy,0]]
  const dKsq_dwz: Mat3 = [
    -2*wz,  0,    wx,
     0,    -2*wz,  wy,
     wx,    wy,    0,
  ];

  // dR/dwi = cs_i * K + s * dK/dwi + cc_i * K² + c * dK²/dwi
  function computeDeriv(
    cs: number, cc: number, dK: Mat3, dKsq: Mat3,
  ): Mat3 {
    const out = new Array<number>(9) as unknown as Mat3;
    for (let i = 0; i < 9; i++) {
      out[i] = cs * K[i]! + s * dK[i]! + cc * Ksq[i]! + c * dKsq[i]!;
    }
    return out;
  }

  return [
    computeDeriv(cs_x, cc_x, dK_dwx, dKsq_dwx),
    computeDeriv(cs_y, cc_y, dK_dwy, dKsq_dwy),
    computeDeriv(cs_z, cc_z, dK_dwz, dKsq_dwz),
  ];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Numerical derivative of rodrigues w.r.t. axis i (0=x, 1=y, 2=z) */
function numericalDeriv(
  wx: number, wy: number, wz: number, axis: 0 | 1 | 2, eps = 1e-5,
): Mat3 {
  const w = [wx, wy, wz] as [number, number, number];

  const wPlus = [...w] as [number, number, number];
  wPlus[axis] += eps;
  const rPlus = rodrigues(...wPlus);

  const wMinus = [...w] as [number, number, number];
  wMinus[axis] -= eps;
  const rMinus = rodrigues(...wMinus);

  const out = new Array<number>(9) as unknown as Mat3;
  for (let i = 0; i < 9; i++) {
    out[i] = (rPlus[i]! - rMinus[i]!) / (2 * eps);
  }
  return out;
}

function maxAbsDiff(a: Mat3, b: Mat3): number {
  let max = 0;
  for (let i = 0; i < 9; i++) {
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
  }
  return max;
}

function mat3Det(m: Mat3): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('rodrigues()', () => {
  it('returns identity for zero rotation', () => {
    const R = rodrigues(0, 0, 0);
    expect(R[0]).toBeCloseTo(1, 5); // [0,0]
    expect(R[4]).toBeCloseTo(1, 5); // [1,1]
    expect(R[8]).toBeCloseTo(1, 5); // [2,2]
    expect(R[1]).toBeCloseTo(0, 5);
    expect(R[2]).toBeCloseTo(0, 5);
    expect(R[3]).toBeCloseTo(0, 5);
    expect(R[5]).toBeCloseTo(0, 5);
    expect(R[6]).toBeCloseTo(0, 5);
    expect(R[7]).toBeCloseTo(0, 5);
  });

  it('produces 90° rotation around Z: [[0,-1,0],[1,0,0],[0,0,1]]', () => {
    const angle = Math.PI / 2;
    const R = rodrigues(0, 0, angle);
    // Row 0: [0, -1, 0]
    expect(R[0]).toBeCloseTo(0, 4);
    expect(R[1]).toBeCloseTo(-1, 4);
    expect(R[2]).toBeCloseTo(0, 4);
    // Row 1: [1, 0, 0]
    expect(R[3]).toBeCloseTo(1, 4);
    expect(R[4]).toBeCloseTo(0, 4);
    expect(R[5]).toBeCloseTo(0, 4);
    // Row 2: [0, 0, 1]
    expect(R[6]).toBeCloseTo(0, 4);
    expect(R[7]).toBeCloseTo(0, 4);
    expect(R[8]).toBeCloseTo(1, 4);
  });

  it('produces orthogonal matrix for arbitrary rotation', () => {
    const R = rodrigues(0.3, -0.5, 0.7);
    // R^T R should equal I — check via dot products of rows
    // row0 · row0 = 1
    const r0sq = R[0]*R[0] + R[1]*R[1] + R[2]*R[2];
    expect(r0sq).toBeCloseTo(1, 5);
    // row0 · row1 = 0
    const r01 = R[0]*R[3] + R[1]*R[4] + R[2]*R[5];
    expect(r01).toBeCloseTo(0, 5);
    // det = 1
    expect(mat3Det(R)).toBeCloseTo(1, 5);
  });
});

describe('rodriguesDeriv() — analytical vs numerical', () => {
  it('zero rotation: dR should match dK (limit of s=1, c=0)', () => {
    // At theta → 0: s → 1, c → 0, so dR/dwi → dK/dwi
    // Numerical check confirms this
    const [dRdwx, dRdwy, dRdwz] = rodriguesDeriv(0, 0, 0);
    const ndx = numericalDeriv(0, 0, 0, 0);
    const ndy = numericalDeriv(0, 0, 0, 1);
    const ndz = numericalDeriv(0, 0, 0, 2);

    expect(maxAbsDiff(dRdwx, ndx)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwy, ndy)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwz, ndz)).toBeLessThan(1e-4);
  });

  it('small rotation [0.1, 0, 0]: analytical matches numerical', () => {
    const [dRdwx, dRdwy, dRdwz] = rodriguesDeriv(0.1, 0, 0);
    const ndx = numericalDeriv(0.1, 0, 0, 0);
    const ndy = numericalDeriv(0.1, 0, 0, 1);
    const ndz = numericalDeriv(0.1, 0, 0, 2);

    expect(maxAbsDiff(dRdwx, ndx)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwy, ndy)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwz, ndz)).toBeLessThan(1e-4);
  });

  it('large rotation [0.5, -0.3, 0.7]: analytical matches numerical', () => {
    const [dRdwx, dRdwy, dRdwz] = rodriguesDeriv(0.5, -0.3, 0.7);
    const ndx = numericalDeriv(0.5, -0.3, 0.7, 0);
    const ndy = numericalDeriv(0.5, -0.3, 0.7, 1);
    const ndz = numericalDeriv(0.5, -0.3, 0.7, 2);

    expect(maxAbsDiff(dRdwx, ndx)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwy, ndy)).toBeLessThan(1e-4);
    expect(maxAbsDiff(dRdwz, ndz)).toBeLessThan(1e-4);
  });

  it('10 random rotations: median max-error < 1e-4', () => {
    // Deterministic seed via a simple LCG so the test is reproducible
    let seed = 42;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
      return (seed / 0x7fffffff) - 0.5; // [-0.5, 0.5)
    };

    const maxErrors: number[] = [];

    for (let trial = 0; trial < 10; trial++) {
      const wx = rand() * 2;
      const wy = rand() * 2;
      const wz = rand() * 2;

      const [dRdwx, dRdwy, dRdwz] = rodriguesDeriv(wx, wy, wz);
      const ndx = numericalDeriv(wx, wy, wz, 0);
      const ndy = numericalDeriv(wx, wy, wz, 1);
      const ndz = numericalDeriv(wx, wy, wz, 2);

      const e = Math.max(
        maxAbsDiff(dRdwx, ndx),
        maxAbsDiff(dRdwy, ndy),
        maxAbsDiff(dRdwz, ndz),
      );
      maxErrors.push(e);
    }

    maxErrors.sort((a, b) => a - b);
    const median = maxErrors[Math.floor(maxErrors.length / 2)]!;
    expect(median).toBeLessThan(1e-4);
  });

  it('dR matrices have determinant close to 0 (they are Jacobians, not rotation matrices)', () => {
    const [dRdwx, dRdwy, dRdwz] = rodriguesDeriv(0.3, -0.4, 0.5);
    // det of a derivative matrix has no reason to be 1 — it should generally be
    // much smaller in magnitude than 1.  We just verify it is not ≈ 1.
    const detX = Math.abs(mat3Det(dRdwx));
    const detY = Math.abs(mat3Det(dRdwy));
    const detZ = Math.abs(mat3Det(dRdwz));
    // Each det should be nowhere near 1 (they are tiny — O(1e-2) or less)
    expect(detX).toBeLessThan(0.1);
    expect(detY).toBeLessThan(0.1);
    expect(detZ).toBeLessThan(0.1);
  });
});

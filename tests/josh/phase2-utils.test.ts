import { describe, it, expect } from 'vitest';
import {
  mat3ToQuat,
  quatToMat3,
  slerpQuat,
  interpolateSE3,
  interpolatePose,
} from '../../demo/josh/utils/se3-interpolation.ts';
import {
  mat4Multiply,
  concatChunkTrajectory,
} from '../../demo/josh/utils/chunk-concat.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function identityMat3(): Float32Array {
  return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

function identityMat4(): Float32Array {
  // column-major 4×4 identity
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Build a column-major 4×4 pure-translation matrix. */
function translationMat4(tx: number, ty: number, tz: number): Float32Array {
  const T = identityMat4();
  T[12] = tx; T[13] = ty; T[14] = tz;
  return T;
}

/** Build a column-major 4×4 rotation around the Z axis by `angle` radians. */
function rotZMat4(angle: number): Float32Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const T = identityMat4();
  // col 0
  T[0] = c; T[1] = s;
  // col 1
  T[4] = -s; T[5] = c;
  return T;
}

/** Build a row-major 3×3 rotation around Z by `angle` radians. */
function rotZMat3(angle: number): Float32Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([
    c, -s, 0,
    s,  c, 0,
    0,  0, 1,
  ]);
}

function approx(a: number, b: number, tol = 1e-5): boolean {
  return Math.abs(a - b) < tol;
}

// ---------------------------------------------------------------------------
// 1. mat3ToQuat on identity → [0, 0, 0, 1]
// ---------------------------------------------------------------------------
describe('mat3ToQuat', () => {
  it('identity matrix → identity quaternion [0,0,0,1]', () => {
    const q = mat3ToQuat(identityMat3());
    expect(approx(q[0], 0)).toBe(true);
    expect(approx(q[1], 0)).toBe(true);
    expect(approx(q[2], 0)).toBe(true);
    expect(approx(q[3], 1)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 2. quatToMat3([0,0,0,1]) → identity 3×3
// ---------------------------------------------------------------------------
describe('quatToMat3', () => {
  it('identity quaternion → identity 3×3 matrix', () => {
    const m = quatToMat3([0, 0, 0, 1]);
    const id = identityMat3();
    for (let i = 0; i < 9; i++) {
      expect(approx(m[i], id[i])).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 3. Round-trip: rotation matrix → quat → mat3 should match original
// ---------------------------------------------------------------------------
describe('mat3ToQuat / quatToMat3 round-trip', () => {
  it('45° Z rotation round-trips within 1e-5', () => {
    const original = rotZMat3(Math.PI / 4);
    const q = mat3ToQuat(original);
    const recovered = quatToMat3(q);
    for (let i = 0; i < 9; i++) {
      expect(approx(recovered[i], original[i], 1e-5)).toBe(true);
    }
  });

  it('arbitrary rotation round-trips within 1e-5', () => {
    // 60° around X axis, row-major
    const a = Math.PI / 3;
    const c = Math.cos(a), s = Math.sin(a);
    const original = new Float32Array([
      1, 0,  0,
      0, c, -s,
      0, s,  c,
    ]);
    const q = mat3ToQuat(original);
    const recovered = quatToMat3(q);
    for (let i = 0; i < 9; i++) {
      expect(approx(recovered[i], original[i], 1e-5)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 4. slerpQuat at t=0 → q0, at t=1 → q1
// ---------------------------------------------------------------------------
describe('slerpQuat endpoints', () => {
  const q0: [number, number, number, number] = [0, 0, 0, 1];
  const q1: [number, number, number, number] = [0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4)]; // 90° around Z

  it('t=0 returns q0', () => {
    const r = slerpQuat(q0, q1, 0);
    for (let i = 0; i < 4; i++) expect(approx(r[i], q0[i])).toBe(true);
  });

  it('t=1 returns q1 (or its negative)', () => {
    const r = slerpQuat(q0, q1, 1);
    // Accept either q1 or -q1 (both represent the same rotation)
    const pos = q1.every((v, i) => approx(r[i], v));
    const neg = q1.every((v, i) => approx(r[i], -v));
    expect(pos || neg).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 5. slerpQuat 90° Z rotation at t=0.5 → 45° (z ≈ sin(22.5°) ≈ 0.1951)
// ---------------------------------------------------------------------------
describe('slerpQuat midpoint', () => {
  it('90° Z rotation at t=0.5 gives 45° rotation (z ≈ sin(22.5°))', () => {
    const q0: [number, number, number, number] = [0, 0, 0, 1];
    // 90° around Z: [0, 0, sin(45°), cos(45°)]
    const q1: [number, number, number, number] = [
      0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4),
    ];
    const r = slerpQuat(q0, q1, 0.5);
    // Expected: 45° around Z → [0, 0, sin(22.5°), cos(22.5°)]
    const expected_z = Math.sin(Math.PI / 8); // ≈ 0.3827
    expect(approx(r[0], 0, 1e-5)).toBe(true);
    expect(approx(r[1], 0, 1e-5)).toBe(true);
    expect(approx(Math.abs(r[2]), expected_z, 1e-5)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 6. interpolateSE3 between identity and T(2,0,0) at t=0.5 → T(1,0,0)
// ---------------------------------------------------------------------------
describe('interpolateSE3', () => {
  it('midpoint between identity and T(2,0,0) gives T(1,0,0)', () => {
    const T0 = identityMat4();
    const T1 = translationMat4(2, 0, 0);
    const Tm = interpolateSE3(T0, T1, 0.5);
    // Translation components are at indices 12, 13, 14
    expect(approx(Tm[12], 1)).toBe(true);
    expect(approx(Tm[13], 0)).toBe(true);
    expect(approx(Tm[14], 0)).toBe(true);
    // Rotation should still be identity
    expect(approx(Tm[0], 1)).toBe(true);
    expect(approx(Tm[5], 1)).toBe(true);
    expect(approx(Tm[10], 1)).toBe(true);
  });

  it('midpoint between identity and 90° Z rotation gives 45° rotation', () => {
    const T0 = identityMat4();
    const T1 = rotZMat4(Math.PI / 2);
    const Tm = interpolateSE3(T0, T1, 0.5);
    const c45 = Math.cos(Math.PI / 4);
    const s45 = Math.sin(Math.PI / 4);
    // col-major: T[0]=cos, T[1]=sin, T[4]=-sin, T[5]=cos
    expect(approx(Tm[0], c45, 1e-5)).toBe(true);
    expect(approx(Tm[1], s45, 1e-5)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 7. interpolatePose: midpoint of zeros and ones → all 0.5
// ---------------------------------------------------------------------------
describe('interpolatePose', () => {
  it('midpoint of all-zeros and all-ones → all 0.5', () => {
    const n = 72;
    const pose0 = new Float32Array(n).fill(0);
    const pose1 = new Float32Array(n).fill(1);
    const result = interpolatePose(pose0, pose1, 0.5);
    for (let i = 0; i < n; i++) {
      expect(approx(result[i], 0.5)).toBe(true);
    }
  });

  it('t=0 returns pose0', () => {
    const pose0 = new Float32Array([1, 2, 3]);
    const pose1 = new Float32Array([4, 5, 6]);
    const r = interpolatePose(pose0, pose1, 0);
    expect(Array.from(r)).toEqual([1, 2, 3]);
  });

  it('t=1 returns pose1', () => {
    const pose0 = new Float32Array([1, 2, 3]);
    const pose1 = new Float32Array([4, 5, 6]);
    const r = interpolatePose(pose0, pose1, 1);
    expect(Array.from(r)).toEqual([4, 5, 6]);
  });
});

// ---------------------------------------------------------------------------
// 8. mat4Multiply: identity × anything = anything
// ---------------------------------------------------------------------------
describe('mat4Multiply', () => {
  it('identity × T(3,4,5) = T(3,4,5)', () => {
    const I = identityMat4();
    const T = translationMat4(3, 4, 5);
    const result = mat4Multiply(I, T);
    for (let i = 0; i < 16; i++) {
      expect(approx(result[i], T[i])).toBe(true);
    }
  });

  it('T × identity = T', () => {
    const I = identityMat4();
    const T = translationMat4(1, 2, 3);
    const result = mat4Multiply(T, I);
    for (let i = 0; i < 16; i++) {
      expect(approx(result[i], T[i])).toBe(true);
    }
  });

  it('T(1,0,0) × T(2,0,0) = T(3,0,0)', () => {
    const T1 = translationMat4(1, 0, 0);
    const T2 = translationMat4(2, 0, 0);
    const result = mat4Multiply(T1, T2);
    expect(approx(result[12], 3)).toBe(true);
    expect(approx(result[13], 0)).toBe(true);
    expect(approx(result[14], 0)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 9. concatChunkTrajectory: chunk1=[I], chunk2=[T(1,0,0)] → T(1,0,0) in world
// ---------------------------------------------------------------------------
describe('concatChunkTrajectory', () => {
  it('chunk1=[I], chunk2=[T(1,0,0)] → result contains T(1,0,0) anchored at I', () => {
    const I = identityMat4();
    const T1 = translationMat4(1, 0, 0);
    const result = concatChunkTrajectory([I], [T1]);
    expect(result.length).toBe(2);
    // First element is unchanged I
    for (let i = 0; i < 16; i++) expect(approx(result[0][i], I[i])).toBe(true);
    // Second element = I × T(1,0,0) = T(1,0,0)
    expect(approx(result[1][12], 1)).toBe(true);
    expect(approx(result[1][13], 0)).toBe(true);
    expect(approx(result[1][14], 0)).toBe(true);
  });

  it('chunk2 with anchor T(5,0,0) shifts chunk2 poses to world frame', () => {
    const anchor = translationMat4(5, 0, 0);
    const local = translationMat4(2, 0, 0);
    const result = concatChunkTrajectory([identityMat4(), anchor], [local]);
    // anchor × T(2,0,0) = T(7,0,0)
    const world = result[result.length - 1];
    expect(approx(world[12], 7)).toBe(true);
  });

  it('empty chunk1 returns chunk2 unchanged', () => {
    const T = translationMat4(3, 0, 0);
    const result = concatChunkTrajectory([], [T]);
    expect(result.length).toBe(1);
    for (let i = 0; i < 16; i++) expect(approx(result[0][i], T[i])).toBe(true);
  });

  it('empty chunk2 returns chunk1 unchanged', () => {
    const I = identityMat4();
    const result = concatChunkTrajectory([I], []);
    expect(result.length).toBe(1);
  });
});

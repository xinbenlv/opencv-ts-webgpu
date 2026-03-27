/**
 * Minimal 4x4 matrix math helpers for WebGPU rendering.
 * All matrices are column-major Float32Array[16], matching WGSL mat4x4<f32>.
 */

export function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/**
 * Perspective projection (column-major, WebGPU NDC: depth [0,1]).
 */
export function mat4Perspective(
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1.0 / Math.tan(fovY / 2);
  const rangeInv = 1.0 / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0,          f, 0, 0,
    0,          0, far * rangeInv,        -1,
    0,          0, far * near * rangeInv,  0,
  ]);
}

/**
 * Look-at view matrix (column-major).
 */
export function mat4LookAt(
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number],
): Float32Array {
  const ex = eye[0], ey = eye[1], ez = eye[2];
  const tx = target[0], ty = target[1], tz = target[2];
  const ux = up[0], uy = up[1], uz = up[2];

  // forward = normalize(eye - target)
  let fx = ex - tx, fy = ey - ty, fz = ez - tz;
  let len = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= len; fy /= len; fz /= len;

  // right = normalize(cross(up, forward))
  let rx = uy * fz - uz * fy;
  let ry = uz * fx - ux * fz;
  let rz = ux * fy - uy * fx;
  len = Math.sqrt(rx * rx + ry * ry + rz * rz);
  rx /= len; ry /= len; rz /= len;

  // true up = cross(forward, right)
  const ux2 = fy * rz - fz * ry;
  const uy2 = fz * rx - fx * rz;
  const uz2 = fx * ry - fy * rx;

  // prettier-ignore
  return new Float32Array([
    rx,              ux2,             fx,              0,
    ry,              uy2,             fy,              0,
    rz,              uz2,             fz,              0,
    -(rx*ex+ry*ey+rz*ez), -(ux2*ex+uy2*ey+uz2*ez), -(fx*ex+fy*ey+fz*ez), 1,
  ]);
}

/**
 * Multiply two column-major 4x4 matrices: result = a * b.
 */
export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row]! * b[col * 4 + k]!;
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

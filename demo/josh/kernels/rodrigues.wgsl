// Rodrigues forward pass: axis-angle → rotation matrices for all joints
//
// For each of 24 joints, converts pose axis-angle [wx, wy, wz] to a 3×3
// rotation matrix using the numerically stable sinc/cosc formulation:
//
//   R = I + sinc(θ)·K + cosc(θ)·K²
//
// where K is the skew-symmetric cross-product matrix and:
//   sinc(θ) = sin(θ)/θ   (limit 1 at θ=0)
//   cosc(θ) = (1-cos(θ))/θ²  (limit 1/2 at θ=0)
//
// Output is row-major [R00,R01,R02, R10,R11,R12, R20,R21,R22] to match
// the convention expected by smpl-joints.wgsl.

@group(0) @binding(0) var<storage, read> pose: array<f32>;            // [72] = 24 joints × 3
@group(0) @binding(1) var<storage, read_write> local_rots: array<f32>; // [24, 9] row-major

@compute @workgroup_size(24, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= 24u) { return; }

  let wx = pose[j * 3u];
  let wy = pose[j * 3u + 1u];
  let wz = pose[j * 3u + 2u];

  let eps: f32 = 1e-12;
  let theta2 = wx*wx + wy*wy + wz*wz;
  let theta  = sqrt(theta2 + eps);

  // sinc(θ) = sin(θ)/θ, smooth at θ=0 (limit = 1)
  let s = sin(theta) / (theta + eps);
  // cosc(θ) = (1-cos(θ))/θ², smooth at θ=0 (limit = 1/2)
  let c = (1.0 - cos(theta)) / (theta2 + eps);

  // Precompute cross products for K² diagonal terms
  let wxwy = wx * wy;
  let wxwz = wx * wz;
  let wywz = wy * wz;
  let wx2  = wx * wx;
  let wy2  = wy * wy;
  let wz2  = wz * wz;

  // R = I + s·K + c·K²  (row-major storage)
  // K  = [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]
  // K² = [[-(wy²+wz²), wxwy, wxwz],
  //        [wxwy, -(wx²+wz²), wywz],
  //        [wxwz, wywz, -(wx²+wy²)]]

  let out_base = j * 9u;

  // Row 0
  local_rots[out_base + 0u] = 1.0 + c * (-(wy2 + wz2));  // R[0,0]
  local_rots[out_base + 1u] = -s * wz + c * wxwy;          // R[0,1]
  local_rots[out_base + 2u] =  s * wy + c * wxwz;          // R[0,2]
  // Row 1
  local_rots[out_base + 3u] =  s * wz + c * wxwy;          // R[1,0]
  local_rots[out_base + 4u] = 1.0 + c * (-(wx2 + wz2));  // R[1,1]
  local_rots[out_base + 5u] = -s * wx + c * wywz;          // R[1,2]
  // Row 2
  local_rots[out_base + 6u] = -s * wy + c * wxwz;          // R[2,0]
  local_rots[out_base + 7u] =  s * wx + c * wywz;          // R[2,1]
  local_rots[out_base + 8u] = 1.0 + c * (-(wx2 + wy2));  // R[2,2]
}

// JOSH 2D Reprojection Loss (L_2D) — Phase 0C
//
// Penalises discrepancy between projected SMPL joints and detected 2D keypoints:
//   L_2D = weight × Σ_k confidence_k × ‖π(j_k, K) - p_k‖²
//
// where π(j, K) = K × j / j.z is the pinhole projection, j_k is the 3D SMPL
// joint corresponding to keypoint k, and p_k is the detected 2D location.
//
// Outputs gradient dL_2D/dj [24×3] additively into dl_djoint buffer.
// Enabled only in Stage 2 of JOSH optimisation (iterations 500-699).

struct Params {
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  num_keypoints: u32,
  weight: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> joint_positions: array<f32>;  // [24*3]
@group(0) @binding(1) var<storage, read> keypoints_2d: array<f32>;     // [K*2] pixel coords (u, v)
@group(0) @binding(2) var<storage, read> confidences: array<f32>;      // [K]
@group(0) @binding(3) var<storage, read> joint_to_smpl: array<u32>;    // [K] keypoint → SMPL joint idx
@group(0) @binding(4) var<storage, read_write> dl_djoint: array<f32>;  // [24*3] gradient
@group(0) @binding(5) var<storage, read_write> loss_out: array<f32>;   // [1]
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ki = gid.x;
  if (ki >= p.num_keypoints) { return; }

  let conf = confidences[ki];
  if (conf < 0.2) { return; }  // Skip low-confidence detections

  let smpl_j = joint_to_smpl[ki];
  if (smpl_j >= 24u) { return; }

  let jx = joint_positions[smpl_j * 3u];
  let jy = joint_positions[smpl_j * 3u + 1u];
  let jz = joint_positions[smpl_j * 3u + 2u];

  if (jz <= 0.01) { return; }  // Behind camera

  // Project 3D joint to 2D pixel: π = (f × X/Z + cx, f × Y/Z + cy)
  let proj_u = p.fx * jx / jz + p.cx;
  let proj_v = p.fy * jy / jz + p.cy;

  let det_u = keypoints_2d[ki * 2u];
  let det_v = keypoints_2d[ki * 2u + 1u];

  let err_u = proj_u - det_u;
  let err_v = proj_v - det_v;
  let sq_err = err_u*err_u + err_v*err_v;

  // dL/dj using chain rule through the projection π:
  //   d(proj_u)/djx = fx/jz,  d(proj_u)/djz = -fx*jx/jz²
  //   d(proj_v)/djy = fy/jz,  d(proj_v)/djz = -fy*jy/jz²
  let jz2 = jz * jz;
  let factor = 2.0 * p.weight * conf;

  let dL_djx = factor * err_u * (p.fx / jz);
  let dL_djy = factor * err_v * (p.fy / jz);
  let dL_djz = factor * (err_u * (-p.fx * jx / jz2) + err_v * (-p.fy * jy / jz2));

  dl_djoint[smpl_j * 3u]      += dL_djx;
  dl_djoint[smpl_j * 3u + 1u] += dL_djy;
  dl_djoint[smpl_j * 3u + 2u] += dL_djz;

  loss_out[0] += p.weight * conf * sq_err;
}

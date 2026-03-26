// JOSH Depth Reprojection Loss Gradient Kernel
//
// Computes gradient of depth reprojection loss: penalizes SMPL mesh vertices
// whose projected depth doesn't match the estimated depth map.
//
// L_reproj = sum_v || z_vertex_v - depth_map(proj(v)) ||^2
//
// This loss ensures the reconstructed human mesh is consistent with
// the monocular depth estimation from Node A.

struct Params {
  width: u32,
  height: u32,
  vertex_count: u32,
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  weight: f32,
}

@group(0) @binding(0) var<storage, read> vertices: array<f32>;       // [V, 3]
@group(0) @binding(1) var<storage, read> depth_map: array<f32>;      // [H, W]
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>; // [param_dim]
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;     // [4]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vid = gid.x;
  if (vid >= params.vertex_count) {
    return;
  }

  let vx = vertices[vid * 3u];
  let vy = vertices[vid * 3u + 1u];
  let vz = vertices[vid * 3u + 2u];

  if (vz <= 0.01) {
    return;
  }

  // Project to image
  let u = params.fx * vx / vz + params.cx;
  let v = params.fy * vy / vz + params.cy;

  let ui = i32(round(u));
  let vi_img = i32(round(v));

  if (ui < 0 || ui >= i32(params.width) || vi_img < 0 || vi_img >= i32(params.height)) {
    return;
  }

  let depth_idx = u32(vi_img) * params.width + u32(ui);
  let scene_depth = depth_map[depth_idx];

  // Skip invalid depth values
  if (scene_depth <= 0.1 || scene_depth >= 99.0) {
    return;
  }

  // Reprojection error
  let error = vz - scene_depth;
  let reproj_loss = params.weight * error * error;

  // Accumulate loss component
  loss[1] += reproj_loss;

  // Gradient contribution to depth scale (parameter 88)
  // d(loss)/d(scale) = weight * 2 * error * d(error)/d(scale)
  gradient[88] += params.weight * 2.0 * error * (-scene_depth);

  // Gradient contribution to camera translation Z (parameter 84)
  gradient[84] += params.weight * 2.0 * error;
}

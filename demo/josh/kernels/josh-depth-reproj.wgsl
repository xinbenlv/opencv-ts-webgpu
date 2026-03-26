// JOSH Depth Reprojection Loss (L_3D) — Phase 0C
//
// Computes gradient of the 3D depth correspondence loss: penalises SMPL
// mesh vertices whose projected depth doesn't match the MASt3R/MiDAS depth map.
//
//   L_3D = Σ_v ‖v_z - depth_map(proj(v))‖²
//
// Gradient dL_3D/dv is output additively into dl_dv [V,3].
// Only the depth (Z) component of each vertex receives gradient.

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
@group(0) @binding(2) var<storage, read_write> dl_dv: array<f32>;    // [V*3] per-vertex gradient
@group(0) @binding(3) var<storage, read_write> loss_out: array<f32>; // [1]
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vid = gid.x;
  if (vid >= p.vertex_count) { return; }

  let vx = vertices[vid * 3u];
  let vy = vertices[vid * 3u + 1u];
  let vz = vertices[vid * 3u + 2u];

  if (vz <= 0.01) { return; }

  // Project vertex to image plane
  let u = p.fx * vx / vz + p.cx;
  let v = p.fy * vy / vz + p.cy;

  let ui = i32(round(u));
  let vi_img = i32(round(v));

  if (ui < 0 || ui >= i32(p.width) || vi_img < 0 || vi_img >= i32(p.height)) { return; }

  let depth_idx = u32(vi_img) * p.width + u32(ui);
  let scene_z = depth_map[depth_idx];
  if (scene_z <= 0.1 || scene_z >= 99.0) { return; }

  // Depth residual
  let err = vz - scene_z;

  // Loss accumulation
  loss_out[0] += p.weight * err * err;

  // Gradient dL/dvz = 2 × weight × err
  dl_dv[vid * 3u + 2u] += 2.0 * p.weight * err;
}

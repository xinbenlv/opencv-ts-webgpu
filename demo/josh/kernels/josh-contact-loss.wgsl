// JOSH Contact Scale Loss (L_c1) — Phase 0C
//
// Penalises contact vertices (foot soles, toes) that are too far from the
// scene depth surface. The loss uses a hinge (ReLU²) formulation:
//
//   L_c1 = wc1 × Σ_i max(0, |v_i_z - z_scene(proj(v_i))| - Δc1)²
//
// Outputs per-vertex gradient dL_c1/dv additively into dl_dv [V,3] buffer.
// Only the Z component of the gradient is non-zero for depth-based contact.

struct Params {
  width: u32,
  height: u32,
  num_contacts: u32,
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  weight: f32,     // wc1
  delta_c1: f32,   // Δc1 = 0 (contact must be exact)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> vertices: array<f32>;         // [V, 3]
@group(0) @binding(1) var<storage, read> depth_map: array<f32>;        // [H, W]
@group(0) @binding(2) var<storage, read> contact_indices: array<u32>;  // [num_contacts]
@group(0) @binding(3) var<storage, read_write> dl_dv: array<f32>;      // [V*3] per-vertex gradient
@group(0) @binding(4) var<storage, read_write> loss_out: array<f32>;   // [1] loss accumulator
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= p.num_contacts) { return; }

  let vi = contact_indices[ci];
  let vx = vertices[vi * 3u];
  let vy = vertices[vi * 3u + 1u];
  let vz = vertices[vi * 3u + 2u];

  if (vz <= 0.01) { return; }  // Behind camera

  // Project contact vertex to image plane
  let u = p.fx * vx / vz + p.cx;
  let v = p.fy * vy / vz + p.cy;

  let ui = i32(round(u));
  let vi_img = i32(round(v));

  if (ui < 0 || ui >= i32(p.width) || vi_img < 0 || vi_img >= i32(p.height)) { return; }

  let depth_idx = u32(vi_img) * p.width + u32(ui);
  let scene_z = depth_map[depth_idx];
  if (scene_z <= 0.1 || scene_z >= 99.0) { return; }

  // Signed depth residual (positive = vertex in front of surface)
  let diff = vz - scene_z;
  let excess = abs(diff) - p.delta_c1;
  if (excess <= 0.0) { return; }

  // Loss: wc1 × (|diff| - Δc1)²
  loss_out[0] += p.weight * excess * excess;

  // Gradient dL/dvz = 2 × wc1 × excess × sign(diff)
  let grad_z = 2.0 * p.weight * excess * sign(diff);
  dl_dv[vi * 3u + 2u] += grad_z;
}

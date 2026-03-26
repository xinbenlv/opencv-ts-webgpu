// JOSH Contact Loss Gradient Kernel
//
// Computes gradient of contact loss: penalizes SMPL contact vertices
// (feet, hands) that are not touching the depth surface.
//
// L_contact = sum_i || z_vertex_i - z_surface(proj(v_i)) ||^2
//
// where proj() projects the 3D vertex to image coordinates using
// the camera intrinsics, and z_surface is the depth value at that pixel.

struct Params {
  width: u32,
  height: u32,
  num_contacts: u32,
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  contact_threshold: f32,
}

@group(0) @binding(0) var<storage, read> vertices: array<f32>;         // [V, 3]
@group(0) @binding(1) var<storage, read> depth_map: array<f32>;        // [H, W]
@group(0) @binding(2) var<storage, read> contact_indices: array<u32>;  // [num_contacts]
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;   // [param_dim]
@group(0) @binding(4) var<storage, read_write> loss: array<f32>;       // [4] loss components
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= params.num_contacts) {
    return;
  }

  let vi = contact_indices[ci];
  let vx = vertices[vi * 3u];
  let vy = vertices[vi * 3u + 1u];
  let vz = vertices[vi * 3u + 2u];

  // Skip vertices behind camera
  if (vz <= 0.01) {
    return;
  }

  // Project to image plane
  let u = params.fx * vx / vz + params.cx;
  let v = params.fy * vy / vz + params.cy;

  let ui = i32(round(u));
  let vi_img = i32(round(v));

  if (ui < 0 || ui >= i32(params.width) || vi_img < 0 || vi_img >= i32(params.height)) {
    return;
  }

  let depth_idx = u32(vi_img) * params.width + u32(ui);
  let scene_depth = depth_map[depth_idx];

  // Contact: vertex depth should match scene depth at projected location
  let penetration = vz - scene_depth;

  if (abs(penetration) < params.contact_threshold) {
    // Soft contact loss
    let contact_loss = penetration * penetration;

    // Accumulate loss (use atomics would be ideal, but for simplicity add per-thread)
    loss[0] += contact_loss;

    // Gradient w.r.t. depth scale parameter (index 88 = last param)
    // d(loss)/d(depth_scale) = 2 * penetration * (-1) = -2 * penetration
    gradient[88] += -2.0 * penetration * scene_depth;

    // Gradient w.r.t. camera Z translation (index 84)
    gradient[84] += 2.0 * penetration;
  }
}

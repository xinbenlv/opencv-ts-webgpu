// JOSH Contact Static Loss (L_c2) — Phase 0C
//
// Penalises movement of contact vertices between consecutive frames.
// Contact vertices (foot soles, toes) should remain stationary once
// they have made contact with the ground.
//
//   L_c2 = wc2 × Σ_i max(0, ‖v_i(t) - v_i(t-1)‖ - Δc2)²
//
// Outputs per-vertex gradient dL_c2/dv [V,3] additively into dl_dv buffer.
// Only contact vertices receive non-zero gradient.

struct Params {
  num_contacts: u32,
  weight: f32,      // wc2 = 20
  delta_c2: f32,    // Δc2 = 0.1m threshold
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> vertices_curr: array<f32>;    // [V, 3]
@group(0) @binding(1) var<storage, read> vertices_prev: array<f32>;    // [V, 3]
@group(0) @binding(2) var<storage, read> contact_indices: array<u32>;  // [num_contacts]
@group(0) @binding(3) var<storage, read_write> dl_dv: array<f32>;      // [V*3] per-vertex gradient
@group(0) @binding(4) var<storage, read_write> loss_out: array<f32>;   // [1]
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ci = gid.x;
  if (ci >= p.num_contacts) { return; }

  let vi = contact_indices[ci];
  let cx = vertices_curr[vi * 3u];
  let cy = vertices_curr[vi * 3u + 1u];
  let cz = vertices_curr[vi * 3u + 2u];
  let px = vertices_prev[vi * 3u];
  let py = vertices_prev[vi * 3u + 1u];
  let pz = vertices_prev[vi * 3u + 2u];

  let diff = vec3<f32>(cx - px, cy - py, cz - pz);
  let dist = length(diff);

  // Hinge loss: only penalise movement beyond delta_c2
  let excess = dist - p.delta_c2;
  if (excess <= 0.0 || dist < 1e-8) { return; }

  // dL/dv_i = 2 × wc2 × excess × (diff / dist)
  let factor = 2.0 * p.weight * excess / dist;
  dl_dv[vi * 3u]      += factor * diff.x;
  dl_dv[vi * 3u + 1u] += factor * diff.y;
  dl_dv[vi * 3u + 2u] += factor * diff.z;

  // Loss accumulation (one thread per contact vertex, potential race — acceptable for approximate scalar)
  loss_out[0] += p.weight * excess * excess;
}

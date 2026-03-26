// SMPL Prior Loss (L_p) — Phase 0C
//
// Penalises deviation of pose/shape parameters from their initial values:
//   L_p = weight × (‖θ - θ₀‖² + ‖β - β₀‖²)
//
// Gradient is simple:
//   dL_p/dθ_i = 2 × weight × (θ_i - θ₀_i)
//   dL_p/dβ_i = 2 × weight × (β_i - β₀_i)
//
// Writes gradients additively into the full 89-dim gradient buffer:
//   [0..71]  pose gradients
//   [72..81] shape gradients

struct Params {
  weight: f32,
  param_count: u32,  // 82 = 72 pose + 10 shape
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> pose: array<f32>;         // [72]
@group(0) @binding(1) var<storage, read> init_pose: array<f32>;    // [72]
@group(0) @binding(2) var<storage, read> shape: array<f32>;        // [10]
@group(0) @binding(3) var<storage, read> init_shape: array<f32>;   // [10]
@group(0) @binding(4) var<storage, read_write> gradient: array<f32>; // [89+] full grad buffer
@group(0) @binding(5) var<storage, read_write> loss_out: array<f32>; // [1] accumulates loss
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(82, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= 82u) { return; }

  var diff: f32;
  if (idx < 72u) {
    diff = pose[idx] - init_pose[idx];
  } else {
    diff = shape[idx - 72u] - init_shape[idx - 72u];
  }

  // Gradient: 2 × weight × diff
  gradient[idx] += 2.0 * params.weight * diff;

  // Loss accumulation (only thread 0 computes the full sum to avoid races)
  if (idx == 0u) {
    var total = 0.0;
    for (var i = 0u; i < 72u; i++) {
      let d = pose[i] - init_pose[i];
      total += d * d;
    }
    for (var i = 0u; i < 10u; i++) {
      let d = shape[i] - init_shape[i];
      total += d * d;
    }
    loss_out[0] += params.weight * total;
  }
}

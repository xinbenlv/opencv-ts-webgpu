// JOSH Temporal Smoothness Loss Gradient Kernel
//
// Computes gradient of temporal smoothness loss: penalizes large changes
// in optimization parameters between consecutive frames.
//
// L_temporal = weight * sum_i || x_i - x_prev_i ||^2
//
// This prevents jitter and ensures smooth motion reconstruction.

struct Params {
  param_dim: u32,
  weight: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> current_params: array<f32>;  // [param_dim]
@group(0) @binding(1) var<storage, read> prev_params: array<f32>;     // [param_dim]
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;  // [param_dim]
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;      // [4]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.param_dim) {
    return;
  }

  let diff = current_params[idx] - prev_params[idx];

  // Gradient: d/dx_i = 2 * weight * (x_i - x_prev_i)
  gradient[idx] += params.weight * 2.0 * diff;

  // Write per-param loss to slot 2 (first thread only, approximate)
  if (idx == 0u) {
    var total_loss = 0.0;
    for (var i = 0u; i < params.param_dim; i++) {
      let d = current_params[i] - prev_params[i];
      total_loss += params.weight * d * d;
    }
    loss[2] = total_loss;
  }
}

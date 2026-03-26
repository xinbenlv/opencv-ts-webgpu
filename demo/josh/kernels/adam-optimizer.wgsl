// Adam Optimizer — GPU-native implementation
//
// Runs entirely on GPU with no CPU readback during the optimization loop.
// Standard Adam update per parameter:
//   m = β₁ × m + (1-β₁) × g
//   v = β₂ × v + (1-β₂) × g²
//   m̂ = m / (1 - β₁ᵗ)
//   v̂ = v / (1 - β₂ᵗ)
//   θ = θ - lr × m̂ / (√v̂ + ε)
//
// Timestep t is read from a storage buffer (counter[0]) which is incremented
// by increment-counter.wgsl before each Adam step. This allows 700+ iterations
// to be encoded into a single command encoder without CPU involvement.

struct AdamConfig {
  lr: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  param_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;    // θ [N]
@group(0) @binding(1) var<storage, read> gradients: array<f32>;       // g [N]
@group(0) @binding(2) var<storage, read_write> m: array<f32>;         // first moment [N]
@group(0) @binding(3) var<storage, read_write> v: array<f32>;         // second moment [N]
@group(0) @binding(4) var<uniform> config: AdamConfig;
@group(0) @binding(5) var<storage, read> timestep: array<u32>;        // [1] incremented by counter kernel

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= config.param_count) { return; }

  let g = gradients[idx];

  // Update biased first moment estimate
  let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * g;
  m[idx] = m_new;

  // Update biased second raw moment estimate
  let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * g * g;
  v[idx] = v_new;

  // Bias correction using GPU-tracked timestep (t ≥ 1 after first increment)
  let t_f = max(f32(timestep[0]), 1.0);
  let m_hat = m_new / (1.0 - pow(config.beta1, t_f));
  let v_hat = v_new / (1.0 - pow(config.beta2, t_f));

  // Parameter update
  params[idx] = params[idx] - config.lr * m_hat / (sqrt(v_hat) + config.epsilon);
}

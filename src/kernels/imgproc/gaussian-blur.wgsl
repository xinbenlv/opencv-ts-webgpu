// Gaussian Blur 5x5 compute shader
// Applies a separable Gaussian blur to a 2D f32 image.

struct Params {
  width: u32,
  height: u32,
  sigma: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// 5-tap Gaussian weights (precomputed for sigma ~1.0)
const KERNEL_SIZE: u32 = 5u;
const HALF_K: i32 = 2;
const WEIGHTS: array<f32, 5> = array<f32, 5>(
  0.06136, 0.24477, 0.38774, 0.24477, 0.06136
);

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;

  if (x >= w || y >= h) {
    return;
  }

  var sum: f32 = 0.0;

  // Horizontal pass (simplified combined 2D kernel for clarity)
  for (var ky: i32 = -HALF_K; ky <= HALF_K; ky = ky + 1) {
    for (var kx: i32 = -HALF_K; kx <= HALF_K; kx = kx + 1) {
      let sx = clamp(i32(x) + kx, 0, i32(w) - 1);
      let sy = clamp(i32(y) + ky, 0, i32(h) - 1);
      let idx = u32(sy) * w + u32(sx);
      let weight = WEIGHTS[u32(kx + HALF_K)] * WEIGHTS[u32(ky + HALF_K)];
      sum = sum + input[idx] * weight;
    }
  }

  let outIdx = y * w + x;
  output[outIdx] = sum;
}

// MiDAS/DPT depth estimation preprocessing
// Normalizes RGB input to model-expected range and layout.
// Input: RGB f32 [H, W, 3] in [0, 1]
// Output: Normalized f32 [H, W, 3] with ImageNet mean/std

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// ImageNet normalization constants
const MEAN: vec3<f32> = vec3<f32>(0.485, 0.456, 0.406);
const STD: vec3<f32> = vec3<f32>(0.229, 0.224, 0.225);

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  let total = params.width * params.height;

  if (pixel >= total) {
    return;
  }

  let base = pixel * 3u;
  let r = (input[base] - MEAN.x) / STD.x;
  let g = (input[base + 1u] - MEAN.y) / STD.y;
  let b = (input[base + 2u] - MEAN.z) / STD.z;

  output[base] = r;
  output[base + 1u] = g;
  output[base + 2u] = b;
}

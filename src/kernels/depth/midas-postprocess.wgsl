// MiDAS/DPT depth estimation postprocessing
// Converts raw inverse depth output to metric depth with scale recovery.
// Input: Raw model output f32 [H, W]
// Output: Depth in meters f32 [H, W]

struct Params {
  width: u32,
  height: u32,
  scale: f32,       // Depth scale factor
  shift: f32,       // Depth shift
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  let total = params.width * params.height;

  if (pixel >= total) {
    return;
  }

  // Convert inverse depth to depth
  let inv_depth = input[pixel];
  let depth = params.scale / max(inv_depth + params.shift, 0.001);

  // Clamp to reasonable range (0.1m to 100m)
  output[pixel] = clamp(depth, 0.1, 100.0);
}

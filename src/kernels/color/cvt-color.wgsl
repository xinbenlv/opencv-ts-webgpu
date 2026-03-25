// Color conversion compute shader
// Supports RGB → Grayscale (BT.709 luminance coefficients)

struct Params {
  width: u32,
  height: u32,
  mode: u32,   // 0 = RGB→Gray, 1 = Gray→RGB
  _pad: u32,
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

  if (params.mode == 0u) {
    // RGB → Grayscale (BT.709)
    let base = pixel * 3u;
    let r = input[base];
    let g = input[base + 1u];
    let b = input[base + 2u];
    output[pixel] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  } else {
    // Grayscale → RGB (replicate)
    let val = input[pixel];
    let base = pixel * 3u;
    output[base] = val;
    output[base + 1u] = val;
    output[base + 2u] = val;
  }
}

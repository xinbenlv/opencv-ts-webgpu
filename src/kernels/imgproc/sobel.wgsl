// Sobel edge detection compute shader
// Computes gradient magnitude from a 2D f32 grayscale image.

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample(x: i32, y: i32) -> f32 {
  let sx = clamp(x, 0, i32(params.width) - 1);
  let sy = clamp(y, 0, i32(params.height) - 1);
  return input[u32(sy) * params.width + u32(sx)];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);

  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  // Sobel X kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
  let gx = -sample(x - 1, y - 1) + sample(x + 1, y - 1)
          - 2.0 * sample(x - 1, y) + 2.0 * sample(x + 1, y)
          - sample(x - 1, y + 1) + sample(x + 1, y + 1);

  // Sobel Y kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
  let gy = -sample(x - 1, y - 1) - 2.0 * sample(x, y - 1) - sample(x + 1, y - 1)
          + sample(x - 1, y + 1) + 2.0 * sample(x, y + 1) + sample(x + 1, y + 1);

  let magnitude = sqrt(gx * gx + gy * gy);
  output[gid.y * params.width + gid.x] = magnitude;
}

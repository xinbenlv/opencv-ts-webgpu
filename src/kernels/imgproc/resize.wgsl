// Bilinear resize compute shader
// Resizes a 2D f32 image to target dimensions using bilinear interpolation.

struct Params {
  src_width: u32,
  src_height: u32,
  dst_width: u32,
  dst_height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dx = gid.x;
  let dy = gid.y;

  if (dx >= params.dst_width || dy >= params.dst_height) {
    return;
  }

  // Map destination pixel to source coordinates
  let sx = (f32(dx) + 0.5) * f32(params.src_width) / f32(params.dst_width) - 0.5;
  let sy = (f32(dy) + 0.5) * f32(params.src_height) / f32(params.dst_height) - 0.5;

  let x0 = u32(max(floor(sx), 0.0));
  let y0 = u32(max(floor(sy), 0.0));
  let x1 = min(x0 + 1u, params.src_width - 1u);
  let y1 = min(y0 + 1u, params.src_height - 1u);

  let fx = sx - floor(sx);
  let fy = sy - floor(sy);

  let v00 = input[y0 * params.src_width + x0];
  let v10 = input[y0 * params.src_width + x1];
  let v01 = input[y1 * params.src_width + x0];
  let v11 = input[y1 * params.src_width + x1];

  let value = v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy;

  output[dy * params.dst_width + dx] = value;
}

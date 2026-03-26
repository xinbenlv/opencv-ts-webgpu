// Point-cloud vertex shader.
// Each logical "point" is drawn as 6 vertices (2 triangles = 1 quad).
// The vertex index within the quad (0-5) selects the corner offset.

struct Uniforms {
  mvp:        mat4x4<f32>,
  point_size: f32,
  _p0:        f32,
  _p1:        f32,
  _p2:        f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>; // [N * 3]
@group(0) @binding(2) var<storage, read> colors:    array<f32>; // [N * 3]

struct VertexOutput {
  @builtin(position) clip_pos:   vec4<f32>,
  @location(0)       color:      vec3<f32>,
  @location(1)       local_uv:   vec2<f32>, // [-1,1] within the quad
}

// Corner table for two CCW triangles forming a unit quad:
//  0---1
//  |  /|
//  | / |
//  |/  |
//  2---3
// tri0: 0,1,2   tri1: 1,3,2
const QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0,  1.0), // 0
  vec2<f32>( 1.0,  1.0), // 1
  vec2<f32>(-1.0, -1.0), // 2
  vec2<f32>( 1.0,  1.0), // 1
  vec2<f32>( 1.0, -1.0), // 3
  vec2<f32>(-1.0, -1.0), // 2
);

@vertex
fn main(@builtin(vertex_index) vid: u32) -> VertexOutput {
  let point_idx   = vid / 6u;
  let corner_idx  = vid % 6u;

  let px = positions[point_idx * 3u + 0u];
  let py = positions[point_idx * 3u + 1u];
  let pz = positions[point_idx * 3u + 2u];

  let clip_center = uniforms.mvp * vec4<f32>(px, py, pz, 1.0);

  // Screen-space offset in NDC: point_size is in pixels; scale by 1/viewport_size
  // We approximate by treating point_size as a NDC fraction multiplied by w.
  // This keeps points a fixed screen size regardless of depth.
  let corner = QUAD_CORNERS[corner_idx];
  // half-size in NDC: uniforms.point_size / 1000.0  (normalised against ~1000px viewport)
  let half_ndc = uniforms.point_size / 1000.0 * clip_center.w;
  let offset   = vec4<f32>(corner * half_ndc, 0.0, 0.0);

  var out: VertexOutput;
  out.clip_pos = clip_center + offset;
  out.color    = vec3<f32>(
    colors[point_idx * 3u + 0u],
    colors[point_idx * 3u + 1u],
    colors[point_idx * 3u + 2u],
  );
  out.local_uv = corner;
  return out;
}

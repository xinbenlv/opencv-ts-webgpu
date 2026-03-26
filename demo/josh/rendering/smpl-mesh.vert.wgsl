struct Uniforms {
  mvp:       mat4x4<f32>,
  model:     mat4x4<f32>,
  light_dir: vec3<f32>,
  _pad:      f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
}

struct VertexOutput {
  @builtin(position) clip_pos:     vec4<f32>,
  @location(0)       world_normal: vec3<f32>,
  @location(1)       world_pos:    vec3<f32>,
}

@vertex
fn main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;

  let world = uniforms.model * vec4<f32>(in.position, 1.0);
  out.clip_pos    = uniforms.mvp * vec4<f32>(in.position, 1.0);
  out.world_pos   = world.xyz;

  // Transform normal by the upper-left 3x3 of the model matrix (no scale here).
  let n = uniforms.model * vec4<f32>(in.normal, 0.0);
  out.world_normal = normalize(n.xyz);

  return out;
}

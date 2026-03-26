struct Uniforms {
  mvp:       mat4x4<f32>,
  model:     mat4x4<f32>,
  light_dir: vec3<f32>,
  _pad:      f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct FragInput {
  @location(0) world_normal: vec3<f32>,
  @location(1) world_pos:    vec3<f32>,
}

// Steel-blue body colour: #7a9fc2
const BODY_COLOR = vec3<f32>(0.478, 0.624, 0.761);

const AMBIENT    = 0.2;
const DIFFUSE    = 0.7;
const SPECULAR   = 0.1;
const SHININESS  = 32.0;

// Fixed camera eye used for specular; matches the JS default orbit distance.
// We pass only mvp/model, so we reconstruct view direction from world_pos
// by embedding the eye position in the uniform pad bytes would require layout
// changes — instead we use a hardcoded approximate eye for specular which is
// acceptable for body-mesh visualisation.
const EYE = vec3<f32>(0.0, 1.0, 3.0);

@fragment
fn main(in: FragInput) -> @location(0) vec4<f32> {
  let N = normalize(in.world_normal);
  let L = normalize(uniforms.light_dir);
  let V = normalize(EYE - in.world_pos);
  let R = reflect(-L, N);

  let ambient  = AMBIENT;
  let diffuse  = DIFFUSE  * max(0.0, dot(N, L));
  let specular = SPECULAR * pow(max(0.0, dot(R, V)), SHININESS);

  let colour = (ambient + diffuse) * BODY_COLOR + vec3<f32>(specular);
  return vec4<f32>(colour, 1.0);
}

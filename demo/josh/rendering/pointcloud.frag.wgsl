// Point-cloud fragment shader.
// Outputs the interpolated per-point colour.
// Discards fragments outside the unit circle so each point looks round.

struct FragInput {
  @location(0) color:    vec3<f32>,
  @location(1) local_uv: vec2<f32>,
}

@fragment
fn main(in: FragInput) -> @location(0) vec4<f32> {
  // Discard pixels outside unit circle centred on the quad
  let dist_sq = dot(in.local_uv, in.local_uv);
  if (dist_sq > 1.0) {
    discard;
  }
  // Soft-edge: slight alpha fade at the border (looks nicer than hard discard)
  let alpha = 1.0 - smoothstep(0.7, 1.0, dist_sq);
  return vec4<f32>(in.color, alpha);
}
